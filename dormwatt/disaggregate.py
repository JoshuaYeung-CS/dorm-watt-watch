"""Whole-home meter -> per-category energy, with classical statistics only.

Model, per clock hour t on a day the occupant is home:

    y_t = B + b_heat*H_t + b_cool*C_t + b_light*L_t + b_cook*K_t + b_elec*E_t + e_t,   all b >= 0

  B        always-on load (fridge + standby + networking). Estimated from days the meter shows
           nobody home (breaks, weekends away): the mean hourly kWh on those days. With no such
           days, the median of each day's lowest hour is used instead.
  H_t      heating degree-hours below a balance point (picked by least squares from 45-65 F),
           weighted toward hours the occupant is active: HDH_t * (0.5 + 0.5*A_t).
  C_t      cooling degree-hours above a balance point (68-80 F), weighted the same way.
  L_t      activity x darkness: A_t * D_t, where D_t comes from the sun's elevation at the home's
           location. Sunset moving through the year is what separates lighting from electronics.
  K_t      activity x a meal-time shape (assumption: peaks at 8:30, 12:30, 18:30, 23:00).
  E_t      activity alone: electronics, chargers, everything else that follows the occupant.
  A_t      activity shape: the average above-baseload kWh for each of the 168 hours of the week
           on mild, home days, smoothed and scaled to mean 1. Re-estimated after each fit with the
           weather and lighting parts removed (3 passes).

Coefficients are solved with non-negative least squares (scipy.optimize.nnls). Each column is
normalised to sum to 1 over the data, so each coefficient is directly "kWh over the period".
The checklist only switches components off (no AC -> no cooling column). 90% intervals come from
a day-block bootstrap. On away days everything is attributed to the always-on load.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from .solar import darkness, sun_elevation_deg

CATEGORIES = ["always_on", "heating", "cooling", "lighting", "electronics"]
LABELS = {
    "always_on": "Always-on (fridge, standby, router)",
    "heating": "Space heating",
    "cooling": "Air conditioning",
    "lighting": "Lighting",
    "electronics": "Electronics, cooking and other plug loads",
}


@dataclass
class Checklist:
    space_heater: bool = True
    window_ac: bool = False
    mini_fridge: bool = True
    fridge_energy_star: bool = False
    lighting: str = "unknown"            # incandescent | halogen | cfl | led | mixed | unknown
    bulbs: int | None = None             # lamps/bulbs you actually switch on in the evening
    desktop_pc: bool = False
    pc_left_on_hours: float = 0.0        # hours/day the PC sits idle but awake
    pc_idle_watts: float | None = None   # measured with a plug meter, if known
    zip: str | None = None
    state: str | None = None
    subregion: str | None = None
    semester_start: str | None = None    # YYYY-MM-DD
    semester_end: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    timezone: str = "America/New_York"

    @classmethod
    def from_dict(cls, d: dict) -> "Checklist":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class Estimate:
    hours: np.ndarray
    y: np.ndarray
    temps: np.ndarray | None
    components: dict[str, np.ndarray]       # kWh per hour per category (sum = metered, where valid)
    totals: dict[str, float]                # kWh over the data period
    intervals: dict[str, tuple[float, float]]
    baseload_kw: float
    baseload_method: str
    away_days: np.ndarray                   # datetime64[D]
    balance_heat_f: float | None
    balance_cool_f: float | None
    r2_hourly: float
    r2_daily: float
    days: float
    warnings: list[str] = field(default_factory=list)
    lighting_method: str = "regression"

    @property
    def total_kwh(self) -> float:
        return float(np.nansum(self.y))

    def window_totals(self, start: np.datetime64 | None, end: np.datetime64 | None) -> dict[str, float]:
        sel = np.ones(len(self.hours), bool)
        if start is not None:
            sel &= self.hours >= start
        if end is not None:
            sel &= self.hours < end
        return {k: float(np.nansum(v[sel])) for k, v in self.components.items()}

    def per_year(self) -> dict[str, float]:
        f = 365.0 / self.days
        return {k: v * f for k, v in self.totals.items()}


def _smooth_circular(x: np.ndarray, k: int = 3) -> np.ndarray:
    pad = np.concatenate([x[-k:], x, x[:k]])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(pad, ker, mode="same")[k:-k]


# Knots (deg F, daily mean) of the non-negative ramp basis: any non-negative combination is a
# heating response that never rises with temperature (and a cooling one that never falls).
HEAT_KNOTS = (35.0, 45.0, 55.0, 65.0, 72.0)
COOL_KNOTS = (65.0, 72.0, 80.0)

MEAL_PEAKS = [(8.5, 0.6), (12.5, 0.8), (18.5, 1.0), (23.0, 0.4)]


def meal_shape(hod: np.ndarray) -> np.ndarray:
    out = np.zeros(len(hod), float)
    for mu, w in MEAL_PEAKS:
        d = np.minimum(np.abs(hod + 0.5 - mu), 24 - np.abs(hod + 0.5 - mu))
        out += w * np.exp(-0.5 * (d / 1.0) ** 2)
    return out


# Typical wattage of a 60 W-equivalent lamp of each type (assumption, see README "Limits").
BULB_WATTS = {"incandescent": 60.0, "halogen": 43.0, "cfl": 14.0, "led": 9.0, "mixed": 30.0}
LAMP_ON_SHARE = 0.75      # assumption: listed lamps are on 3/4 of the hours you are up and in after dark
ACTIVE_KWH = 0.05         # an hour counts as "someone is up" when use is 50 Wh above the always-on load
LIGHT_RANGE = (0.6, 1.4)  # assumption range reported for the bottom-up lighting estimate


def _by_category(names: list[str], Xn: np.ndarray, beta: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i, k in enumerate(names):
        cat = k.split(":")[0]
        out[cat] = out.get(cat, 0.0) + Xn[:, i] * beta[i]
    return out


def _daily(values: np.ndarray, day_idx: np.ndarray, n_days: int, fn) -> np.ndarray:
    out = np.full(n_days, np.nan)
    for d in range(n_days):
        v = values[day_idx == d]
        v = v[np.isfinite(v)]
        if len(v) >= 20:
            out[d] = fn(v)
    return out


def estimate(hours: np.ndarray, y: np.ndarray, checklist: Checklist | None = None,
             temps: np.ndarray | None = None, n_boot: int = 200, seed: int = 0) -> Estimate:
    cl = checklist or Checklist()
    warnings: list[str] = []
    y = np.asarray(y, float)
    valid = np.isfinite(y)
    days_arr = hours.astype("datetime64[D]")
    uniq_days, day_idx = np.unique(days_arr, return_inverse=True)
    n_days = len(uniq_days)
    hod = (hours - days_arr).astype(int)
    dow = ((days_arr.astype(int) + 3) % 7)          # 0 = Monday
    how = dow * 24 + hod

    # --- 1. Always-on load and away days -------------------------------------------------
    dmin = _daily(y, day_idx, n_days, np.min)
    dmean = _daily(y, day_idx, n_days, np.mean)
    dmax = _daily(y, day_idx, n_days, np.max)
    b0 = float(np.nanmedian(dmin))
    b_low = float(np.nanpercentile(dmin, 10))
    away = (dmean <= 1.5 * b_low) & (dmax <= b_low + 0.12)
    away = np.where(np.isfinite(dmean), away, False)
    if away.sum() >= 3:
        vals = y[np.isin(day_idx, np.where(away)[0]) & valid]
        lo, hi = np.percentile(vals, [2, 98])
        B = float(np.mean(vals[(vals >= lo) & (vals <= hi)]))
        b_method = f"mean hourly use on {int(away.sum())} days with nobody home"
    else:
        B = b0
        b_method = "median of each day's lowest hour (no away days found; likely an underestimate)"
        warnings.append("No away days in the data, so the always-on load is taken from overnight "
                        "minima, which tends to read low for a cycling fridge.")
    home_day = ~away
    home = home_day[day_idx]

    # --- 2. Design columns ----------------------------------------------------------------
    lat = cl.latitude if cl.latitude is not None else 42.444
    lon = cl.longitude if cl.longitude is not None else -76.502
    dark = darkness(sun_elevation_deg(hours, lat, lon, cl.timezone))
    meal = meal_shape(hod)
    have_t = temps is not None and np.isfinite(temps).mean() > 0.8
    if temps is not None and not have_t:
        warnings.append("Temperature file covers under 80% of the hours; heating and cooling are not separated.")
    if not have_t:
        warnings.append("No temperature data: heating and cooling cannot be separated from other loads.")
    t = np.where(np.isfinite(temps), temps, np.nanmean(temps)) if have_t else None
    if have_t:
        t_day = np.array([t[day_idx == d].mean() for d in range(n_days)])[day_idx]
        if cl.space_heater and np.mean(t_day[valid & home_day[day_idx]] > 62) < 0.05:
            warnings.append("Few warm days in this file: the steady part of space heating cannot be told apart "
                            "from plug loads, so heating may read low and electronics high. A file that includes "
                            "late summer or early fall fixes this.")

    # Lighting. Evening lights and evening electronics rise and fall together for a student who
    # is only in after dark, so a whole-home meter cannot split them reliably. When the bulb count
    # is known, lighting is estimated bottom-up: bulbs x typical watts x on-share, in every hour the
    # meter shows someone up (use above the always-on load) while it is dark outside.
    light_fixed = bool(cl.bulbs) and cl.lighting in BULB_WATTS
    if light_fixed:
        up = valid & home & ((y - B) > ACTIVE_KWH)
        L = np.where(up, cl.bulbs * BULB_WATTS[cl.lighting] * LAMP_ON_SHARE * dark / 1000.0, 0.0)
        L = np.minimum(L, np.where(valid, np.clip(y - B, 0, None), 0.0))
    else:
        L = np.zeros(len(y))
        warnings.append("Bulb count not given: lighting is separated by regression on darkness, which is "
                        "unreliable when you are mostly home after dark.")
    resid0 = np.where(valid, y - B - L, np.nan)
    fit_rows = valid & home

    def activity(r: np.ndarray) -> np.ndarray:
        sel = fit_rows.copy()
        if have_t:
            sel &= (t >= 50) & (t <= 70)
            if sel.sum() < 24 * 14:
                sel = fit_rows.copy()
        a = np.zeros(168)
        for h in range(168):
            v = r[sel & (how == h)]
            a[h] = np.nanmean(np.clip(v, 0, None)) if len(v) else 0.0
        a = _smooth_circular(a, 1)
        a = np.maximum(a, 0.05 * a.mean() + 1e-9)
        return a / a.mean()

    def columns(A_t: np.ndarray, bh: float | None, bc: float | None) -> dict[str, np.ndarray]:
        An = A_t / A_t.max()
        w = An  # heating while up; the separate "night" column covers heating while asleep
        cols = {}
        if have_t and cl.space_heater and bh is not None:
            cols["heating:degree_hours"] = np.clip(bh - t, 0, None) * w
            for k in HEAT_KNOTS:     # monotone response to the day's mean temperature
                cols[f"heating:ramp{k:.0f}"] = np.clip((k - t_day) / 10.0, 0, 1) * w
            cols["heating:night"] = np.clip(bh - t, 0, None) * (1.0 - An)
        if have_t and cl.window_ac and bc is not None:
            cols["cooling:degree_hours"] = np.clip(t - bc, 0, None) * w
            for k in COOL_KNOTS:
                cols[f"cooling:ramp{k:.0f}"] = np.clip((t_day - k) / 8.0, 0, 1) * w
        if not light_fixed:
            cols["lighting"] = A_t * dark
        cols["electronics"] = A_t.copy()
        cols["electronics:meals"] = meal * w
        for k in cols:
            cols[k] = np.where(home, cols[k], 0.0)
        return cols

    def solve(cols: dict[str, np.ndarray], rows: np.ndarray, target: np.ndarray):
        names = list(cols)
        X = np.column_stack([cols[k] for k in names])
        scale = X[rows].sum(axis=0)
        scale[scale <= 0] = 1.0
        Xn = X / scale
        beta, _ = nnls(Xn[rows], target[rows])
        return names, Xn, beta

    heat_grid = [45.0, 50.0, 55.0, 60.0, 65.0, 70.0] if (have_t and cl.space_heater) else [None]
    cool_grid = [68.0, 72.0, 76.0, 80.0] if (have_t and cl.window_ac) else [None]
    A_t = activity(resid0)[how]
    best = None
    for _ in range(3):
        best = None
        for bh in heat_grid:
            for bc in cool_grid:
                cols = columns(A_t, bh, bc)
                names, Xn, beta = solve(cols, fit_rows, resid0)
                sse = float(np.sum((resid0[fit_rows] - Xn[fit_rows] @ beta) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, bh, bc, names, Xn, beta)
        _, bh, bc, names, Xn, beta = best
        parts = _by_category(names, Xn, beta)
        r_act = resid0.copy()
        for k in ("heating", "cooling", "lighting"):
            if k in parts:
                r_act = r_act - parts[k]
        A_new = activity(r_act)[how]
        if np.allclose(A_new, A_t, rtol=1e-3):
            break
        A_t = 0.5 * A_t + 0.5 * A_new
    _, bh, bc, names, Xn, beta = best

    # --- 3. Components summing to the meter -----------------------------------------------
    comps = {k: np.zeros(len(y)) for k in CATEGORIES}
    comps["always_on"] = np.where(valid, np.minimum(B, y), 0.0)
    comps["lighting"] = L.copy()
    fitted = _by_category(names, Xn, beta)
    above = np.where(valid, np.clip(y - comps["always_on"] - L, 0, None), 0.0)
    ftot = sum(fitted.values()) if fitted else np.zeros(len(y))
    # Hour-by-hour, share whatever is above baseload in proportion to the fitted parts;
    # hours with no fitted activity (e.g. away days) go to always-on.
    share_ok = ftot > 1e-9
    for k, f in fitted.items():
        comps[k] = comps[k] + np.where(share_ok, above * f / np.where(share_ok, ftot, 1.0), 0.0)
    comps["always_on"] = comps["always_on"] + np.where(share_ok, 0.0, above)
    totals = {k: float(v.sum()) for k, v in comps.items()}

    pred = B + L + Xn @ beta
    yy = y[fit_rows]
    r2 = 1 - np.sum((yy - pred[fit_rows]) ** 2) / np.sum((yy - yy.mean()) ** 2)
    dy = _daily(y, day_idx, n_days, np.sum)
    dp = np.array([pred[(day_idx == d)].sum() for d in range(n_days)])
    ok = np.isfinite(dy) & home_day
    r2d = 1 - np.sum((dy[ok] - dp[ok]) ** 2) / np.sum((dy[ok] - dy[ok].mean()) ** 2) if ok.sum() > 2 else np.nan

    # --- 4. Day-block bootstrap for intervals ---------------------------------------------
    rng = np.random.default_rng(seed)
    home_ids = np.where(home_day & np.isfinite(dmean))[0]
    away_ids = np.where(away)[0]
    boots = {k: [] for k in CATEGORIES}
    rows_by_day = [np.where((day_idx == d) & valid)[0] for d in range(n_days)]
    period_hours = valid.sum()
    for _ in range(n_boot):
        pick = rng.choice(home_ids, size=len(home_ids), replace=True)
        rows = np.concatenate([rows_by_day[d] for d in pick])
        if len(away_ids) >= 3:
            ap = rng.choice(away_ids, size=len(away_ids), replace=True)
            arows = np.concatenate([rows_by_day[d] for d in ap])
            Bb = float(np.mean(y[arows]))
        else:
            Bb = float(np.median(rng.choice(dmin[np.isfinite(dmin)], size=np.isfinite(dmin).sum())))
        target = y - Bb - L
        b, _ = nnls(Xn[rows], target[rows])
        # Contribution per category over the full period, rescaled the same way as the point fit.
        contrib = {k: float(v[fit_rows].sum()) for k, v in _by_category(names, Xn, b).items()}
        above_total = float(np.clip(y[valid] - Bb - L[valid], 0, None).sum())
        tot = sum(contrib.values()) or 1.0
        for k in CATEGORIES:
            if k == "always_on":
                boots[k].append(Bb * period_hours)
            else:
                boots[k].append(contrib.get(k, 0.0) * min(1.0, above_total / tot) if tot else 0.0)
    intervals = {}
    for k in CATEGORIES:
        v = np.array(boots[k])
        # Centre the bootstrap spread on the point estimate (basic percentile, shifted).
        if len(v) == 0:
            intervals[k] = (float("nan"), float("nan"))
            continue
        if totals[k] == 0 and not v.any():
            intervals[k] = (0.0, 0.0)
            continue
        lo, hi = np.percentile(v, [5, 95])
        mid = np.median(v)
        intervals[k] = (max(0.0, totals[k] - (mid - lo)), totals[k] + (hi - mid))
    if light_fixed:
        # Bottom-up lighting: the spread comes from the wattage and on-share assumptions, not sampling.
        intervals["lighting"] = (totals["lighting"] * LIGHT_RANGE[0], totals["lighting"] * LIGHT_RANGE[1])

    return Estimate(hours, y, temps, comps, totals, intervals, B, b_method, uniq_days[away],
                    bh, bc, float(r2), float(r2d), float(valid.sum()) / 24.0, warnings,
                    "bottom-up" if light_fixed else "regression")
