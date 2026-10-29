"""Synthetic student households with known per-appliance energy, for validating the estimator.

Simulated minute by minute from physical behaviour, not from the estimator's own templates:
a fridge compressor cycling on a duty that rises with room temperature, a space heater and a
window AC driven by a thermostat on a first-order room-temperature model fed with real Ithaca
weather, lights that come on when the occupant is home, awake and the sun is low, randomly timed
gaming, TV, microwave, kettle and hair-dryer events, school breaks and weekends away.
Every household draws its own schedule and wattages, so the estimator never sees the "true" shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from .disaggregate import Checklist
from .solar import sun_elevation_deg

BULB_W = {"incandescent": 60.0, "halogen": 43.0, "cfl": 14.0, "led": 9.0}
TRUTH_KEYS = ["always_on", "heating", "cooling", "lighting", "electronics"]

# Breaks (whole days away) loosely following a Cornell-style calendar for 2025.
BREAKS_2025 = [(date(2025, 1, 1), date(2025, 1, 19)), (date(2025, 3, 15), date(2025, 3, 23)),
               (date(2025, 11, 26), date(2025, 11, 30)), (date(2025, 12, 20), date(2025, 12, 31))]
SUMMER_2025 = (date(2025, 5, 18), date(2025, 8, 22))


@dataclass
class Household:
    params: dict
    checklist: Checklist
    minutes: np.ndarray             # datetime64[m], one per minute
    watts: dict[str, np.ndarray]    # per truth category, W per minute
    interval_minutes: int

    def interval_kwh(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.interval_minutes
        total = sum(self.watts.values())
        k = total.reshape(-1, n).sum(axis=1) / 60000.0
        k = np.round(k, 3)  # meters report to 1 Wh
        return self.minutes[::n], k

    def truth_kwh(self, start=None, end=None) -> dict[str, float]:
        sel = np.ones(len(self.minutes), bool)
        if start is not None:
            sel &= self.minutes >= np.datetime64(start, "m")
        if end is not None:
            sel &= self.minutes < np.datetime64(end, "m")
        return {k: float(v[sel].sum() / 60000.0) for k, v in self.watts.items()}


def _hour_to_minute(values_hourly: np.ndarray) -> np.ndarray:
    x = np.arange(len(values_hourly) * 60) / 60.0
    return np.interp(x, np.arange(len(values_hourly)), values_hourly)


def simulate(temps_hourly: np.ndarray, hours: np.ndarray, seed: int = 0,
             interval_minutes: int = 15, overrides: dict | None = None) -> Household:
    rng = np.random.default_rng(seed)
    o = overrides or {}
    p = {
        "has_heater": o.get("has_heater", rng.random() < 0.65),
        "heater_w": o.get("heater_w", float(rng.choice([1500.0, 1500.0, 900.0]))),
        "heat_set": o.get("heat_set", rng.uniform(66, 71)),
        "heat_at_night": o.get("heat_at_night", rng.random() < 0.5),
        "has_ac": o.get("has_ac", rng.random() < 0.35),
        "ac_w": rng.uniform(450, 800),
        "cool_set": rng.uniform(72, 77),
        "fridge_w": rng.uniform(65, 110),
        "fridge_duty70": rng.uniform(0.28, 0.45),
        "fridge_period": rng.uniform(35, 60),
        "standby_w": rng.uniform(6, 30),
        "bulb": o.get("bulb", str(rng.choice(["incandescent", "halogen", "cfl", "led", "led"]))),
        "n_bulbs": int(rng.integers(2, 6)),
        "light_elev": rng.uniform(2, 10),
        "has_pc": o.get("has_pc", rng.random() < 0.65),
        "pc_game_w": rng.uniform(150, 350),
        "pc_idle_w": rng.uniform(40, 90),
        "pc_left_on": float(o.get("pc_left_on", rng.choice([0, 2, 6, 12, 24]))),
        "has_tv": rng.random() < 0.5,
        "tv_w": rng.uniform(70, 160),
        "has_kettle": rng.random() < 0.4,
        "has_dryer": rng.random() < 0.5,
        "summer_home": o.get("summer_home", rng.random() < 0.4),
        "bed": rng.normal(1.0, 0.6),     # hours after midnight
        "wake": rng.normal(8.75, 0.6),
    }
    n_h = len(hours)
    n_m = n_h * 60
    minutes = hours[0].astype("datetime64[m]") + np.arange(n_m).astype("timedelta64[m]")
    mday = minutes.astype("datetime64[D]")
    days = np.unique(mday)
    t_out = _hour_to_minute(np.where(np.isfinite(temps_hourly), temps_hourly, np.nanmean(temps_hourly)))
    elev = _hour_to_minute(sun_elevation_deg(hours, minute=0.0))

    # --- occupancy ---------------------------------------------------------------------
    home = np.zeros(n_m, bool)
    awake = np.zeros(n_m, bool)
    for di, d in enumerate(days):
        dd = date.fromisoformat(str(d))
        sl = slice(di * 1440, (di + 1) * 1440)
        away_day = any(a <= dd <= b for a, b in BREAKS_2025)
        if SUMMER_2025[0] <= dd <= SUMMER_2025[1] and not p["summer_home"]:
            away_day = True
        if dd.weekday() >= 5 and rng.random() < 0.08:
            away_day = True
        if away_day:
            continue
        h = np.ones(1440, bool)
        m = np.arange(1440) / 60.0
        bed = p["bed"] + rng.normal(0, 0.5)
        wake = p["wake"] + rng.normal(0, 0.5) + (1.0 if dd.weekday() >= 5 else 0.0)
        aw = ~((m >= max(bed, 0)) & (m < wake)) if bed >= 0 else ~((m < wake) | (m >= 24 + bed))
        in_term = not (SUMMER_2025[0] <= dd <= SUMMER_2025[1])
        if dd.weekday() < 5 and in_term:
            leave = rng.uniform(min(max(wake + 0.3, 9), 11.4), 11.5)
            back = rng.uniform(14, 18.5)
            h &= ~((m >= leave) & (m < back))
            if rng.random() < 0.3:
                h |= (m >= 12.2) & (m < 13.0)
        elif rng.random() < 0.6:
            leave = rng.uniform(11, 16)
            h &= ~((m >= leave) & (m < leave + rng.uniform(2, 5)))
        if rng.random() < 0.35:        # evening out
            s = rng.uniform(19, 22)
            h &= ~((m >= s) & (m < s + rng.uniform(1.5, 3.5)))
        home[sl] = h
        awake[sl] = h & aw
    active = home & awake

    W = {k: np.zeros(n_m) for k in TRUTH_KEYS}

    # --- room temperature, heater, AC (thermostat on a first-order room model) ----------
    # Free-running room temperature: central heat holds ~68 F at 55 F outside and loses ground
    # as it gets colder; sun and people push it to ~72 F by 65 F outside.
    t_drive = np.where(t_out < 55, 68 + 0.15 * (t_out - 55),
                       np.where(t_out > 65, 72 + 0.45 * (t_out - 65), 68 + 0.4 * (t_out - 55)))
    tau = 180.0
    heat_ok = (active | (home & p["heat_at_night"])) & p["has_heater"]
    cool_ok = active & p["has_ac"]
    heat_gain = 4.0 / 60.0 * p["heater_w"] / 1500.0
    cool_gain = 6.0 / 60.0
    T = 67.0
    h_on = False
    c_on = False
    heater = np.zeros(n_m)
    ac = np.zeros(n_m)
    room = np.zeros(n_m)
    hs, cs = p["heat_set"], p["cool_set"]
    for i in range(n_m):
        if heat_ok[i]:
            if T < hs - 0.7:
                h_on = True
            elif T > hs + 0.7:
                h_on = False
        else:
            h_on = False
        if cool_ok[i]:
            if T > cs + 0.7:
                c_on = True
            elif T < cs - 0.7:
                c_on = False
        else:
            c_on = False
        T += (t_drive[i] - T) / tau + (heat_gain if h_on else 0.0) - (cool_gain if c_on else 0.0)
        heater[i] = p["heater_w"] if h_on else 0.0
        ac[i] = p["ac_w"] if c_on else 0.0
        room[i] = T
    W["heating"] = heater
    W["cooling"] = ac

    # --- fridge (duty rises 1.5 percentage points per F above 70) + standby --------------
    duty = np.clip(p["fridge_duty70"] + 0.015 * (room - 70), 0.15, 0.85)
    phase = np.cumsum(1.0 / p["fridge_period"] * np.ones(n_m)) + rng.uniform()
    comp_on = (phase % 1.0) < duty
    W["always_on"] = np.where(comp_on, p["fridge_w"], 0.0) + p["standby_w"] + 1.5 * rng.standard_normal(n_m).clip(-2, 2)
    W["always_on"] = np.clip(W["always_on"], 0, None)

    # --- lighting --------------------------------------------------------------------------
    frac_on = 0.5 + 0.5 * rng.random(len(days))
    lit = active & (elev < p["light_elev"] + rng.normal(0, 1.5))
    W["lighting"] = np.where(lit, BULB_W[p["bulb"]] * p["n_bulbs"] * np.repeat(frac_on, 1440), 0.0)
    # 10% of evenings the desk lamp stays on overnight
    for di in np.where(rng.random(len(days)) < 0.10)[0]:
        sl = slice(di * 1440, di * 1440 + 7 * 60)
        W["lighting"][sl] += np.where(home[sl], BULB_W[p["bulb"]], 0.0)

    # --- electronics -------------------------------------------------------------------------
    elec = np.where(active & (rng.random(n_m) < 0.4), 45.0, 0.0)          # laptop charging
    elec += np.where(home, 3.0, 0.0)                                      # phone chargers etc.
    if p["has_pc"]:
        pc = np.zeros(n_m)
        for di in range(len(days)):
            base = di * 1440
            if not home[base:base + 1440].any():
                continue
            if rng.random() < 0.6:
                s = int(rng.uniform(18.5, 23) * 60)
                d = int(rng.uniform(1, 3.5) * 60)
                idx = np.arange(base + s, min(base + s + d, n_m))
                pc[idx] = np.where(home[idx], p["pc_game_w"], 0.0)
            if p["pc_left_on"] >= 24:
                idle_idx = np.arange(base, base + 1440)
            elif p["pc_left_on"] > 0:
                s = int(rng.uniform(21, 24) * 60)
                idle_idx = np.arange(base + s, min(base + s + int(p["pc_left_on"] * 60), n_m))
            else:
                idle_idx = np.array([], int)
            idle_idx = idle_idx[pc[idle_idx] == 0]
            pc[idle_idx] = p["pc_idle_w"]
        elec += pc
    if p["has_tv"]:
        for di in range(len(days)):
            base = di * 1440
            if rng.random() < 0.5:
                s = int(rng.uniform(19, 22.5) * 60)
                idx = np.arange(base + s, min(base + s + int(rng.uniform(1, 3) * 60), n_m))
                elec[idx] += np.where(active[idx], p["tv_w"], 0.0)
    if p["has_dryer"]:
        for di in range(len(days)):
            base = di * 1440
            if rng.random() < 0.6:
                s = int((p["wake"] + rng.uniform(0, 1)) * 60)
                idx = np.arange(base + s, min(base + s + 7, n_m))
                elec[idx] += np.where(active[idx], 1500.0, 0.0)
    W["electronics"] = elec

    # --- cooking -------------------------------------------------------------------------
    cook = np.zeros(n_m)
    for di in range(len(days)):
        base = di * 1440
        for mu in (8.5, 12.5, 18.5, 23.0):
            if rng.random() < 0.5:
                s = int(np.clip(rng.normal(mu, 0.8), 0, 23.9) * 60)
                idx = np.arange(base + s, min(base + s + int(rng.uniform(2, 6)), n_m))
                cook[idx] += np.where(active[idx], 1100.0, 0.0)
        if p["has_kettle"] and rng.random() < 0.7:
            s = int((p["wake"] + 0.3) * 60)
            idx = np.arange(base + s, min(base + s + 3, n_m))
            cook[idx] += np.where(active[idx], 1500.0, 0.0)
    W["electronics"] = W["electronics"] + cook

    # Pad to whole intervals
    rem = n_m % interval_minutes
    if rem:
        minutes = minutes[:-rem]
        W = {k: v[:-rem] for k, v in W.items()}

    pc_left = p["pc_left_on"] if p["has_pc"] else 0.0
    cl = Checklist(space_heater=bool(p["has_heater"]), window_ac=bool(p["has_ac"]), mini_fridge=True,
                   fridge_energy_star=False, lighting=p["bulb"], bulbs=p["n_bulbs"], desktop_pc=bool(p["has_pc"]),
                   pc_left_on_hours=float(min(pc_left, 24.0)), pc_idle_watts=None)
    return Household(p, cl, minutes, W, interval_minutes)


def to_datetimes(starts: np.ndarray) -> list[datetime]:
    base = datetime.fromisoformat(str(starts[0].astype("datetime64[m]")))
    off = (starts - starts[0]).astype("timedelta64[m]").astype(int)
    return [base + timedelta(minutes=int(m)) for m in off]
