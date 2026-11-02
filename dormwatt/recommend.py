"""Turn the per-category estimate into ranked swaps and habits, each with a stated method.

Every saving is (estimated kWh in the affected category) x (a reduction factor). The factor is
either sourced (see sources.py) or explicitly labelled an assumption.

Two things are uncertain: the estimated kWh in each category, and the reduction factor (DOE's
thermostat figure is a ceiling, so the low end uses half of it; the low PC idle draw is the ENERGY STAR
median for certified desktops). Each option's low-high range moves both together, using that
category's 80th-percentile error in the validation run (docs/validation.json). The top-three total
uses the low factors and the 80th-percentile error of the top-three saving itself, measured
end to end in the same run: summing the per-option extremes would assume every category is wrong in
the same direction, which the validation shows does not happen. These are bounding ranges, not
confidence intervals.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .disaggregate import Checklist, Estimate
from .factors import GridFactors, energystar

PC_IDLE_W_ASSUMED = 60.0          # assumption when the user has not measured their PC
STANDBY_CUT_ASSUMED = 0.25        # assumption: share of standby a switched strip removes
MIXED_INCANDESCENT_SHARE = 0.5    # assumption for "mixed" bulbs
BREAK_DAYS_ASSUMED = 30           # assumption when the data shows no away days
FRIDGE_SHARE_CAP = 0.8            # fridge can be at most this share of the always-on load
HEATER_SETBACK = 0.10             # DOE: "as much as 10%" a year for a 7-10 F, 8 h/day setback (a ceiling)
HEATER_SETBACK_LOW = 0.05         # assumption: half of DOE's ceiling, the low end of the range (heat and AC)
LED_CUT = 0.75                    # DOE: LEDs use "at least 75%" less (a floor, so no higher value is used)


@dataclass
class Saving:
    key: str
    title: str
    action: str
    kwh_year: float
    kwh_semester: float
    method: str
    sources: list[str]
    assumptions: list[str] = field(default_factory=list)
    usd_year: float = 0.0
    usd_semester: float = 0.0
    kg_year: float = 0.0
    kg_semester: float = 0.0
    kg_year_marginal: float = 0.0
    kwh_year_low: float = 0.0
    kwh_year_high: float = 0.0
    kwh_semester_low: float = 0.0
    kwh_semester_high: float = 0.0

    def price(self, g: GridFactors) -> "Saving":
        self.usd_year = self.kwh_year * g.price_usd_per_kwh
        self.usd_semester = self.kwh_semester * g.price_usd_per_kwh
        self.kg_year = self.kwh_year * g.kg_co2e_per_kwh
        self.kg_semester = self.kwh_semester * g.kg_co2e_per_kwh
        self.kg_year_marginal = self.kwh_year * g.kg_co2e_per_kwh_nonbaseload
        return self


@dataclass
class Plan:
    savings: list[Saving]           # all candidates, ranked
    top: list[Saving]               # the best three
    semester: tuple[np.datetime64, np.datetime64]
    semester_note: str
    fridge_kwh_year: float
    fridge_method: str
    ranged: bool = False            # True when the ranges come from validation errors
    top_low: tuple[float, float] = (0.0, 0.0)    # (year, semester) kWh, low end of the top three
    top_high: tuple[float, float] = (0.0, 0.0)   # (year, semester) kWh, high end of the top three
    top_error: float = 0.0                        # validated 80th-percentile error of the top-three saving

    def top_totals(self) -> dict[str, tuple[float, float, float]]:
        """(low, central, high) kWh of the top three, per year and per semester."""
        y = sum(s.kwh_year for s in self.top)
        m = sum(s.kwh_semester for s in self.top)
        return {"year": (self.top_low[0], y, self.top_high[0]), "semester": (self.top_low[1], m, self.top_high[1])}


@dataclass(frozen=True)
class Scenario:
    """How far to push each uncertain input. scale[c] multiplies category c's estimated kWh."""
    scale: dict
    setback: float = HEATER_SETBACK
    pc_idle_default_w: float = PC_IDLE_W_ASSUMED


def category_errors(validation: dict | None) -> dict[str, float] | None:
    """80th-percentile absolute error (as a fraction) per category, and for the top-three saving
    under the key "_top3", from the full-year 15-minute run in docs/validation.json."""
    st = (validation or {}).get("settings", {}).get("year_15min", {})
    out = {}
    for c in ("always_on", "heating", "cooling", "lighting", "electronics", "_top3_savings"):
        v = st.get(c, {}).get("p80_ape_pct")
        if v is None:
            return None
        out["_top3" if c == "_top3_savings" else c] = v / 100.0
    return out


def with_totals(est: Estimate, totals: dict[str, float]) -> Estimate:
    """A copy of the estimate whose category totals (and hourly parts, proportionally) are replaced."""
    e = copy.copy(est)
    e.totals = dict(totals)
    e.components = {c: est.components[c] * (totals[c] / est.totals[c]) if est.totals[c] > 0 else est.components[c]
                    for c in est.components}
    return e


def semester_window(est: Estimate, cl: Checklist) -> tuple[np.datetime64, np.datetime64, str]:
    if cl.semester_start and cl.semester_end:
        s, e = np.datetime64(cl.semester_start, "h"), np.datetime64(cl.semester_end, "h")
        return s, e, "semester dates from the checklist"
    last = est.hours[-1].astype("datetime64[Y]").astype(int) + 1970
    for yr in (last, last - 1):
        s, e = np.datetime64(f"{yr}-08-25T00", "h"), np.datetime64(f"{yr}-12-20T00", "h")
        if s >= est.hours[0] and e <= est.hours[-1] + 1:
            return s, e, f"fall semester {yr} (Aug 25 - Dec 19) found in the data"
    s, e = est.hours[0], est.hours[-1] + 1
    return s, e, "data does not cover a full fall semester; semester figures use the whole file scaled to 16 weeks"


def _candidates(est: Estimate, cl: Checklist, sc: Scenario):
    es = energystar()
    s0, s1, sem_note = semester_window(est, cl)
    year_f = 365.0 / est.days
    tot = {k: v * sc.scale.get(k, 1.0) for k, v in est.totals.items()}
    sem = {k: v * sc.scale.get(k, 1.0) for k, v in est.window_totals(s0, s1).items()}
    sem_hours = int(np.sum((est.hours >= s0) & (est.hours < s1) & np.isfinite(est.y)))
    if "scaled" in sem_note:
        sem = {k: v * (16 * 7 * 24) / max(1, sem_hours) for k, v in tot.items()}
        sem_hours = 16 * 7 * 24
    sem_days = sem_hours / 24.0
    away = est.away_days
    away_sem = int(np.sum((away >= s0.astype("datetime64[D]")) & (away < s1.astype("datetime64[D]"))))
    home_days_year = (est.days - len(away)) * year_f
    home_days_sem = sem_days - away_sem
    out: list[Saving] = []

    # Fridge share of the always-on load (prior-based split, flagged)
    if cl.mini_fridge:
        rated = es["mini_fridge_energy_star_median_kwh_yr"] if cl.fridge_energy_star \
            else es["mini_fridge_federal_standard_median_kwh_yr"]
        fridge_year = min(rated, FRIDGE_SHARE_CAP * tot["always_on"] * year_f)
        fridge_method = (f"{'ENERGY STAR median' if cl.fridge_energy_star else 'federal-standard median'} "
                         f"for 2.5-4.5 cu ft compact fridges ({rated:.0f} kWh/yr, ENERGY STAR dataset), "
                         f"capped at {FRIDGE_SHARE_CAP:.0%} of the measured always-on load. The meter cannot "
                         "tell the fridge from other always-on devices, so this split is a prior, not a measurement.")
    else:
        fridge_year, fridge_method = 0.0, "no fridge in the checklist"

    # 1. Space heater setback
    if tot["heating"] > 0:
        f = sc.setback
        out.append(Saving(
            "heater_setback", "Put the space heater on a thermostat timer",
            "Use a plug-in thermostat or timer: 7-10 F lower for the 8 hours you sleep, and off when you leave.",
            tot["heating"] * year_f * f, sem["heating"] * f,
            f"Estimated space-heating kWh x {f:.0%}. DOE: turning a thermostat back 7-10 F for 8 hours a day "
            "saves as much as 10% a year on heating.",
            ["doe_thermostat"],
            ["DOE's figure is for a home thermostat; applying it to a space heater assumes the same setback.",
             f"DOE's {HEATER_SETBACK:.0%} is a ceiling (\"as much as\"); the low end of the range assumes "
             f"{HEATER_SETBACK_LOW:.0%}."]))

    # 2. LED lighting
    if tot["lighting"] > 0 and cl.lighting in ("incandescent", "halogen", "mixed", "unknown"):
        share = 1.0 if cl.lighting in ("incandescent", "halogen") else MIXED_INCANDESCENT_SHARE
        f = LED_CUT * share
        assum = []
        if cl.lighting == "mixed":
            assum.append(f"Assumes {MIXED_INCANDESCENT_SHARE:.0%} of lighting energy is from incandescent or halogen bulbs.")
        if cl.lighting == "unknown":
            assum.append("Bulb type not given: assumes half the lighting energy is incandescent or halogen. "
                         "Tick your bulb type for a firmer number.")
        out.append(Saving(
            "led", "Swap incandescent and halogen bulbs for LEDs",
            "Replace each incandescent or halogen bulb you use in the evening with an LED of the same brightness.",
            tot["lighting"] * year_f * f, sem["lighting"] * f,
            f"Estimated lighting kWh x {share:.0%} incandescent share x 75%. DOE: residential LEDs use at "
            "least 75% less energy than incandescent lighting.",
            ["doe_led"], assum))

    # 3. Computer sleep settings
    if cl.desktop_pc and cl.pc_left_on_hours > 0:
        idle = cl.pc_idle_watts or sc.pc_idle_default_w
        sleep_w = es["desktop_sleep_watts_median"]
        per_day = min(cl.pc_left_on_hours, 24) * (idle - sleep_w) / 1000.0
        cap_y, cap_s = 0.5 * tot["electronics"] * year_f, 0.5 * sem["electronics"]
        assum = [] if cl.pc_idle_watts else [
            f"Idle draw of {PC_IDLE_W_ASSUMED:.0f} W assumed; a plug meter gives your real figure. The low end of "
            f"the range uses {es['desktop_short_idle_watts_median']:.1f} W, the median idle draw of ENERGY STAR "
            "certified desktops."]
        assum.append("Capped at half of the measured electronics load so it cannot exceed what the meter saw.")
        out.append(Saving(
            "pc_sleep", "Let the computer sleep instead of idling",
            "Set the display to sleep after 15 minutes and the computer after 30 (the ENERGY STAR defaults).",
            min(per_day * home_days_year, cap_y), min(per_day * home_days_sem, cap_s),
            f"{cl.pc_left_on_hours:g} idle hours/day x ({idle:.0f} W idle - {sleep_w:.1f} W sleep) x days at home. "
            "Sleep draw is the median of ENERGY STAR certified desktops.",
            ["es_computers", "es_computers_page"], assum))

    # 4. Fridge over breaks
    if fridge_year > 0:
        if len(away) >= 3:
            bd_year, bd_sem, bd_note = len(away) * year_f, away_sem, "away days the meter shows"
            assum = []
        else:
            bd_year, bd_sem, bd_note = BREAK_DAYS_ASSUMED, BREAK_DAYS_ASSUMED * sem_days / 365, "assumed break days"
            assum = [f"No away days in the data: assumes {BREAK_DAYS_ASSUMED} break days a year."]
        assum.append("Fridge share of the always-on load is a prior (see Method).")
        per_day = fridge_year / 365.0
        out.append(Saving(
            "fridge_breaks", "Empty and unplug the mini-fridge over breaks",
            "Before Thanksgiving, winter and spring break: eat it down, unplug, prop the door open.",
            per_day * bd_year, per_day * bd_sem,
            f"Fridge kWh/day ({per_day:.2f}) x {bd_note}.", ["es_fridge"], assum))

    # 5. Replace an older fridge
    if fridge_year > 0 and not cl.fridge_energy_star:
        es_med = es["mini_fridge_energy_star_median_kwh_yr"]
        d_year = max(0.0, fridge_year - es_med)
        out.append(Saving(
            "fridge_energy_star", "Choose an ENERGY STAR mini-fridge next time",
            "When the fridge is replaced, pick an ENERGY STAR certified compact model (or share one with a roommate).",
            d_year, d_year * sem_days / 365.0,
            f"Fridge kWh/yr ({fridge_year:.0f}) - ENERGY STAR median for compact fridges ({es_med:.0f} kWh/yr).",
            ["es_fridge"], ["Fridge share of the always-on load is a prior (see Method)."]))

    # 6. AC setpoint
    if tot["cooling"] > 0:
        f = sc.setback
        out.append(Saving(
            "ac_setback", "Set the AC warmer when you sleep or leave",
            "Raise the setpoint 7-10 F for 8 hours a day, or switch it off when you go out.",
            tot["cooling"] * year_f * f, sem["cooling"] * f,
            f"Estimated cooling kWh x {f:.0%} (DOE thermostat setback figure, which covers cooling too).",
            ["doe_thermostat"], [f"DOE's {HEATER_SETBACK:.0%} is a ceiling; the low end of the range assumes "
                                 f"{HEATER_SETBACK_LOW:.0%}."]))

    # 7. Standby
    standby_year = max(0.0, tot["always_on"] * year_f - fridge_year)
    if standby_year > 0:
        out.append(Saving(
            "standby_strip", "Switch off standby loads at a power strip",
            "Put the TV, console, speakers and monitor on one strip and switch it off at night and when away.",
            standby_year * STANDBY_CUT_ASSUMED, standby_year * STANDBY_CUT_ASSUMED * sem_days / 365.0,
            f"Always-on kWh not attributed to the fridge ({standby_year:.0f} kWh/yr) x {STANDBY_CUT_ASSUMED:.0%}.",
            [], [f"The {STANDBY_CUT_ASSUMED:.0%} cut is an assumption, not a sourced figure."]))

    return out, (s0, s1), sem_note, fridge_year, fridge_method


def plan(est: Estimate, cl: Checklist, g: GridFactors, errors: dict[str, float] | None = None) -> Plan:
    """Rank every option by central kWh/yr. With `errors` (see category_errors), each option also
    gets a low-high range; without it, low = high = central."""
    out, window, sem_note, fridge_year, fridge_method = _candidates(est, cl, Scenario(scale={}))
    central = {s.key: s for s in out}
    lo_by = hi_by = factor_lo = central
    if errors:
        cat = {c: e for c, e in errors.items() if not c.startswith("_")}
        low_factors = dict(setback=HEATER_SETBACK_LOW, pc_idle_default_w=energystar()["desktop_short_idle_watts_median"])
        factor_lo = {s.key: s for s in _candidates(est, cl, Scenario(scale={}, **low_factors))[0]}
        lo_by = {s.key: s for s in _candidates(est, cl, Scenario(
            scale={c: max(0.0, 1 - e) for c, e in cat.items()}, **low_factors))[0]}
        hi_by = {s.key: s for s in _candidates(est, cl, Scenario(scale={c: 1 + e for c, e in cat.items()}))[0]}
    for s in out:
        lo, hi = lo_by.get(s.key, s), hi_by.get(s.key, s)
        s.kwh_year_low, s.kwh_year_high = min(lo.kwh_year, s.kwh_year), max(hi.kwh_year, s.kwh_year)
        s.kwh_semester_low, s.kwh_semester_high = (min(lo.kwh_semester, s.kwh_semester),
                                                   max(hi.kwh_semester, s.kwh_semester))
        s.price(g)
    ranked = sorted(out, key=lambda s: s.kwh_year, reverse=True)
    top = ranked[:3]
    e3 = (errors or {}).get("_top3", 0.0)
    flo = [factor_lo.get(s.key, s) for s in top]
    top_low = (min(sum(s.kwh_year for s in flo), sum(s.kwh_year for s in top)) * (1 - e3),
               min(sum(s.kwh_semester for s in flo), sum(s.kwh_semester for s in top)) * (1 - e3))
    top_high = (sum(s.kwh_year for s in top) * (1 + e3), sum(s.kwh_semester for s in top) * (1 + e3))
    return Plan(ranked, top, window, sem_note, fridge_year, fridge_method, ranged=bool(errors),
                top_low=top_low, top_high=top_high, top_error=e3)
