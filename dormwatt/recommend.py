"""Turn the per-category estimate into ranked swaps and habits, each with a stated method.

Every saving is (estimated kWh in the affected category) x (a reduction factor). The factor is
either sourced (see sources.py) or explicitly labelled an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .disaggregate import Checklist, Estimate
from .factors import GridFactors, energystar

PC_IDLE_W_ASSUMED = 60.0          # assumption when the user has not measured their PC
STANDBY_CUT_ASSUMED = 0.25        # assumption: share of standby a switched strip removes
MIXED_INCANDESCENT_SHARE = 0.5    # assumption for "mixed" bulbs
BREAK_DAYS_ASSUMED = 30           # assumption when the data shows no away days
FRIDGE_SHARE_CAP = 0.8            # fridge can be at most this share of the always-on load


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


def plan(est: Estimate, cl: Checklist, g: GridFactors) -> Plan:
    es = energystar()
    s0, s1, sem_note = semester_window(est, cl)
    year_f = 365.0 / est.days
    tot = est.totals
    sem = est.window_totals(s0, s1)
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
        f = 0.10
        out.append(Saving(
            "heater_setback", "Put the space heater on a thermostat timer",
            "Use a plug-in thermostat or timer: 7-10 F lower for the 8 hours you sleep, and off when you leave.",
            tot["heating"] * year_f * f, sem["heating"] * f,
            f"Estimated space-heating kWh x {f:.0%}. DOE: turning a thermostat back 7-10 F for 8 hours a day "
            "saves as much as 10% a year on heating.",
            ["doe_thermostat"],
            ["DOE's figure is for a home thermostat; applying it to a space heater assumes the same setback."]))

    # 2. LED lighting
    if tot["lighting"] > 0 and cl.lighting in ("incandescent", "halogen", "mixed", "unknown"):
        share = 1.0 if cl.lighting in ("incandescent", "halogen") else MIXED_INCANDESCENT_SHARE
        f = 0.75 * share
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
        idle = cl.pc_idle_watts or PC_IDLE_W_ASSUMED
        sleep_w = es["desktop_sleep_watts_median"]
        per_day = min(cl.pc_left_on_hours, 24) * (idle - sleep_w) / 1000.0
        cap_y, cap_s = 0.5 * tot["electronics"] * year_f, 0.5 * sem["electronics"]
        assum = [] if cl.pc_idle_watts else [f"Idle draw of {PC_IDLE_W_ASSUMED:.0f} W assumed; a plug meter gives your real figure."]
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
        f = 0.10
        out.append(Saving(
            "ac_setback", "Set the AC warmer when you sleep or leave",
            "Raise the setpoint 7-10 F for 8 hours a day, or switch it off when you go out.",
            tot["cooling"] * year_f * f, sem["cooling"] * f,
            f"Estimated cooling kWh x {f:.0%} (DOE thermostat setback figure, which covers cooling too).",
            ["doe_thermostat"], []))

    # 7. Standby
    standby_year = max(0.0, tot["always_on"] * year_f - fridge_year)
    if standby_year > 0:
        out.append(Saving(
            "standby_strip", "Switch off standby loads at a power strip",
            "Put the TV, console, speakers and monitor on one strip and switch it off at night and when away.",
            standby_year * STANDBY_CUT_ASSUMED, standby_year * STANDBY_CUT_ASSUMED * sem_days / 365.0,
            f"Always-on kWh not attributed to the fridge ({standby_year:.0f} kWh/yr) x {STANDBY_CUT_ASSUMED:.0%}.",
            [], [f"The {STANDBY_CUT_ASSUMED:.0%} cut is an assumption, not a sourced figure."]))

    for s in out:
        s.price(g)
    ranked = sorted(out, key=lambda s: s.kwh_year, reverse=True)
    return Plan(ranked, ranked[:3], (s0, s1), sem_note, fridge_year, fridge_method)
