import json
import math
from pathlib import Path

import numpy as np
import pytest

from dormwatt import disaggregate as dg
from dormwatt import synth
from dormwatt.cli import SAMPLE_CHECKLIST, SAMPLE_CSV, SAMPLE_TEMPS, main
from dormwatt.factors import LB_TO_KG, energystar, resolve
from dormwatt.io import read_usage, read_usage_text, to_hourly
from dormwatt.recommend import plan
from dormwatt.solar import sun_elevation_deg
from dormwatt.validate import YEAR_HOURS, load_year_temps, run_one, score

ROOT = Path(__file__).resolve().parent.parent


# ---------- parsing ----------

def _rows(n_hours=72, fmt="pge"):
    lines = []
    if fmt == "pge":
        lines += ["Name,Test", "Address,Nowhere", "", "TYPE,DATE,START TIME,END TIME,USAGE (kWh),COST,NOTES"]
        for h in range(n_hours):
            for q in range(4):
                d = 1 + h // 24
                hh = h % 24
                lines.append(f"Electric usage,2025-03-{d:02d},{hh:02d}:{q * 15:02d},{hh:02d}:{q * 15 + 14:02d},0.250,$0.07,")
    elif fmt == "units_wh":
        lines += ["TYPE,DATE,START TIME,END TIME,USAGE,UNITS,COST,NOTES"]
        for h in range(n_hours):
            d = 1 + h // 24
            lines.append(f"Electric usage,03/{d:02d}/2025,{h % 24}:00,{h % 24}:59,1000,Wh,,")
    elif fmt == "iso":
        lines += ["Interval Start,Consumption (kWh)"]
        for h in range(n_hours):
            d = 1 + h // 24
            lines.append(f"2025-03-{d:02d}T{h % 24:02d}:00:00-05:00,1.0")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("fmt,minutes,per_hour", [("pge", 15, 1.0), ("units_wh", 60, 1.0), ("iso", 60, 1.0)])
def test_parse_common_shapes(fmt, minutes, per_hour):
    u = read_usage_text(_rows(fmt=fmt))
    assert u.interval_minutes == minutes
    assert len(u.hourly_kwh) == 72
    assert np.allclose(u.hourly_kwh, per_hour)
    assert u.total_kwh == pytest.approx(72 * per_hour)


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        read_usage_text("hello,world\n1,2\n")


def test_incomplete_hours_are_nan():
    st = np.array(["2025-01-01T00:00", "2025-01-01T00:15", "2025-01-01T00:30", "2025-01-01T00:45",
                   "2025-01-01T01:00"], dtype="datetime64[m]")
    hrs, y = to_hourly(st, np.ones(5), 15)
    assert y[0] == pytest.approx(4.0) and math.isnan(y[1])


def test_sample_file_parses():
    u = read_usage(SAMPLE_CSV)
    assert u.interval_minutes == 15
    assert u.days == pytest.approx(365, abs=0.5)


# ---------- factors ----------

def test_ithaca_defaults_to_nyup_and_ny_price():
    g = resolve()
    assert g.subregion == "NYUP"
    assert g.co2e_lb_per_mwh == pytest.approx(242.776)
    assert g.price_usd_per_kwh == pytest.approx(0.2926)
    # consumption rate = generation rate grossed up for grid loss
    assert g.kg_co2e_per_kwh == pytest.approx(242.776 * LB_TO_KG / 1000 / (1 - g.grid_gross_loss))
    assert g.kg_co2e_per_kwh_nonbaseload > g.kg_co2e_per_kwh


def test_zip_and_subregion_choice():
    assert resolve(zip_code="10001").subregion == "NYCW"
    assert resolve(subregion="camx", state="CA").price_usd_per_kwh > 0.2
    with pytest.raises(ValueError):
        resolve(subregion="NOPE")


def test_energystar_figures_present():
    es = energystar()
    assert es["mini_fridge_energy_star_median_kwh_yr"] < es["mini_fridge_federal_standard_median_kwh_yr"]
    assert 0 < es["desktop_sleep_watts_median"] < 5


# ---------- solar ----------

def test_sun_elevation_ithaca_noon_and_midnight():
    h = np.array(["2025-06-21T12", "2025-06-21T00", "2025-12-21T12"], dtype="datetime64[h]")
    el = sun_elevation_deg(h, minute=30)
    # solar noon at Ithaca in June (EDT) is ~13:10, Dec (EST) ~12:05
    assert 60 < el[0] < 72
    assert el[1] < -20
    assert 20 < el[2] < 25


# ---------- estimator on a controlled synthetic home ----------

def _toy(heater_kw=1.0, bulbs_w=120.0, base_kw=0.05, seed=0):
    rng = np.random.default_rng(seed)
    hours = np.arange(np.datetime64("2025-01-01T00", "h"), np.datetime64("2025-07-01T00", "h"))
    n = len(hours)
    hod = (hours - hours.astype("datetime64[D]")).astype(int)
    day = (hours.astype("datetime64[D]") - hours[0].astype("datetime64[D]")).astype(int)
    temps = 25 + 50 * day / day.max() + 8 * np.sin((hod - 15) / 24 * 2 * np.pi) + rng.normal(0, 3, n)
    away_days = day < 10
    home = (~away_days) & ((hod >= 17) | (hod <= 1))
    heat = np.where(home, heater_kw * np.clip((60 - temps) / 30, 0, 1), 0.0)
    elec = np.where(home, 0.1 + 0.05 * rng.random(n), 0.0)
    y = base_kw + heat + elec + rng.normal(0, 0.005, n)
    return hours, np.clip(y, 0, None), temps, {"base": base_kw * n, "heat": heat.sum(), "elec": elec.sum()}


def test_estimator_recovers_toy_household():
    hours, y, temps, truth = _toy()
    cl = dg.Checklist(space_heater=True, window_ac=False, lighting="led", bulbs=1)
    est = dg.estimate(hours, y, cl, temps, n_boot=20)
    assert est.lighting_method == "bottom-up"
    # categories add up to the meter
    assert sum(est.totals.values()) == pytest.approx(np.nansum(y), rel=1e-6)
    assert est.totals["always_on"] == pytest.approx(truth["base"], rel=0.10)
    assert est.totals["heating"] == pytest.approx(truth["heat"], rel=0.15)
    assert est.totals["cooling"] == 0
    assert len(est.away_days) >= 8
    for c in dg.CATEGORIES:
        lo, hi = est.intervals[c]
        assert lo <= est.totals[c] + 1e-9 <= hi + 1e-9


def test_regression_lighting_path_warns():
    hours, y, temps, _ = _toy()
    est = dg.estimate(hours, y, dg.Checklist(bulbs=None), temps, n_boot=0)
    assert est.lighting_method == "regression"
    assert any("bulb count" in w.lower() for w in est.warnings)


def test_nonnegative_components():
    hours, y, temps, _ = _toy(seed=3)
    est = dg.estimate(hours, y, dg.Checklist(), temps, n_boot=0)
    for v in est.components.values():
        assert (v >= -1e-12).all()


def test_heating_season_only_warns():
    hours, y, temps, _ = _toy()
    cold = hours < np.datetime64("2025-04-01T00", "h")
    est = dg.estimate(hours[cold], y[cold], dg.Checklist(bulbs=1, lighting="led"), temps[cold], n_boot=0)
    assert any("warm days" in w for w in est.warnings)


def test_no_temperature_warns_and_has_no_heating():
    hours, y, _, _ = _toy()
    est = dg.estimate(hours, y, dg.Checklist(space_heater=True), None, n_boot=0)
    assert est.totals["heating"] == 0
    assert any("temperature" in w.lower() for w in est.warnings)


# ---------- validation on the minute-level simulator ----------

@pytest.fixture(scope="module")
def year_temps():
    return load_year_temps()


def test_validation_accuracy_bounds(year_temps):
    runs = [run_one(s, year_temps) for s in range(8)]
    sc = score(runs)
    assert sc["always_on"]["median_ape_pct"] < 20
    assert sc["heating"]["n_present"] == 0 or sc["heating"]["median_ape_pct"] < 15
    assert sc["electronics"]["median_ape_pct"] < 50
    tops = [abs(r["top3_kwh_est"] - r["top3_kwh_true"]) / max(r["top3_kwh_true"], 1) for r in runs]
    assert np.median(tops) < 0.2


def test_simulator_truth_matches_meter(year_temps):
    hh = synth.simulate(year_temps, YEAR_HOURS, seed=5)
    st, k = hh.interval_kwh()
    # the meter rounds each 15-min reading to 1 Wh, so allow a small rounding gap
    assert sum(hh.truth_kwh().values()) == pytest.approx(k.sum(), rel=1e-2)


# ---------- savings arithmetic ----------

def test_savings_use_sourced_factors():
    hours, y, temps, _ = _toy()
    cl = dg.Checklist(space_heater=True, lighting="incandescent", bulbs=4, desktop_pc=True,
                      pc_left_on_hours=4, pc_idle_watts=50)
    est = dg.estimate(hours, y, cl, temps, n_boot=0)
    g = resolve()
    p = plan(est, cl, g)
    by = {s.key: s for s in p.savings}
    f = 365 / est.days
    assert by["heater_setback"].kwh_year == pytest.approx(0.10 * est.totals["heating"] * f)
    assert by["led"].kwh_year == pytest.approx(0.75 * est.totals["lighting"] * f)
    for s in p.savings:
        assert s.usd_year == pytest.approx(s.kwh_year * g.price_usd_per_kwh)
        assert s.kg_year == pytest.approx(s.kwh_year * g.kg_co2e_per_kwh)
        assert s.kwh_year >= 0 and s.kwh_semester >= 0
    assert [s.kwh_year for s in p.savings] == sorted((s.kwh_year for s in p.savings), reverse=True)
    assert len(p.top) == 3


# ---------- end to end ----------

def test_cli_report_end_to_end(tmp_path):
    out = tmp_path / "r.html"
    js = tmp_path / "r.json"
    rc = main(["report", str(SAMPLE_CSV), "--checklist", str(SAMPLE_CHECKLIST), "--temps", str(SAMPLE_TEMPS),
               "--out", str(out), "--json", str(js), "--boot", "20"])
    assert rc == 0
    h = out.read_text()
    assert "Your three biggest wins" in h and "data:image/png;base64," in h and "eGRID" in h
    d = json.loads(js.read_text())
    assert len(d["top3"]) == 3
    truth = json.loads((ROOT / "data" / "sample_truth.json").read_text())["kwh"]
    assert d["kwh_per_year"]["heating"] == pytest.approx(truth["heating"], rel=0.15)
