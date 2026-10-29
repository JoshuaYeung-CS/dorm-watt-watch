"""Run the estimator on synthetic households with known answers and score it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import copy

from . import disaggregate as dg
from . import synth
from .factors import resolve
from .recommend import plan as make_plan
from .io import read_temps, to_hourly

DATA = Path(__file__).resolve().parent.parent / "data"
YEAR_HOURS = np.arange(np.datetime64("2025-01-01T00", "h"), np.datetime64("2026-01-01T00", "h"))
FALL = (np.datetime64("2025-08-25T00", "h"), np.datetime64("2025-12-20T00", "h"))


def load_year_temps() -> np.ndarray:
    return read_temps(DATA / "ithaca_hourly_temp_2025.csv", YEAR_HOURS)


def run_one(seed: int, temps: np.ndarray, interval: int = 15, window: tuple | None = None,
            n_boot: int = 0) -> dict:
    hh = synth.simulate(temps, YEAR_HOURS, seed=seed, interval_minutes=interval)
    st, k = hh.interval_kwh()
    hrs, y = to_hourly(st, k, interval)
    t = temps
    if window is not None:
        sel = (hrs >= window[0]) & (hrs < window[1])
        hrs, y, t = hrs[sel], y[sel], temps[sel]
        truth = hh.truth_kwh(str(window[0]), str(window[1]))
    else:
        truth = hh.truth_kwh()
    est = dg.estimate(hrs, y, hh.checklist, t, n_boot=n_boot, seed=seed)
    # Same recommendation engine fed with the true category totals: how far off is the saving?
    g = resolve()
    p_est = make_plan(est, hh.checklist, g)
    true_est = copy.copy(est)
    true_est.totals = dict(truth)
    true_est.components = {c: est.components[c] * (truth[c] / est.totals[c]) if est.totals[c] > 0 else est.components[c]
                           for c in est.components}
    p_true = make_plan(true_est, hh.checklist, g)
    true_by_key = {s.key: s.kwh_year for s in p_true.savings}
    top3_est = sum(s.kwh_year for s in p_est.top)
    top3_true = sum(true_by_key.get(s.key, 0.0) for s in p_est.top)
    out = {"seed": seed, "truth": truth, "est": est.totals, "total": float(np.nansum(y)),
           "top3_kwh_est": top3_est, "top3_kwh_true": top3_true,
           "top3_same_as_true_ranking": sorted(s.key for s in p_est.top) == sorted(s.key for s in p_true.top),
           "params": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in hh.params.items()}}
    if n_boot:
        out["intervals"] = est.intervals
    return out


def score(runs: list[dict]) -> dict:
    """Per category: mean absolute error (kWh), mean error (bias), and median absolute % error
    over households where the category is present (truth >= 20 kWh)."""
    res = {}
    for c in dg.CATEGORIES:
        tr = np.array([r["truth"][c] for r in runs])
        es = np.array([r["est"][c] for r in runs])
        tot = np.array([r["total"] for r in runs])
        present = tr >= 20
        ape = np.abs(es[present] - tr[present]) / tr[present] * 100
        res[c] = {
            "n_present": int(present.sum()),
            "mae_kwh": float(np.mean(np.abs(es - tr))),
            "bias_kwh": float(np.mean(es - tr)),
            "median_ape_pct": float(np.median(ape)) if present.any() else None,
            "p80_ape_pct": float(np.percentile(ape, 80)) if present.any() else None,
            "mae_share_pts": float(np.mean(np.abs(es - tr) / tot) * 100),
        }
        if runs and "intervals" in runs[0]:
            cov = [r["intervals"][c][0] <= r["truth"][c] <= r["intervals"][c][1] for r in runs]
            res[c]["interval_coverage"] = float(np.mean(cov))
    return res


def baseline_prior_only(runs: list[dict]) -> dict:
    """A checklist-only guess with no meter data (same totals split by fixed typical shares),
    so the meter-based estimate has something honest to beat."""
    shares = {"always_on": 0.30, "heating": 0.35, "cooling": 0.05, "lighting": 0.12, "electronics": 0.18}
    fake = []
    for r in runs:
        p = r["params"]
        s = dict(shares)
        if not p["has_heater"]:
            s["heating"] = 0
        if not p["has_ac"]:
            s["cooling"] = 0
        z = sum(s.values())
        fake.append({k: v for k, v in r.items() if k != "intervals"} | {"est": {k: r["total"] * v / z for k, v in s.items()}})
    return score(fake)


def run(n: int = 40, out: Path | None = None, n_boot: int = 100) -> dict:
    temps = load_year_temps()
    report = {"n_households": n, "settings": {}}
    for name, interval, window in [("year_15min", 15, None), ("year_hourly", 60, None),
                                   ("fall_semester_15min", 15, FALL)]:
        runs = [run_one(s, temps, interval, window, n_boot=n_boot if name == "year_15min" else 0)
                for s in range(n)]
        report["settings"][name] = score(runs)
        te = np.array([r["top3_kwh_est"] for r in runs])
        tt = np.array([r["top3_kwh_true"] for r in runs])
        ok = tt > 1
        report["settings"][name]["_top3_savings"] = {
            "median_ape_pct": float(np.median(np.abs(te[ok] - tt[ok]) / tt[ok] * 100)),
            "p80_ape_pct": float(np.percentile(np.abs(te[ok] - tt[ok]) / tt[ok] * 100, 80)),
            "bias_pct": float(np.median((te[ok] - tt[ok]) / tt[ok] * 100)),
            "same_top3_share": float(np.mean([r["top3_same_as_true_ranking"] for r in runs])),
        }
        if name == "year_15min":
            report["settings"]["checklist_only_baseline"] = baseline_prior_only(runs)
            report["year_15min_runs"] = [{"seed": r["seed"], "truth": r["truth"], "est": r["est"]} for r in runs]
    if out:
        Path(out).write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report
