"""Command line: report, validate, sample, demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import disaggregate as dg
from .factors import resolve
from .io import read_temps, read_usage, write_green_button_csv
from .recommend import plan as make_plan

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
SAMPLE_CSV = DATA / "sample_greenbutton.csv"
SAMPLE_CHECKLIST = DATA / "sample_checklist.json"
SAMPLE_TEMPS = DATA / "ithaca_hourly_temp_2025.csv"
VALIDATION_JSON = DOCS / "validation.json"
SAMPLE_SEED = 17  # the 3-bulb candidate (of seeds 0-19) closest to the median error; see README
SAMPLE_OVERRIDES = {"has_heater": True, "heat_at_night": False, "has_ac": False, "bulb": "halogen",
                    "has_pc": True, "pc_left_on": 6, "summer_home": False}


def _checklist(path: str | None, args) -> dg.Checklist:
    d = json.loads(Path(path).read_text()) if path else {}
    for k in ("zip", "state", "subregion"):
        if getattr(args, k, None):
            d[k] = getattr(args, k)
    return dg.Checklist.from_dict(d)


def run_report(csv_path, checklist_path=None, temps_path=None, out="report.html", json_out=None,
               args=None, title_note="", chart_dir=None, n_boot=200):
    from .report import build_html, load_validation, to_json

    usage = read_usage(csv_path)
    cl = _checklist(checklist_path, args)
    temps = read_temps(temps_path, usage.hours) if temps_path else None
    est = dg.estimate(usage.hours, usage.hourly_kwh, cl, temps, n_boot=n_boot)
    g = resolve(cl.zip, cl.subregion, cl.state)
    p = make_plan(est, cl, g)
    val = load_validation(VALIDATION_JSON)
    Path(out).write_text(build_html(est, p, cl, g, title_note, val, chart_dir))
    summary = to_json(est, p, g)
    if json_out:
        Path(json_out).write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return est, p, g, summary


def make_sample() -> None:
    from . import synth
    from .validate import YEAR_HOURS

    temps = read_temps(SAMPLE_TEMPS, YEAR_HOURS)
    hh = synth.simulate(temps, YEAR_HOURS, seed=SAMPLE_SEED, overrides=SAMPLE_OVERRIDES)
    st, k = hh.interval_kwh()
    header = ["Name,Sample household (synthetic data for demonstration)",
              "Address,Ithaca NY 14850", "Account Number,SAMPLE-0001", "Service,Electric 1"]
    write_green_button_csv(SAMPLE_CSV, synth.to_datetimes(st), k, 15, price=None, header_lines=header)
    truth = hh.truth_kwh()
    (DATA / "sample_truth.json").write_text(json.dumps(
        {"note": "True per-category kWh of the synthetic sample household (2025), for checking the estimate.",
         "kwh": truth, "params": {kk: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for kk, v in hh.params.items()}},
        indent=2, default=float) + "\n")
    print(f"wrote {SAMPLE_CSV} ({len(k)} readings, {k.sum():.0f} kWh)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dormwatt", description="Estimate per-appliance energy from a utility usage CSV "
                                 "and rank the swaps that save the most kWh, dollars and CO2e.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="build an HTML report from a usage CSV")
    r.add_argument("csv")
    r.add_argument("--checklist", help="appliance checklist JSON (see data/sample_checklist.json)")
    r.add_argument("--temps", help="hourly outdoor temperature CSV (timestamp,temp_f)")
    r.add_argument("--out", default="report.html")
    r.add_argument("--json", dest="json_out", help="also write the numbers as JSON")
    r.add_argument("--zip", help="ZIP code, picks the eGRID subregion (default 14850, Ithaca)")
    r.add_argument("--subregion", help="eGRID subregion acronym, e.g. NYUP, overrides --zip")
    r.add_argument("--state", help="two-letter state for the EIA price (default NY)")
    r.add_argument("--boot", type=int, default=200, help="bootstrap resamples (default 200)")
    v = sub.add_parser("validate", help="score the estimator on synthetic households")
    v.add_argument("--n", type=int, default=40)
    v.add_argument("--out", default=str(VALIDATION_JSON))
    sub.add_parser("sample", help="rebuild data/sample_greenbutton.csv")
    sub.add_parser("demo", help="rebuild docs/index.html and docs/*.png from the sample")
    a = ap.parse_args(argv)

    if a.cmd == "report":
        temps = a.temps
        if temps is None and Path(a.csv).resolve() == SAMPLE_CSV.resolve():
            temps = str(SAMPLE_TEMPS)
        est, p, g, s = run_report(a.csv, a.checklist, temps, a.out, a.json_out, a, n_boot=a.boot)
        f = 365 / est.days
        print(f"{est.days:.0f} days, {est.total_kwh * f:,.0f} kWh/yr. Grid {g.subregion}: {g.kg_co2e_per_kwh:.3f} kg CO2e/kWh; "
              f"price ${g.price_usd_per_kwh:.4f}/kWh ({g.state}).")
        for c in dg.CATEGORIES:
            lo, hi = est.intervals[c]
            print(f"  {dg.LABELS[c]:45s} {est.totals[c] * f:7.0f} kWh/yr  [{lo * f:.0f}-{hi * f:.0f}]")
        print("Top 3:")
        for i, x in enumerate(p.top, 1):
            print(f"  {i}. {x.title}: {x.kwh_year:.0f} kWh/yr, ${x.usd_year:.0f}/yr, {x.kg_year:.1f} kg CO2e/yr "
                  f"(semester {x.kwh_semester:.0f} kWh)")
        print(f"wrote {a.out}")
    elif a.cmd == "validate":
        from .validate import run

        rep = run(a.n, Path(a.out))
        for name, sc in rep["settings"].items():
            print(name)
            for c, m in sc.items():
                if c == "_top3_savings":
                    print(f"  top-3 savings: median |err| {m['median_ape_pct']:.1f}%  p80 {m['p80_ape_pct']:.1f}%  "
                          f"median bias {m['bias_pct']:+.1f}%  same top 3 as truth {m['same_top3_share']:.0%}")
                    continue
                if m["n_present"]:
                    cov = f"  90% interval coverage {m['interval_coverage']:.0%}" if "interval_coverage" in m else ""
                    print(f"  {c:12s} n={m['n_present']:2d} median |err| {m['median_ape_pct']:5.1f}%  "
                          f"p80 {m['p80_ape_pct']:5.1f}%  MAE {m['mae_kwh']:6.1f} kWh  bias {m['bias_kwh']:+6.1f}{cov}")
        print(f"wrote {a.out}")
    elif a.cmd == "sample":
        make_sample()
    elif a.cmd == "demo":
        DOCS.mkdir(exist_ok=True)
        run_report(SAMPLE_CSV, SAMPLE_CHECKLIST, SAMPLE_TEMPS, DOCS / "index.html", DOCS / "sample_report.json",
                   title_note="Demo: this report is built from a synthetic sample household, not a real customer.",
                   chart_dir=DOCS)
        print(f"wrote {DOCS / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
