"""Rebuild the small factor tables in data/factors/ from the public source files.

Every number the estimator uses for CO2, price or appliance efficiency comes from a
file produced here, so the provenance is one script long.

Usage (downloads are done with curl first; see README "Rebuilding the factor tables"):

    python scripts/build_factors.py --egrid-zip-tool power_profiler_zipcode_tool_v14.3.xlsx \
        --eia-table epm_5_6_b.html --fridges fridge_compact.json \
        --desktops comp_desktop.json --room-ac rac.json

openpyxl is only needed for this script, not for the estimator.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "factors"

EGRID_URL = "https://www.epa.gov/system/files/documents/2025-06/power_profiler_zipcode_tool_v14.3.xlsx"
EGRID_SUMMARY_URL = "https://www.epa.gov/system/files/documents/2025-06/summary_tables_rev2.xlsx"
EIA_URL = "https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_b"
ES_FRIDGE_URL = "https://data.energystar.gov/resource/p5st-her9.json"
ES_DESKTOP_URL = "https://data.energystar.gov/resource/rxdj-2c88.json"
ES_RAC_URL = "https://data.energystar.gov/resource/5xn2-dv4h.json"

STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def build_egrid(path: Path) -> None:
    import openpyxl  # only needed here

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rates = wb["Subregion Rates (lbs-MWh)"]
    rows = list(rates.iter_rows(values_only=True))
    hdr_i = next(i for i, r in enumerate(rows) if r and r[1] == "SUBRGN")
    out_rows = []
    for r in rows[hdr_i + 1:]:
        if not r or not r[1] or not isinstance(r[2], (int, float)):
            continue
        out_rows.append({
            "subregion": r[1], "name": str(r[0]).strip(),
            "co2_lb_per_mwh": r[2], "co2e_lb_per_mwh": r[5], "grid_gross_loss": r[8],
        })
    # Non-baseload CO2e (a marginal-style rate) lives in summary Table 1; the values below
    # are read from the same eGRID2023 rev2 release by build_nonbaseload().
    with open(OUT / "egrid2023_subregions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    zs = wb["Zip-subregion"]
    with open(OUT / "zip_to_egrid_subregion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zip", "subregion_1", "subregion_2", "subregion_3"])
        for i, r in enumerate(zs.iter_rows(values_only=True)):
            if i == 0 or not r or not r[0]:
                continue
            subs = [s for s in r[1:4] if s]
            w.writerow([str(r[0]).zfill(5)] + subs + [""] * (3 - len(subs)))
    print(f"eGRID: {len(out_rows)} subregions")


def build_nonbaseload(path: Path) -> None:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Table 1"]
    nb = {}
    for r in ws.iter_rows(values_only=True):
        # Columns: (blank, acronym, name, CO2, CH4, N2O, CO2e, NOx, ozone NOx, SO2,
        #           non-baseload CO2, CH4, N2O, CO2e, ...)
        if r and isinstance(r[1], str) and len(r[1]) == 4 and r[1].isupper() and isinstance(r[3], (int, float)):
            nb[r[1]] = r[13]
    src = OUT / "egrid2023_subregions.csv"
    rows = list(csv.DictReader(open(src)))
    for row in rows:
        row["nonbaseload_co2e_lb_per_mwh"] = nb.get(row["subregion"], "")
    with open(src, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"non-baseload rates for {len(nb)} subregions")


def build_eia(path: Path) -> None:
    t = path.read_text(errors="ignore")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    cells = [html.unescape(c).strip() for c in re.split(r"<[^>]+>", t)]
    cells = [c for c in cells if c]
    prices = {}
    for i, c in enumerate(cells):
        if c in STATE_ABBR and i + 1 < len(cells):
            try:
                prices[STATE_ABBR[c]] = float(cells[i + 1])
            except ValueError:
                pass
    with open(OUT / "eia_residential_price.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "residential_cents_per_kwh"])
        for k in sorted(prices):
            w.writerow([k, prices[k]])
    print(f"EIA: {len(prices)} states, NY={prices.get('NY')}")


def build_energystar(fridges: Path, desktops: Path, rac: Path) -> None:
    fr = json.loads(fridges.read_text())
    mini = [r for r in fr if r.get("type") == "Compact Refrigerator"
            and 2.5 <= float(r["capacity_total_volume_ft3"]) < 4.5]
    dk = json.loads(desktops.read_text())
    ac = json.loads(rac.read_text())
    ac_small = [r for r in ac if r.get("cooling_capacity_btu_hour")
                and 5000 <= float(r["cooling_capacity_btu_hour"]) <= 8000
                and r.get("percent_less_energy_use_than_us_federal_standard")]
    figs = {
        "mini_fridge_energy_star_median_kwh_yr": statistics.median(float(r["annual_energy_use_kwh_yr"]) for r in mini),
        "mini_fridge_federal_standard_median_kwh_yr": statistics.median(float(r["us_federal_standard_kwh_yr"]) for r in mini),
        "mini_fridge_models": len(mini),
        "mini_fridge_filter": "type = Compact Refrigerator, 2.5 <= total volume < 4.5 cu ft",
        "desktop_sleep_watts_median": statistics.median(float(r["sleep_mode_watts"]) for r in dk if r.get("sleep_mode_watts")),
        "desktop_short_idle_watts_median": statistics.median(float(r["short_idle_watts"]) for r in dk),
        "desktop_models": len(dk),
        "room_ac_pct_less_than_federal_median": statistics.median(float(r["percent_less_energy_use_than_us_federal_standard"]) for r in ac_small),
        "room_ac_models": len(ac_small),
        "room_ac_filter": "cooling capacity 5,000-8,000 Btu/h",
    }
    (OUT / "energystar_figures.json").write_text(json.dumps(figs, indent=2) + "\n")
    print("ENERGY STAR:", figs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--egrid-zip-tool", type=Path, required=True)
    ap.add_argument("--egrid-summary", type=Path, required=True)
    ap.add_argument("--eia-table", type=Path, required=True)
    ap.add_argument("--fridges", type=Path, required=True)
    ap.add_argument("--desktops", type=Path, required=True)
    ap.add_argument("--room-ac", type=Path, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    build_egrid(a.egrid_zip_tool)
    build_nonbaseload(a.egrid_summary)
    build_eia(a.eia_table)
    build_energystar(a.fridges, a.desktops, a.room_ac)


if __name__ == "__main__":
    main()
