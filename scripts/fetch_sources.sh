#!/usr/bin/env bash
# Download the public source files behind data/factors/ and data/ithaca_hourly_temp_2025.csv,
# then rebuild the factor tables. About 5 MB in total.
set -euo pipefail
cd "$(dirname "$0")/.."
D="${1:-$(mktemp -d)}"
mkdir -p "$D"
UA="Mozilla/5.0"
curl -sSL -A "$UA" -o "$D/pp_zip.xlsx" https://www.epa.gov/system/files/documents/2025-06/power_profiler_zipcode_tool_v14.3.xlsx
curl -sSL -A "$UA" -o "$D/summary_tables_rev2.xlsx" https://www.epa.gov/system/files/documents/2025-06/summary_tables_rev2.xlsx
curl -sSL -A "$UA" -o "$D/epm_5_6_b.html" "https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_b"
curl -sSL -G -o "$D/fridge_compact.json" https://data.energystar.gov/resource/p5st-her9.json \
  --data-urlencode '$where=compact="Yes"' --data-urlencode '$limit=5000'
curl -sSL -G -o "$D/comp_desktop.json" https://data.energystar.gov/resource/rxdj-2c88.json \
  --data-urlencode '$where=type="Desktop"' --data-urlencode '$limit=5000'
curl -sSL -G -o "$D/rac.json" https://data.energystar.gov/resource/5xn2-dv4h.json --data-urlencode '$limit=3000'
curl -sSL -o "$D/ithaca_2025.json" "https://archive-api.open-meteo.com/v1/archive?latitude=42.4440&longitude=-76.5019&start_date=2025-01-01&end_date=2025-12-31&hourly=temperature_2m&timezone=America%2FNew_York&temperature_unit=fahrenheit"
python scripts/build_factors.py --egrid-zip-tool "$D/pp_zip.xlsx" --egrid-summary "$D/summary_tables_rev2.xlsx" \
  --eia-table "$D/epm_5_6_b.html" --fridges "$D/fridge_compact.json" --desktops "$D/comp_desktop.json" --room-ac "$D/rac.json"
python - "$D/ithaca_2025.json" <<'PY'
import csv, json, sys
h = json.load(open(sys.argv[1]))["hourly"]
with open("data/ithaca_hourly_temp_2025.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "temp_f"])
    for t, v in zip(h["time"], h["temperature_2m"]):
        w.writerow([t.replace("T", " "), v])
PY
echo "rebuilt data/factors/ and data/ithaca_hourly_temp_2025.csv from $D"
