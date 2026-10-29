"""Read utility interval-usage CSVs (Green Button "Download My Data" style) and weather CSVs.

Supported shapes (header row is found automatically; any preamble lines above it are skipped):

1. Utility export with separate date and time columns, e.g.
       TYPE,DATE,START TIME,END TIME,USAGE (kWh),COST,NOTES
       Electric usage,2025-01-01,00:00,00:14,0.05,$0.01,
   A separate UNITS column (kWh or Wh) is honoured when present.
2. One timestamp column plus a usage column, e.g.
       Interval Start,Consumption (kWh)      or      timestamp,kwh
Timestamps may be ISO (2025-01-01 00:00, 2025-01-01T00:00:00-05:00) or US style (1/1/2025 0:00).
Time-zone offsets are dropped: everything is treated as local wall-clock time.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%m/%d/%Y %I:%M %p", "%m/%d/%Y", "%m/%d/%y %H:%M", "%m/%d/%y",
]


@dataclass
class Usage:
    """Interval readings as parsed (start times, kWh) plus the hourly series used for fitting."""
    starts: np.ndarray          # datetime64[m]
    kwh: np.ndarray             # float, per interval
    interval_minutes: int
    hours: np.ndarray           # datetime64[h], contiguous
    hourly_kwh: np.ndarray      # float, NaN where no data
    source_name: str = ""

    @property
    def days(self) -> float:
        return float(np.isfinite(self.hourly_kwh).sum()) / 24.0

    @property
    def total_kwh(self) -> float:
        return float(np.nansum(self.hourly_kwh))


def parse_time(s: str) -> datetime:
    s = s.strip().strip('"')
    s = re.sub(r"(Z|[+-]\d{2}:?\d{2})$", "", s.replace("T", " ")).strip()
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date/time: {s!r}")


def _num(s: str) -> float:
    s = s.strip().replace("$", "").replace(",", "")
    return float(s) if s else float("nan")


def _find_header(rows: list[list[str]]) -> int:
    for i, r in enumerate(rows):
        low = [c.strip().lower() for c in r]
        has_time = any(("date" in c) or ("start" in c) or ("timestamp" in c) or c == "time" for c in low)
        has_val = any(("usage" in c) or ("kwh" in c) or ("consumption" in c) or c in ("value", "import")
                      or c.startswith("import") for c in low)
        if has_time and has_val and len(r) >= 2:
            return i
    raise ValueError("Could not find a header row with a date/time column and a usage column")


def _col(low: list[str], *keys: str, exclude: tuple[str, ...] = ()) -> int | None:
    for k in keys:
        for j, c in enumerate(low):
            if k in c and not any(e in c for e in exclude):
                return j
    return None


def read_usage_text(text: str, source_name: str = "") -> Usage:
    rows = list(csv.reader(io.StringIO(text)))
    h = _find_header(rows)
    low = [c.strip().lower() for c in rows[h]]
    i_date = _col(low, "date", exclude=("end",))
    i_start = _col(low, "start time", "start", "timestamp", "interval", exclude=("date",) if i_date is not None else ())
    if i_start == i_date:
        i_start = None
    if i_date is None and i_start is None:
        i_start = _col(low, "time")
    i_end = _col(low, "end time", "end")
    i_val = _col(low, "usage", "consumption", "import", "kwh", "value", exclude=("cost", "export"))
    i_units = _col(low, "unit")
    if i_val is None:
        raise ValueError("No usage column found")
    starts, ends, vals = [], [], []
    for r in rows[h + 1:]:
        if len(r) <= i_val or not any(c.strip() for c in r):
            continue
        try:
            if i_date is not None and i_start is not None and re.search(r"\d:\d", r[i_start]) \
                    and not re.search(r"\d[-/]\d", r[i_start]):
                t0 = parse_time(f"{r[i_date].strip()} {r[i_start].strip()}")
                t1 = parse_time(f"{r[i_date].strip()} {r[i_end].strip()}") if i_end is not None and r[i_end].strip() else None
            else:
                t0 = parse_time(r[i_start if i_start is not None else i_date])
                t1 = parse_time(r[i_end]) if i_end is not None and re.search(r"\d[-/]\d", r[i_end]) else None
            v = _num(r[i_val])
        except (ValueError, IndexError):
            continue  # footer lines, totals, notes
        if i_units is not None and len(r) > i_units and r[i_units].strip().lower() == "wh":
            v /= 1000.0
        starts.append(t0)
        ends.append(t1)
        vals.append(v)
    if len(starts) < 48:
        raise ValueError(f"Only {len(starts)} readings parsed; need at least two days of interval data")
    order = np.argsort(np.array(starts, dtype="datetime64[m]"))
    st = np.array(starts, dtype="datetime64[m]")[order]
    kwh = np.array(vals, dtype=float)[order]
    # Interval length: END-START when given (END is often inclusive, e.g. 00:00-00:14), else the mode gap.
    durs = [((e - s).total_seconds() / 60.0) for s, e in zip(starts, ends) if e is not None]
    if durs:
        d = float(np.median(durs))
        minutes = int(round(d + 1)) if round(d) % 5 == 4 else int(round(d))
    else:
        gaps = np.diff(st).astype(int)
        gaps = gaps[gaps > 0]
        vals_, counts = np.unique(gaps, return_counts=True)
        minutes = int(vals_[np.argmax(counts)])
    if minutes <= 0 or 60 % minutes != 0 and minutes != 60:
        raise ValueError(f"Unsupported interval length {minutes} min (need 5, 15, 30 or 60)")
    hours, hourly = to_hourly(st, kwh, minutes)
    return Usage(st, kwh, minutes, hours, hourly, source_name)


def read_usage(path: str | Path) -> Usage:
    p = Path(path)
    return read_usage_text(p.read_text(encoding="utf-8-sig", errors="replace"), p.name)


def to_hourly(starts: np.ndarray, kwh: np.ndarray, minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum intervals into clock hours. Hours with incomplete coverage are set to NaN."""
    hrs = starts.astype("datetime64[h]")
    h0, h1 = hrs.min(), hrs.max()
    n = int((h1 - h0).astype(int)) + 1
    idx = (hrs - h0).astype(int)
    total = np.zeros(n)
    count = np.zeros(n)
    good = np.isfinite(kwh)
    np.add.at(total, idx[good], kwh[good])
    np.add.at(count, idx[good], 1)
    need = 60 // minutes
    out = np.where(count >= need, total, np.nan)  # DST fall-back hour has 2x readings: kept, summed
    return h0 + np.arange(n).astype("timedelta64[h]"), out


def read_temps(path: str | Path, hours: np.ndarray) -> np.ndarray:
    """Hourly outdoor temperature (deg F) aligned to `hours`; NaN where missing.

    CSV: a timestamp column and a temperature column (temp_f, or temp_c which is converted).
    """
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    low = [c.strip().lower() for c in rows[0]]
    i_t = _col(low, "time", "date")
    i_f = _col(low, "temp_f", "fahrenheit", "(f)", "°f")
    i_c = _col(low, "temp_c", "celsius", "(c)", "°c")
    if i_f is None and i_c is None:
        i_f = _col(low, "temp")
    if i_t is None or (i_f is None and i_c is None):
        raise ValueError("Temperature CSV needs a timestamp column and temp_f or temp_c")
    lookup = {}
    for r in rows[1:]:
        try:
            t = np.datetime64(parse_time(r[i_t]), "h")
            v = _num(r[i_f]) if i_f is not None else _num(r[i_c]) * 9 / 5 + 32
        except (ValueError, IndexError):
            continue
        lookup[t] = v
    return np.array([lookup.get(h, np.nan) for h in hours], dtype=float)


def write_green_button_csv(path: str | Path, starts: list[datetime], kwh: np.ndarray,
                           minutes: int, price: float | None = None, header_lines: list[str] | None = None) -> None:
    """Write readings in the common utility export shape (used for the shipped sample)."""
    with open(path, "w", newline="") as f:
        for line in header_lines or []:
            f.write(line + "\n")
        if header_lines:
            f.write("\n")
        w = csv.writer(f)
        w.writerow(["TYPE", "DATE", "START TIME", "END TIME", "USAGE (kWh)", "COST", "NOTES"])
        for t, v in zip(starts, kwh):
            e = t + timedelta(minutes=minutes - 1)
            cost = f"${v * price:.2f}" if price else ""
            w.writerow(["Electric usage", t.strftime("%Y-%m-%d"), t.strftime("%H:%M"), e.strftime("%H:%M"),
                        f"{v:.3f}", cost, ""])
