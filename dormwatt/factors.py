"""Load the sourced conversion factors (CO2 rate, price, ENERGY STAR figures)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

FACTOR_DIR = Path(__file__).resolve().parent.parent / "data" / "factors"
LB_TO_KG = 0.45359237
DEFAULT_ZIP = "14850"  # Ithaca, NY
DEFAULT_STATE = "NY"


@lru_cache(maxsize=1)
def egrid_table() -> dict[str, dict]:
    out = {}
    with open(FACTOR_DIR / "egrid2023_subregions.csv") as f:
        for r in csv.DictReader(f):
            out[r["subregion"]] = {
                "name": r["name"],
                "co2e_lb_per_mwh": float(r["co2e_lb_per_mwh"]),
                "co2_lb_per_mwh": float(r["co2_lb_per_mwh"]),
                "grid_gross_loss": float(r["grid_gross_loss"]),
                "nonbaseload_co2e_lb_per_mwh": float(r["nonbaseload_co2e_lb_per_mwh"] or "nan"),
            }
    return out


@lru_cache(maxsize=1)
def zip_table() -> dict[str, list[str]]:
    out = {}
    with open(FACTOR_DIR / "zip_to_egrid_subregion.csv") as f:
        for r in csv.DictReader(f):
            out[r["zip"]] = [s for s in (r["subregion_1"], r["subregion_2"], r["subregion_3"]) if s]
    return out


@lru_cache(maxsize=1)
def price_table() -> dict[str, float]:
    with open(FACTOR_DIR / "eia_residential_price.csv") as f:
        return {r["state"]: float(r["residential_cents_per_kwh"]) for r in csv.DictReader(f)}


@lru_cache(maxsize=1)
def energystar() -> dict:
    return json.loads((FACTOR_DIR / "energystar_figures.json").read_text())


@dataclass(frozen=True)
class GridFactors:
    subregion: str
    subregion_name: str
    co2e_lb_per_mwh: float          # eGRID total output rate (generation)
    grid_gross_loss: float          # fraction lost between plant and meter
    nonbaseload_co2e_lb_per_mwh: float
    state: str
    price_usd_per_kwh: float
    note: str = ""

    @property
    def kg_co2e_per_kwh(self) -> float:
        """Consumption-based rate: generation rate grossed up for line losses.

        One kWh at the meter needs 1 / (1 - GGL) kWh at the plant.
        """
        return self.co2e_lb_per_mwh * LB_TO_KG / 1000.0 / (1.0 - self.grid_gross_loss)

    @property
    def kg_co2e_per_kwh_nonbaseload(self) -> float:
        return self.nonbaseload_co2e_lb_per_mwh * LB_TO_KG / 1000.0 / (1.0 - self.grid_gross_loss)


def resolve(zip_code: str | None = None, subregion: str | None = None,
            state: str | None = None) -> GridFactors:
    """Pick the eGRID subregion (explicit > ZIP > Ithaca default) and the state price."""
    eg = egrid_table()
    note = ""
    if subregion:
        sub = subregion.upper()
        if sub not in eg:
            raise ValueError(f"Unknown eGRID subregion {subregion!r}; choose one of {sorted(eg)}")
    else:
        z = (zip_code or DEFAULT_ZIP).strip()[:5].zfill(5)
        subs = zip_table().get(z)
        if not subs:
            raise ValueError(f"ZIP {z} is not in the EPA Power Profiler ZIP table; pass --subregion")
        sub = subs[0]
        if len(subs) > 1:
            note = (f"ZIP {z} spans {', '.join(subs)}; using {sub} (EPA's first listed). "
                    "Pass --subregion to choose another.")
    st = (state or DEFAULT_STATE).upper()
    prices = price_table()
    if st not in prices:
        raise ValueError(f"No EIA residential price for state {st!r}")
    r = eg[sub]
    return GridFactors(sub, r["name"], r["co2e_lb_per_mwh"], r["grid_gross_loss"],
                       r["nonbaseload_co2e_lb_per_mwh"], st, prices[st] / 100.0, note)
