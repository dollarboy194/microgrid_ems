"""Assembles the hybrid dataset: real NASA POWER weather + synthetic load.

One hourly frame covering 2018-2025, split chronologically -- never randomly --
into training (2018-2023), validation (2024) and a held-out test year (2025)
that is touched exactly once for final reported metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .load import calibrate, generate_load
from .nasa_power import load_weather
from .pvmodel import attach_pv

TARGETS = ("pv_kw", "load_kw")
WEATHER_COLS = ("ghi_w_m2", "ghi_clear_w_m2", "temp_c", "humidity_pct", "wind_speed_ms")


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def years(self, index: pd.DatetimeIndex) -> list[int]:
        return sorted({int(t.year) for t in index})

    def summary(self) -> dict[str, object]:
        f = self.frame
        return {
            "records": len(f),
            "span": f"{f.index[0]:%Y-%m-%d} to {f.index[-1]:%Y-%m-%d}",
            "train_years": self.years(self.train),
            "validation_years": self.years(self.validation),
            "test_years": self.years(self.test),
            "mean_pv_kw": float(f["pv_kw"].mean()),
            "mean_load_kw": float(f["load_kw"].mean()),
            "annual_load_kwh": float(f["load_kw"].resample("YE").sum().mean()),
            "annual_pv_kwh": float(f["pv_kw"].resample("YE").sum().mean()),
        }


def build_dataset(cfg: ExperimentConfig, cache_dir: Path = Path("data")) -> Dataset:
    """Weather, PV and load on one hourly index, with the chronological split."""
    weather = load_weather(
        latitude=cfg.site.latitude,
        longitude=cfg.site.longitude,
        start_year=min(cfg.split.train_years),
        end_year=max(cfg.split.test_years),
        cache_dir=cache_dir,
    )

    frame = weather.copy()
    frame["pv_kw"] = attach_pv(weather, cfg.pv)

    demand = generate_load(weather.index, cfg.load, seed=cfg.split.seed)
    scale = calibrate(demand["load_kw"], cfg.load)
    for col in ("load_basic_kw", "load_productive_kw", "load_kw"):
        demand[col] *= scale
    frame = frame.join(demand)

    year = frame.index.year
    pick = lambda years: frame.index[year.isin(years)]
    return Dataset(
        frame=frame,
        train=pick(cfg.split.train_years),
        validation=pick(cfg.split.validation_years),
        test=pick(cfg.split.test_years),
    )
