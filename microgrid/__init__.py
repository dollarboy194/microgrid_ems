"""Forecast-driven EMS for a rural solar-battery-diesel microgrid (Tamale, Ghana).

Implements the forecast-optimize-evaluate framework: short-term ML forecasts of
PV and load feed a genetic-algorithm dispatch optimiser over a rolling 24-hour
horizon, re-solved every 6 hours; the committed decisions are then replayed
against ground-truth PV and load so forecast error is honestly reflected in the
reported reliability and cost.
"""

from .config import (
    DIESEL_COST_PER_KWH_GHS,
    BatteryConfig,
    DieselConfig,
    EMSConfig,
    ExperimentConfig,
    GAConfig,
    LoadConfig,
    PVConfig,
    SiteConfig,
    SplitConfig,
)

__version__ = "0.2.0"

__all__ = [
    "DIESEL_COST_PER_KWH_GHS",
    "BatteryConfig",
    "DieselConfig",
    "EMSConfig",
    "ExperimentConfig",
    "GAConfig",
    "LoadConfig",
    "PVConfig",
    "SiteConfig",
    "SplitConfig",
]
