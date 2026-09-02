"""Synthetic hourly load for a 175-household rural community.

No public granular consumption dataset exists for rural Ghana, so the profile
is constructed and calibrated against published benchmarks, following the
approach the paper describes:

  * 149 basic residential households at ~500 kWh/yr each -- **bimodal**, with a
    morning cooking/lighting peak and a larger evening peak.
  * 26 productive-use customers at ~1,700 kWh/yr each (>3x the residential
    figure, as seen in metered West African mini-grid data) -- **daytime
    dominant**, tracking shop and mill operating hours.

On top of the deterministic shape sit stochastic household noise and occasional
productive-use spikes (grain milling, welding). Those spikes matter: they are
low-probability, high-magnitude events that no forecaster predicts well, and in
the paper they are the dominant driver of unmet load under a soft reliability
penalty.

The combined annual demand calibrates to 149*500 + 26*1700 = 118,700 kWh/yr.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LoadConfig

HOURS_PER_YEAR = 8766.0  # 365.25 * 24, so leap years do not bias the calibration

# Hour-of-day shape for a basic residential household: dark-hours baseline,
# a morning peak around 06:00-07:00, and a dominant evening peak at 19:00-20:00.
BASIC_SHAPE = np.array([
    0.020, 0.018, 0.017, 0.017, 0.020, 0.030,   # 00-05
    0.055, 0.060, 0.045, 0.028, 0.025, 0.025,   # 06-11
    0.026, 0.025, 0.025, 0.026, 0.032, 0.050,   # 12-17
    0.085, 0.110, 0.105, 0.080, 0.045, 0.030,   # 18-23
])

# Productive-use customers: shops, grain mills, welding, cold storage. Cold
# storage keeps a small overnight baseline; everything else follows daylight.
PRODUCTIVE_SHAPE = np.array([
    0.008, 0.008, 0.008, 0.008, 0.008, 0.012,   # 00-05
    0.030, 0.055, 0.080, 0.110, 0.120, 0.115,   # 06-11
    0.085, 0.085, 0.110, 0.100, 0.075, 0.040,   # 12-17
    0.018, 0.010, 0.008, 0.008, 0.008, 0.008,   # 18-23
])


def _seasonal_factor(index: pd.DatetimeIndex) -> np.ndarray:
    """Mild seasonal swing: hottest months (Mar-Apr) draw more fan load.

    Centred on 1.0 so it redistributes demand across the year without changing
    the annual total.
    """
    doy = index.dayofyear.to_numpy(dtype=float)
    return 1.0 + 0.06 * np.cos(2.0 * np.pi * (doy - 95.0) / 365.25)


def generate_load(index: pd.DatetimeIndex, cfg: LoadConfig, seed: int = 20260709) -> pd.DataFrame:
    """Hourly community demand in kW, with the two customer classes separated."""
    rng = np.random.default_rng(seed)
    hour = index.hour.to_numpy()
    season = _seasonal_factor(index)

    # Sharpening raises the peak-to-trough contrast, which raises the fraction
    # of demand variance a forecaster can explain. Renormalised so the daily
    # energy per customer is untouched.
    gamma = cfg.shape_sharpness
    basic_shape = BASIC_SHAPE**gamma
    basic_shape /= basic_shape.sum()
    productive_shape = PRODUCTIVE_SHAPE**gamma
    productive_shape /= productive_shape.sum()

    # Daily energy per customer, spread over the day by the shape. A shape that
    # sums to 1 over 24 hours means shape[h] * daily_kwh is the kW drawn in
    # hour h (since each hour is 1 h wide).
    basic_daily = cfg.kwh_per_basic_year / (HOURS_PER_YEAR / 24.0)
    productive_daily = cfg.kwh_per_productive_year / (HOURS_PER_YEAR / 24.0)

    basic = cfg.n_basic * basic_daily * basic_shape[hour] * season
    productive = cfg.n_productive * productive_daily * productive_shape[hour] * season

    # Sundays: markets and mills largely idle. Compensate the weekday level so
    # the annual productive-use energy still calibrates to the benchmark.
    is_sunday = index.dayofweek.to_numpy() == 6
    week_factor = np.where(is_sunday, cfg.weekend_productive_factor, 1.0)
    productive = productive * week_factor / ((6.0 + cfg.weekend_productive_factor) / 7.0)

    # Stochastic household heterogeneity, applied at the aggregate. Lognormal so
    # demand stays positive; de-biased so the multiplicative noise does not lift
    # the annual mean.
    sigma = cfg.household_noise_cv
    noise = rng.lognormal(mean=0.0, sigma=sigma, size=len(index)) / np.exp(sigma**2 / 2.0)
    basic = basic * noise

    noise_p = rng.lognormal(mean=0.0, sigma=sigma, size=len(index)) / np.exp(sigma**2 / 2.0)
    productive = productive * noise_p

    # Milling and welding: brief, large, essentially unpredictable. Only during
    # working hours, and only for productive-use customers.
    working = (hour >= 7) & (hour <= 17)
    spikes = (rng.random(len(index)) < cfg.spike_probability) & working
    productive = productive * np.where(spikes, cfg.spike_multiplier, 1.0)
    # Remove the energy the spikes add, so the calibration target still holds.
    productive /= 1.0 + cfg.spike_probability * (cfg.spike_multiplier - 1.0) * working.mean()

    frame = pd.DataFrame(
        {"load_basic_kw": basic, "load_productive_kw": productive, "load_spike": spikes},
        index=index,
    )
    frame["load_kw"] = frame["load_basic_kw"] + frame["load_productive_kw"]
    return frame


def calibrate(load_kw: pd.Series, cfg: LoadConfig) -> float:
    """Scale factor that pins mean annual demand to the benchmark total.

    The stochastic terms are de-biased individually, but rounding and the
    Sunday correction leave a fraction of a percent on the table. Applying one
    global factor keeps the shape untouched while making the annual energy
    exactly the figure the paper calibrates against.
    """
    years = load_kw.resample("YE").sum()
    hours = load_kw.resample("YE").count()
    whole = years[hours >= 8760]
    return float(cfg.annual_kwh / whole.mean())
