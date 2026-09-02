"""Temperature-corrected PV output model.

Cell temperature from the NOCT rating:

    T_cell = T_air + (NOCT - 20) / 800 * GHI

and power from the standard first-order efficiency model:

    P = P_stc * (GHI / 1000) * (1 + gamma * (T_cell - 25)) * derate

with gamma = -0.4%/degC and an 86% system derate covering inverter efficiency,
wiring losses, soiling and mismatch. In Tamale this matters: cell temperatures
reach ~60 degC at midday, costing roughly 14% of nameplate output exactly when
irradiance peaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PVConfig


def cell_temperature_c(air_temp_c: np.ndarray, ghi_w_m2: np.ndarray, cfg: PVConfig) -> np.ndarray:
    """NOCT-based estimate of module cell temperature."""
    return np.asarray(air_temp_c, dtype=float) + (cfg.noct_c - 20.0) / 800.0 * np.asarray(ghi_w_m2, dtype=float)


def pv_power_kw(air_temp_c: np.ndarray, ghi_w_m2: np.ndarray, cfg: PVConfig) -> np.ndarray:
    """Hourly AC power in kW from irradiance and air temperature."""
    ghi = np.asarray(ghi_w_m2, dtype=float)
    t_cell = cell_temperature_c(air_temp_c, ghi, cfg)

    temp_factor = 1.0 + cfg.temp_coeff_per_c * (t_cell - cfg.stc_temp_c)
    power = cfg.capacity_kwp * (ghi / cfg.stc_irradiance_w_m2) * temp_factor * cfg.derate

    # No output in darkness, and a hot module never generates negative power.
    return np.clip(power, 0.0, None)


def attach_pv(weather: pd.DataFrame, cfg: PVConfig) -> pd.Series:
    """PV generation series aligned to the weather record."""
    return pd.Series(
        pv_power_kw(weather["temp_c"].to_numpy(), weather["ghi_w_m2"].to_numpy(), cfg),
        index=weather.index,
        name="pv_kw",
    )


def specific_yield_kwh_per_kwp(pv_kw: pd.Series, cfg: PVConfig) -> float:
    """Annual energy per kWp installed, averaged over whole years in the series."""
    annual = pv_kw.resample("YE").sum()
    # Drop partial years so the average is not dragged down by a stub.
    hours = pv_kw.resample("YE").count()
    whole = annual[hours >= 8760]
    return float(whole.mean() / cfg.capacity_kwp)
