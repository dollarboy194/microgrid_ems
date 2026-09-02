"""NASA POWER hourly weather retrieval for the Tamale case study.

Downloads the exact variables the paper uses -- all-sky and clear-sky downward
shortwave irradiance, 2 m air temperature, 2 m relative humidity and 2 m wind
speed -- for the case-study coordinates, and caches them to disk so the study
is reproducible without re-hitting the API.

The hourly endpoint is requested one calendar year at a time: NASA POWER caps
the span of a single hourly request, and per-year chunks also make the cache
resumable if a download is interrupted.

Reference: NASA POWER (Prediction of Worldwide Energy Resources),
https://power.larc.nasa.gov/ -- community "RE" (renewable energy).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# NASA POWER names -> the names used throughout this package.
PARAMETERS = {
    "ALLSKY_SFC_SW_DWN": "ghi_w_m2",       # all-sky downward shortwave, W/m^2
    "CLRSKY_SFC_SW_DWN": "ghi_clear_w_m2",  # clear-sky downward shortwave, W/m^2
    "T2M": "temp_c",                        # air temperature at 2 m, deg C
    "RH2M": "humidity_pct",                 # relative humidity at 2 m, %
    "WS2M": "wind_speed_ms",                # wind speed at 2 m, m/s
}

# NASA POWER writes missing values as -999.
MISSING = -999.0


def _fetch_year(lat: float, lon: float, year: int, timeout: float, retries: int = 4) -> pd.DataFrame:
    url = (
        f"{API}?parameters={','.join(PARAMETERS)}&community=RE"
        f"&longitude={lon}&latitude={lat}"
        f"&start={year}0101&end={year}1231&format=JSON"
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.load(response)
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))  # the API rate-limits under load
    else:
        raise RuntimeError(f"NASA POWER request failed for {year}") from last

    block = payload["properties"]["parameter"]
    frame = pd.DataFrame({dest: pd.Series(block[src]) for src, dest in PARAMETERS.items()})
    # Index keys are local strings "YYYYMMDDHH".
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H")
    return frame.sort_index()


def load_weather(
    latitude: float = 9.4008,
    longitude: float = -0.8393,
    start_year: int = 2018,
    end_year: int = 2025,
    cache_dir: Path = Path("data"),
    timeout: float = 60.0,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the hourly weather record, downloading and caching as needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"nasa_power_{latitude:.4f}_{longitude:.4f}"

    frames = []
    for year in range(start_year, end_year + 1):
        path = cache_dir / f"{tag}_{year}.csv"
        if path.exists() and not refresh:
            frames.append(pd.read_csv(path, index_col=0, parse_dates=True))
            continue
        frame = _fetch_year(latitude, longitude, year, timeout)
        frame.to_csv(path)
        frames.append(frame)

    weather = pd.concat(frames).sort_index()
    weather = weather.replace(MISSING, np.nan)

    if weather.isna().any().any():
        counts = weather.isna().sum()
        raise ValueError(f"NASA POWER returned missing values:\n{counts[counts > 0]}")

    # Irradiance cannot be negative; the record occasionally carries tiny
    # negative all-sky values from the retrieval algorithm at dawn and dusk.
    for col in ("ghi_w_m2", "ghi_clear_w_m2"):
        weather[col] = weather[col].clip(lower=0.0)

    return weather


def describe(weather: pd.DataFrame) -> dict[str, float]:
    """Summary statistics used to cross-check the record against the paper."""
    daily_kwh_m2 = weather["ghi_w_m2"].resample("D").sum() / 1000.0
    return {
        "records": float(len(weather)),
        "missing": float(weather.isna().sum().sum()),
        "mean_daily_kwh_m2": float(daily_kwh_m2.mean()),
        "mean_temp_c": float(weather["temp_c"].mean()),
        "max_ghi_w_m2": float(weather["ghi_w_m2"].max()),
        "start": weather.index[0],
        "end": weather.index[-1],
    }
