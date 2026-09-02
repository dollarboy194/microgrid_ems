"""Leakage-safe feature engineering.

The paper's discipline, restated precisely: a forecast **issued at time t0** for
target time ``t0 + h`` may use

  * observations of PV, load and weather up to and including ``t0``;
  * rolling statistics over windows **ending at t0**;
  * deterministic calendar attributes of the target time (hour, day-of-week,
    month, and their cyclical encodings), which are legitimately known ahead.

It may never touch a realised value after ``t0``, and it may never see any
downstream EMS output -- state of charge, dispatch decisions, unmet load,
curtailment -- because those are consequences of the forecast, not inputs to it.

For a one-step-ahead forecast (``h = 1``) the lags below sit at exactly the
paper's 1, 2, 3, 24, 48 and 168 hours before the prediction time. They are
expressed relative to the *issue* time so the same construction extends to the
full 24-hour horizon the dispatch optimiser needs, where a "1-hour lag relative
to prediction time" would be a value that has not happened yet.

`tests/test_leakage.py` implements the five verification tests the paper
describes, including a bit-exact check against an independently reconstructed
time shift and an empirical check that poisoning the future moves no feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Relative to the issue time t0. At h = 1 these are the paper's
# 1, 2, 3, 24, 48, 168 hours before the predicted hour.
LAGS = (0, 1, 2, 23, 47, 167)
ROLLING_WINDOWS = (3, 24, 168)

# Observed drivers whose recent history is informative. Only lagged values are
# ever used: the contemporaneous weather at the target time is not known when
# the forecast is issued.
WEATHER_FEATURES = ("ghi_w_m2", "temp_c", "humidity_pct", "wind_speed_ms")

# Anything produced by the EMS is forbidden as a forecast input.
FORBIDDEN_FEATURES = (
    "soc", "battery_kw", "diesel_kw", "unmet_kw", "curtailed_kw", "charge_kw", "discharge_kw",
)


def calendar_features(index: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    """Deterministic time attributes of the target hour."""
    hour = index.hour.to_numpy(dtype=float)
    dow = index.dayofweek.to_numpy(dtype=float)
    month = index.month.to_numpy(dtype=float)
    doy = index.dayofyear.to_numpy(dtype=float)

    block = np.column_stack([
        hour, dow, month,
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        (dow == 6).astype(float),
    ])
    names = [
        "hour", "dayofweek", "month",
        "sin_hour", "cos_hour", "sin_dow", "cos_dow", "sin_doy", "cos_doy", "is_sunday",
    ]
    return block, names


def _lag_matrix(values: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """Column j holds ``values[t0 - lags[j]]``; rows before the lag are NaN."""
    n = len(values)
    out = np.full((n, len(lags)), np.nan)
    for j, lag in enumerate(lags):
        if lag == 0:
            out[:, j] = values
        else:
            out[lag:, j] = values[:-lag]
    return out


def _rolling_matrix(values: np.ndarray, windows: tuple[int, ...]) -> np.ndarray:
    """Trailing mean/std/min/max over windows that end at the issue time."""
    series = pd.Series(values)
    cols = []
    for w in windows:
        roll = series.rolling(w, min_periods=w)
        cols.extend([roll.mean().to_numpy(), roll.std().to_numpy(),
                     roll.min().to_numpy(), roll.max().to_numpy()])
    return np.column_stack(cols)


def _shift_forward(values: np.ndarray, k: int) -> np.ndarray:
    """``out[t] = values[t + k]``, tail padded with NaN."""
    n = len(values)
    out = np.full(n if values.ndim == 1 else (n, values.shape[1]), np.nan)
    if k < n:
        out[: n - k] = values[k:]
    return out


def issue_time_features(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, list[str]]:
    """Everything knowable at the issue time t0."""
    y = frame[target].to_numpy()

    blocks = [_lag_matrix(y, LAGS), _rolling_matrix(y, ROLLING_WINDOWS)]
    names = [f"{target}_lag{l}" for l in LAGS]
    for w in ROLLING_WINDOWS:
        names += [f"{target}_roll{w}_{s}" for s in ("mean", "std", "min", "max")]

    for var in WEATHER_FEATURES:
        if var not in frame.columns:
            continue
        blocks.append(_lag_matrix(frame[var].to_numpy(), (0, 1, 23)))
        names += [f"{var}_lag{l}" for l in (0, 1, 23)]

    return np.column_stack(blocks), names


def build_design(
    frame: pd.DataFrame, target: str, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Design matrix for forecasting `target` exactly `horizon` hours ahead.

    Row t corresponds to a forecast issued at ``frame.index[t]`` for
    ``frame.index[t + horizon]``. Returns ``(X, y, valid, names)``.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 hour ahead")

    x_issue, names = issue_time_features(frame, target)

    cal_all, cal_names = calendar_features(frame.index)
    cal = _shift_forward(cal_all, horizon)
    names = names + [f"target_{c}" for c in cal_names]

    x = np.column_stack([x_issue, cal])
    y = _shift_forward(frame[target].to_numpy(), horizon)

    leaked = [n for n in names if any(bad in n for bad in FORBIDDEN_FEATURES)]
    if leaked:
        raise ValueError(f"EMS outputs must never be forecast features: {leaked}")

    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    valid[len(frame) - horizon:] = False  # no observable label in the tail
    return x, y, valid, names
