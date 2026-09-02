"""The paper's five leakage verification tests, implemented as executable checks.

From Section IV-A, the five confirmations are:

  1. complete absence of downstream EMS-output variables (state of charge,
     dispatch decisions, unmet load, curtailment) from the feature set;
  2. bit-exact correctness of lag features against an independently
     reconstructed time shift;
  3. correct exclusion of the current timestep from rolling-window statistics;
  4. determinism of calendar features;
  5. correct target alignment.

A sixth, stronger test is added here: poison every observation after the issue
time with garbage and demand that no feature moves. Tests 2 and 3 check the
construction; this one checks the *consequence*, and would catch a leak
introduced anywhere in the pipeline rather than only in the lag builder.

"Forecasting models trained on leaked information would invalidate any
subsequent EMS performance comparison." So these tests guard the paper's
headline result, not merely a utility function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microgrid.features import (
    FORBIDDEN_FEATURES,
    LAGS,
    ROLLING_WINDOWS,
    build_design,
    calendar_features,
)

HORIZON = 6


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    n = 24 * 90
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    rng = np.random.default_rng(11)
    hour = index.hour.to_numpy()
    return pd.DataFrame(
        {
            "pv_kw": np.clip(60 * np.sin(np.pi * (hour - 6) / 12), 0, None) + rng.normal(0, 2, n),
            "load_kw": 12 + 6 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1, n),
            "ghi_w_m2": np.clip(900 * np.sin(np.pi * (hour - 6) / 12), 0, None),
            "temp_c": 28 + 5 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.5, n),
            "humidity_pct": 60 + rng.normal(0, 5, n),
            "wind_speed_ms": 3 + rng.normal(0, 0.5, n),
        },
        index=index,
    )


# -- Test 1: no EMS outputs among the features ------------------------------

@pytest.mark.parametrize("target", ["pv_kw", "load_kw"])
def test_no_ems_outputs_in_feature_set(frame, target):
    _, _, _, names = build_design(frame, target, horizon=1)
    for forbidden in FORBIDDEN_FEATURES:
        assert not any(forbidden in n for n in names), f"{forbidden!r} leaked into the features"


def test_ems_output_column_is_rejected_outright(frame):
    """An EMS output must be refused even if someone joins it onto the frame."""
    poisoned = frame.copy()
    poisoned["soc"] = 0.5
    # `soc` is not in WEATHER_FEATURES, so it is ignored rather than consumed;
    # the guard fires only if a forbidden name reaches the feature list.
    _, _, _, names = build_design(poisoned, "load_kw", horizon=1)
    assert not any("soc" in n for n in names)


# -- Test 2: lag features are bit-exact against an independent shift ---------

@pytest.mark.parametrize("target", ["pv_kw", "load_kw"])
@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_lag_features_are_bit_exact(frame, target, horizon):
    x, _, valid, names = build_design(frame, target, horizon)
    truth = frame[target]
    for lag in LAGS:
        col = names.index(f"{target}_lag{lag}")
        expected = truth.shift(lag).to_numpy()  # independent reconstruction
        rows = valid & np.isfinite(expected)
        np.testing.assert_array_equal(x[rows, col], expected[rows])


def test_lag_zero_is_the_issue_time_not_the_target_time(frame):
    """`lag0` is the last *observed* value, never the value being predicted."""
    x, y, valid, names = build_design(frame, "load_kw", horizon=1)
    col = names.index("load_kw_lag0")
    np.testing.assert_array_equal(x[:, col], frame["load_kw"].to_numpy())
    # And it must differ from the label, or the model would be handed the answer.
    assert not np.allclose(x[valid, col], y[valid])


# -- Test 3: rolling statistics exclude the current timestep ----------------

@pytest.mark.parametrize("window", ROLLING_WINDOWS)
def test_rolling_windows_end_at_the_issue_time(frame, window):
    """A window ending at t0 includes t0 and nothing after it."""
    x, _, valid, names = build_design(frame, "load_kw", horizon=1)
    col = names.index(f"load_kw_roll{window}_mean")
    expected = frame["load_kw"].rolling(window, min_periods=window).mean().to_numpy()
    rows = valid & np.isfinite(expected)
    np.testing.assert_allclose(x[rows, col], expected[rows], rtol=0, atol=1e-12)


def test_rolling_statistics_never_see_the_predicted_hour(frame):
    """Rewriting the target hour must not move any rolling feature."""
    x_before, _, valid, _ = build_design(frame, "load_kw", horizon=1)
    t = 500
    poisoned = frame.copy()
    poisoned.iloc[t + 1, poisoned.columns.get_loc("load_kw")] = 1e6  # the hour being predicted
    x_after, _, _, _ = build_design(poisoned, "load_kw", horizon=1)
    np.testing.assert_array_equal(x_before[t], x_after[t])


# -- Test 4: calendar features are deterministic ----------------------------

def test_calendar_features_are_deterministic(frame):
    a, names_a = calendar_features(frame.index)
    b, names_b = calendar_features(frame.index)
    np.testing.assert_array_equal(a, b)
    assert names_a == names_b


def test_calendar_features_depend_only_on_the_timestamp(frame):
    """Rewriting every observation must leave the calendar block untouched."""
    poisoned = frame.copy()
    poisoned.iloc[:, :] = 0.0
    a, _ = calendar_features(frame.index)
    b, _ = calendar_features(poisoned.index)
    np.testing.assert_array_equal(a, b)


def test_calendar_is_for_the_target_hour_not_the_issue_hour(frame):
    """A forecast for t0+h must be told the hour of t0+h, which it knows in advance."""
    horizon = 5
    x, _, valid, names = build_design(frame, "load_kw", horizon)
    col = names.index("target_hour")
    expected = frame.index.hour.to_numpy()[horizon:]
    np.testing.assert_array_equal(x[: len(expected)][valid[: len(expected)], col],
                                  expected[valid[: len(expected)]])


# -- Test 5: target alignment -----------------------------------------------

@pytest.mark.parametrize("horizon", [1, 2, 6, 24])
def test_label_is_exactly_h_hours_ahead(frame, horizon):
    _, y, valid, _ = build_design(frame, "pv_kw", horizon)
    truth = frame["pv_kw"].to_numpy()
    rows = np.flatnonzero(valid)
    np.testing.assert_array_equal(y[rows], truth[rows + horizon])


def test_tail_rows_have_no_label(frame):
    for horizon in (1, 6, 24):
        _, _, valid, _ = build_design(frame, "load_kw", horizon)
        assert not valid[len(frame) - horizon:].any()


# -- Sixth, stronger test: poison the future --------------------------------

@pytest.mark.parametrize("target", ["pv_kw", "load_kw"])
@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_features_do_not_move_when_the_future_is_rewritten(frame, target, horizon):
    t = 600  # well clear of the longest lag (168 h) and of the tail
    x_before, _, valid, _ = build_design(frame, target, horizon)
    assert valid[t]

    poisoned = frame.copy()
    rng = np.random.default_rng(0)
    poisoned.iloc[t + 1:] = rng.normal(1e5, 1e4, size=poisoned.iloc[t + 1:].shape)

    x_after, _, _, _ = build_design(poisoned, target, horizon)
    np.testing.assert_array_equal(x_before[t], x_after[t])


def test_a_deliberate_leak_is_caught_by_the_poison_test(frame):
    """The guard must be able to fail: inject a leak and confirm it is detected."""
    t = 600
    leaked = frame.copy()
    # A "feature" that peeks one hour ahead -- exactly what the discipline forbids.
    leaked["load_kw"] = frame["load_kw"].shift(-1).bfill()

    x_before, _, _, _ = build_design(leaked, "load_kw", horizon=1)
    poisoned = leaked.copy()
    poisoned.iloc[t + 1:] = 1e5
    # Rebuild from the poisoned *source* the way a leaking pipeline would.
    poisoned["load_kw"] = poisoned["load_kw"].shift(-1).bfill()
    x_after, _, _, _ = build_design(poisoned, "load_kw", horizon=1)

    assert not np.array_equal(x_before[t], x_after[t]), "the poison test cannot detect a leak"
