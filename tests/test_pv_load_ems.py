"""PV model, load calibration, the rolling EMS loop, and the reported metrics.

The PV and load tests check this implementation against the paper's own
published figures, so a drift in either model shows up immediately rather than
being absorbed silently into the EMS comparison.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microgrid.config import ExperimentConfig, LoadConfig, PVConfig
from microgrid.dataset import build_dataset
from microgrid.ems import issue_index_for, run_baseline, run_ga_ems, usable_test_index
from microgrid.forecast import build_forecaster
from microgrid.load import generate_load
from microgrid.metrics import ems_kpis
from microgrid.pvmodel import cell_temperature_c, pv_power_kw, specific_yield_kwh_per_kwp

DATA = Path("data")
pytestmark = pytest.mark.skipif(
    not (DATA / "nasa_power_9.4008_-0.8393_2025.csv").exists(),
    reason="NASA POWER cache absent; run `python -c 'from microgrid.nasa_power import load_weather; load_weather()'`",
)


@pytest.fixture(scope="module")
def ds():
    return build_dataset(ExperimentConfig(), cache_dir=DATA)


# -- PV model ---------------------------------------------------------------

def test_specific_yield_matches_the_paper(ds):
    """The paper reports 1,486 kWh/kWp/yr for this site and PV model."""
    cfg = PVConfig()
    yield_ = specific_yield_kwh_per_kwp(ds.frame["pv_kw"], cfg)
    assert yield_ == pytest.approx(1486.0, abs=8.0)


def test_pv_is_zero_without_irradiance():
    cfg = PVConfig()
    assert pv_power_kw(np.array([30.0]), np.array([0.0]), cfg)[0] == pytest.approx(0.0)


def test_pv_derates_with_cell_temperature():
    """A hot module produces less: -0.4%/degC above 25 degC."""
    cfg = PVConfig()
    cool = pv_power_kw(np.array([25.0]), np.array([1000.0]), cfg)[0]
    hot = pv_power_kw(np.array([45.0]), np.array([1000.0]), cfg)[0]
    assert hot < cool
    t_cool = cell_temperature_c(np.array([25.0]), np.array([1000.0]), cfg)[0]
    t_hot = cell_temperature_c(np.array([45.0]), np.array([1000.0]), cfg)[0]
    expected = (1 + cfg.temp_coeff_per_c * (t_hot - 25)) / (1 + cfg.temp_coeff_per_c * (t_cool - 25))
    assert hot / cool == pytest.approx(expected)


def test_cell_temperature_exceeds_air_temperature_under_sun():
    cfg = PVConfig()
    assert cell_temperature_c(np.array([30.0]), np.array([800.0]), cfg)[0] > 30.0


def test_pv_never_exceeds_nameplate(ds):
    assert ds.frame["pv_kw"].max() <= PVConfig().capacity_kwp


# -- load model -------------------------------------------------------------

def test_annual_demand_matches_the_calibration_target(ds):
    """149 * 500 + 26 * 1700 = 118,700 kWh/yr."""
    annual = ds.frame["load_kw"].resample("YE").sum()
    assert LoadConfig().annual_kwh == pytest.approx(118700.0)
    assert annual.mean() == pytest.approx(118700.0, rel=0.005)


def test_demand_is_bimodal_with_a_dominant_evening_peak(ds):
    by_hour = ds.frame.groupby(ds.frame.index.hour)["load_kw"].mean()
    assert by_hour.loc[17:21].max() > by_hour.loc[9:15].max()   # evening beats midday
    assert by_hour.loc[5:8].max() > by_hour.loc[0:4].max()      # a morning shoulder exists
    assert by_hour.idxmax() in range(17, 22)


def test_productive_customers_consume_more_than_three_times_the_basic_household(ds):
    cfg = LoadConfig()
    per_basic = ds.frame["load_basic_kw"].resample("YE").sum().mean() / cfg.n_basic
    per_productive = ds.frame["load_productive_kw"].resample("YE").sum().mean() / cfg.n_productive
    assert per_productive / per_basic > 3.0


def test_productive_demand_is_daytime_dominant(ds):
    by_hour = ds.frame.groupby(ds.frame.index.hour)["load_productive_kw"].mean()
    assert by_hour.loc[8:16].mean() > 3.0 * by_hour.loc[0:4].mean()


def test_sundays_are_quieter_for_productive_use(ds):
    f = ds.frame
    sunday = f[f.index.dayofweek == 6]["load_productive_kw"].mean()
    weekday = f[f.index.dayofweek < 5]["load_productive_kw"].mean()
    assert sunday < weekday


def test_load_is_reproducible_from_its_seed():
    index = pd.date_range("2021-01-01", periods=24 * 40, freq="h")
    cfg = LoadConfig()
    pd.testing.assert_frame_equal(generate_load(index, cfg, seed=5), generate_load(index, cfg, seed=5))


def test_spikes_only_occur_in_working_hours():
    index = pd.date_range("2021-01-01", periods=24 * 120, freq="h")
    frame = generate_load(index, LoadConfig(), seed=7)
    spiking = frame[frame["load_spike"]]
    assert spiking.index.hour.min() >= 7
    assert spiking.index.hour.max() <= 17


# -- dataset splits ---------------------------------------------------------

def test_splits_are_chronological_and_disjoint(ds):
    assert ds.train.max() < ds.validation.min() < ds.test.min()
    assert len(ds.train.intersection(ds.test)) == 0
    assert ds.years(ds.test) == [2025]


def test_record_count_matches_the_paper(ds):
    assert len(ds.frame) == 70128


# -- EMS loop ---------------------------------------------------------------

@pytest.fixture(scope="module")
def short(ds):
    cfg = ExperimentConfig()
    test = usable_test_index(ds.frame, ds.test[: 12 * 24], cfg)
    return cfg, ds.frame, test


def test_baseline_never_sheds_load(short):
    cfg, frame, test = short
    log = run_baseline(frame, test, cfg)
    assert log["unmet_kw"].max() == pytest.approx(0.0)
    assert ems_kpis(log, cfg)["lpsp_pct"] == pytest.approx(0.0)


def test_baseline_energy_balance_closes(short):
    cfg, frame, test = short
    log = run_baseline(frame, test, cfg)
    residual = (log["pv_kw"] - log["curtailed_kw"] + log["discharge_kw"] + log["diesel_kw"]
                + log["unmet_kw"] - log["load_kw"] - log["charge_kw"])
    assert residual.abs().max() < 1e-9


def test_perfect_forecast_reproduces_the_baseline(short):
    """The control that makes the diesel premium interpretable.

    Handed the true future, the GA framework must match the reactive baseline.
    Any gap is a defect in the commitment loop, not a cost of forecast error.
    """
    cfg, frame, test = short
    issue = issue_index_for(frame, test, cfg)
    bundle = build_forecaster("perfect", cfg.ems.horizon_hours, cfg.pv.capacity_kwp).predict(frame, issue)

    base = ems_kpis(run_baseline(frame, test, cfg), cfg)
    ga = ems_kpis(run_ga_ems(frame, test, bundle, cfg), cfg)
    assert ga["diesel_kwh"] == pytest.approx(base["diesel_kwh"], rel=1e-6)
    assert ga["curtailed_kwh"] == pytest.approx(base["curtailed_kwh"], rel=1e-6)
    assert ga["lpsp_pct"] == pytest.approx(base["lpsp_pct"], abs=1e-9)


def test_following_generator_keeps_lpsp_at_zero(short):
    cfg, frame, test = short
    issue = issue_index_for(frame, test, cfg)
    bundle = build_forecaster("persistence", cfg.ems.horizon_hours, cfg.pv.capacity_kwp).predict(frame, issue)
    log = run_ga_ems(frame, test, bundle, cfg)
    assert ems_kpis(log, cfg)["lpsp_pct"] == pytest.approx(0.0)


def test_freezing_the_generator_breaks_reliability(short):
    """Why the generator must balance: a frozen 6-hour schedule strands every
    under-predicted load spike as unserved energy."""
    cfg, frame, test = short
    frozen = dataclasses.replace(cfg, ems=dataclasses.replace(cfg.ems, diesel_follows_deficit=False))
    issue = issue_index_for(frame, test, frozen)
    bundle = build_forecaster("persistence", frozen.ems.horizon_hours, frozen.pv.capacity_kwp).predict(frame, issue)
    log = run_ga_ems(frame, test, bundle, frozen)
    assert ems_kpis(log, frozen)["lpsp_pct"] > 0.0


def test_soc_stays_inside_the_operating_window(short):
    cfg, frame, test = short
    issue = issue_index_for(frame, test, cfg)
    bundle = build_forecaster("perfect", cfg.ems.horizon_hours, cfg.pv.capacity_kwp).predict(frame, issue)
    log = run_ga_ems(frame, test, bundle, cfg)
    assert log["soc"].min() >= cfg.battery.soc_min - 1e-9
    assert log["soc"].max() <= cfg.battery.soc_max + 1e-9


def test_usable_test_index_leaves_room_for_the_horizon(ds):
    """The invariant belongs on the last *block start*, not the last committed hour.

    A block beginning at position p plans p .. p + horizon - 1. The final
    committed hour sits `commit - 1` hours after that start, so checking the
    horizon from the last hour would demand `commit - 1` rows that no forecast
    ever needs.
    """
    cfg = ExperimentConfig()
    trimmed = usable_test_index(ds.frame, ds.test, cfg)
    assert len(trimmed) % cfg.ems.commit_hours == 0

    last_hour = ds.frame.index.get_loc(trimmed[-1])
    last_block_start = last_hour - (cfg.ems.commit_hours - 1)
    assert last_block_start + cfg.ems.horizon_hours <= len(ds.frame)

    # And the forecast issued for that block must exist: issued one hour before it.
    assert last_block_start - 1 >= 0


# -- metrics ----------------------------------------------------------------

def test_kpis_measure_what_they_claim():
    cfg = ExperimentConfig()
    index = pd.date_range("2025-01-01", periods=4, freq="h")
    log = pd.DataFrame(
        {
            "pv_kw": [0.0, 50.0, 50.0, 0.0],
            "load_kw": [10.0, 10.0, 10.0, 10.0],
            "charge_kw": [0.0, 20.0, 0.0, 0.0],
            "discharge_kw": [0.0, 0.0, 0.0, 5.0],
            "battery_kw": [0.0, -20.0, 0.0, 5.0],
            "diesel_kw": [10.0, 0.0, 0.0, 5.0],
            "unmet_kw": [0.0, 0.0, 0.0, 0.0],
            "curtailed_kw": [0.0, 20.0, 40.0, 0.0],
            "soc": [0.5, 0.6, 0.6, 0.55],
        },
        index=index,
    )
    k = ems_kpis(log, cfg)
    assert k["demand_kwh"] == pytest.approx(40.0)
    assert k["diesel_kwh"] == pytest.approx(15.0)
    assert k["diesel_cost_ghs"] == pytest.approx(15.0 * 5.28)
    assert k["curtailed_kwh"] == pytest.approx(60.0)
    assert k["lpsp_pct"] == pytest.approx(0.0)
    assert k["renewable_fraction_pct"] == pytest.approx(100.0 * (1 - 15.0 / 40.0))
    assert k["battery_cycles"] == pytest.approx(25.0 / (2 * cfg.battery.usable_kwh))


def test_lpsp_counts_unserved_energy_as_a_fraction_of_demand():
    cfg = ExperimentConfig()
    index = pd.date_range("2025-01-01", periods=2, freq="h")
    log = pd.DataFrame(
        {"pv_kw": 0.0, "load_kw": [10.0, 10.0], "charge_kw": 0.0, "discharge_kw": 0.0,
         "battery_kw": 0.0, "diesel_kw": [10.0, 5.0], "unmet_kw": [0.0, 5.0],
         "curtailed_kw": 0.0, "soc": 0.1},
        index=index,
    )
    k = ems_kpis(log, cfg)
    assert k["lpsp_pct"] == pytest.approx(100.0 * 5.0 / 20.0)
    assert k["loss_of_load_hours"] == 1.0
