"""Battery model: 130 kWh nominal, 117 kWh usable at 90% depth of discharge."""

from __future__ import annotations

import pytest

from microgrid.battery import Battery
from microgrid.config import BatteryConfig


@pytest.fixture
def cfg():
    return BatteryConfig(
        nominal_kwh=100.0, max_depth_of_discharge=0.90,
        p_charge_max_kw=50.0, p_discharge_max_kw=50.0,
        eta_charge=0.9, eta_discharge=0.9, soc_init=0.5,
    )


def test_depth_of_discharge_sets_the_floor_and_the_usable_capacity():
    cfg = BatteryConfig()  # the paper's 130 kWh pack
    assert cfg.soc_min == pytest.approx(0.10)
    assert cfg.soc_max == pytest.approx(1.00)
    assert cfg.usable_kwh == pytest.approx(117.0)


def test_charging_stores_energy_net_of_efficiency(cfg):
    b = Battery(cfg)
    b.step(-10.0)  # absorb 10 kWh at the terminals
    assert b.energy_kwh == pytest.approx(50.0 + 10.0 * 0.9)


def test_discharging_drains_more_than_it_delivers(cfg):
    b = Battery(cfg)
    b.step(9.0)  # deliver 9 kWh at the terminals
    assert b.energy_kwh == pytest.approx(50.0 - 9.0 / 0.9)


def test_round_trip_loses_exactly_the_round_trip_efficiency(cfg):
    b = Battery(cfg)
    start = b.energy_kwh
    b.step(-20.0)
    b.step(20.0)
    assert start - b.energy_kwh == pytest.approx(20.0 / 0.9 - 20.0 * 0.9)


def test_power_is_clipped_to_rate_limits(cfg):
    """With ample energy and headroom, the rate limit is what binds.

    The SOC window must be opened first: at 50% SOC only 40 kWh is withdrawable,
    so an hour-long 50 kW discharge would be stopped by the energy limit and the
    test would silently assert the wrong thing.
    """
    b = Battery(cfg)
    b.state.soc = 1.0  # 90 kWh withdrawable
    assert b.step(999.0) == pytest.approx(cfg.p_discharge_max_kw)

    b.state.soc = 0.1  # 90 kWh of headroom
    assert b.step(-999.0) == pytest.approx(-cfg.p_charge_max_kw)


def test_energy_limit_binds_before_the_rate_limit_when_nearly_empty(cfg):
    b = Battery(cfg)
    b.state.soc = 0.15  # 5 kWh usable
    realised = b.step(50.0)
    assert realised == pytest.approx(5.0 * cfg.eta_discharge)
    assert realised < cfg.p_discharge_max_kw
    assert b.soc == pytest.approx(cfg.soc_min)


def test_soc_never_leaves_its_operating_window(cfg):
    b = Battery(cfg)
    for _ in range(50):
        b.step(50.0)
    assert b.soc == pytest.approx(cfg.soc_min)

    b.reset()
    for _ in range(50):
        b.step(-50.0)
    assert b.soc == pytest.approx(cfg.soc_max)


def test_feasible_power_agrees_with_step(cfg):
    b = Battery(cfg)
    b.state.soc = 0.15
    feasible = b.feasible_power(40.0)
    assert b.step(feasible) == pytest.approx(feasible)


def test_throughput_and_cycles_count_against_usable_capacity(cfg):
    b = Battery(cfg)
    b.step(-10.0)
    b.step(10.0)
    assert b.state.throughput_kwh == pytest.approx(20.0)
    assert b.equivalent_full_cycles() == pytest.approx(20.0 / (2 * cfg.usable_kwh))


def test_an_idle_battery_holds_its_charge(cfg):
    b = Battery(cfg)
    before = b.energy_kwh
    b.step(0.0)
    assert b.energy_kwh == pytest.approx(before)


def test_invalid_configs_are_rejected():
    with pytest.raises(ValueError):
        BatteryConfig(max_depth_of_discharge=1.5)
    with pytest.raises(ValueError):
        BatteryConfig(soc_init=0.05)  # below the 10% floor implied by 90% DoD
