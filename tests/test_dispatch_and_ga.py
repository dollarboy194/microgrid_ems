"""Physics, merit order, and the genetic algorithm."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from microgrid.config import BatteryConfig, DieselConfig, EMSConfig, GAConfig
from microgrid.dispatch import energy_balance_residual, objective, simulate
from microgrid.ga import greedy_merit_order, optimise


@pytest.fixture
def assets():
    return BatteryConfig(), DieselConfig(), EMSConfig(), GAConfig(population=40, generations=12)


def _day(pv_peak=60.0, load_base=12.0):
    hour = np.arange(24)
    pv = np.clip(pv_peak * np.sin(np.pi * (hour - 6) / 12), 0.0, None)
    load = load_base + 10.0 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    return pv, load


# -- energy balance ---------------------------------------------------------

def test_energy_balance_closes_for_any_plan(assets):
    battery, diesel, _, _ = assets
    pv, load = _day()
    rng = np.random.default_rng(0)
    b = rng.uniform(-200, 200, (64, 24))
    d = rng.uniform(-200, 200, (64, 24))
    res = simulate(pv, load, b, d, 0.5, battery, diesel)
    assert np.abs(energy_balance_residual(pv, load, res)).max() < 1e-9


def test_absurd_plans_stay_physical(assets):
    battery, diesel, _, _ = assets
    pv, load = _day()
    rng = np.random.default_rng(1)
    res = simulate(pv, load, rng.uniform(-1e5, 1e5, (32, 24)), rng.uniform(-1e5, 1e5, (32, 24)),
                   0.5, battery, diesel)
    assert res.soc.min() >= battery.soc_min - 1e-12
    assert res.soc.max() <= battery.soc_max + 1e-12
    assert res.diesel_kw.max() <= diesel.rated_kw + 1e-9
    assert res.diesel_kw.min() >= 0.0
    assert res.charge_kw.max() <= battery.p_charge_max_kw + 1e-9
    assert res.discharge_kw.max() <= battery.p_discharge_max_kw + 1e-9


def test_soc_floor_respects_the_90_percent_depth_of_discharge():
    battery = BatteryConfig()
    assert battery.soc_min == pytest.approx(0.10)
    assert battery.usable_kwh == pytest.approx(117.0)


# -- merit order is enforced by the physics, not hoped for ------------------

def test_diesel_never_runs_when_supply_already_covers_demand(assets):
    """Gratuitous diesel: the bug a repair operator is supposed to remove."""
    battery, diesel, _, _ = assets
    pv = np.full(6, 50.0)
    load = np.full(6, 10.0)
    # Ask for full diesel output during a large PV surplus.
    res = simulate(pv, load, np.zeros((1, 6)), np.full((1, 6), diesel.rated_kw), 0.5, battery, diesel)
    assert res.diesel_kw.max() == pytest.approx(0.0)
    assert res.curtailed_kw.sum() > 0


def test_diesel_never_exceeds_the_genuine_deficit(assets):
    battery, diesel, _, _ = assets
    pv = np.zeros(4)
    load = np.full(4, 12.0)
    res = simulate(pv, load, np.zeros((1, 4)), np.full((1, 4), diesel.rated_kw), 0.5, battery, diesel)
    np.testing.assert_allclose(res.diesel_kw[0], 12.0)
    assert res.curtailed_kw.max() == pytest.approx(0.0)
    assert res.unmet_kw.max() == pytest.approx(0.0)


def test_repair_does_not_discharge_the_battery_into_a_pv_surplus(assets):
    """The paper's reported bug: battery discharged during PV surplus.

    A genuine deficit is computed only after PV is applied, so a surplus hour is
    never seen as one needing battery support. With an idle battery setpoint the
    surplus is curtailed, the generator stays off, and the state of charge does
    not move -- nothing "repairs" the surplus by draining the pack.
    """
    battery, diesel, _, _ = assets
    pv = np.full(5, 40.0)
    load = np.full(5, 10.0)
    idle = simulate(pv, load, np.zeros((1, 5)), np.full((1, 5), diesel.rated_kw), 0.5, battery, diesel)

    assert idle.discharge_kw.max() == pytest.approx(0.0)
    assert idle.diesel_kw.max() == pytest.approx(0.0)
    assert idle.curtailed_kw.min() > 0
    np.testing.assert_allclose(idle.soc[0], 0.5)  # an idle setpoint moves no energy


def test_diesel_may_charge_the_battery_when_the_plan_asks(assets):
    """Charging raises the deficit, so the generator can legitimately serve it."""
    battery, diesel, _, _ = assets
    pv = np.zeros(3)
    load = np.full(3, 5.0)
    res = simulate(pv, load, np.full((1, 3), -20.0), np.full((1, 3), diesel.rated_kw),
                   0.5, battery, diesel)
    assert res.charge_kw.min() > 0
    np.testing.assert_allclose(res.diesel_kw[0], 25.0)  # 5 kW load + 20 kW charging


def test_unmet_load_appears_only_when_nothing_else_can_serve(assets):
    battery, diesel, _, _ = assets
    pv = np.zeros(3)
    load = np.full(3, 100.0)  # far above diesel rating
    res = simulate(pv, load, np.zeros((1, 3)), np.full((1, 3), diesel.rated_kw),
                   battery.soc_min, battery, diesel)
    assert res.diesel_kw.min() == pytest.approx(diesel.rated_kw)
    assert res.unmet_kw.min() > 0


# -- greedy merit-order heuristic -------------------------------------------

def test_greedy_charges_from_surplus_and_discharges_into_deficit(assets):
    battery, diesel, _, _ = assets
    pv, load = _day()
    b, d = greedy_merit_order(pv, load, 0.5, battery, diesel)
    surplus = pv > load
    assert (b[surplus] <= 1e-9).all(), "greedy must not discharge during surplus"
    assert (b[~surplus] >= -1e-9).all(), "greedy must not charge during deficit"


def test_greedy_uses_diesel_only_after_the_battery(assets):
    battery, diesel, _, _ = assets
    pv = np.zeros(3)
    load = np.full(3, 20.0)
    b, d = greedy_merit_order(pv, load, 1.0, battery, diesel)
    assert (b > 0).all()
    # `x == pytest.approx(0.0)` on an array collapses to a single bool; compare
    # elementwise or the assertion silently tests the wrong thing.
    np.testing.assert_allclose(d, 0.0, atol=1e-12)


# -- objective --------------------------------------------------------------

def test_unmet_load_dominates_the_objective(assets):
    """~950x the diesel marginal cost makes reliability a near-hard constraint.

    The most a cost-minimising optimiser can ever save by shedding one kWh is
    the diesel it would have burned to serve it. Pricing the shed kWh at 950x
    that saving means no reliable plan is ever undercut by an unreliable one --
    which is precisely what a soft, comparable-magnitude penalty fails to do.
    """
    _, diesel, ems, _ = assets
    max_saving_per_kwh_shed = diesel.cost_per_kwh_ghs
    assert ems.unmet_penalty_per_kwh == pytest.approx(950.0 * max_saving_per_kwh_shed)
    assert ems.unmet_penalty_per_kwh / max_saving_per_kwh_shed >= 900.0


def test_a_soft_penalty_would_rationally_accept_unmet_load(assets):
    """The paper's methodological finding, as an executable statement.

    With the penalty set near the diesel marginal cost, shedding load scores
    *better* than serving it, because the tie-breaking curtailment and wear
    terms tip the balance. No tuning of such a weight reproduces the baseline's
    reliability; the constraint has to be hard.
    """
    battery, diesel, ems, _ = assets
    soft = dataclasses.replace(ems, unmet_penalty_multiplier=1.0)
    pv = np.zeros(3)
    load = np.full(3, 20.0)
    empty = battery.soc_min

    serve = simulate(pv, load, np.zeros((1, 3)), np.full((1, 3), diesel.rated_kw), empty, battery, diesel)
    shed = simulate(pv, load, np.zeros((1, 3)), np.zeros((1, 3)), empty, battery, diesel)

    assert objective(shed, diesel, soft)[0] <= objective(serve, diesel, soft)[0]
    # ... whereas the hard constraint reverses the preference.
    assert objective(serve, diesel, ems)[0] < objective(shed, diesel, ems)[0]


def test_objective_prefers_diesel_over_unmet_load(assets):
    battery, diesel, ems, _ = assets
    pv = np.zeros(3)
    load = np.full(3, 20.0)
    empty = battery.soc_min

    serve = simulate(pv, load, np.zeros((1, 3)), np.full((1, 3), diesel.rated_kw), empty, battery, diesel)
    shed = simulate(pv, load, np.zeros((1, 3)), np.zeros((1, 3)), empty, battery, diesel)
    assert objective(serve, diesel, ems)[0] < objective(shed, diesel, ems)[0]


# -- genetic algorithm ------------------------------------------------------

def test_ga_never_loses_to_its_greedy_seed(assets):
    battery, diesel, ems, ga = assets
    pv, load = _day()
    plan = optimise(pv, load, 0.5, battery, diesel, ems, ga)
    assert plan.fitness <= plan.seeded_fitness + 1e-9


def test_ga_plan_is_within_actuator_limits(assets):
    battery, diesel, ems, ga = assets
    pv, load = _day()
    plan = optimise(pv, load, 0.5, battery, diesel, ems, ga)
    assert plan.battery_kw.min() >= -battery.p_charge_max_kw - 1e-9
    assert plan.battery_kw.max() <= battery.p_discharge_max_kw + 1e-9
    assert plan.diesel_kw.min() >= -1e-9
    assert plan.diesel_kw.max() <= diesel.rated_kw + 1e-9


def test_ga_fitness_is_monotone_non_increasing(assets):
    """Elitism guarantees the best-so-far never worsens."""
    battery, diesel, ems, ga = assets
    pv, load = _day()
    plan = optimise(pv, load, 0.5, battery, diesel, ems, ga)
    assert (np.diff(plan.history) <= 1e-9).all()


def test_seeded_initialisation_beats_pure_random(assets):
    """The paper's finding: random init converges to a much worse optimum."""
    battery, diesel, ems, ga = assets
    pv, load = _day(pv_peak=10.0)  # a scarce day, where scheduling actually matters
    seeded = optimise(pv, load, 0.5, battery, diesel, ems, ga)
    random_init = optimise(pv, load, 0.5, battery, diesel, ems,
                           dataclasses.replace(ga, seed_fraction=0.0))
    assert seeded.fitness < random_init.fitness


def test_pure_random_initialisation_is_a_valid_configuration(assets):
    """seed_fraction=0 must not crash: it is how the failure mode is reproduced."""
    battery, diesel, ems, ga = assets
    pv, load = _day()
    plan = optimise(pv, load, 0.5, battery, diesel, ems, dataclasses.replace(ga, seed_fraction=0.0))
    assert np.isfinite(plan.fitness)


def test_fully_seeded_initialisation_is_a_valid_configuration(assets):
    battery, diesel, ems, ga = assets
    pv, load = _day()
    plan = optimise(pv, load, 0.5, battery, diesel, ems, dataclasses.replace(ga, seed_fraction=1.0))
    assert np.isfinite(plan.fitness)


def test_ga_is_reproducible_from_its_seed(assets):
    battery, diesel, ems, ga = assets
    pv, load = _day()
    a = optimise(pv, load, 0.5, battery, diesel, ems, ga, rng=np.random.default_rng(3))
    b = optimise(pv, load, 0.5, battery, diesel, ems, ga, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(a.battery_kw, b.battery_kw)
    assert a.fitness == b.fitness
