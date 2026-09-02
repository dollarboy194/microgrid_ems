"""NumPy-vectorised genetic algorithm for dispatch over the forecast horizon.

Decision variables are the battery power (signed: positive discharges) and the
diesel generator output at each hour of the horizon, so a genome is a
``(2, horizon)`` array flattened into ``2 * horizon`` genes. The whole
population is evaluated in one call to `dispatch.simulate`, which is what makes
the search affordable on constrained hardware.

Two implementation details from the paper are load-bearing:

**Seeded initialisation.** A purely random initial population converges
unreliably to poor local optima -- the search space is 48-dimensional and
almost all of it is dominated by unmet-load penalties. 70% of the population is
therefore seeded from a greedy merit-order heuristic (PV, then battery, then
diesel) plus Gaussian-perturbed variants of it; the rest is random, for
diversity. The heuristic alone is a strong solution, so the GA starts near a
good basin and spends its budget refining rather than finding one.

**Repair.** Diesel is clamped to the genuine deficit remaining after PV and
battery discharge. The clamp lives in `dispatch.simulate`, which means it
applies identically during the GA's fitness evaluation and during the
ground-truth replay -- an optimiser cannot win fitness by exploiting physics
the evaluator will not grant it. Computing the deficit *after* PV is applied is
what stops the repair from discharging the battery into a PV surplus, the bug
the paper reports finding and fixing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BatteryConfig, DieselConfig, EMSConfig, GAConfig
from .dispatch import DispatchResult, objective, simulate


@dataclass
class GAResult:
    battery_kw: np.ndarray      # (horizon,) best plan, signed
    diesel_kw: np.ndarray       # (horizon,)
    fitness: float
    history: np.ndarray         # best fitness per generation
    seeded_fitness: float       # fitness of the greedy heuristic that seeded it


def greedy_merit_order(
    pv_kw: np.ndarray, load_kw: np.ndarray, soc0: float,
    battery: BatteryConfig, diesel: DieselConfig, dt_hours: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """PV first, then the battery, then diesel. The seed, and the baseline's logic.

    Surplus PV charges the battery and is otherwise curtailed; a deficit is met
    from the battery first and only then from the generator. No foresight is
    used: each hour is resolved from its own supply and demand.
    """
    n = len(pv_kw)
    b_plan = np.zeros(n)
    d_plan = np.zeros(n)
    soc = float(soc0)
    cap = battery.nominal_kwh

    for h in range(n):
        net = pv_kw[h] - load_kw[h]
        if net >= 0.0:
            headroom = (battery.soc_max - soc) * cap / (battery.eta_charge * dt_hours)
            charge = min(net, battery.p_charge_max_kw, max(headroom, 0.0))
            b_plan[h] = -charge
            soc += charge * battery.eta_charge * dt_hours / cap
        else:
            deficit = -net
            available = (soc - battery.soc_min) * cap * battery.eta_discharge / dt_hours
            discharge = min(deficit, battery.p_discharge_max_kw, max(available, 0.0))
            b_plan[h] = discharge
            soc -= discharge / battery.eta_discharge * dt_hours / cap
            d_plan[h] = min(deficit - discharge, diesel.rated_kw)
        soc = min(max(soc, battery.soc_min), battery.soc_max)

    return b_plan, d_plan


def _initial_population(
    seed_b: np.ndarray, seed_d: np.ndarray, cfg: GAConfig,
    battery: BatteryConfig, diesel: DieselConfig, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = len(seed_b)
    n_seed = int(round(cfg.seed_fraction * cfg.population))
    n_seed = min(max(n_seed, 0), cfg.population)

    b = np.empty((cfg.population, horizon))
    d = np.empty((cfg.population, horizon))

    if n_seed > 0:
        # One exact copy of the heuristic, the rest of the seeded share jittered
        # around it. `seed_fraction = 0` is a legitimate configuration -- it is
        # how the paper's random-initialisation failure is reproduced -- so the
        # exact copy must not be written unconditionally.
        b[0], d[0] = seed_b, seed_d
        n_jitter = n_seed - 1
        if n_jitter > 0:
            b[1:n_seed] = seed_b + rng.normal(
                0.0, cfg.seed_jitter * battery.p_discharge_max_kw, (n_jitter, horizon))
            d[1:n_seed] = seed_d + rng.normal(
                0.0, cfg.seed_jitter * diesel.rated_kw, (n_jitter, horizon))

    n_rand = cfg.population - n_seed
    if n_rand > 0:
        b[n_seed:] = rng.uniform(-battery.p_charge_max_kw, battery.p_discharge_max_kw, (n_rand, horizon))
        d[n_seed:] = rng.uniform(0.0, diesel.rated_kw, (n_rand, horizon))

    return _clip(b, d, battery, diesel)


def _clip(b: np.ndarray, d: np.ndarray, battery: BatteryConfig, diesel: DieselConfig):
    return (
        np.clip(b, -battery.p_charge_max_kw, battery.p_discharge_max_kw),
        np.clip(d, 0.0, diesel.rated_kw),
    )


def _tournament(fitness: np.ndarray, k: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Select n parents; lower fitness wins."""
    picks = rng.integers(0, len(fitness), size=(n, k))
    best = np.argmin(fitness[picks], axis=1)
    return picks[np.arange(n), best]


def optimise(
    pv_kw: np.ndarray, load_kw: np.ndarray, soc0: float,
    battery: BatteryConfig, diesel: DieselConfig, ems: EMSConfig, cfg: GAConfig,
    dt_hours: float = 1.0, rng: np.random.Generator | None = None,
) -> GAResult:
    """Minimise the dispatch objective over the horizon, given forecast PV/load."""
    rng = rng or np.random.default_rng(cfg.random_state)
    horizon = len(pv_kw)

    def evaluate(b: np.ndarray, d: np.ndarray) -> np.ndarray:
        result = simulate(pv_kw, load_kw, b, d, soc0, battery, diesel, dt_hours)
        return objective(result, diesel, ems, dt_hours)

    seed_b, seed_d = greedy_merit_order(pv_kw, load_kw, soc0, battery, diesel, dt_hours)
    seeded_fitness = float(evaluate(seed_b[None, :], seed_d[None, :])[0])

    b, d = _initial_population(seed_b, seed_d, cfg, battery, diesel, rng)
    fitness = evaluate(b, d)

    n_elite = max(1, int(round(cfg.elite_fraction * cfg.population)))
    history = np.empty(cfg.generations)

    for gen in range(cfg.generations):
        order = np.argsort(fitness)
        elite_b, elite_d = b[order[:n_elite]].copy(), d[order[:n_elite]].copy()

        n_children = cfg.population - n_elite
        pa = _tournament(fitness, cfg.tournament_size, n_children, rng)
        pb = _tournament(fitness, cfg.tournament_size, n_children, rng)

        # Uniform crossover, per gene and per decision variable.
        mask_b = rng.random((n_children, horizon)) < 0.5
        mask_d = rng.random((n_children, horizon)) < 0.5
        do_cross = rng.random((n_children, 1)) < cfg.crossover_rate
        child_b = np.where(do_cross & mask_b, b[pa], b[pb])
        child_d = np.where(do_cross & mask_d, d[pa], d[pb])

        # Gaussian mutation on a random subset of genes.
        mut_b = rng.random((n_children, horizon)) < cfg.mutation_rate
        mut_d = rng.random((n_children, horizon)) < cfg.mutation_rate
        child_b += mut_b * rng.normal(0.0, cfg.mutation_scale * battery.p_discharge_max_kw,
                                      (n_children, horizon))
        child_d += mut_d * rng.normal(0.0, cfg.mutation_scale * diesel.rated_kw,
                                      (n_children, horizon))

        child_b, child_d = _clip(child_b, child_d, battery, diesel)

        b = np.vstack([elite_b, child_b])
        d = np.vstack([elite_d, child_d])
        fitness = evaluate(b, d)
        history[gen] = fitness.min()

    best = int(np.argmin(fitness))
    return GAResult(
        battery_kw=b[best], diesel_kw=d[best], fitness=float(fitness[best]),
        history=history, seeded_fitness=seeded_fitness,
    )
