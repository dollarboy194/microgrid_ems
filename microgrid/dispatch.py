"""Physical dispatch simulation for the islanded microgrid.

One function does the physics, vectorised over a population of candidate plans.
The genetic algorithm calls it with hundreds of genomes against *forecast* PV
and load; the evaluation loop calls it with a single genome against *actual* PV
and load. Sharing one implementation is what makes "plan against the forecast,
score against the truth" an honest comparison rather than two subtly different
models.

Hourly energy balance, all terms non-negative except the signed battery power:

    pv - curtailed + discharge + diesel + unmet  ==  load + charge

Merit order is enforced structurally rather than hoped for. Diesel is clamped to
the *genuine* deficit that remains after PV and any battery discharge:

  * it can never run when supply already covers demand (gratuitous diesel), and
  * it can never generate more than the shortfall (over-dispatch).

Leaving that clamp out is what makes a naive optimiser burn fuel into
curtailment. The paper reports the mirror-image bug -- a repair operator that
discharged the battery during PV surplus -- which the same clamp prevents,
because a genuine deficit is computed *after* PV is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BatteryConfig, DieselConfig, EMSConfig


@dataclass
class DispatchResult:
    """Every array has shape ``(n_plans, n_hours)``; `soc` is end-of-hour."""

    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    diesel_kw: np.ndarray
    unmet_kw: np.ndarray
    curtailed_kw: np.ndarray
    soc: np.ndarray
    final_soc: np.ndarray

    @property
    def battery_kw(self) -> np.ndarray:
        """Signed battery power: positive discharging."""
        return self.discharge_kw - self.charge_kw

    @property
    def throughput_kwh(self) -> np.ndarray:
        return (self.charge_kw + self.discharge_kw).sum(axis=1)


def simulate(
    pv_kw: np.ndarray,
    load_kw: np.ndarray,
    battery_request_kw: np.ndarray,
    diesel_request_kw: np.ndarray,
    soc0: np.ndarray | float,
    battery: BatteryConfig,
    diesel: DieselConfig,
    dt_hours: float = 1.0,
) -> DispatchResult:
    """Apply a dispatch plan to a PV/load trajectory.

    `pv_kw` and `load_kw` have shape ``(n_hours,)``; the two request arrays have
    shape ``(n_plans, n_hours)``. Requests are projected onto what the plant can
    physically do, so any plan -- however absurd -- yields a feasible trajectory.
    """
    pv = np.asarray(pv_kw, dtype=float)
    load = np.asarray(load_kw, dtype=float)
    b_req = np.atleast_2d(np.asarray(battery_request_kw, dtype=float))
    d_req = np.atleast_2d(np.asarray(diesel_request_kw, dtype=float))

    n_plans, n_hours = b_req.shape
    if pv.shape != (n_hours,) or load.shape != (n_hours,):
        raise ValueError(f"pv/load must have shape ({n_hours},)")
    if d_req.shape != b_req.shape:
        raise ValueError("battery and diesel requests must have the same shape")

    cap = battery.nominal_kwh
    soc = np.full(n_plans, float(soc0)) if np.isscalar(soc0) else np.array(soc0, dtype=float)

    out = {k: np.zeros((n_plans, n_hours)) for k in
           ("charge", "discharge", "diesel", "unmet", "curtailed", "soc")}

    for h in range(n_hours):
        # -- battery, clipped by rate then by stored energy / headroom --------
        want = b_req[:, h]
        discharge = np.clip(want, 0.0, battery.p_discharge_max_kw)
        charge = np.clip(-want, 0.0, battery.p_charge_max_kw)

        max_discharge = (soc - battery.soc_min) * cap * battery.eta_discharge / dt_hours
        max_charge = (battery.soc_max - soc) * cap / (battery.eta_charge * dt_hours)
        discharge = np.minimum(discharge, np.maximum(max_discharge, 0.0))
        charge = np.minimum(charge, np.maximum(max_charge, 0.0))

        # -- load has priority over storing energy ---------------------------
        # A committed charge setpoint is planned on a forecast. If the sky turns
        # out cloudier, that charging adds to the deficit and can push it past
        # the generator's rating -- shedding load in order to fill a battery.
        # No real installation does that: the charger backs off first. Without
        # this rule the framework strands a few kWh a year and cannot reach the
        # zero LPSP the paper reports.
        deficit = load[h] + charge - pv[h] - discharge
        unservable = np.maximum(deficit - diesel.rated_kw, 0.0)
        relief = np.minimum(charge, unservable)
        charge = charge - relief
        deficit = deficit - relief

        # -- diesel: last resort, sized to the genuine deficit ---------------
        allowed = np.clip(np.maximum(deficit, 0.0), 0.0, diesel.rated_kw)
        gen = np.clip(d_req[:, h], 0.0, allowed)
        if diesel.min_load_fraction > 0.0:
            floor = diesel.min_load_fraction * diesel.rated_kw
            gen = np.where(gen <= 0.0, 0.0, np.maximum(gen, np.minimum(floor, allowed)))

        unmet = np.maximum(deficit - gen, 0.0)
        curtailed = np.maximum(-deficit, 0.0)

        # -- state of charge --------------------------------------------------
        soc = soc + (charge * battery.eta_charge - discharge / battery.eta_discharge) * dt_hours / cap
        soc = np.clip(soc, battery.soc_min, battery.soc_max)

        out["charge"][:, h] = charge
        out["discharge"][:, h] = discharge
        out["diesel"][:, h] = gen
        out["unmet"][:, h] = unmet
        out["curtailed"][:, h] = curtailed
        out["soc"][:, h] = soc

    return DispatchResult(
        charge_kw=out["charge"], discharge_kw=out["discharge"], diesel_kw=out["diesel"],
        unmet_kw=out["unmet"], curtailed_kw=out["curtailed"], soc=out["soc"], final_soc=soc,
    )


def objective(
    result: DispatchResult, diesel: DieselConfig, ems: EMSConfig, dt_hours: float = 1.0
) -> np.ndarray:
    """Cost of each plan, in GHS. Lower is better.

    Diesel fuel cost, plus an unmet-load penalty priced at ~950x the diesel
    marginal cost, plus small tie-breaking penalties on battery throughput
    (a degradation proxy) and curtailment.

    The unmet-load weight is the paper's central methodological finding: as a
    soft, comparable-magnitude penalty it never reproduces the baseline's
    reliability, because a cost-minimising optimiser will rationally accept
    some unserved load. Pricing it far above any achievable saving turns it
    into a near-hard constraint, leaving the optimiser free to minimise cost,
    cycling and curtailment among the plans that are already reliable.
    """
    diesel_cost = result.diesel_kw.sum(axis=1) * dt_hours * diesel.cost_per_kwh_ghs
    unmet_cost = result.unmet_kw.sum(axis=1) * dt_hours * ems.unmet_penalty_per_kwh
    curtail_cost = result.curtailed_kw.sum(axis=1) * dt_hours * ems.curtailment_penalty_per_kwh
    wear_cost = result.throughput_kwh * dt_hours * ems.throughput_penalty_per_kwh
    return diesel_cost + unmet_cost + curtail_cost + wear_cost


def energy_balance_residual(
    pv_kw: np.ndarray, load_kw: np.ndarray, result: DispatchResult, dt_hours: float = 1.0
) -> np.ndarray:
    """``pv - curtailed + discharge + diesel + unmet - load - charge``, per hour."""
    return (
        pv_kw[None, :] - result.curtailed_kw + result.discharge_kw + result.diesel_kw
        + result.unmet_kw - load_kw[None, :] - result.charge_kw
    )
