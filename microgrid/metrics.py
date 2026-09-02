"""Evaluation metrics.

Forecast accuracy (Table I) and EMS performance (Table II) are computed
separately, so an improvement in one can be attributed to -- or dissociated
from -- the other. That separation is the point of the paper: all three
forecasters score comparably, yet the framework still pays a diesel premium,
which locates the premium in the commitment structure rather than in forecast
skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .dataset import TARGETS
from .forecast import ForecastBundle

_UNMET_TOL_KW = 1e-6


# --------------------------------------------------------------------------
# Forecast accuracy
# --------------------------------------------------------------------------

def _aligned_actuals(frame: pd.DataFrame, bundle: ForecastBundle, target: str) -> np.ndarray:
    """`out[t, h-1]` is the realised value the forecast at lead h was aiming at."""
    rows = frame.index.get_indexer(bundle.index)
    y = frame[target].to_numpy()
    out = np.empty((len(rows), bundle.horizon))
    for h in range(1, bundle.horizon + 1):
        out[:, h - 1] = y[np.minimum(rows + h, len(y) - 1)]
    return out


def forecast_errors(frame: pd.DataFrame, bundle: ForecastBundle, leads: tuple[int, ...] | None = None) -> pd.DataFrame:
    """MAE, RMSE, R^2 and nRMSE per target.

    `nRMSE` and `nMAE` are normalised by the **mean** of the actual series, the
    convention that reproduces the paper's Table I magnitudes (PV nRMSE ~12.6%
    against a mean PV output of ~15 kW).

    The paper does not state whether its Table I is one-step-ahead or averaged
    across the 24-hour dispatch horizon, so both are reported: pass
    ``leads=(1,)`` for one-step, or leave it as None for the full horizon.
    """
    records = []
    for target in TARGETS:
        actual = _aligned_actuals(frame, bundle, target)
        pred = bundle.values[target]
        if leads is not None:
            cols = [h - 1 for h in leads]
            actual, pred = actual[:, cols], pred[:, cols]

        err = (pred - actual).ravel()
        truth = actual.ravel()
        mean = truth.mean()
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((truth - mean) ** 2))

        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        records.append({
            "target": target,
            "mae_kw": mae,
            "rmse_kw": rmse,
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "nmae_pct": 100.0 * mae / mean,
            "nrmse_pct": 100.0 * rmse / mean,
            "bias_kw": float(np.mean(err)),
        })
    return pd.DataFrame(records)


def errors_by_lead(frame: pd.DataFrame, bundle: ForecastBundle) -> pd.DataFrame:
    """Per-lead-time error, to show how skill decays across the commit window."""
    records = []
    for target in TARGETS:
        actual = _aligned_actuals(frame, bundle, target)
        pred = bundle.values[target]
        for h in range(1, bundle.horizon + 1):
            err = pred[:, h - 1] - actual[:, h - 1]
            records.append({
                "target": target,
                "lead_hours": h,
                "mae_kw": float(np.mean(np.abs(err))),
                "rmse_kw": float(np.sqrt(np.mean(err**2))),
            })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# EMS performance
# --------------------------------------------------------------------------

def ems_kpis(log: pd.DataFrame, cfg: ExperimentConfig, dt_hours: float = 1.0) -> dict[str, float]:
    """The Table II metrics for one closed-loop run."""
    demand_kwh = float(log["load_kw"].sum() * dt_hours)
    unmet_kwh = float(log["unmet_kw"].sum() * dt_hours)
    diesel_kwh = float(log["diesel_kw"].sum() * dt_hours)
    curtailed_kwh = float(log["curtailed_kw"].sum() * dt_hours)
    pv_kwh = float(log["pv_kw"].sum() * dt_hours)
    charge_kwh = float(log["charge_kw"].sum() * dt_hours)
    discharge_kwh = float(log["discharge_kw"].sum() * dt_hours)

    served_kwh = demand_kwh - unmet_kwh
    pv_used_kwh = pv_kwh - curtailed_kwh

    # Loss of Power Supply Probability: unserved energy as a fraction of demand.
    lpsp = unmet_kwh / demand_kwh if demand_kwh else float("nan")

    # Throughput is measured at the terminals; a full cycle is one charge and
    # one discharge of the *usable* pack.
    throughput_kwh = charge_kwh + discharge_kwh
    cycles = throughput_kwh / (2.0 * cfg.battery.usable_kwh)

    return {
        "lpsp_pct": 100.0 * lpsp,
        "loss_of_load_hours": float((log["unmet_kw"] > _UNMET_TOL_KW).sum()),
        "unmet_kwh": unmet_kwh,
        "demand_kwh": demand_kwh,
        "served_kwh": served_kwh,
        "diesel_kwh": diesel_kwh,
        "diesel_cost_ghs": diesel_kwh * cfg.diesel.cost_per_kwh_ghs,
        "diesel_hours": float((log["diesel_kw"] > 1e-6).sum()),
        # Two conventions; the paper does not say which it uses, so report both.
        "renewable_fraction_pct": 100.0 * (1.0 - diesel_kwh / served_kwh) if served_kwh else float("nan"),
        "renewable_fraction_gen_pct": 100.0 * pv_used_kwh / (pv_used_kwh + diesel_kwh)
        if (pv_used_kwh + diesel_kwh) else float("nan"),
        "pv_kwh": pv_kwh,
        "pv_used_kwh": pv_used_kwh,
        "curtailed_kwh": curtailed_kwh,
        "curtailment_pct": 100.0 * curtailed_kwh / pv_kwh if pv_kwh else float("nan"),
        "battery_cycles": cycles,
        "battery_throughput_kwh": throughput_kwh,
        "discharge_kwh": discharge_kwh,
        "min_soc": float(log["soc"].min()),
    }


def compare(logs: dict[str, pd.DataFrame], cfg: ExperimentConfig) -> pd.DataFrame:
    """KPI table with one column per controller, plus the change versus baseline."""
    table = pd.DataFrame({name: ems_kpis(log, cfg) for name, log in logs.items()})
    if "baseline" in table.columns and len(table.columns) > 1:
        for col in table.columns:
            if col == "baseline":
                continue
            base = table["baseline"]
            with np.errstate(divide="ignore", invalid="ignore"):
                table[f"{col}_vs_baseline_pct"] = 100.0 * (table[col] - base) / base.abs()
    return table
