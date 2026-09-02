"""Energy management: the rolling forecast-optimize-evaluate loop.

Every `commit_hours` (6 by default) the EMS issues a fresh forecast covering the
next `horizon_hours` (24), runs the genetic algorithm against that forecast, and
**commits only the first 6 hours** of the resulting plan. Those committed
decisions are then replayed against the *actual* PV and load.

The asymmetry this creates is the paper's central mechanism. The rule-based
baseline reacts to actual conditions every hour, with perfect hindsight. The
framework commits six hours ahead on an imperfect forecast and cannot revise
mid-block, so any error inside a commitment window -- above all an
under-predicted milling or welding spike -- must be absorbed as extra diesel or,
failing that, as unmet load. That is where the diesel premium comes from, and
scoring the plan against ground truth is what makes it visible instead of
hiding it inside a self-consistent evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .dispatch import simulate
from .forecast import ForecastBundle
from .ga import GAResult, greedy_merit_order, optimise

LOG_COLUMNS = (
    "pv_kw", "load_kw", "charge_kw", "discharge_kw", "battery_kw",
    "diesel_kw", "unmet_kw", "curtailed_kw", "soc",
)


def _log_frame(index: pd.DatetimeIndex, pv, load, result) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "pv_kw": pv,
            "load_kw": load,
            "charge_kw": result.charge_kw[0],
            "discharge_kw": result.discharge_kw[0],
            "battery_kw": result.battery_kw[0],
            "diesel_kw": result.diesel_kw[0],
            "unmet_kw": result.unmet_kw[0],
            "curtailed_kw": result.curtailed_kw[0],
            "soc": result.soc[0],
        },
        index=index,
    )
    return frame


def run_baseline(frame: pd.DataFrame, test_index: pd.DatetimeIndex, cfg: ExperimentConfig) -> pd.DataFrame:
    """Conventional rule-based EMS: reactive, hourly, perfect information.

    PV serves load first; surplus charges the battery and is otherwise
    curtailed; a deficit draws on the battery and then the generator. It never
    forecasts, and it never has to: it sees the actual hour it is dispatching.
    That is precisely why it is a demanding reliability benchmark rather than a
    straw man -- it is reliable by brute force.
    """
    pv = frame.loc[test_index, "pv_kw"].to_numpy()
    load = frame.loc[test_index, "load_kw"].to_numpy()

    b_plan, d_plan = greedy_merit_order(pv, load, cfg.battery.soc_init, cfg.battery, cfg.diesel)
    result = simulate(pv, load, b_plan[None, :], d_plan[None, :], cfg.battery.soc_init,
                      cfg.battery, cfg.diesel)

    log = _log_frame(test_index, pv, load, result)
    log.attrs["controller"] = "baseline_rule_based"
    return log


def _apply_margin(pv: np.ndarray, load: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Plan pessimistically: less sun than forecast, more demand than forecast."""
    if margin <= 0.0:
        return pv, load
    return pv * (1.0 - margin), load * (1.0 + margin)


def run_ga_ems(
    frame: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    bundle: ForecastBundle,
    cfg: ExperimentConfig,
    collect_ga: bool = False,
) -> pd.DataFrame:
    """Rolling GA dispatch: re-plan every commit block, replay against truth."""
    ems, ga_cfg = cfg.ems, cfg.ga
    horizon, commit = ems.horizon_hours, ems.commit_hours

    positions = frame.index.get_indexer(test_index)
    if (positions < 0).any():
        raise ValueError("test_index contains timestamps missing from the frame")

    pv_actual = frame["pv_kw"].to_numpy()
    load_actual = frame["load_kw"].to_numpy()

    # A forecast for the block starting at t0 is issued at t0 - 1, the last hour
    # whose observation is available. Lead h then lands on t0 + h - 1.
    block_starts = positions[::commit]
    issue_positions = block_starts - 1
    issue_index = frame.index[issue_positions]
    if not bundle.index.equals(issue_index):
        raise ValueError(
            "forecast bundle was not issued at the block boundaries this loop expects"
        )
    if bundle.horizon < horizon:
        raise ValueError(f"bundle horizon {bundle.horizon} < EMS horizon {horizon}")

    rng = np.random.default_rng(ga_cfg.random_state)
    soc = cfg.battery.soc_init
    logs: list[pd.DataFrame] = []
    ga_runs: list[GAResult] = []

    for block, start in enumerate(block_starts):
        n_commit = min(commit, positions[-1] + 1 - start)
        n_plan = min(horizon, len(frame) - start)

        pv_fc = bundle.values["pv_kw"][block, :n_plan]
        load_fc = bundle.values["load_kw"][block, :n_plan]
        pv_fc, load_fc = _apply_margin(pv_fc, load_fc, ems.forecast_margin)

        plan = optimise(pv_fc, load_fc, soc, cfg.battery, cfg.diesel, ems, ga_cfg, rng=rng)
        if collect_ga:
            ga_runs.append(plan)

        # Commit the first `n_commit` hours and replay them against the truth.
        # The battery follows the plan exactly -- that is the commitment, and
        # the cost of planning it on a forecast. The generator either follows
        # the residual deficit (it is the balancing unit) or is frozen too.
        window = slice(start, start + n_commit)
        if ems.diesel_follows_deficit:
            # `simulate` clamps the generator to the genuine remaining deficit,
            # so requesting its full rating means "serve whatever is left".
            diesel_request = np.full(n_commit, cfg.diesel.rated_kw)
        else:
            diesel_request = plan.diesel_kw[:n_commit]

        result = simulate(
            pv_actual[window], load_actual[window],
            plan.battery_kw[None, :n_commit], diesel_request[None, :],
            soc, cfg.battery, cfg.diesel,
        )
        soc = float(result.final_soc[0])
        logs.append(_log_frame(frame.index[window], pv_actual[window], load_actual[window], result))

    log = pd.concat(logs)
    log.attrs["controller"] = "ga_forecast_ems"
    if collect_ga:
        log.attrs["ga_runs"] = ga_runs
    return log


def issue_index_for(frame: pd.DataFrame, test_index: pd.DatetimeIndex, cfg: ExperimentConfig) -> pd.DatetimeIndex:
    """Timestamps at which the EMS issues a forecast (one per commit block)."""
    positions = frame.index.get_indexer(test_index)
    return frame.index[positions[:: cfg.ems.commit_hours] - 1]


def usable_test_index(frame: pd.DataFrame, test_index: pd.DatetimeIndex, cfg: ExperimentConfig) -> pd.DatetimeIndex:
    """Trim the test window so every block can be issued a full-horizon forecast.

    The last block of the year would need calendar attributes for hours beyond
    the end of the record. Rather than silently short-planning it, drop the
    tail: at most `horizon - commit` hours of an 8,760-hour year, and the loop
    is then uniform across every block it does evaluate.
    """
    ems = cfg.ems
    n = len(test_index) - (ems.horizon_hours - ems.commit_hours)
    n -= n % ems.commit_hours
    if n <= 0:
        raise ValueError("test window shorter than one commit block plus the horizon tail")
    return test_index[:n]
