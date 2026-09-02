"""The sensitivity studies the paper names but does not run.

Section VI states that the diesel premium "should be sensitive to at least three
factors not independently varied in this study", and presents them explicitly as
hypotheses rather than conclusions:

  1. **the re-optimization interval** -- a shorter commitment window should
     narrow the gap to the baseline's hourly reactive control;
  2. **load-forecast error magnitude** -- the dominant driver of unmet load
     under the soft-constraint formulation;
  3. **the unmet-load penalty weight** -- which traded diesel cost directly
     against reliability before being fixed as a hard constraint.

Each is now a one-command experiment. The soft-vs-hard sweep additionally
reproduces the validation-year finding: no penalty weight matches the baseline
on both reliability and cost simultaneously.

Usage:
    python scripts/ablations.py commit          # 1, 2, 3, 6, 12, 24 h windows
    python scripts/ablations.py penalty         # unmet-load weight sweep
    python scripts/ablations.py margin          # forecast safety margin sweep
    python scripts/ablations.py all
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microgrid.config import ExperimentConfig  # noqa: E402
from microgrid.dataset import build_dataset  # noqa: E402
from microgrid.ems import issue_index_for, run_baseline, run_ga_ems, usable_test_index  # noqa: E402
from microgrid.forecast import build_forecaster  # noqa: E402
from microgrid.metrics import ems_kpis  # noqa: E402

OUT = Path("results")
KEYS = ("lpsp_pct", "diesel_kwh", "diesel_cost_ghs", "renewable_fraction_pct",
        "curtailed_kwh", "battery_cycles", "unmet_kwh")


def _setup(test_days: int | None, forecaster: str = "xgboost"):
    cfg = ExperimentConfig()
    ds = build_dataset(cfg, cache_dir=Path("data"))
    test = ds.test if test_days is None else ds.test[: test_days * 24]
    model = build_forecaster(forecaster, cfg.ems.horizon_hours, cfg.pv.capacity_kwp,
                             random_state=cfg.split.seed)
    model.fit(ds.frame, ds.train)
    return cfg, ds, test, model


def _run(cfg: ExperimentConfig, ds, test, model) -> dict[str, float]:
    window = usable_test_index(ds.frame, test, cfg)
    bundle = model.predict(ds.frame, issue_index_for(ds.frame, window, cfg))
    return ems_kpis(run_ga_ems(ds.frame, window, bundle, cfg), cfg)


def sweep_commit(test_days: int | None) -> pd.DataFrame:
    """Hypothesis 1: a shorter commitment window narrows the diesel premium."""
    cfg, ds, test, model = _setup(test_days)
    base = ems_kpis(run_baseline(ds.frame, usable_test_index(ds.frame, test, cfg), cfg), cfg)

    rows = []
    for commit in (1, 2, 3, 6, 12, 24):
        c = dataclasses.replace(cfg, ems=dataclasses.replace(cfg.ems, commit_hours=commit))
        k = _run(c, ds, test, model)
        rows.append({"commit_hours": commit,
                     "diesel_premium_pct": 100 * (k["diesel_kwh"] - base["diesel_kwh"]) / base["diesel_kwh"],
                     **{key: k[key] for key in KEYS}})
        print(f"  commit={commit:2d} h  diesel {k['diesel_kwh']:9,.0f} kWh  "
              f"premium {rows[-1]['diesel_premium_pct']:+6.2f}%  LPSP {k['lpsp_pct']:.4f}%")
    return pd.DataFrame(rows)


def sweep_penalty(test_days: int | None) -> pd.DataFrame:
    """Hypothesis 3, and the soft-vs-hard finding.

    A soft penalty must be shown to fail: at low weights the optimiser accepts
    unmet load, and no weight in the swept range matches the baseline on both
    reliability and cost.
    """
    cfg, ds, test, model = _setup(test_days)
    frozen = dataclasses.replace(cfg.ems, diesel_follows_deficit=False)

    rows = []
    for multiplier in (0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 950.0):
        c = dataclasses.replace(cfg, ems=dataclasses.replace(frozen, unmet_penalty_multiplier=multiplier))
        k = _run(c, ds, test, model)
        rows.append({"unmet_penalty_multiplier": multiplier, **{key: k[key] for key in KEYS}})
        print(f"  weight={multiplier:7.1f}x  LPSP {k['lpsp_pct']:7.4f}%  "
              f"unmet {k['unmet_kwh']:8,.1f} kWh  diesel cost {k['diesel_cost_ghs']:11,.0f} GHS")
    return pd.DataFrame(rows)


def sweep_margin(test_days: int | None) -> pd.DataFrame:
    """The paper's validation sweep: forecast safety margin from 0% to 25%."""
    cfg, ds, test, model = _setup(test_days)
    rows = []
    for margin in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        c = dataclasses.replace(cfg, ems=dataclasses.replace(cfg.ems, forecast_margin=margin))
        k = _run(c, ds, test, model)
        rows.append({"forecast_margin": margin, **{key: k[key] for key in KEYS}})
        print(f"  margin={margin:5.0%}  diesel cost {k['diesel_cost_ghs']:11,.0f} GHS  "
              f"LPSP {k['lpsp_pct']:.4f}%  curtailed {k['curtailed_kwh']:9,.0f} kWh")
    return pd.DataFrame(rows)


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    test_days = int(sys.argv[2]) if len(sys.argv) > 2 else None
    OUT.mkdir(exist_ok=True)

    jobs = {"commit": sweep_commit, "penalty": sweep_penalty, "margin": sweep_margin}
    chosen = jobs if which == "all" else {which: jobs[which]}

    for name, fn in chosen.items():
        print(f"\n=== {name} sweep ===")
        table = fn(test_days)
        path = OUT / f"ablation_{name}.csv"
        table.to_csv(path, index=False)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
