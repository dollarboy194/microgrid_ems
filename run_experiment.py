"""Reproduce the paper: forecast comparison (Table I) and EMS evaluation (Table II).

Pipeline
--------
1. Assemble the hybrid dataset: real NASA POWER weather (2018-2025) + a
   literature-calibrated synthetic load for 175 households.
2. Train the forecasting module's three candidate algorithms on 2018-2023,
   score them once on the held-out 2025 test year.
3. Instantiate the module with XGBoost, run the rolling GA dispatch (24 h
   horizon, 6 h commitment) across the test year, and replay the committed
   decisions against ground-truth PV and load.
4. Compare against the rule-based baseline and against the published figures.

Two control runs are included because they are what make the comparison
interpretable:

  * `ga_perfect` -- the same GA loop handed the true future. It must match the
    reactive baseline; any gap would be a defect in the commitment loop rather
    than a cost of forecast error.
  * `baseline` -- reactive, hourly, perfect-information. Reliable by brute force.

Usage:
    python run_experiment.py
    python run_experiment.py --forecaster rf --skip-table1
    python run_experiment.py --test-days 60          # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from microgrid import plots
from microgrid.config import ExperimentConfig
from microgrid.dataset import build_dataset
from microgrid.ems import issue_index_for, run_baseline, run_ga_ems, usable_test_index
from microgrid.forecast import build_forecaster
from microgrid.metrics import compare, ems_kpis, errors_by_lead, forecast_errors

# Published values, for a side-by-side rather than a claim of exact equality.
PAPER_TABLE1 = {
    ("rf", "pv_kw"): (0.78, 1.93, 0.991, 12.73),
    ("lstm", "pv_kw"): (0.81, 1.86, 0.991, 12.27),
    ("xgboost", "pv_kw"): (0.82, 1.91, 0.991, 12.64),
    ("rf", "load_kw"): (1.03, 1.91, 0.937, 13.89),
    ("lstm", "load_kw"): (1.04, 1.93, 0.935, 14.05),
    ("xgboost", "load_kw"): (1.02, 1.90, 0.937, 13.84),
}
PAPER_TABLE2 = {
    "lpsp_pct": (0.00, 0.00),
    "diesel_kwh": (21440.0, 24813.0),
    "diesel_cost_ghs": (113205.0, 131011.0),
    "renewable_fraction_pct": (82.1, 79.3),
    "curtailed_kwh": (30367.0, 33662.0),
    "battery_cycles": (339.1, 352.0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forecaster", default="xgboost", choices=["xgboost", "rf", "lstm"],
                   help="algorithm instantiating the forecasting module for the EMS run")
    p.add_argument("--skip-table1", action="store_true", help="skip the three-way forecaster comparison")
    p.add_argument("--test-days", type=int, default=None, help="shorten the test window (smoke runs)")
    p.add_argument("--outdir", type=Path, default=Path("results"))
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def _fmt(value: float, width: int = 10, dp: int = 2) -> str:
    return f"{value:>{width},.{dp}f}"


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()

    # -- 1. data -----------------------------------------------------------
    print("building dataset (NASA POWER + synthetic load) ...")
    ds = build_dataset(cfg, cache_dir=Path("data"))
    frame = ds.frame
    for key, value in ds.summary().items():
        print(f"  {key:20s} {value}")

    test = ds.test if args.test_days is None else ds.test[: args.test_days * 24]
    test = usable_test_index(frame, test, cfg)
    print(f"  evaluated test window {test[0]:%Y-%m-%d %H:%M} .. {test[-1]:%Y-%m-%d %H:%M} ({len(test)} h)")

    # -- 2. Table I: three-way forecaster comparison -----------------------
    table1 = None
    fitted: dict[str, object] = {}
    if not args.skip_table1:
        print("\ntraining the forecasting module's three candidates ...")
        rows = []
        eval_issue = ds.test[: -cfg.ems.horizon_hours]
        for kind in ("rf", "lstm", "xgboost"):
            t0 = time.perf_counter()
            model = build_forecaster(kind, cfg.ems.horizon_hours, cfg.pv.capacity_kwp,
                                     random_state=cfg.split.seed)
            model.fit(frame, ds.train)
            fitted[kind] = model
            fit_s = time.perf_counter() - t0
            bundle = model.predict(frame, eval_issue)

            one_step = forecast_errors(frame, bundle, leads=(1,))
            horizon = forecast_errors(frame, bundle)
            for _, r in one_step.iterrows():
                paper = PAPER_TABLE1[(kind, r["target"])]
                rows.append({
                    "model": kind, "target": r["target"], "scope": "h=1",
                    "mae_kw": r["mae_kw"], "rmse_kw": r["rmse_kw"], "r2": r["r2"],
                    "nrmse_pct": r["nrmse_pct"], "fit_seconds": fit_s,
                    "paper_mae": paper[0], "paper_rmse": paper[1],
                    "paper_r2": paper[2], "paper_nrmse": paper[3],
                })
            for _, r in horizon.iterrows():
                rows.append({
                    "model": kind, "target": r["target"], "scope": "h=1..24",
                    "mae_kw": r["mae_kw"], "rmse_kw": r["rmse_kw"], "r2": r["r2"],
                    "nrmse_pct": r["nrmse_pct"], "fit_seconds": fit_s,
                })
            print(f"  {kind:<8s} fitted in {fit_s:6.0f}s")

        table1 = pd.DataFrame(rows)
        table1.to_csv(args.outdir / "table1_forecasters.csv", index=False)

        print("\nTABLE I  three-way forecasting comparison, held-out test year 2025")
        print("  one-hour-ahead (the scope that reproduces the paper's magnitudes)")
        print(f"  {'model':<9}{'task':<7}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'nRMSE%':>9}   "
              f"{'| paper: MAE':>12}{'RMSE':>7}{'R2':>7}{'nRMSE%':>8}")
        for _, r in table1[table1["scope"] == "h=1"].iterrows():
            task = r["target"].replace("_kw", "")
            print(f"  {r['model']:<9}{task:<7}{_fmt(r['mae_kw'],8)}{_fmt(r['rmse_kw'],8)}"
                  f"{_fmt(r['r2'],8,3)}{_fmt(r['nrmse_pct'],9)}   |{_fmt(r['paper_mae'],11)}"
                  f"{_fmt(r['paper_rmse'],7)}{_fmt(r['paper_r2'],7,3)}{_fmt(r['paper_nrmse'],8)}")

    # -- 3. EMS runs -------------------------------------------------------
    print(f"\nrunning the EMS over the test year (forecaster: {args.forecaster}) ...")
    issue = issue_index_for(frame, test, cfg)

    t0 = time.perf_counter()
    if args.forecaster in fitted:
        model = fitted[args.forecaster]  # already trained for Table I; refitting would be waste
    else:
        model = build_forecaster(args.forecaster, cfg.ems.horizon_hours, cfg.pv.capacity_kwp,
                                 random_state=cfg.split.seed)
        model.fit(frame, ds.train)
    bundle = model.predict(frame, issue)
    print(f"  forecasts ready in {time.perf_counter() - t0:.0f}s")

    oracle = build_forecaster("perfect", cfg.ems.horizon_hours, cfg.pv.capacity_kwp)
    oracle_bundle = oracle.predict(frame, issue)

    logs: dict[str, pd.DataFrame] = {}
    t0 = time.perf_counter()
    logs["baseline"] = run_baseline(frame, test, cfg)
    print(f"  baseline           {time.perf_counter() - t0:5.0f}s")

    t0 = time.perf_counter()
    logs["ga_perfect"] = run_ga_ems(frame, test, oracle_bundle, cfg)
    print(f"  ga_perfect         {time.perf_counter() - t0:5.0f}s  (control: must match baseline)")

    t0 = time.perf_counter()
    logs[f"ga_{args.forecaster}"] = run_ga_ems(frame, test, bundle, cfg)
    print(f"  ga_{args.forecaster:<15s} {time.perf_counter() - t0:5.0f}s")

    for name, log in logs.items():
        log.to_csv(args.outdir / f"dispatch_{name}.csv")

    # -- 4. Table II -------------------------------------------------------
    kpis = compare(logs, cfg)
    kpis.to_csv(args.outdir / "table2_ems.csv")

    proposed = f"ga_{args.forecaster}"
    base_k, prop_k = ems_kpis(logs["baseline"], cfg), ems_kpis(logs[proposed], cfg)

    print("\nTABLE II  proposed framework vs rule-based baseline, held-out test year")
    print(f"  {'metric':<26}{'baseline':>12}{'proposed':>12}{'change':>10}   "
          f"{'| paper base':>13}{'paper prop':>12}{'paper chg':>11}")
    for key, (p_base, p_prop) in PAPER_TABLE2.items():
        b, p = base_k[key], prop_k[key]
        # A percentage change off a zero baseline is meaningless, and float noise
        # turns 0 -> 0 into a spectacular number. Report the absolute delta.
        chg = 100.0 * (p - b) / b if abs(b) > 1e-9 else float("nan")
        p_chg = 100.0 * (p_prop - p_base) / p_base if abs(p_base) > 1e-9 else float("nan")
        chg_s = f"{p - b:+9.2f} " if not np.isfinite(chg) else f"{chg:+9.1f}%"
        pchg_s = f"{p_prop - p_base:+10.2f} " if not np.isfinite(p_chg) else f"{p_chg:+10.1f}%"
        print(f"  {key:<26}{_fmt(b,12)}{_fmt(p,12)}{chg_s}   |{_fmt(p_base,12)}{_fmt(p_prop,12)}{pchg_s}")

    ctrl = ems_kpis(logs["ga_perfect"], cfg)
    gap = abs(ctrl["diesel_kwh"] - base_k["diesel_kwh"]) / max(base_k["diesel_kwh"], 1e-9)
    print(f"\n  control: ga_perfect diesel differs from baseline by {100*gap:.3f}% "
          f"({'OK' if gap < 0.005 else 'PROBLEM: the commitment loop, not forecast error'})")

    summary = {
        "config": cfg.to_dict(),
        "forecaster": args.forecaster,
        "test_window": [str(test[0]), str(test[-1])],
        "kpis": {k: {m: float(v) for m, v in col.items()} for k, col in kpis.to_dict().items()},
        "paper_table2": {k: {"baseline": v[0], "proposed": v[1]} for k, v in PAPER_TABLE2.items()},
    }
    with open(args.outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # -- 5. figures --------------------------------------------------------
    if not args.skip_plots:
        print("\nwriting figures ...")
        plots.plot_resource_and_demand(frame, ds.test, cfg, args.outdir / "fig2_pv_vs_load.png")
        plots.plot_forecast_week(frame, bundle, args.outdir / "fig4_forecast_vs_actual.png")
        plots.plot_ems_comparison(logs, cfg, args.outdir / "fig5_ems_comparison.png")
        plots.plot_soc_and_diesel(logs, cfg, args.outdir / "fig5b_soc_monthly_diesel.png")
        if table1 is not None:
            plots.plot_forecast_errors(table1, args.outdir / "fig3_forecast_errors.png")
        errors_by_lead(frame, bundle).to_csv(args.outdir / "forecast_error_by_lead.csv", index=False)

    print(f"\nwrote results to {args.outdir.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
