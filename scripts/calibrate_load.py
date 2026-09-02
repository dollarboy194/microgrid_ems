"""Calibrate the synthetic load generator against the paper's Table I.

The paper's load profile is synthetic and not published. What *is* published is
how predictable it is: one-hour-ahead RMSE ~1.90 kW and R^2 ~0.937 for XGBoost
on the held-out test year. Since the EMS results depend directly on how much of
the demand a forecaster can anticipate, the honest way to reproduce them is to
tune the generator's stochastic parameters until its forecastability matches --
rather than tune them to taste and then report an EMS gap that is really an
artefact of the noise model.

Only the one-hour-ahead load model is fitted, so the sweep is cheap.

Usage:  python scripts/calibrate_load.py
"""

from __future__ import annotations

import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microgrid.config import ExperimentConfig  # noqa: E402
from microgrid.dataset import build_dataset  # noqa: E402
from microgrid.features import build_design  # noqa: E402

TARGET_RMSE = 1.90
TARGET_R2 = 0.937


def score(cfg: ExperimentConfig) -> tuple[float, float, float]:
    ds = build_dataset(cfg, cache_dir=Path("data"))
    frame = ds.frame

    x, y, valid, _ = build_design(frame, "load_kw", horizon=1)
    train_mask = np.zeros(len(frame), dtype=bool)
    train_mask[frame.index.get_indexer(ds.train)] = True
    test_mask = np.zeros(len(frame), dtype=bool)
    test_mask[frame.index.get_indexer(ds.test)] = True

    model = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.85,
        colsample_bytree=0.85, tree_method="hist", n_jobs=-1, verbosity=0, random_state=0,
    ).fit(x[train_mask & valid], y[train_mask & valid])

    rows = test_mask & valid
    pred = model.predict(x[rows])
    truth = y[rows]
    err = pred - truth
    rmse = float(np.sqrt(np.mean(err**2)))
    r2 = 1.0 - float(np.sum(err**2)) / float(np.sum((truth - truth.mean()) ** 2))
    return rmse, r2, float(truth.std())


def main() -> None:
    grid = list(itertools.product(
        (0.04, 0.05, 0.06, 0.07),   # household_noise_cv
        (0.010, 0.015, 0.020),      # spike_probability
        (2.0, 2.5),                 # spike_multiplier
        (1.0, 1.15, 1.3),           # shape_sharpness
    ))

    records = []
    for cv, sp, sm, gamma in grid:
        base = ExperimentConfig()
        cfg = dataclasses.replace(
            base,
            load=dataclasses.replace(
                base.load, household_noise_cv=cv, spike_probability=sp,
                spike_multiplier=sm, shape_sharpness=gamma,
            ),
        )
        rmse, r2, sd = score(cfg)
        loss = abs(rmse - TARGET_RMSE) / TARGET_RMSE + abs(r2 - TARGET_R2) / TARGET_R2
        records.append(dict(cv=cv, spike_p=sp, spike_mult=sm, gamma=gamma,
                            rmse=rmse, r2=r2, load_sd=sd, loss=loss))
        sys.stdout.write(
            f"cv={cv:.2f} p={sp:.3f} m={sm:.1f} g={gamma:.2f} -> "
            f"RMSE {rmse:.3f}  R2 {r2:.4f}  sd {sd:.2f}  loss {loss:.4f}\n"
        )
        sys.stdout.flush()

    table = pd.DataFrame(records).sort_values("loss")
    table.to_csv("results/load_calibration.csv", index=False)
    best = table.iloc[0]
    sys.stdout.write("\nbest: %s\n" % best.to_dict())
    sys.stdout.write(f"target RMSE {TARGET_RMSE}, R2 {TARGET_R2}\n")


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    main()
