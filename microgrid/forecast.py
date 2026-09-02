"""Forecasting module: Random Forest, XGBoost, and LSTM.

The framework's claim is that its value lies in the forecast-optimize-evaluate
*structure*, not in any one forecaster. So all three candidates implement the
same contract: fit on a training window, then emit, for every issue time, a
vector of predictions covering the next `horizon` hours -- exactly what the
dispatch optimiser consumes.

RF and XGBoost use the leakage-safe tabular feature set, with an independent
model per lead time (**direct multi-horizon**). This costs more to train than a
recursive one-step model but does not compound its own errors.

The LSTM instead reads a 24-hour sliding window of raw sequence values and
emits all 24 lead times at once, with inputs scaled using **training-set-only**
statistics -- computing those statistics over the whole series would leak test
information into training.

Note on the paper's Table I: it does not state whether the reported errors are
one-step-ahead or averaged across the dispatch horizon. `forecast_errors` in
`metrics.py` reports both, and `run_experiment.py` prints them side by side.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .dataset import TARGETS
from .features import build_design


@dataclass
class ForecastBundle:
    """`values[target]` has shape ``(n_issue_times, horizon)``; column h-1 is t0+h."""

    index: pd.DatetimeIndex
    horizon: int
    values: dict[str, np.ndarray]

    def at(self, issue_position: int) -> dict[str, np.ndarray]:
        return {t: self.values[t][issue_position] for t in self.values}


def _clip_physical(target: str, values: np.ndarray, pv_capacity_kw: float) -> np.ndarray:
    """Forecasts must stay inside what the plant can physically do."""
    if target == "pv_kw":
        return np.clip(values, 0.0, pv_capacity_kw)
    return np.maximum(values, 0.0)


class Forecaster(abc.ABC):
    name = "base"

    def __init__(self, horizon: int, pv_capacity_kw: float) -> None:
        self.horizon = horizon
        self.pv_capacity_kw = pv_capacity_kw

    @abc.abstractmethod
    def fit(self, frame: pd.DataFrame, train_index: pd.DatetimeIndex) -> "Forecaster": ...

    @abc.abstractmethod
    def predict(self, frame: pd.DataFrame, issue_index: pd.DatetimeIndex) -> ForecastBundle: ...

    def _finalise(self, issue_index, values) -> ForecastBundle:
        for target in values:
            values[target] = _clip_physical(target, values[target], self.pv_capacity_kw)
        return ForecastBundle(issue_index, self.horizon, values)


class PerfectForecaster(Forecaster):
    """Oracle. The reactive baseline in the paper is a perfect-information
    controller, so this is what makes that comparison reproducible."""

    name = "perfect"

    def fit(self, frame, train_index):
        return self

    def predict(self, frame, issue_index) -> ForecastBundle:
        rows = frame.index.get_indexer(issue_index)
        values = {}
        for target in TARGETS:
            y = frame[target].to_numpy()
            out = np.empty((len(rows), self.horizon))
            for h in range(1, self.horizon + 1):
                out[:, h - 1] = y[np.minimum(rows + h, len(y) - 1)]
            values[target] = out
        return self._finalise(issue_index, values)


class PersistenceForecaster(Forecaster):
    """Yesterday's value at the same hour -- the standard naive reference."""

    name = "persistence"

    def fit(self, frame, train_index):
        return self

    def predict(self, frame, issue_index) -> ForecastBundle:
        rows = frame.index.get_indexer(issue_index)
        values = {}
        for target in TARGETS:
            y = frame[target].to_numpy()
            out = np.empty((len(rows), self.horizon))
            for h in range(1, self.horizon + 1):
                out[:, h - 1] = y[np.clip(rows + h - 24, 0, len(y) - 1)]
            values[target] = out
        return self._finalise(issue_index, values)


class _TabularForecaster(Forecaster):
    """Direct multi-horizon: one estimator per (target, lead time)."""

    def __init__(self, horizon: int, pv_capacity_kw: float, random_state: int = 0) -> None:
        super().__init__(horizon, pv_capacity_kw)
        self.random_state = random_state
        self._models: dict[tuple[str, int], object] = {}

    @abc.abstractmethod
    def _make_estimator(self): ...

    def fit(self, frame, train_index):
        train_rows = frame.index.get_indexer(train_index)
        mask = np.zeros(len(frame), dtype=bool)
        mask[train_rows] = True

        for target in TARGETS:
            for h in range(1, self.horizon + 1):
                x, y, valid, _ = build_design(frame, target, h)
                rows = mask & valid
                self._models[(target, h)] = self._make_estimator().fit(x[rows], y[rows])
        return self

    def predict(self, frame, issue_index) -> ForecastBundle:
        rows = frame.index.get_indexer(issue_index)
        values = {t: np.empty((len(rows), self.horizon)) for t in TARGETS}
        for target in TARGETS:
            for h in range(1, self.horizon + 1):
                x, _, _, _ = build_design(frame, target, h)
                # Predicting needs the features to exist, not the label. The
                # label is only required for fitting, and demanding it here
                # would refuse to forecast the very hours the EMS runs on.
                finite = np.isfinite(x).all(axis=1)
                if not finite[rows].all():
                    bad = issue_index[~finite[rows]]
                    raise ValueError(
                        f"features undefined at {len(bad)} issue times, e.g. {bad[0]}; "
                        f"lead {h} h needs calendar attributes beyond the end of the record"
                    )
                values[target][:, h - 1] = self._models[(target, h)].predict(x[rows])
        return self._finalise(issue_index, values)


class RandomForestForecaster(_TabularForecaster):
    name = "rf"

    def __init__(self, horizon, pv_capacity_kw, random_state=0, n_estimators=120, max_depth=None):
        super().__init__(horizon, pv_capacity_kw, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _make_estimator(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=4, n_jobs=-1, random_state=self.random_state,
        )


class XGBoostForecaster(_TabularForecaster):
    """Selected in the paper to instantiate the module for the final EMS run."""

    name = "xgboost"

    def __init__(self, horizon, pv_capacity_kw, random_state=0,
                 n_estimators=400, learning_rate=0.05, max_depth=6):
        super().__init__(horizon, pv_capacity_kw, random_state)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def _make_estimator(self):
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, subsample=0.85, colsample_bytree=0.85,
            reg_lambda=1.0, tree_method="hist", n_jobs=-1,
            random_state=self.random_state, verbosity=0,
        )


class LSTMForecaster(Forecaster):
    """Sequence-to-sequence LSTM over a 24-hour input window.

    Reads the raw recent history of PV, load and weather, and emits the whole
    24-hour horizon in one pass. Scaling statistics come from the training rows
    only.
    """

    name = "lstm"

    # The paper reports the LSTM taking "minutes versus seconds" to train and
    # achieving marginally the best PV RMSE. A 12-epoch, 64-unit network trains
    # in ~80 s and lands well short of the tree ensembles; the capacity and the
    # schedule below are what it takes to make the three-way comparison fair.
    def __init__(self, horizon, pv_capacity_kw, random_state=0,
                 lookback=24, hidden=96, layers=2, epochs=30, batch_size=256, lr=2e-3):
        super().__init__(horizon, pv_capacity_kw)
        self.random_state = random_state
        self.lookback = lookback
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._models: dict[str, object] = {}
        self._stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._channels = ("pv_kw", "load_kw", "temp_c", "ghi_w_m2")

    # -- windowing ---------------------------------------------------------
    def _sequences(self, frame: pd.DataFrame, rows: np.ndarray, target: str, with_labels: bool):
        """Input windows ending at the issue time; labels are the next `horizon` hours."""
        raw = frame[list(self._channels)].to_numpy()
        y = frame[target].to_numpy()

        keep = rows[(rows >= self.lookback - 1) & (rows + self.horizon < len(frame))] if with_labels \
            else rows[rows >= self.lookback - 1]
        if len(keep) == 0:
            raise ValueError("no usable LSTM windows; series too short for the lookback")

        offsets = np.arange(-(self.lookback - 1), 1)
        xs = raw[keep[:, None] + offsets[None, :], :]           # (n, lookback, channels)
        if not with_labels:
            return keep, xs, None
        lead = np.arange(1, self.horizon + 1)
        ys = y[keep[:, None] + lead[None, :]]                    # (n, horizon)
        return keep, xs, ys

    def fit(self, frame, train_index):
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        train_rows = frame.index.get_indexer(train_index)

        for target in TARGETS:
            _, xs, ys = self._sequences(frame, train_rows, target, with_labels=True)

            # Scale with training statistics only. Computing them over the full
            # series would leak the test year into the fit.
            mu = xs.reshape(-1, xs.shape[-1]).mean(axis=0)
            sd = xs.reshape(-1, xs.shape[-1]).std(axis=0)
            sd[sd < 1e-8] = 1.0
            self._stats[target] = (mu, sd)

            y_mu, y_sd = ys.mean(), max(ys.std(), 1e-8)
            self._stats[target + "_y"] = (np.array([y_mu]), np.array([y_sd]))

            xt = torch.tensor((xs - mu) / sd, dtype=torch.float32)
            yt = torch.tensor((ys - y_mu) / y_sd, dtype=torch.float32)

            model = _LSTMNet(len(self._channels), self.hidden, self.layers, self.horizon)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
            loss_fn = nn.MSELoss()

            dataset = torch.utils.data.TensorDataset(xt, yt)
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            model.train()
            for _ in range(self.epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    opt.step()
                sched.step()
            model.eval()
            self._models[target] = model
        return self

    def predict(self, frame, issue_index) -> ForecastBundle:
        import torch

        rows = frame.index.get_indexer(issue_index)
        if (rows < self.lookback - 1).any():
            raise ValueError("issue times precede the LSTM lookback window")

        values = {}
        for target in TARGETS:
            _, xs, _ = self._sequences(frame, rows, target, with_labels=False)
            mu, sd = self._stats[target]
            y_mu, y_sd = self._stats[target + "_y"]
            xt = torch.tensor((xs - mu) / sd, dtype=torch.float32)
            with torch.no_grad():
                pred = self._models[target](xt).numpy()
            values[target] = pred * y_sd[0] + y_mu[0]
        return self._finalise(issue_index, values)


class _LSTMNet:
    """Constructed lazily so importing this module never requires torch."""

    def __new__(cls, n_channels: int, hidden: int, layers: int, horizon: int):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_channels, hidden, num_layers=layers, batch_first=True)
                self.head = nn.Linear(hidden, horizon)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        return Net()


def build_forecaster(kind: str, horizon: int, pv_capacity_kw: float, random_state: int = 0) -> Forecaster:
    kinds = {
        "perfect": PerfectForecaster,
        "persistence": PersistenceForecaster,
        "rf": RandomForestForecaster,
        "xgboost": XGBoostForecaster,
        "lstm": LSTMForecaster,
    }
    if kind not in kinds:
        raise ValueError(f"unknown forecaster {kind!r}; choose from {sorted(kinds)}")
    cls = kinds[kind]
    if kind in ("perfect", "persistence"):
        return cls(horizon, pv_capacity_kw)
    return cls(horizon, pv_capacity_kw, random_state=random_state)
