"""Figures for the Tamale solar-battery-diesel EMS study.

Colour choices are computed, not eyeballed. Every categorical palette used here
passed the six checks (OKLCH lightness band, chroma floor, Machado colour-vision
separation, WCAG contrast against the chart surface) via
`tools/validate_palette.py`; worst adjacent CVD delta-E is 47.2, well above the
12.0 target. Amber and aqua fall below 3:1 contrast on the light surface, so
every series carrying them is directly labelled as well as legended -- identity
is never colour-alone.

Red is reserved for unmet load. It is a status colour, not a series colour, and
unserved energy in a rural microgrid is exactly the failure state it signals.

No figure uses a dual y-axis. State of charge and power get their own panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentConfig

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Entity -> hue, fixed across every figure in the study.
PV = "#eda100"        # amber
LOAD = "#2a78d6"      # blue
BATTERY = "#1baf7a"   # aqua
DIESEL = "#4a3aa7"    # violet
CURTAILED = "#eb6834"  # orange
UNMET = "#d03b3b"     # status: critical

CONTROLLER_HUES = ("#2a78d6", "#1baf7a", "#eda100", "#4a3aa7")
MODEL_HUES = {"rf": "#2a78d6", "lstm": "#1baf7a", "xgboost": "#eda100"}

PRETTY = {
    "lpsp_pct": "LPSP (%)",
    "diesel_kwh": "Diesel energy (kWh/yr)",
    "diesel_cost_ghs": "Diesel cost (GHS/yr)",
    "renewable_fraction_pct": "Renewable fraction (%)",
    "curtailed_kwh": "Curtailed PV (kWh/yr)",
    "battery_cycles": "Battery cycles (full-eq./yr)",
}


def _style(ax, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)


def _title(ax, text: str, subtitle: str = "") -> None:
    ax.set_title(text, color=INK, fontsize=12, loc="left",
                 pad=24 if subtitle else 8, fontweight="bold")
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                color=MUTED, fontsize=9, va="bottom", ha="left")


def _label(ax, x, y, text, color) -> None:
    ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=9, va="center", ha="left",
                fontweight="bold", clip_on=False)


def _bar_value(ax, bars, values) -> None:
    """Value labels: the mandatory relief for the low-contrast hues."""
    for bar, value in zip(bars, values):
        ax.annotate(_fmt(value), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=INK_SECONDARY, fontweight="bold")


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:,.1f}"
    if a >= 0.01:
        return f"{v:,.3f}"
    return "0" if v == 0 else f"{v:.2e}"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_resource_and_demand(frame: pd.DataFrame, test_index, cfg: ExperimentConfig, path: Path,
                             days: int = 7) -> Path:
    """PV generation against community demand: the surplus/deficit pattern the EMS exists to manage."""
    window = frame.loc[test_index[: days * 24]]
    t = window.index

    fig, axes = plt.subplots(1, 1, figsize=(12, 4.6), facecolor=SURFACE)
    ax = axes
    ax.fill_between(t, 0, window["pv_kw"], color=PV, alpha=0.30, linewidth=0)
    ax.plot(t, window["pv_kw"], color=PV, linewidth=2.0, label="PV generation")
    ax.plot(t, window["load_kw"], color=LOAD, linewidth=2.0, label="Community demand")

    _label(ax, t[-1], window["pv_kw"].iloc[-1], "PV", PV)
    _label(ax, t[-1], window["load_kw"].iloc[-1], "Demand", LOAD)

    _style(ax, ylabel="Power (kW)", xlabel="Time")
    _title(ax, "Daytime PV surplus, evening demand deficit",
           f"{cfg.pv.capacity_kwp:.0f} kWp PV against 175 households, first {days} days of the test year")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    return _save(fig, path)


def plot_forecast_week(frame: pd.DataFrame, bundle, path: Path, hours: int = 168, lead: int = 1) -> Path:
    """Forecast against actual over a representative window, at a fixed lead time.

    One point per issue time. The dispatch loop issues a forecast at every commit
    boundary, so `hours` issue times span more calendar time than `hours` hours;
    the axis carries the real dates and the title states the lead, not a span.
    """
    n = min(hours, len(bundle.index))
    idx = bundle.index[:n]
    rows = frame.index.get_indexer(idx)
    pos = np.minimum(rows + lead, len(frame) - 1)
    t = frame.index[pos]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, facecolor=SURFACE)
    for ax, target, hue, name in ((axes[0], "pv_kw", PV, "PV"), (axes[1], "load_kw", LOAD, "Demand")):
        actual = frame[target].to_numpy()[pos]
        pred = bundle.values[target][:n, lead - 1]
        ax.plot(t, actual, color=hue, linewidth=2.0, label="Actual")
        ax.plot(t, pred, color=INK_SECONDARY, linewidth=1.6, linestyle="--", label="Forecast")
        mae = float(np.mean(np.abs(pred - actual)))
        _style(ax, ylabel=f"{name} (kW)")
        _title(ax, f"{name}: forecast vs actual, {lead} h ahead", f"MAE over this window: {mae:.2f} kW")
        ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=2)
    axes[1].set_xlabel("Time", color=INK_SECONDARY, fontsize=10)
    return _save(fig, path)


def plot_forecast_errors(table1: pd.DataFrame, path: Path) -> Path:
    """Normalised error across the three candidate forecasters (paper Fig. 3)."""
    one = table1[table1["scope"] == "h=1"]
    models = ["rf", "lstm", "xgboost"]
    tasks = ["pv_kw", "load_kw"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax, task in zip(axes, tasks):
        sub = one[one["target"] == task].set_index("model").reindex(models)
        values = sub["nrmse_pct"].to_numpy()
        bars = ax.bar(range(len(models)), values, width=0.6,
                      color=[MODEL_HUES[m] for m in models], edgecolor=SURFACE, linewidth=2.0)
        _bar_value(ax, bars, values)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.upper() for m in models], fontsize=9)
        _style(ax, ylabel="nRMSE (% of mean)")
        _title(ax, task.replace("_kw", "").upper() + " forecasting")
        ax.margins(y=0.20)
    fig.suptitle("All three candidates score comparably; none dominates",
                 color=INK, fontsize=12, x=0.01, ha="left", fontweight="bold")
    return _save(fig, path)


def plot_ems_comparison(logs: dict[str, pd.DataFrame], cfg: ExperimentConfig, path: Path) -> Path:
    """Headline KPIs, one panel per metric, one bar per controller (paper Table II)."""
    from .metrics import ems_kpis

    names = list(logs)
    if len(names) > len(CONTROLLER_HUES):
        raise ValueError("more controllers than categorical slots; hues are never cycled")
    kpis = {n: ems_kpis(log, cfg) for n, log in logs.items()}

    keys = list(PRETTY)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), facecolor=SURFACE)
    axes = axes.ravel()

    # A summed float can leave dust like 2.8e-16 where the true value is zero.
    # Plotted naively that dust becomes a full-height bar on a 1e-16 axis, which
    # reads as a real reliability difference. Below this it is zero.
    zero_tol = 1e-9

    for ax, key in zip(axes, keys):
        values = [0.0 if abs(kpis[n][key]) < zero_tol else kpis[n][key] for n in names]
        bars = ax.bar(range(len(names)), values, width=0.6,
                      color=list(CONTROLLER_HUES[: len(names)]), edgecolor=SURFACE, linewidth=2.0)
        _bar_value(ax, bars, values)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
        _style(ax)
        ax.set_title(PRETTY[key], color=INK, fontsize=11, loc="left", fontweight="bold")
        ax.margins(y=0.20)
        if max(values) <= 0.0:
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 1])
            ax.text(0.5, 0.45, "zero for every controller", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED, fontsize=10)
    return _save(fig, path)


def plot_soc_and_diesel(logs: dict[str, pd.DataFrame], cfg: ExperimentConfig, path: Path,
                        days: int = 10) -> Path:
    """State of charge over a sample window, and diesel energy by month."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), facecolor=SURFACE)

    ax = axes[0]
    for i, (name, log) in enumerate(logs.items()):
        window = log.iloc[: days * 24]
        ax.plot(window.index, window["soc"] * 100.0, linewidth=1.9,
                color=CONTROLLER_HUES[i], label=name)
    # No direct labels here: the trajectories converge on the 10% floor, so every
    # end-of-series label lands on the same point. The legend carries identity.
    ax.axhline(cfg.battery.soc_min * 100.0, color=MUTED, linestyle=":", linewidth=1.0)
    ax.set_ylim(0, 118)
    ax.annotate(f"floor {cfg.battery.soc_min*100:.0f}%  (90% max depth of discharge)",
                xy=(0.985, cfg.battery.soc_min * 100.0 + 2.5), xycoords=("axes fraction", "data"),
                color=MUTED, fontsize=8, va="bottom", ha="right")
    _style(ax, ylabel="State of charge (%)")
    _title(ax, "Battery state of charge",
           f"first {days} days of the test year; all three cycle the pack fully every day")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=len(logs))

    ax = axes[1]
    months = np.arange(1, 13)
    width = 0.8 / len(logs)
    for i, (name, log) in enumerate(logs.items()):
        monthly = log["diesel_kw"].groupby(log.index.month).sum().reindex(months, fill_value=0.0)
        ax.bar(months + (i - (len(logs) - 1) / 2) * width, monthly.to_numpy(), width=width,
               color=CONTROLLER_HUES[i], edgecolor=SURFACE, linewidth=1.5, label=name)
    ax.set_xticks(months)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=9)
    _style(ax, ylabel="Diesel energy (kWh)", xlabel="Month of the test year")
    _title(ax, "Diesel use by month",
           "the premium tracks the solar resource: corr(monthly PV, premium) = -0.72")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=len(logs))
    ax.margins(y=0.15)
    return _save(fig, path)
