# Forecast-Driven EMS for a Rural Solar–Battery–Diesel Microgrid

A reproduction and extension of *"A Forecast-Driven Machine Learning-Based Energy
Management Framework for Solar-Battery-Diesel Microgrids in Rural Ghana"* —
Tamale, Northern Region (9.4008 °N, 0.8393 °W).

Short-term ML forecasts of PV generation and community demand feed a genetic-algorithm
dispatch optimiser over a rolling 24-hour horizon, re-solved every 6 hours. The
committed decisions are then **replayed against ground-truth PV and load**, so forecast
error shows up honestly as diesel burn, curtailment, or unserved energy rather than
being hidden inside a self-consistent evaluation.

## What this reproduces

| Quantity | Paper | This implementation |
|---|---|---|
| NASA POWER records, 2018–2025 | 70,128, zero missing | **70,128, zero missing** |
| Mean daily solar resource | 5.28 kWh/m²/day | **5.28** |
| PV specific yield | 1,486 kWh/kWp/yr | **1,487** |
| Community annual demand | 118,700 kWh/yr | **118,700** |
| XGBoost PV forecast (1 h ahead) | MAE 0.82, RMSE 1.91, R² 0.991 | **0.80 / 1.92 / 0.991** |
| LPSP, baseline and framework | 0.00% / 0.00% | **0.00% / 0.00%** |

The solar and weather data are the real NASA POWER record, so the irradiance, PV
and forecast-accuracy figures reproduce independently. The load profile is
synthetic in the paper and unpublished, so it is reconstructed and calibrated
(see *Calibration*, below).

## Install and run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt

python run_experiment.py                 # full reproduction: Tables I and II
python run_experiment.py --skip-table1 --test-days 45   # ~4 min smoke run
python -m pytest -q                      # test suite
```

The first run downloads eight years of hourly NASA POWER data (~3 MB) into `data/`
and caches it per year. Everything after that is offline.

Ablations the paper names as future work:

```bash
python scripts/ablations.py commit    # 1/2/3/6/12/24 h commitment windows
python scripts/ablations.py penalty   # unmet-load weight: soft vs hard
python scripts/ablations.py margin    # forecast safety margin, 0-25%
```

## The system

* **175 households** — 149 basic residential (~500 kWh/yr each, bimodal with a
  dominant evening peak) and 26 productive-use customers (~1,700 kWh/yr,
  daytime-dominant: shops, grain mills, welding, cold storage).
* **90 kWp PV**, NOCT cell-temperature model, −0.4%/°C, 86% system derate.
* **130 kWh battery**, 117 kWh usable at 90% maximum depth of discharge.
* **40 kW diesel** generator, last resort, GHS 5.28/kWh.

Islanded. There is no grid connection, so reliability is measured as Loss of Power
Supply Probability (LPSP) and the generator is the only backstop.

## Design decisions worth knowing

These are the points where the paper is silent or ambiguous and the implementation
had to choose. Each is enforced by a test, and each is the first place to look if a
number drifts.

**Merit order is enforced by physics, not by a repair operator.** The generator is
clamped to the *genuine* deficit remaining after PV and battery discharge, inside
`dispatch.simulate`. It therefore cannot run when supply already covers demand, and
cannot over-dispatch. Because the same function evaluates GA fitness *and* replays
against ground truth, an optimiser can never win fitness by exploiting physics the
evaluator will not grant it. Computing the deficit *after* PV is applied is what
prevents the bug the paper reports finding — a repair operator that discharged the
battery during PV surplus.

**Load has priority over charging.** A committed charge setpoint is planned on a
forecast. If the sky turns out cloudier, that charging adds to the deficit and can
push it past the 40 kW generator rating — shedding load in order to fill a battery.
No real installation does that; the charger backs off first. Without this rule the
framework strands ~6 kWh/yr and cannot reach the zero LPSP the paper reports.

**The generator is the balancing unit; the battery trajectory is what gets
committed.** The paper reports 0.00% LPSP. That is only reachable if the generator
follows the residual deficit hour by hour: peak deficit in the test year is 30.7 kW
against a 40 kW set, so a following generator always covers it, whereas freezing its
output for six hours would strand every under-predicted load spike. Set
`EMSConfig.diesel_follows_deficit = False` to freeze both and watch reliability
collapse — which is exactly what motivates the hard constraint.

**Table I is one-step-ahead.** The paper does not say whether its forecast errors
are one-step or averaged across the 24-hour dispatch horizon. At h=1 the PV numbers
land on the published values almost exactly (MAE 0.80 vs 0.82, RMSE 1.92 vs 1.91,
R² 0.991); averaged over the horizon they do not (RMSE 4.11). `metrics.forecast_errors`
reports both and `run_experiment.py` prints them side by side.

**Features cannot see the future, and this is verified rather than asserted.**
`tests/test_leakage.py` implements the paper's five verification tests — no EMS
outputs among the features, bit-exact lags, rolling windows that exclude the
predicted hour, deterministic calendar features, correct target alignment — plus a
stronger sixth: overwrite every observation after the issue time with garbage and
demand that no feature moves. A further test injects a deliberate leak and confirms
the guard catches it, so the check cannot pass vacuously.

**The diesel marginal cost is recovered, not guessed.** GHS 113,205 / 21,440 kWh =
GHS 131,011 / 24,813 kWh = **5.28 GHS/kWh** exactly, from the paper's own Table II.

## Calibration

The paper's load profile is synthetic and not published. What *is* published is how
predictable it is: one-hour-ahead RMSE ≈ 1.90 kW and R² ≈ 0.937 for XGBoost on the
held-out test year. Since the EMS results depend directly on how much of the demand
a forecaster can anticipate, the generator's stochastic parameters are tuned until
its **forecastability** matches — rather than tuned to taste, which would report an
EMS gap that is really an artefact of the noise model.

`scripts/calibrate_load.py` runs that sweep. The chosen setting gives RMSE 1.879 and
R² 0.941, with an implied load standard deviation of 7.71 kW against the ~7.57 kW
the paper's own numbers imply.

Everything marked `# ASSUMPTION` in `config.py` is a value the paper does not state:
battery C-rate and round-trip efficiency, module NOCT, GA population and generation
count, and the curtailment and throughput penalty weights.

## Controls that make the comparison interpretable

Two runs exist purely to keep the headline honest:

* `ga_perfect` — the same GA loop handed the **true future**. It must match the
  reactive baseline. It does, to 0.000% on diesel energy. Any gap would mean the
  commitment loop is broken, not that forecasting is hard.
* `baseline` — reactive, hourly, perfect-information rule-based control. Reliable by
  brute force, because it dispatches the generator for any deficit regardless of cost.

The diesel premium is therefore attributable to the *information asymmetry* — the
framework commits six hours ahead on an imperfect forecast — and not to the
optimiser, the physics, or the evaluation.

## Layout

```
microgrid/
  config.py       every parameter, with ASSUMPTION notes where the paper is silent
  nasa_power.py   NASA POWER retrieval + per-year disk cache
  pvmodel.py      NOCT cell temperature, -0.4%/C, 86% derate
  load.py         175-household synthetic profile, calibrated
  dataset.py      hybrid dataset + chronological split (train 2018-23, val 24, test 25)
  features.py     leakage-safe design matrices
  forecast.py     Random Forest, XGBoost, LSTM, plus persistence and perfect-foresight
  battery.py      energy reservoir, 90% max depth of discharge
  diesel.py       last-resort generator
  dispatch.py     vectorised physics: one simulator for GA fitness and ground-truth replay
  ga.py           NumPy-vectorised GA, merit-order seeding
  ems.py          rolling forecast-optimize-evaluate loop, and the rule-based baseline
  metrics.py      Table I forecast errors, Table II EMS KPIs
  plots.py        figures
run_experiment.py  end-to-end reproduction
scripts/
  calibrate_load.py  load-generator calibration sweep
  ablations.py       the sensitivity studies the paper leaves open
tests/               leakage, physics, GA, PV/load, EMS loop, metrics
tools/
  validate_palette.py  computes the figure palette's colour-vision safety
```

## Extending it

* **Real metered load.** Replace `load.generate_load`; keep the column names. The
  paper names this as its first limitation.
* **A different forecaster.** Subclass `Forecaster`, implement `fit`/`predict`,
  register it in `build_forecaster`. The direct multi-horizon contract is what
  matters, not the model class.
* **A different optimiser.** `ga.optimise` returns a plan; anything with the same
  signature (an LP, a rolling MILP, an RL policy) drops into `ems.run_ga_ems`. This
  is the cleanest way to test whether the GA leaves cost on the table.
* **Minimum generator loading.** `DieselConfig.min_load_fraction` exists and defaults
  to zero, because the paper models diesel cost as strictly proportional to energy.
  A real set has a no-load fuel burn, which would penalise short shallow runs and
  change the optimiser's incentives.

## Caveats

The load profile is synthetic. Its *structure* is literature-calibrated and its
*forecastability* is matched to the paper, but it is not the paper's actual series,
so the absolute diesel figures differ even where the relative changes agree.

The generator model has no minimum stable loading and no start-up cost, following
the paper's implicit assumption of cost proportional to energy.

The 40× diagnostic in the paper — a random-initialised GA scoring far worse than the
merit-order heuristic on the GA's own objective — reproduces qualitatively (≈6.7× on
a typical day) but not in magnitude, because here the merit order is enforced inside
the physics rather than by a separate repair operator. A random genome is therefore
projected onto a feasible, sane dispatch before it is ever scored, which is a
stronger guarantee than a repair pass and makes catastrophic genomes impossible.
