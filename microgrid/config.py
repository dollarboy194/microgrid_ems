"""Configuration for the Tamale solar-battery-diesel microgrid.

Values are taken from the paper wherever it states them. Where it does not,
the parameter carries an explicit `# ASSUMPTION` note: those are choices this
implementation had to make, and they are the first things to check if a
reproduced number drifts from the published one.

Units: power kW, energy kWh, irradiance W/m^2, money GHS (Ghana cedi).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# The paper reports diesel energy of 21,440 kWh/yr costing GHS 113,205 for the
# baseline, and 24,813 kWh/yr costing GHS 131,011 for the framework. Both give
# 113205/21440 = 131011/24813 = 5.28 GHS/kWh, so the marginal diesel cost is
# recovered exactly rather than guessed.
DIESEL_COST_PER_KWH_GHS = 5.28


@dataclass(frozen=True)
class SiteConfig:
    """Case-study location: a rural community near Tamale, Northern Ghana."""

    name: str = "Tamale, Northern Region, Ghana"
    latitude: float = 9.4008
    longitude: float = -0.8393
    utc_offset_hours: float = 0.0  # Ghana observes UTC year-round


@dataclass(frozen=True)
class PVConfig:
    """Temperature-corrected PV model with a NOCT cell-temperature estimate.

    The paper specifies 90 kWp, a -0.4%/degC power temperature coefficient, an
    86% system derate (inverter, wiring, soiling, mismatch), and a NOCT-based
    cell temperature. It reports a resulting specific yield of 1,486
    kWh/kWp/yr, which `tests/test_pv_and_load.py` checks.
    """

    capacity_kwp: float = 90.0
    temp_coeff_per_c: float = -0.004
    derate: float = 0.86
    noct_c: float = 45.0            # ASSUMPTION: standard module NOCT; paper says "NOCT-based"
    stc_irradiance_w_m2: float = 1000.0
    stc_temp_c: float = 25.0


@dataclass(frozen=True)
class BatteryConfig:
    """130 kWh nominal, 117 kWh usable at 90% maximum depth of discharge."""

    nominal_kwh: float = 130.0
    max_depth_of_discharge: float = 0.90

    # ASSUMPTION: the paper gives no C-rate. 0.5C on the nominal capacity is
    # typical for rural mini-grid lithium installations and never binds before
    # the energy limit in this system.
    p_charge_max_kw: float = 65.0
    p_discharge_max_kw: float = 65.0

    # ASSUMPTION: not stated. 95%/95% gives a ~90% round trip, the usual figure
    # quoted for these installations.
    eta_charge: float = 0.95
    eta_discharge: float = 0.95

    soc_init: float = 0.50

    @property
    def soc_min(self) -> float:
        """A 90% maximum depth of discharge floors the state of charge at 10%."""
        return 1.0 - self.max_depth_of_discharge

    @property
    def soc_max(self) -> float:
        return 1.0

    @property
    def usable_kwh(self) -> float:
        return self.nominal_kwh * self.max_depth_of_discharge

    def __post_init__(self) -> None:
        if not 0.0 < self.max_depth_of_discharge <= 1.0:
            raise ValueError("max_depth_of_discharge must lie in (0, 1]")
        if not self.soc_min <= self.soc_init <= self.soc_max:
            raise ValueError("soc_init must lie inside the operating window")


@dataclass(frozen=True)
class DieselConfig:
    """40 kW last-resort backup generator."""

    rated_kw: float = 40.0
    cost_per_kwh_ghs: float = DIESEL_COST_PER_KWH_GHS

    # ASSUMPTION: no minimum loading is imposed. The paper models diesel cost as
    # proportional to energy, which a minimum-load constraint would break.
    min_load_fraction: float = 0.0


@dataclass(frozen=True)
class LoadConfig:
    """175 households: 149 basic residential + 26 productive-use customers.

    The per-customer annual figures are the paper's calibration targets and
    multiply out to exactly the reported community demand:
        149 * 500 + 26 * 1700 = 74,500 + 44,200 = 118,700 kWh/yr
    """

    n_basic: int = 149
    n_productive: int = 26
    kwh_per_basic_year: float = 500.0
    kwh_per_productive_year: float = 1700.0

    # ASSUMPTION. The paper's load profile is synthetic and not published, so
    # these are calibrated so that the profile's *forecastability* matches the
    # accuracy reported in its Table I (load RMSE ~1.90 kW, R^2 ~0.937 at one
    # hour ahead). That matters more than matching any single shape parameter:
    # the EMS results are driven by how much of the load is predictable, and a
    # profile that is too noisy would overstate the diesel premium.
    # See `scripts/calibrate_load.py` for the sweep behind these values.
    household_noise_cv: float = 0.07       # per-hour coefficient of variation
    spike_probability: float = 0.020       # per productive-use working hour
    spike_multiplier: float = 2.5
    weekend_productive_factor: float = 0.55  # markets/mills quiet on Sundays

    # Sharpens the hour-of-day shape (>1 deepens the night trough and lifts the
    # evening peak), raising the share of demand variance that is predictable.
    shape_sharpness: float = 1.30

    @property
    def annual_kwh(self) -> float:
        return self.n_basic * self.kwh_per_basic_year + self.n_productive * self.kwh_per_productive_year


@dataclass(frozen=True)
class GAConfig:
    """Genetic algorithm for dispatch over the forecast horizon.

    The paper specifies the seeding strategy (70% greedy merit-order plus
    Gaussian-perturbed variants, remainder random) and that the GA is
    NumPy-vectorized. Population size, generation count and the genetic
    operators are ASSUMPTIONS chosen for convergence on this problem size.
    """

    population: int = 120
    generations: int = 60
    seed_fraction: float = 0.70        # from the paper
    elite_fraction: float = 0.08
    tournament_size: int = 3
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    mutation_scale: float = 0.20       # of the relevant power limit
    seed_jitter: float = 0.10          # Gaussian perturbation of seeded genomes
    random_state: int = 20260709


@dataclass(frozen=True)
class EMSConfig:
    """Rolling-horizon dispatch: plan 24 h ahead, commit the next 6 h.

    `unmet_penalty_multiplier` is the paper's central methodological finding:
    a soft, cost-weighted unmet-load penalty never matches the baseline's
    reliability, so unmet load is priced at ~950x the diesel marginal cost,
    making reliability a near-hard constraint while leaving the optimiser free
    to minimise cost, cycling and curtailment among all reliable plans.
    """

    horizon_hours: int = 24
    commit_hours: int = 6
    unmet_penalty_multiplier: float = 950.0

    # Which setpoints are actually frozen for the commitment window.
    #
    # The paper reports 0.00% LPSP for the framework. That is only reachable if
    # the generator balances the residual deficit hour by hour: the peak deficit
    # in the test year is 30.7 kW against a 40 kW set, so a following generator
    # always covers it, whereas freezing the generator's output for six hours
    # would strand every under-predicted load spike as unserved energy.
    #
    # So the committed decision is the *battery trajectory*, and the generator
    # is the balancing unit -- standard practice for a diesel-backed island.
    # The GA still plans generator output, because the unmet-load penalty is
    # what shapes the battery plan. Set this False to freeze both, which
    # reproduces the reliability collapse that motivates the hard constraint.
    diesel_follows_deficit: bool = True

    # ASSUMPTION: the paper names these objective terms but not their weights.
    # Both are small enough to break ties without competing with diesel cost.
    curtailment_penalty_per_kwh: float = 0.05
    throughput_penalty_per_kwh: float = 0.02

    # Forecast safety margin: inflate load / deflate PV before planning. The
    # validation sweep in the paper varies this from 0% to 25%; the final
    # configuration relies on the hard constraint instead.
    forecast_margin: float = 0.0

    @property
    def unmet_penalty_per_kwh(self) -> float:
        return self.unmet_penalty_multiplier * DIESEL_COST_PER_KWH_GHS

    def __post_init__(self) -> None:
        if self.commit_hours > self.horizon_hours:
            raise ValueError("commit_hours cannot exceed horizon_hours")
        if self.horizon_hours % self.commit_hours:
            raise ValueError("horizon_hours should be a whole number of commit blocks")


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split. The test year is touched exactly once."""

    train_years: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023)
    validation_years: tuple[int, ...] = (2024,)
    test_years: tuple[int, ...] = (2025,)
    seed: int = 20260709


@dataclass(frozen=True)
class ExperimentConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    pv: PVConfig = field(default_factory=PVConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    diesel: DieselConfig = field(default_factory=DieselConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    ems: EMSConfig = field(default_factory=EMSConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
