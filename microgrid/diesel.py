"""Diesel generator: the last-resort backup.

A 40 kW unit whose cost is proportional to the energy it produces, at the
marginal cost recovered from the paper's own figures (GHS 113,205 / 21,440 kWh
= GHS 131,011 / 24,813 kWh = 5.28 GHS/kWh exactly).

Modelling cost as strictly proportional to energy is the paper's implicit
assumption. A real generator also has a minimum stable loading and a no-load
fuel burn, which would make short, shallow diesel runs disproportionately
expensive and would change the optimiser's incentives; `min_load_fraction`
exists so that can be explored, and defaults to zero to stay faithful.
"""

from __future__ import annotations

import numpy as np

from .config import DieselConfig


class Diesel:
    def __init__(self, cfg: DieselConfig) -> None:
        self.cfg = cfg

    def feasible_power(self, power_kw: float) -> float:
        """Clip a request to the generator's rating and minimum stable loading."""
        if power_kw <= 0.0:
            return 0.0
        p = min(power_kw, self.cfg.rated_kw)
        floor = self.cfg.min_load_fraction * self.cfg.rated_kw
        if floor > 0.0 and p < floor:
            # Below minimum load the set either runs at the floor or not at all;
            # running it is only worth it if the demand is at least half the floor.
            return floor if p >= 0.5 * floor else 0.0
        return p

    def cost_ghs(self, energy_kwh: float | np.ndarray) -> float | np.ndarray:
        return energy_kwh * self.cfg.cost_per_kwh_ghs
