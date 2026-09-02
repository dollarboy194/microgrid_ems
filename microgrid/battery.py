"""Battery energy storage model.

Sign convention: **positive power = discharge** (battery supplies the microgrid),
negative = charge.

130 kWh nominal with a 90% maximum depth of discharge gives 117 kWh usable, so
the state of charge is floored at 10%. Round-trip losses are split between the
charge and discharge legs.

`step` returns the power the battery *actually* delivered, which may be smaller
in magnitude than the requested setpoint once a rate or energy limit binds. That
gap is where a dispatch plan built on an imperfect forecast meets reality, and
it must not be hidden -- the diesel generator or unmet load absorbs the
difference.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import BatteryConfig


@dataclass
class BatteryState:
    soc: float
    throughput_kwh: float = 0.0  # cumulative energy in + out, measured at the terminals


class Battery:
    def __init__(self, cfg: BatteryConfig) -> None:
        self.cfg = cfg
        self.state = BatteryState(soc=cfg.soc_init)

    def reset(self, soc: float | None = None) -> None:
        self.state = BatteryState(soc=self.cfg.soc_init if soc is None else soc)

    @property
    def soc(self) -> float:
        return self.state.soc

    @property
    def energy_kwh(self) -> float:
        return self.state.soc * self.cfg.nominal_kwh

    def headroom_kwh(self) -> float:
        """Energy that can still be stored before hitting soc_max."""
        return (self.cfg.soc_max - self.state.soc) * self.cfg.nominal_kwh

    def available_kwh(self) -> float:
        """Energy that can still be withdrawn before hitting soc_min."""
        return (self.state.soc - self.cfg.soc_min) * self.cfg.nominal_kwh

    def feasible_power(self, power_kw: float, dt_hours: float = 1.0) -> float:
        """Clip a requested setpoint to what rate and energy limits allow."""
        cfg = self.cfg
        if power_kw >= 0.0:
            p = min(power_kw, cfg.p_discharge_max_kw)
            # Delivering p kW at the terminals drains p / eta_d from the pack.
            limit = self.available_kwh() * cfg.eta_discharge / dt_hours
            return max(min(p, limit), 0.0)

        p = max(power_kw, -cfg.p_charge_max_kw)
        # Absorbing |p| kW at the terminals stores |p| * eta_c in the pack.
        limit = self.headroom_kwh() / (cfg.eta_charge * dt_hours)
        return -max(min(-p, limit), 0.0)

    def step(self, power_kw: float, dt_hours: float = 1.0) -> float:
        """Apply a setpoint for one interval; return the realised power (kW)."""
        cfg = self.cfg
        p = self.feasible_power(power_kw, dt_hours)

        if p >= 0.0:
            delta = -p * dt_hours / cfg.eta_discharge
        else:
            delta = -p * dt_hours * cfg.eta_charge

        soc = (self.energy_kwh + delta) / cfg.nominal_kwh
        self.state.soc = min(max(soc, cfg.soc_min), cfg.soc_max)
        self.state.throughput_kwh += abs(p) * dt_hours
        return p

    def equivalent_full_cycles(self) -> float:
        """Throughput in equivalent full cycles, on the *usable* capacity.

        The paper reports ~339 cycles/yr against a 117 kWh usable pack; scoring
        against the 130 kWh nominal would understate cycling by 10%.
        """
        return self.state.throughput_kwh / (2.0 * self.cfg.usable_kwh)
