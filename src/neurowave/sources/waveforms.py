"""
Electromagnetic excitation sources for FDTD simulation.

Source Types: GaussianSource, SinusoidalSource, ModulatedGaussianSource.
All sources use soft injection (additive) to avoid artificial reflections.

References
----------
.. [1] A. Taflove, "Computational Electrodynamics," 3rd ed., Artech House, 2005, Ch5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from neurowave.core.base import BaseSource
from neurowave.core.constants import TWO_PI


@dataclass
class GaussianSource(BaseSource):
    """Gaussian pulse: J(t) = A·exp(−((t−t₀)/τ)²). Broadband, DC to f_max≈1/(πτ)."""

    x: int
    y: int
    frequency_max: float
    amplitude: float = 1.0
    delay_factor: float = 6.0
    field_component: str = "Ez"
    z: int = 0

    def __post_init__(self) -> None:
        self._tau: float = 1.0 / (math.pi * self.frequency_max)
        self._t0: float = self.delay_factor * self._tau

    def value_at(self, timestep: int, dt: float) -> float:
        t = timestep * dt
        return self.amplitude * math.exp(-((t - self._t0) / self._tau) ** 2)

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def component(self) -> str:
        return self.field_component

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def t0(self) -> float:
        return self._t0

    def waveform(self, num_steps: int, dt: float) -> np.ndarray:
        t = np.arange(num_steps) * dt
        return self.amplitude * np.exp(-((t - self._t0) / self._tau) ** 2)


@dataclass
class SinusoidalSource(BaseSource):
    """CW source: J(t) = A·sin(2πf₀t)·ramp(t). Smooth ramp avoids transients."""

    x: int
    y: int
    frequency: float
    amplitude: float = 1.0
    field_component: str = "Ez"
    z: int = 0

    def value_at(self, timestep: int, dt: float) -> float:
        t = timestep * dt
        period = 1.0 / self.frequency
        ramp = 1.0 - math.exp(-((t / (2.0 * period)) ** 2))
        return self.amplitude * math.sin(TWO_PI * self.frequency * t) * ramp

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def component(self) -> str:
        return self.field_component

    def waveform(self, num_steps: int, dt: float) -> np.ndarray:
        t = np.arange(num_steps) * dt
        period = 1.0 / self.frequency
        ramp = 1.0 - np.exp(-((t / (2.0 * period)) ** 2))
        return self.amplitude * np.sin(TWO_PI * self.frequency * t) * ramp


@dataclass
class ModulatedGaussianSource(BaseSource):
    """Band-limited pulse: J(t) = A·exp(−((t−t₀)/τ)²)·sin(2πf₀t). Δf≈1/(πτ)."""

    x: int
    y: int
    frequency: float
    bandwidth: float
    amplitude: float = 1.0
    delay_factor: float = 6.0
    field_component: str = "Ez"
    z: int = 0

    def __post_init__(self) -> None:
        self._tau: float = 1.0 / (math.pi * self.bandwidth)
        self._t0: float = self.delay_factor * self._tau

    def value_at(self, timestep: int, dt: float) -> float:
        t = timestep * dt
        envelope = math.exp(-((t - self._t0) / self._tau) ** 2)
        carrier = math.sin(TWO_PI * self.frequency * t)
        return self.amplitude * envelope * carrier

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def component(self) -> str:
        return self.field_component

    def waveform(self, num_steps: int, dt: float) -> np.ndarray:
        t = np.arange(num_steps) * dt
        envelope = np.exp(-((t - self._t0) / self._tau) ** 2)
        carrier = np.sin(TWO_PI * self.frequency * t)
        return self.amplitude * envelope * carrier
