"""
Abstract base classes for NeuroWave components.

This module defines the interfaces that all concrete implementations
must follow. This ensures modularity and consistent API across different
backends and implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ceep.core.config import GridConfig, SimulationConfig


class BaseSource(ABC):
    """Abstract base class for electromagnetic excitation sources.

    All sources must implement the `value_at` method to provide the
    temporal waveform at a given timestep.
    """

    @abstractmethod
    def value_at(self, timestep: int, dt: float) -> float:
        """Calculate source value at a given timestep.

        Parameters
        ----------
        timestep : int
            Current simulation timestep index.
        dt : float
            Simulation timestep size [s].

        Returns
        -------
        float
            Source amplitude at the given time.
        """
        ...

    @property
    @abstractmethod
    def position(self) -> Tuple[int, ...]:
        """Grid position of the source (i, j) or (i, j, k)."""
        ...

    @property
    @abstractmethod
    def component(self) -> str:
        """Field component this source excites (e.g., 'Ez', 'Hx')."""
        ...


class BaseBoundary(ABC):
    """Abstract base class for boundary conditions.

    Boundary conditions are applied after each field update to absorb
    or reflect outgoing waves at the grid edges.
    """

    @abstractmethod
    def apply_e_field(
        self,
        e_fields: dict[str, npt.NDArray[np.float64]],
        h_fields: dict[str, npt.NDArray[np.float64]],
        config: SimulationConfig,
    ) -> None:
        """Apply boundary condition to E-field components.

        Parameters
        ----------
        e_fields : dict
            Dictionary of E-field component arrays.
        h_fields : dict
            Dictionary of H-field component arrays.
        config : SimulationConfig
            Simulation configuration.
        """
        ...

    @abstractmethod
    def apply_h_field(
        self,
        e_fields: dict[str, npt.NDArray[np.float64]],
        h_fields: dict[str, npt.NDArray[np.float64]],
        config: SimulationConfig,
    ) -> None:
        """Apply boundary condition to H-field components.

        Parameters
        ----------
        e_fields : dict
            Dictionary of E-field component arrays.
        h_fields : dict
            Dictionary of H-field component arrays.
        config : SimulationConfig
            Simulation configuration.
        """
        ...


class BaseMaterial(ABC):
    """Abstract base class for material definitions.

    Materials define the spatially-varying electromagnetic properties
    (permittivity, permeability, conductivity) on the simulation grid.
    """

    @abstractmethod
    def get_permittivity(self, grid: GridConfig) -> npt.NDArray[np.float64]:
        """Return relative permittivity array on the grid.

        Parameters
        ----------
        grid : GridConfig
            Grid configuration defining array shape.

        Returns
        -------
        numpy.ndarray
            Relative permittivity εᵣ at each grid cell.
        """
        ...

    @abstractmethod
    def get_permeability(self, grid: GridConfig) -> npt.NDArray[np.float64]:
        """Return relative permeability array on the grid.

        Parameters
        ----------
        grid : GridConfig
            Grid configuration defining array shape.

        Returns
        -------
        numpy.ndarray
            Relative permeability μᵣ at each grid cell.
        """
        ...

    @abstractmethod
    def get_conductivity(self, grid: GridConfig) -> npt.NDArray[np.float64]:
        """Return electric conductivity array on the grid.

        Parameters
        ----------
        grid : GridConfig
            Grid configuration defining array shape.

        Returns
        -------
        numpy.ndarray
            Electric conductivity σ [S/m] at each grid cell.
        """
        ...


class BaseSolver(ABC):
    """Abstract base class for FDTD solvers.

    Solvers orchestrate the main simulation loop, managing field updates,
    source injection, and boundary condition application.
    """

    @abstractmethod
    def initialize(self, config: SimulationConfig) -> None:
        """Initialize solver with the given configuration.

        Allocates field arrays, sets up material coefficients, and
        prepares the solver for time-stepping.

        Parameters
        ----------
        config : SimulationConfig
            Complete simulation configuration.
        """
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one timestep.

        Performs the leapfrog update:
        1. Update H-fields from E-fields
        2. Apply H-field boundary conditions
        3. Update E-fields from H-fields
        4. Inject sources
        5. Apply E-field boundary conditions
        """
        ...

    @abstractmethod
    def run(self, num_steps: Optional[int] = None) -> None:
        """Run the simulation for the specified number of steps.

        Parameters
        ----------
        num_steps : int, optional
            Number of steps to run. If None, uses config.num_steps.
        """
        ...

    @abstractmethod
    def get_field(self, component: str) -> npt.NDArray[np.float64]:
        """Get current field component array.

        Parameters
        ----------
        component : str
            Field component name ('Ez', 'Hx', 'Hy', etc.).

        Returns
        -------
        numpy.ndarray
            Current field values.
        """
        ...

    @property
    @abstractmethod
    def current_step(self) -> int:
        """Current simulation timestep index."""
        ...

    @property
    @abstractmethod
    def current_time(self) -> float:
        """Current simulation time [s]."""
        ...
