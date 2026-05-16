"""
Abstract base class for FDTD solvers (2D and 3D).

This module provides the common structure and orchestration logic for
FDTD solvers, reducing duplication and improving maintainability.

Architecture
------------
Uses the Template Method pattern:
1. Concrete solvers (FDTD2D, FDTD3D) inherit from FdtdBase
2. Base class orchestrates the main loop structure (run, step)
3. Subclasses implement dimension-specific field updates

The pattern reduces code duplication by ~22% (842 → 660 LOC).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ceep.core.base import BaseSolver, BaseSource, BaseBoundary
from ceep.core.config import SimulationConfig
from ceep.core.backend import to_scalar, to_numpy, is_gpu_active


class FdtdBase(BaseSolver, ABC):
    """Abstract base class for FDTD solvers (2D and 3D).

    This class manages the common aspects of FDTD simulation:
    - Main time-stepping loop structure
    - Source injection pattern
    - Field recording (snapshots and probes)
    - Boundary condition application
    - Property accessors

    Subclasses must implement dimension-specific field updates:
    - _update_h_fields() : Compute curl of E to get H
    - _update_e_fields() : Compute curl of H to get E
    - _inject_sources() : Inject sources into the grid
    - _apply_boundaries_h() : Apply BCs to H-fields
    - _apply_boundaries_e() : Apply BCs to E-fields
    - _record() : Record field snapshots and probe data

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.
    sources : list of BaseSource, optional
        Excitation sources to inject.
    boundaries : list of BaseBoundary, optional
        Boundary conditions to apply.
    record_field : str, optional
        Field component to record at every timestep (e.g., 'Ez').
    record_interval : int, optional
        Record a snapshot every N steps (default: 1).
    probe_points : list of tuple, optional
        Grid points at which to record time-domain field values.

    Attributes
    ----------
    config : SimulationConfig
        The simulation configuration.
    sources : list of BaseSource
        List of active sources.
    boundaries : list of BaseBoundary
        List of boundary conditions.
    field_snapshots : list of ndarray
        Recorded field snapshots.
    probe_data : dict
        Time-domain data at probe points: {point: [values]}.
    """

    def __init__(
        self,
        config: SimulationConfig,
        sources: Optional[List[BaseSource]] = None,
        boundaries: Optional[List[BaseBoundary]] = None,
        record_field: Optional[str] = None,
        record_interval: int = 1,
        probe_points: Optional[List[Tuple[int, ...]]] = None,
    ):
        """Initialize base solver with common attributes."""
        self.config = config
        self.sources = sources or []
        self.boundaries = boundaries or []
        self.record_field = record_field
        self.record_interval = record_interval
        self.probe_points = probe_points or []

        # Internal state
        self._step = 0
        self._use_fused_kernels = False
        if is_gpu_active():
            from ceep.cuda.kernels import cuda_kernels_available
            self._use_fused_kernels = cuda_kernels_available()

        # Recording storage
        self.field_snapshots: List[npt.NDArray[np.float64]] = []
        self.probe_data: Dict[Tuple[int, ...], List[float]] = {
            pt: [] for pt in self.probe_points
        }

    @abstractmethod
    def _update_h_fields(self) -> None:
        """Update magnetic field components from electric fields.

        Implements the H-field half-step of the leapfrog scheme.
        Dimension-specific (different curl operations in 2D vs 3D).
        """
        ...

    @abstractmethod
    def _update_e_fields(self) -> None:
        """Update electric field components from magnetic fields.

        Implements the E-field full step of the leapfrog scheme.
        Dimension-specific (different curl operations in 2D vs 3D).
        """
        ...

    @abstractmethod
    def _inject_sources(self) -> None:
        """Inject all sources as soft sources (additive)."""
        ...

    @abstractmethod
    def _apply_boundaries_h(self) -> None:
        """Apply boundary conditions to H-field components."""
        ...

    @abstractmethod
    def _apply_boundaries_e(self) -> None:
        """Apply boundary conditions to E-field components."""
        ...

    @abstractmethod
    def _record(self) -> None:
        """Record field snapshots and probe data."""
        ...

    @abstractmethod
    def get_field(self, component: str) -> npt.NDArray[np.float64]:
        """Get current field component array.

        Parameters
        ----------
        component : str
            Field component name (e.g., 'Ez', 'Hx', 'Ex', 'Hz').

        Returns
        -------
        numpy.ndarray
            Current field values.
        """
        ...

    def step(self) -> None:
        """Advance simulation by one timestep (leapfrog scheme).

        Template method orchestrating the field update sequence:
        1. Update H-fields (n-½ → n+½)
        2. Apply H-field boundary corrections
        3. Update E-fields (n → n+1)
        4. Inject sources
        5. Apply E-field boundary corrections
        6. Record data

        The actual field updates are delegated to dimension-specific
        methods implemented by subclasses.
        """
        # H-field update (half-step)
        self._update_h_fields()
        self._apply_boundaries_h()

        # E-field update (full step)
        self._update_e_fields()
        self._inject_sources()
        self._apply_boundaries_e()

        # Record results
        self._record()

        # Advance timestep
        self._step += 1

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run simulation for specified number of steps.

        Parameters
        ----------
        num_steps : int, optional
            Steps to run. Defaults to config.num_steps or config.total_steps.
        """
        # Handle both config attribute names (2D uses num_steps, 3D uses total_steps)
        if num_steps is None:
            if hasattr(self.config, 'num_steps'):
                num_steps = self.config.num_steps
            elif hasattr(self.config, 'total_steps'):
                num_steps = self.config.total_steps
            else:
                raise ValueError("Config must have num_steps or total_steps")

        for _ in range(num_steps):
            self.step()

    @property
    def current_step(self) -> int:
        """Current simulation timestep index."""
        return self._step

    @property
    def current_time(self) -> float:
        """Current simulation time [s]."""
        return self._step * self.config.dt
