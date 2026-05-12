"""
2D FDTD Solver — TMz and TEz modes.

Implements the complete FDTD time-stepping loop with:
- Leapfrog E/H field updates
- Soft source injection
- Boundary condition application
- Field recording and snapshots

TMz Update Equations (Ez, Hx, Hy)
----------------------------------
H-field update (half-step ahead of E):
    Hx^{n+½}(i,j) = Da·Hx^{n-½}(i,j) - Db/Δy · [Ez^n(i,j+1) - Ez^n(i,j)]
    Hy^{n+½}(i,j) = Da·Hy^{n-½}(i,j) + Db/Δx · [Ez^n(i+1,j) - Ez^n(i,j)]

E-field update:
    Ez^{n+1}(i,j) = Ca·Ez^n(i,j) + Cb · [
        (Hy^{n+½}(i,j) - Hy^{n+½}(i-1,j))/Δx
      - (Hx^{n+½}(i,j) - Hx^{n+½}(i,j-1))/Δy
    ]

References
----------
.. [1] Taflove & Hagness, "Computational Electrodynamics," 3rd ed., 2005.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from neurowave.core.base import BaseBoundary, BaseSource, BaseSolver
from neurowave.core.config import SimulationConfig, SimulationMode
from neurowave.core.grid import Grid2D


@dataclass
class FDTD2D(BaseSolver):
    """2D FDTD solver supporting TMz and TEz modes.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.
    sources : list of BaseSource
        Excitation sources to inject.
    boundaries : list of BaseBoundary
        Boundary conditions to apply.
    record_field : str, optional
        Field component to record at every timestep (e.g., 'Ez').
    record_interval : int
        Record a snapshot every N steps (default: 1 = every step).
    probe_points : list of (int, int), optional
        Grid points at which to record time-domain field values.

    Attributes
    ----------
    grid : Grid2D
        The Yee grid with field arrays.
    field_snapshots : list of ndarray
        Recorded field snapshots.
    probe_data : dict
        Time-domain data at probe points.
    """

    config: SimulationConfig
    sources: List[BaseSource] = field(default_factory=list)
    boundaries: List[BaseBoundary] = field(default_factory=list)
    record_field: Optional[str] = None
    record_interval: int = 1
    probe_points: List[Tuple[int, int]] = field(default_factory=list)

    # Internal state
    grid: Grid2D = field(init=False, repr=False)
    _step: int = field(init=False, default=0)
    field_snapshots: List[npt.NDArray[np.float64]] = field(
        init=False, default_factory=list, repr=False
    )
    probe_data: Dict[Tuple[int, int], List[float]] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        """Auto-initialize on creation."""
        self.initialize(self.config)

    def initialize(self, config: SimulationConfig) -> None:
        """Initialize the solver: allocate grid and prepare recording."""
        self.config = config
        self.grid = Grid2D(config)
        self._step = 0
        self.field_snapshots = []
        self.probe_data = {pt: [] for pt in self.probe_points}

    def _update_h_fields_tmz(self) -> None:
        """Update Hx and Hy for TMz mode using vectorized NumPy.

        Hx^{n+½}(i,j) = Da(i,j)·Hx^{n-½}(i,j)
                       - Db(i,j)/Δy · [Ez^n(i,j+1) - Ez^n(i,j)]

        Hy^{n+½}(i,j) = Da(i,j)·Hy^{n-½}(i,j)
                       + Db(i,j)/Δx · [Ez^n(i+1,j) - Ez^n(i,j)]

        Indexing: Hx uses j to j+1 (y-derivative of Ez)
                  Hy uses i to i+1 (x-derivative of Ez)
        """
        dx = self.config.grid.dx
        dy = self.config.grid.dy
        da = self.grid._da
        db = self.grid._db

        # Hx update: dEz/dy
        self.grid.hx[:, :-1] = (
            da[:, :-1] * self.grid.hx[:, :-1]
            - db[:, :-1] / dy * (self.grid.ez[:, 1:] - self.grid.ez[:, :-1])
        )

        # Hy update: dEz/dx
        self.grid.hy[:-1, :] = (
            da[:-1, :] * self.grid.hy[:-1, :]
            + db[:-1, :] / dx * (self.grid.ez[1:, :] - self.grid.ez[:-1, :])
        )

    def _update_e_fields_tmz(self) -> None:
        """Update Ez for TMz mode using vectorized NumPy.

        Ez^{n+1}(i,j) = Ca(i,j)·Ez^n(i,j) + Cb(i,j) · [
            (Hy^{n+½}(i,j) - Hy^{n+½}(i-1,j)) / Δx
          - (Hx^{n+½}(i,j) - Hx^{n+½}(i,j-1)) / Δy
        ]
        """
        dx = self.config.grid.dx
        dy = self.config.grid.dy
        ca = self.grid._ca
        cb = self.grid._cb

        # curl_h = dHy/dx - dHx/dy
        self.grid.ez[1:, 1:] = (
            ca[1:, 1:] * self.grid.ez[1:, 1:]
            + cb[1:, 1:] * (
                (self.grid.hy[1:, 1:] - self.grid.hy[:-1, 1:]) / dx
                - (self.grid.hx[1:, 1:] - self.grid.hx[1:, :-1]) / dy
            )
        )

    def _update_h_fields_tez(self) -> None:
        """Update Hz for TEz mode using vectorized NumPy.

        Hz^{n+½}(i,j) = Da(i,j)·Hz^{n-½}(i,j)
                       + Db(i,j) · [(Ex^n(i,j+1) - Ex^n(i,j))/Δy
                                  - (Ey^n(i+1,j) - Ey^n(i,j))/Δx]
        """
        dx = self.config.grid.dx
        dy = self.config.grid.dy
        da = self.grid._da
        db = self.grid._db

        # Hz update: dEx/dy - dEy/dx
        self.grid.hz[:-1, :-1] = (
            da[:-1, :-1] * self.grid.hz[:-1, :-1]
            + db[:-1, :-1] * (
                (self.grid.ex[:-1, 1:] - self.grid.ex[:-1, :-1]) / dy
                - (self.grid.ey[1:, :-1] - self.grid.ey[:-1, :-1]) / dx
            )
        )

    def _update_e_fields_tez(self) -> None:
        """Update Ex and Ey for TEz mode using vectorized NumPy.

        Ex^{n+1}(i,j) = Ca(i,j)·Ex^n(i,j) + Cb(i,j)/Δy · [Hz^{n+½}(i,j) - Hz^{n+½}(i,j-1)]
        Ey^{n+1}(i,j) = Ca(i,j)·Ey^n(i,j) - Cb(i,j)/Δx · [Hz^{n+½}(i,j) - Hz^{n+½}(i-1,j)]
        """
        dx = self.config.grid.dx
        dy = self.config.grid.dy
        ca = self.grid._ca
        cb = self.grid._cb

        # Ex update: dHz/dy
        self.grid.ex[:, 1:] = (
            ca[:, 1:] * self.grid.ex[:, 1:]
            + cb[:, 1:] / dy * (self.grid.hz[:, 1:] - self.grid.hz[:, :-1])
        )

        # Ey update: -dHz/dx
        self.grid.ey[1:, :] = (
            ca[1:, :] * self.grid.ey[1:, :]
            - cb[1:, :] / dx * (self.grid.hz[1:, :] - self.grid.hz[:-1, :])
        )

    def _inject_sources(self) -> None:
        """Inject all sources as soft sources (additive)."""
        dt = self.config.dt
        for src in self.sources:
            val = src.value_at(self._step, dt)
            x, y = src.position
            comp = src.component

            if comp == "Ez" and self.config.mode == SimulationMode.TMZ:
                self.grid.ez[x, y] += val
            elif comp == "Hz" and self.config.mode == SimulationMode.TEZ:
                self.grid.hz[x, y] += val
            elif comp == "Ex":
                self.grid.ex[x, y] += val
            elif comp == "Ey":
                self.grid.ey[x, y] += val

    def _apply_boundaries_h(self) -> None:
        """Apply boundary conditions to H-fields."""
        e_dict, h_dict = self._field_dicts()
        for bc in self.boundaries:
            bc.apply_h_field(e_dict, h_dict, self.config)

    def _apply_boundaries_e(self) -> None:
        """Apply boundary conditions to E-fields."""
        e_dict, h_dict = self._field_dicts()
        for bc in self.boundaries:
            bc.apply_e_field(e_dict, h_dict, self.config)

    def _field_dicts(self) -> Tuple[Dict, Dict]:
        """Build field dictionaries for boundary condition interface."""
        if self.config.mode == SimulationMode.TMZ:
            e = {"Ez": self.grid.ez}
            h = {"Hx": self.grid.hx, "Hy": self.grid.hy}
        else:
            e = {"Ex": self.grid.ex, "Ey": self.grid.ey}
            h = {"Hz": self.grid.hz}
        return e, h

    def _record(self) -> None:
        """Record field snapshots and probe data."""
        if self.record_field and self._step % self.record_interval == 0:
            field_arr = self.get_field(self.record_field)
            self.field_snapshots.append(field_arr.copy())

        for pt in self.probe_points:
            comp = self.record_field or "Ez"
            field_arr = self.get_field(comp)
            self.probe_data[pt].append(float(field_arr[pt[0], pt[1]]))

    def step(self) -> None:
        """Advance simulation by one timestep (leapfrog scheme).

        Order of operations:
        1. Update H-fields (n-½ → n+½)
        2. Apply H-field boundary corrections
        3. Update E-fields (n → n+1)
        4. Inject sources
        5. Apply E-field boundary corrections
        6. Record data
        """
        if self.config.mode == SimulationMode.TMZ:
            self._update_h_fields_tmz()
            self._apply_boundaries_h()
            self._update_e_fields_tmz()
            self._inject_sources()
            self._apply_boundaries_e()
        elif self.config.mode == SimulationMode.TEZ:
            self._update_h_fields_tez()
            self._apply_boundaries_h()
            self._update_e_fields_tez()
            self._inject_sources()
            self._apply_boundaries_e()
        else:
            raise NotImplementedError(f"Solver not implemented for mode {self.config.mode}")

        self._record()
        self._step += 1

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run simulation for specified number of steps.

        Parameters
        ----------
        num_steps : int, optional
            Steps to run. Defaults to config.num_steps.
        """
        steps = num_steps or self.config.num_steps
        for _ in range(steps):
            self.step()

    def get_field(self, component: str) -> npt.NDArray[np.float64]:
        """Get current field component array.

        Parameters
        ----------
        component : str
            One of 'Ez', 'Hx', 'Hy', 'Hz', 'Ex', 'Ey'.
        """
        field_map = {
            "Ez": self.grid.ez, "Hx": self.grid.hx, "Hy": self.grid.hy,
            "Hz": self.grid.hz, "Ex": self.grid.ex, "Ey": self.grid.ey,
        }
        if component not in field_map:
            raise ValueError(f"Unknown component '{component}'. Use: {list(field_map)}")
        return field_map[component]

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def current_time(self) -> float:
        return self._step * self.config.dt
