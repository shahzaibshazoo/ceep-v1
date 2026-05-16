"""
Full 3D FDTD Solver Implementation
===================================

Complete 3D finite-difference time-domain electromagnetic solver with:
- All 6 field components (Ex, Ey, Ez, Hx, Hy, Hz)
- Yee staggered grid
- CPML absorbing boundaries
- Arbitrary sources
- Material heterogeneity
- Field recording and visualization

Author: NeuroWave Development Team
Date: 2026-05-13
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ceep.core.config import SimulationConfig
from ceep.core.grid_3d import Grid3D
from ceep.core.base import BaseSource, BaseBoundary
from ceep.core.backend import to_scalar, to_numpy, get_backend_module, is_gpu_active
from ceep.solvers.fdtd_base import FdtdBase
from ceep.boundaries.absorbing import CPML


class FDTD3D(FdtdBase):
    """Complete 3D FDTD electromagnetic solver.

    Implements the full 3D Yee algorithm with all 6 field components,
    CPML absorbing boundaries, and support for complex materials.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration with 3D grid parameters.
    sources : list of BaseSource, optional
        List of electromagnetic sources to inject.
    boundaries : list of BaseBoundary, optional
        Boundary conditions (CPML recommended for 3D).
    record_field : str, optional
        Which field component to snapshot ('Ex', 'Ey', 'Ez', etc.).
    probe_points : list of tuple, optional
        List of (x, y, z) probe locations for time-domain recording.

    Examples
    --------
    >>> # 3D cavity resonator
    >>> grid = GridConfig(nx=100, ny=100, nz=100, dx=1e-3, dy=1e-3, dz=1e-3)
    >>> config = SimulationConfig(grid=grid, total_steps=1000)
    >>> source = PointSource(x=50, y=50, z=50, component='Ez', ...)
    >>> solver = FDTD3D(config, sources=[source], boundaries=[CPML(thickness=10)])
    >>> solver.run()
    """

    def __init__(
        self,
        config: SimulationConfig,
        sources: Optional[List[BaseSource]] = None,
        boundaries: Optional[List[BaseBoundary]] = None,
        record_field: Optional[str] = None,
        probe_points: Optional[List[Tuple[int, int, int]]] = None
    ):
        """Initialize the 3D FDTD solver."""
        # Call parent class initialization
        super().__init__(
            config=config,
            sources=sources,
            boundaries=boundaries,
            record_field=record_field,
            record_interval=1,  # 3D hardcodes every 10 steps in record()
            probe_points=probe_points,
        )

        # Initialize the 3D grid
        self.grid = Grid3D(config.grid, config.dt)

        # Initialize CPML boundaries
        for boundary in self.boundaries:
            if isinstance(boundary, CPML):
                boundary.initialize_3d(self.grid, self.config.dt)

        # Advanced probe tracking (complementary to base class probe_data)
        self._probes: Dict[str, dict] = {}

    def add_probe(self, x: int, y: int, z: int, component: str) -> None:
        """Add a time-domain probe at a specific grid location."""
        key = f"{x}_{y}_{z}_{component}"
        if key not in self._probes:
            self._probes[key] = {"loc": (x, y, z), "comp": component, "data": []}

    def _update_h_fields(self) -> None:
        """Update Hx, Hy, Hz for 3D."""
        if is_gpu_active() and self._use_fused_kernels:
            from ceep.cuda.kernels import launch_update_h_3d
            nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
            launch_update_h_3d(
                self.grid.hx, self.grid.hy, self.grid.hz,
                self.grid.ex, self.grid.ey, self.grid.ez,
                self.grid._da, self.grid._db,
                nx, ny, nz,
                self.config.grid.dx, self.config.grid.dy, self.config.grid.dz
            )
            return

        dx, dy, dz = self.config.grid.dx, self.config.grid.dy, self.config.grid.dz
        da, db = self.grid._da, self.grid._db

        ex, ey, ez = self.grid.ex, self.grid.ey, self.grid.ez
        hx, hy, hz = self.grid.hx, self.grid.hy, self.grid.hz

        # Hx = Da*Hx - Db*(dEz/dy - dEy/dz)
        hx[:, :-1, :-1] = da[:, :-1, :-1] * hx[:, :-1, :-1] - db[:, :-1, :-1] * (
            (ez[:, 1:, :-1] - ez[:, :-1, :-1]) / dy -
            (ey[:, :-1, 1:] - ey[:, :-1, :-1]) / dz
        )

        # Hy = Da*Hy - Db*(dEx/dz - dEz/dx)
        hy[:-1, :, :-1] = da[:-1, :, :-1] * hy[:-1, :, :-1] - db[:-1, :, :-1] * (
            (ex[:-1, :, 1:] - ex[:-1, :, :-1]) / dz -
            (ez[1:, :, :-1] - ez[:-1, :, :-1]) / dx
        )

        # Hz = Da*Hz - Db*(dEy/dx - dEx/dy)
        hz[:-1, :-1, :] = da[:-1, :-1, :] * hz[:-1, :-1, :] - db[:-1, :-1, :] * (
            (ey[1:, :-1, :] - ey[:-1, :-1, :]) / dx -
            (ex[:-1, 1:, :] - ex[:-1, :-1, :]) / dy
        )

    def _update_e_fields(self) -> None:
        """Update Ex, Ey, Ez for 3D."""
        if is_gpu_active() and self._use_fused_kernels:
            from ceep.cuda.kernels import launch_update_e_3d
            nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
            launch_update_e_3d(
                self.grid.ex, self.grid.ey, self.grid.ez,
                self.grid.hx, self.grid.hy, self.grid.hz,
                self.grid._ca, self.grid._cb,
                nx, ny, nz,
                self.config.grid.dx, self.config.grid.dy, self.config.grid.dz
            )
            return

        dx, dy, dz = self.config.grid.dx, self.config.grid.dy, self.config.grid.dz
        ca, cb = self.grid._ca, self.grid._cb

        ex, ey, ez = self.grid.ex, self.grid.ey, self.grid.ez
        hx, hy, hz = self.grid.hx, self.grid.hy, self.grid.hz

        # Ex = Ca*Ex + Cb*(dHz/dy - dHy/dz)
        ex[1:, 1:, 1:] = ca[1:, 1:, 1:] * ex[1:, 1:, 1:] + cb[1:, 1:, 1:] * (
            (hz[1:, 1:, 1:] - hz[1:, :-1, 1:]) / dy -
            (hy[1:, 1:, 1:] - hy[1:, 1:, :-1]) / dz
        )

        # Ey = Ca*Ey + Cb*(dHx/dz - dHz/dx)
        ey[1:, 1:, 1:] = ca[1:, 1:, 1:] * ey[1:, 1:, 1:] + cb[1:, 1:, 1:] * (
            (hx[1:, 1:, 1:] - hx[1:, 1:, :-1]) / dz -
            (hz[1:, 1:, 1:] - hz[:-1, 1:, 1:]) / dx
        )

        # Ez = Ca*Ez + Cb*(dHy/dx - dHx/dy)
        ez[1:, 1:, 1:] = ca[1:, 1:, 1:] * ez[1:, 1:, 1:] + cb[1:, 1:, 1:] * (
            (hy[1:, 1:, 1:] - hy[:-1, 1:, 1:]) / dx -
            (hx[1:, 1:, 1:] - hx[1:, :-1, 1:]) / dy
        )

    def _inject_sources(self) -> None:
        """Inject sources into the 3D grid."""
        for src in self.sources:
            # Check if source has 3D coordinates
            if not hasattr(src, 'z'):
                # 2D source, skip
                continue

            val = src.value_at(self._step, self.config.dt)
            if val == 0.0:
                continue

            field_map = {
                "Ex": self.grid.ex, "Ey": self.grid.ey, "Ez": self.grid.ez,
                "Hx": self.grid.hx, "Hy": self.grid.hy, "Hz": self.grid.hz,
            }

            if not hasattr(src, 'component') or src.component not in field_map:
                continue

            arr = field_map[src.component]
            # Soft source
            arr[src.x, src.y, src.z] += val

    def _record(self) -> None:
        """Record fields at probes and snapshots."""
        field_map = {
            "Ex": self.grid.ex, "Ey": self.grid.ey, "Ez": self.grid.ez,
            "Hx": self.grid.hx, "Hy": self.grid.hy, "Hz": self.grid.hz,
        }

        # Record field snapshots (every 10 steps to match original behavior)
        if self.record_field and (self._step % 10 == 0):
            if self.record_field in field_map:
                self.field_snapshots.append(to_numpy(field_map[self.record_field]))

        # Record from probe_data dict (simple x,y,z tuples, default to Ez)
        if self.probe_data:
            field_component = self.record_field or "Ez"
            arr = field_map.get(field_component, self.grid.ez)
            for point in self.probe_points:
                if point in self.probe_data:
                    self.probe_data[point].append(to_scalar(arr[point]))

        # Record from _probes dict (advanced probes with component specification)
        for probe in self._probes.values():
            loc = probe["loc"]
            arr = field_map[probe["comp"]]
            probe["data"].append(to_scalar(arr[loc]))

    def _apply_boundaries_h(self) -> None:
        """Apply boundary conditions to H-fields."""
        for boundary in self.boundaries:
            if isinstance(boundary, CPML):
                boundary.apply_h_3d(self.grid)

    def _apply_boundaries_e(self) -> None:
        """Apply boundary conditions to E-fields."""
        for boundary in self.boundaries:
            if isinstance(boundary, CPML):
                boundary.apply_e_3d(self.grid)


    def get_probe_data(self, x: int, y: int, z: int) -> npt.NDArray[np.float64]:
        """Get recorded time-domain data at a probe point.

        Parameters
        ----------
        x, y, z : int
            Probe grid coordinates.

        Returns
        -------
        data : ndarray
            Time-domain field values at the probe.
        """
        point = (x, y, z)
        if point not in self.probe_data:
            raise KeyError(f"No probe at ({x}, {y}, {z})")
        return np.array(self.probe_data[point])

    def get_field_snapshot(self, component: str, step_index: int = -1) -> npt.NDArray[np.float64]:
        """Get a 3D field snapshot at a specific timestep.

        Parameters
        ----------
        component : str
            Field component ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz').
        step_index : int
            Snapshot index (-1 for latest).

        Returns
        -------
        field : ndarray (nx, ny, nz)
            3D field array.
        """
        if not self.field_snapshots:
            raise ValueError("No field snapshots recorded")

        if component != self.record_field:
            raise ValueError(f"Only {self.record_field} was recorded")

        return self.field_snapshots[step_index]

    def get_slice_2d(
        self,
        component: str,
        plane: str,
        index: int
    ) -> npt.NDArray[np.float64]:
        """Extract a 2D slice from the current 3D field.

        Parameters
        ----------
        component : str
            Field component ('Ex', 'Ey', 'Ez', etc.).
        plane : str
            Slice plane ('xy', 'xz', or 'yz').
        index : int
            Slice index along the perpendicular axis.

        Returns
        -------
        slice_2d : ndarray
            2D field slice.

        Examples
        --------
        >>> # Get xy-plane slice at z=50
        >>> ez_slice = solver.get_slice_2d('Ez', 'xy', 50)
        """
        field_map = {
            "Ex": self.grid.ex, "Ey": self.grid.ey, "Ez": self.grid.ez,
            "Hx": self.grid.hx, "Hy": self.grid.hy, "Hz": self.grid.hz,
        }

        if component not in field_map:
            raise ValueError(f"Unknown component: {component}")

        field = field_map[component]

        if plane == 'xy':
            return field[:, :, index]
        elif plane == 'xz':
            return field[:, index, :]
        elif plane == 'yz':
            return field[index, :, :]
        else:
            raise ValueError(f"Unknown plane: {plane}. Use 'xy', 'xz', or 'yz'.")

    def compute_energy(self) -> float:
        """Compute total electromagnetic energy in the grid.

        Returns
        -------
        energy : float
            Total energy = ∫ (ε|E|² + μ|H|²)/2 dV
        """
        from ceep.core.constants import EPS_0, MU_0
        xp = get_backend_module()

        # Electric energy density
        e_energy = 0.5 * EPS_0 * to_scalar(
            xp.sum(self.grid.ex ** 2) +
            xp.sum(self.grid.ey ** 2) +
            xp.sum(self.grid.ez ** 2)
        )

        # Magnetic energy density
        h_energy = 0.5 * MU_0 * to_scalar(
            xp.sum(self.grid.hx ** 2) +
            xp.sum(self.grid.hy ** 2) +
            xp.sum(self.grid.hz ** 2)
        )

        # Total energy (multiply by cell volume)
        dV = self.config.grid.dx * self.config.grid.dy * self.config.grid.dz
        return (e_energy + h_energy) * dV


def visualize_slice_3d(
    solver: FDTD3D,
    component: str,
    plane: str,
    index: int,
    title: Optional[str] = None
):
    """Visualize a 2D slice from 3D simulation.

    Parameters
    ----------
    solver : FDTD3D
        Completed simulation.
    component : str
        Field component to visualize.
    plane : str
        Slice plane ('xy', 'xz', 'yz').
    index : int
        Slice index.
    title : str, optional
        Plot title.
    """
    import matplotlib.pyplot as plt

    field_slice = solver.get_slice_2d(component, plane, index)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use symmetric colormap
    vmax = np.max(np.abs(field_slice))
    im = ax.imshow(
        field_slice.T,
        origin='lower',
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
        aspect='equal'
    )

    plt.colorbar(im, ax=ax, label=f'{component} (V/m or A/m)')

    if title is None:
        title = f'{component} field - {plane.upper()} plane at index {index}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    if plane == 'xy':
        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')
    elif plane == 'xz':
        ax.set_xlabel('X index')
        ax.set_ylabel('Z index')
    else:  # yz
        ax.set_xlabel('Y index')
        ax.set_ylabel('Z index')

    plt.tight_layout()
    return fig
