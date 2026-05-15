"""
Anatomically Realistic Head Phantom Models
===========================================

Provides multilayer head phantoms for microwave imaging validation:
1. Simple 3-layer skin model
2. Detailed brain model (skull, CSF, gray/white matter)
3. Hemorrhage models (blood clot simulation)

All models use Gabriel tissue database for accurate dielectric properties.

Author: NeuroWave Development Team
Date: 2026-05-13
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import numpy.typing as npt

from ceep.materials.tissue_database import TissueDatabase, TissueProperties


@dataclass
class PhantomGeometry:
    """Base class for phantom geometry specifications.

    Parameters
    ----------
    name : str
        Phantom name/description.
    grid_shape : tuple
        (nx, ny) for 2D or (nx, ny, nz) for 3D.
    dx : float
        Grid spacing in meters.
    """
    name: str
    grid_shape: Tuple[int, ...]
    dx: float

    @property
    def is_3d(self) -> bool:
        """Check if phantom is 3D."""
        return len(self.grid_shape) == 3


class SimpleHeadPhantom:
    """Simple 3-layer circular head model.

    Layers (from outside to inside):
    1. Skin (3 mm)
    2. Skull (7 mm)
    3. Brain (gray matter)

    This is a simplified model useful for initial testing and algorithm
    development. For realistic studies, use DetailedBrainPhantom.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    dx : float
        Grid spacing (meters).
    head_radius_mm : float
        Outer radius of head (default: 90 mm, typical adult).

    Examples
    --------
    >>> phantom = SimpleHeadPhantom(200, 200, dx=1e-3)
    >>> tissue_map = phantom.create_tissue_map()
    >>> eps_r_10ghz = phantom.get_permittivity_map(10e9)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        head_radius_mm: float = 90.0
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.head_radius = head_radius_mm * 1e-3  # Convert to meters

        # Layer thicknesses (meters)
        self.skin_thickness = 3e-3
        self.skull_thickness = 7e-3

        # Get tissue properties from database
        self.db = TissueDatabase()
        self.skin = self.db.get('skin_wet')
        self.skull = self.db.get('skull_cortical_bone')
        self.brain = self.db.get('brain_gray_matter')

    def create_tissue_map(self) -> npt.NDArray[np.object_]:
        """Create a map assigning tissue types to each grid cell.

        Returns
        -------
        tissue_map : ndarray of TissueProperties
            Grid array where each cell contains the tissue at that location.
        """
        tissue_map = np.empty((self.nx, self.ny), dtype=object)

        # Center of head
        cx, cy = self.nx // 2, self.ny // 2

        # Create circular layers
        for i in range(self.nx):
            for j in range(self.ny):
                # Distance from center (in meters)
                r = np.sqrt(((i - cx) * self.dx) ** 2 +
                           ((j - cy) * self.dx) ** 2)

                if r > self.head_radius:
                    tissue_map[i, j] = None  # Outside head (free space)
                elif r > self.head_radius - self.skin_thickness:
                    tissue_map[i, j] = self.skin
                elif r > self.head_radius - self.skin_thickness - self.skull_thickness:
                    tissue_map[i, j] = self.skull
                else:
                    tissue_map[i, j] = self.brain

        return tissue_map

    def get_permittivity_map(
        self,
        frequency: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get complex permittivity map at a specific frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        eps_real, eps_imag : ndarray
            Real and imaginary parts of relative permittivity.
        """
        tissue_map = self.create_tissue_map()
        eps_real = np.ones((self.nx, self.ny), dtype=np.float64)
        eps_imag = np.zeros((self.nx, self.ny), dtype=np.float64)

        for i in range(self.nx):
            for j in range(self.ny):
                tissue = tissue_map[i, j]
                if tissue is not None:
                    eps_complex = tissue.permittivity(np.array([frequency]))[0]
                    eps_real[i, j] = eps_complex.real
                    eps_imag[i, j] = -eps_complex.imag  # Note: negative for loss

        return eps_real, eps_imag


class DetailedBrainPhantom:
    """Anatomically detailed brain phantom with multiple tissue layers.

    Layers (from outside to inside):
    1. Skin (3 mm)
    2. Skull - cortical bone (7 mm)
    3. Cerebrospinal fluid - CSF (2 mm)
    4. Gray matter (10 mm outer cortex)
    5. White matter (interior)

    Supports optional pathology insertion (hemorrhage, tumor).

    Parameters
    ----------
    nx, ny, nz : int
        3D grid dimensions.
    dx : float
        Grid spacing (meters).
    head_shape : str
        'spherical' or 'ellipsoidal'.

    Examples
    --------
    >>> phantom = DetailedBrainPhantom(200, 200, 200, dx=1e-3)
    >>> phantom.add_hemorrhage(x=110, y=100, z=100, radius_mm=10)
    >>> tissue_map = phantom.create_tissue_map()
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        head_shape: str = 'ellipsoidal'
    ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.head_shape = head_shape

        # Ellipsoidal head dimensions (meters)
        self.radius_x = 90e-3   # Width
        self.radius_y = 110e-3  # Length (front-back)
        self.radius_z = 85e-3   # Height

        # Layer thicknesses
        self.skin_thickness = 3e-3
        self.skull_thickness = 7e-3
        self.csf_thickness = 2e-3
        self.gray_matter_thickness = 10e-3

        # Get tissues from database
        self.db = TissueDatabase()
        self.skin = self.db.get('skin_wet')
        self.skull = self.db.get('skull_cortical_bone')
        self.csf = self.db.get('cerebrospinal_fluid')
        self.gray_matter = self.db.get('brain_gray_matter')
        self.white_matter = self.db.get('brain_white_matter')

        # Optional pathologies
        self.hemorrhages: List[Dict] = []
        self.tumors: List[Dict] = []

    def add_hemorrhage(
        self,
        x: int,
        y: int,
        z: int,
        radius_mm: float
    ) -> None:
        """Add a hemorrhage (blood clot) to the phantom.

        Parameters
        ----------
        x, y, z : int
            Grid coordinates of hemorrhage center.
        radius_mm : float
            Radius of hemorrhage in millimeters.
        """
        self.hemorrhages.append({
            'center': (x, y, z),
            'radius': radius_mm * 1e-3,
            'tissue': self.db.get('hemorrhage')
        })

    def add_tumor(
        self,
        x: int,
        y: int,
        z: int,
        radius_mm: float
    ) -> None:
        """Add a tumor to the phantom.

        Parameters
        ----------
        x, y, z : int
            Grid coordinates of tumor center.
        radius_mm : float
            Radius of tumor in millimeters.
        """
        # Tumors have higher water content than normal tissue
        self.tumors.append({
            'center': (x, y, z),
            'radius': radius_mm * 1e-3,
            'tissue': self.db.get('edema')  # Use edema as tumor proxy
        })

    def _ellipsoidal_distance(
        self,
        i: int,
        j: int,
        k: int,
        cx: int,
        cy: int,
        cz: int
    ) -> float:
        """Compute normalized ellipsoidal distance from center.

        Returns
        -------
        r_normalized : float
            Distance normalized by ellipsoid radii (< 1 = inside).
        """
        dx_norm = ((i - cx) * self.dx) / self.radius_x
        dy_norm = ((j - cy) * self.dx) / self.radius_y
        dz_norm = ((k - cz) * self.dx) / self.radius_z
        return np.sqrt(dx_norm**2 + dy_norm**2 + dz_norm**2)

    def create_tissue_map(self) -> npt.NDArray[np.object_]:
        """Create 3D tissue map.

        Returns
        -------
        tissue_map : ndarray (nx, ny, nz) of TissueProperties
            3D grid with tissue assignment.
        """
        tissue_map = np.empty((self.nx, self.ny, self.nz), dtype=object)

        cx, cy, cz = self.nx // 2, self.ny // 2, self.nz // 2

        # Assign tissues layer by layer
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    if self.head_shape == 'ellipsoidal':
                        r_norm = self._ellipsoidal_distance(i, j, k, cx, cy, cz)
                        r = r_norm * self.radius_x  # Use x-radius as reference
                    else:  # spherical
                        r = np.sqrt(((i - cx) * self.dx) ** 2 +
                                   ((j - cy) * self.dx) ** 2 +
                                   ((k - cz) * self.dx) ** 2)
                        r_norm = r / self.radius_x

                    # Assign tissue based on depth
                    outer_radius = self.radius_x

                    if r_norm > 1.0:
                        tissue_map[i, j, k] = None  # Free space
                    elif r > outer_radius - self.skin_thickness:
                        tissue_map[i, j, k] = self.skin
                    elif r > outer_radius - self.skin_thickness - self.skull_thickness:
                        tissue_map[i, j, k] = self.skull
                    elif r > outer_radius - self.skin_thickness - self.skull_thickness - self.csf_thickness:
                        tissue_map[i, j, k] = self.csf
                    elif r > outer_radius - self.skin_thickness - self.skull_thickness - self.csf_thickness - self.gray_matter_thickness:
                        tissue_map[i, j, k] = self.gray_matter
                    else:
                        tissue_map[i, j, k] = self.white_matter

        # Add hemorrhages
        for hemorrhage in self.hemorrhages:
            center = hemorrhage['center']
            radius = hemorrhage['radius']
            tissue = hemorrhage['tissue']

            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        r = np.sqrt(((i - center[0]) * self.dx) ** 2 +
                                   ((j - center[1]) * self.dx) ** 2 +
                                   ((k - center[2]) * self.dx) ** 2)
                        if r <= radius:
                            tissue_map[i, j, k] = tissue

        # Add tumors
        for tumor in self.tumors:
            center = tumor['center']
            radius = tumor['radius']
            tissue = tumor['tissue']

            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        r = np.sqrt(((i - center[0]) * self.dx) ** 2 +
                                   ((j - center[1]) * self.dx) ** 2 +
                                   ((k - center[2]) * self.dx) ** 2)
                        if r <= radius:
                            tissue_map[i, j, k] = tissue

        return tissue_map

    def get_permittivity_map(
        self,
        frequency: float
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get 3D complex permittivity map at specific frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        eps_real, eps_imag : ndarray (nx, ny, nz)
            Real and imaginary parts of relative permittivity.
        """
        tissue_map = self.create_tissue_map()
        eps_real = np.ones((self.nx, self.ny, self.nz), dtype=np.float64)
        eps_imag = np.zeros((self.nx, self.ny, self.nz), dtype=np.float64)

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    tissue = tissue_map[i, j, k]
                    if tissue is not None:
                        eps_complex = tissue.permittivity(np.array([frequency]))[0]
                        eps_real[i, j, k] = eps_complex.real
                        eps_imag[i, j, k] = -eps_complex.imag

        return eps_real, eps_imag


class SkinLayerPhantom:
    """Multilayer skin phantom for melanoma detection.

    Layers:
    1. Epidermis (0.1 mm)
    2. Dermis (2 mm)
    3. Fat/subcutaneous (5+ mm)

    Supports tumor insertion at various depths.

    Parameters
    ----------
    nx, ny : int
        2D grid dimensions.
    dx : float
        Grid spacing (meters), should be fine (< 0.1 mm).
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx

        # Layer thicknesses
        self.epidermis_depth = 0.1e-3
        self.dermis_depth = 2e-3

        # Tissues (using available proxies)
        self.db = TissueDatabase()
        self.skin = self.db.get('skin_wet')
        self.fat = self.db.get('fat')
        self.tumor = self.db.get('breast_tumor')  # Proxy for melanoma

        self.tumor_positions: List[Dict] = []

    def add_tumor(
        self,
        x: int,
        y: int,
        depth_mm: float,
        width_mm: float,
        height_mm: float
    ) -> None:
        """Add elliptical tumor at specified depth.

        Parameters
        ----------
        x, y : int
            Center position (x is depth direction).
        depth_mm : float
            Depth below surface (mm).
        width_mm, height_mm : float
            Tumor dimensions (mm).
        """
        self.tumor_positions.append({
            'center': (x, y),
            'depth': depth_mm * 1e-3,
            'width': width_mm * 1e-3,
            'height': height_mm * 1e-3
        })

    def create_tissue_map(self) -> npt.NDArray[np.object_]:
        """Create 2D cross-section tissue map.

        Returns
        -------
        tissue_map : ndarray (nx, ny)
            Tissue assignment (x is depth direction).
        """
        tissue_map = np.empty((self.nx, self.ny), dtype=object)

        for i in range(self.nx):
            depth = i * self.dx
            for j in range(self.ny):
                if depth < self.epidermis_depth:
                    tissue_map[i, j] = self.skin
                elif depth < self.epidermis_depth + self.dermis_depth:
                    tissue_map[i, j] = self.skin
                else:
                    tissue_map[i, j] = self.fat

        # Add tumors
        for tumor in self.tumor_positions:
            cx, cy = tumor['center']
            width, height = tumor['width'], tumor['height']

            for i in range(self.nx):
                for j in range(self.ny):
                    dx_norm = ((i - cx) * self.dx) / (width / 2)
                    dy_norm = ((j - cy) * self.dx) / (height / 2)
                    if dx_norm**2 + dy_norm**2 <= 1.0:
                        tissue_map[i, j] = self.tumor

        return tissue_map


class BrainPhantom2D:
    """2D Brain Phantom for batched FDTD simulations.

    Convenience class matching the Quick Start API. Creates a 2D circular
    head cross-section with optional hemorrhage for microwave imaging.

    Parameters
    ----------
    nx, ny : int, optional
        Grid dimensions (inferred from solver if using set_phantom).
    dx : float, optional
        Grid spacing in meters.
    hemorrhage_location : tuple of (float, float), optional
        Hemorrhage center (cm from head center).
    hemorrhage_radius : float, optional
        Hemorrhage radius in cm.
    head_radius_cm : float, optional
        Head radius in cm (default: 9 cm).
    use_gabriel_database : bool
        Use Gabriel tissue database for properties (default True).
    """

    def __init__(
        self,
        nx: int = 600,
        ny: int = 600,
        dx: float = 0.5e-3,
        hemorrhage_location: Optional[Tuple[float, float]] = None,
        hemorrhage_radius: float = 1.0,
        head_radius_cm: float = 9.0,
        use_gabriel_database: bool = True,
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.head_radius = head_radius_cm * 1e-2
        self.hemorrhage_location = hemorrhage_location
        self.hemorrhage_radius = hemorrhage_radius * 1e-2
        self.use_gabriel = use_gabriel_database

        # Layer thicknesses
        self.skin_thickness = 3e-3
        self.skull_thickness = 7e-3
        self.csf_thickness = 2e-3
        self.gray_thickness = 10e-3

        if use_gabriel_database:
            self.db = TissueDatabase()

    def get_eps_map(
        self, frequency: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 2D permittivity and conductivity maps.

        Parameters
        ----------
        frequency : float
            Center frequency in Hz.

        Returns
        -------
        eps_r : ndarray (nx, ny)
            Relative permittivity map.
        sigma_e : ndarray (nx, ny)
            Effective conductivity map (S/m).
        """
        eps_r = np.ones((self.nx, self.ny), dtype=np.float64)
        sigma_e = np.zeros((self.nx, self.ny), dtype=np.float64)

        cx, cy = self.nx // 2, self.ny // 2

        if self.use_gabriel:
            tissues = {
                'skin': self.db.get('skin_wet'),
                'skull': self.db.get('skull_cortical_bone'),
                'csf': self.db.get('cerebrospinal_fluid'),
                'gray': self.db.get('brain_gray_matter'),
                'white': self.db.get('brain_white_matter'),
                'blood': self.db.get('hemorrhage'),
            }
            freq_arr = np.array([frequency])
            eps_values = {}
            for name, tissue in tissues.items():
                eps_c = tissue.permittivity(freq_arr)[0]
                eps_values[name] = (eps_c.real, -eps_c.imag * 2 * np.pi * frequency * 8.854e-12)
        else:
            eps_values = {
                'skin': (40.0, 1.0),
                'skull': (12.0, 0.2),
                'csf': (68.0, 2.5),
                'gray': (52.0, 1.8),
                'white': (38.0, 1.2),
                'blood': (61.0, 2.0),
            }

        # Build concentric layers using vectorized operations
        x_grid, y_grid = np.meshgrid(
            np.arange(self.nx), np.arange(self.ny), indexing='ij'
        )
        r = np.sqrt(((x_grid - cx) * self.dx)**2 + ((y_grid - cy) * self.dx)**2)

        r_skin = self.head_radius
        r_skull = r_skin - self.skin_thickness
        r_csf = r_skull - self.skull_thickness
        r_gray = r_csf - self.csf_thickness
        r_white = r_gray - self.gray_thickness

        # Assign layers (outside-in)
        mask_skin = (r <= r_skin) & (r > r_skull)
        mask_skull = (r <= r_skull) & (r > r_csf)
        mask_csf = (r <= r_csf) & (r > r_gray)
        mask_gray = (r <= r_gray) & (r > r_white)
        mask_white = (r <= r_white)

        for mask, name in [
            (mask_skin, 'skin'), (mask_skull, 'skull'),
            (mask_csf, 'csf'), (mask_gray, 'gray'), (mask_white, 'white')
        ]:
            eps_r[mask] = eps_values[name][0]
            sigma_e[mask] = eps_values[name][1]

        # Add hemorrhage
        if self.hemorrhage_location is not None:
            hx_cm, hy_cm = self.hemorrhage_location
            hx_m = hx_cm * 1e-2
            hy_m = hy_cm * 1e-2
            r_hem = np.sqrt(
                ((x_grid - cx) * self.dx - hx_m)**2 +
                ((y_grid - cy) * self.dx - hy_m)**2
            )
            mask_hem = r_hem <= self.hemorrhage_radius
            eps_r[mask_hem] = eps_values['blood'][0]
            sigma_e[mask_hem] = eps_values['blood'][1]

        return eps_r, sigma_e


def visualize_phantom_slice(
    tissue_map: npt.NDArray[np.object_],
    frequency: float = 10e9,
    slice_axis: str = 'z',
    slice_index: Optional[int] = None
):
    """Visualize phantom slice with tissue dielectric properties.

    Parameters
    ----------
    tissue_map : ndarray
        Tissue map from phantom.create_tissue_map().
    frequency : float
        Frequency for permittivity calculation (Hz).
    slice_axis : str
        For 3D: 'x', 'y', or 'z' axis to slice.
    slice_index : int, optional
        Slice index (default: center).
    """
    import matplotlib.pyplot as plt

    # Extract slice
    if tissue_map.ndim == 2:
        slice_2d = tissue_map
    else:  # 3D
        if slice_index is None:
            slice_index = tissue_map.shape[{'x': 0, 'y': 1, 'z': 2}[slice_axis]] // 2

        if slice_axis == 'x':
            slice_2d = tissue_map[slice_index, :, :]
        elif slice_axis == 'y':
            slice_2d = tissue_map[:, slice_index, :]
        else:  # z
            slice_2d = tissue_map[:, :, slice_index]

    # Compute permittivity map
    eps_map = np.ones(slice_2d.shape, dtype=np.float64)

    for i in range(slice_2d.shape[0]):
        for j in range(slice_2d.shape[1]):
            tissue = slice_2d[i, j]
            if tissue is not None:
                eps = tissue.permittivity(np.array([frequency]))[0].real
                eps_map[i, j] = eps

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(eps_map.T, origin='lower', cmap='viridis', aspect='equal')
    plt.colorbar(im, ax=ax, label="Relative Permittivity ε'")
    ax.set_title(f'Phantom Cross-Section at {frequency/1e9:.1f} GHz',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X index')
    ax.set_ylabel('Y index')
    plt.tight_layout()

    return fig
