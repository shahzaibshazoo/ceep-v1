"""
Field visualization tools for 2D FDTD simulation.

Provides:
- FieldPlotter: Static 2D field snapshots (publication quality)
- FieldAnimator: Animated field evolution over time
- SourcePlotter: Source waveform visualization
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize, TwoSlopeNorm


def plot_field_2d(
    field: npt.NDArray[np.float64],
    title: str = "Field",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    colorbar_label: str = "Amplitude",
    show_grid: bool = False,
) -> plt.Figure:
    """Plot a 2D field snapshot.

    Parameters
    ----------
    field : ndarray
        2D array of field values, shape (nx, ny).
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap. 'RdBu_r' is good for diverging fields.
    vmin, vmax : float, optional
        Color scale limits. Auto-symmetric if None.
    dx, dy : float
        Grid spacings for axis labeling [m].
    save_path : str, optional
        If given, save figure to this path.
    figsize : tuple
        Figure size in inches.
    colorbar_label : str
        Label for the colorbar.
    show_grid : bool
        If True, overlay grid lines.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    nx, ny = field.shape
    extent = [0, ny * dy * 1e3, 0, nx * dx * 1e3]  # Convert to mm

    # Symmetric color scale for wave fields
    if vmin is None and vmax is None:
        abs_max = np.abs(field).max()
        if abs_max > 0:
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -1, 1

    im = ax.imshow(
        field,
        cmap=cmap,
        extent=extent,
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )

    ax.set_xlabel("y [mm]")
    ax.set_ylabel("x [mm]")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(colorbar_label)

    if show_grid:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_field_snapshots(
    snapshots: List[npt.NDArray[np.float64]],
    step_indices: Optional[List[int]] = None,
    ncols: int = 4,
    title_prefix: str = "Step",
    cmap: str = "RdBu_r",
    dx: float = 1.0,
    dy: float = 1.0,
    save_path: Optional[str] = None,
    figsize_per_plot: Tuple[float, float] = (4, 3),
) -> plt.Figure:
    """Plot multiple field snapshots in a grid layout.

    Parameters
    ----------
    snapshots : list of ndarray
        Field snapshots to plot.
    step_indices : list of int, optional
        Step numbers for titles. Auto-generated if None.
    ncols : int
        Number of columns in subplot grid.
    save_path : str, optional
        Path to save the figure.
    """
    n = len(snapshots)
    if n == 0:
        return plt.figure()

    if step_indices is None:
        step_indices = list(range(n))

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    # Global color scale
    all_max = max(np.abs(s).max() for s in snapshots)
    if all_max == 0:
        all_max = 1.0

    for idx, (snap, step) in enumerate(zip(snapshots, step_indices)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        nx, ny = snap.shape
        extent = [0, ny * dy * 1e3, 0, nx * dx * 1e3]
        ax.imshow(
            snap, cmap=cmap, extent=extent, origin="lower",
            aspect="equal", vmin=-all_max, vmax=all_max,
            interpolation="bilinear",
        )
        ax.set_title(f"{title_prefix} {step}", fontsize=10)
        ax.set_xlabel("y [mm]", fontsize=8)
        ax.set_ylabel("x [mm]", fontsize=8)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_animation(
    snapshots: List[npt.NDArray[np.float64]],
    dx: float = 1.0,
    dy: float = 1.0,
    dt: float = 1.0,
    interval: int = 50,
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: str = "FDTD Simulation",
) -> animation.FuncAnimation:
    """Create an animation from field snapshots.

    Parameters
    ----------
    snapshots : list of ndarray
        Sequence of 2D field arrays.
    dx, dy : float
        Grid spacings [m].
    dt : float
        Time between snapshots [s].
    interval : int
        Delay between frames in ms.
    save_path : str, optional
        Save as .mp4 or .gif.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    nx, ny = snapshots[0].shape
    extent = [0, ny * dy * 1e3, 0, nx * dx * 1e3]

    all_max = max(np.abs(s).max() for s in snapshots)
    if all_max == 0:
        all_max = 1.0

    im = ax.imshow(
        snapshots[0], cmap=cmap, extent=extent, origin="lower",
        aspect="equal", vmin=-all_max, vmax=all_max,
        interpolation="bilinear",
    )
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("x [mm]")
    time_text = ax.set_title(f"{title} — t = 0.00 ns")
    fig.colorbar(im, ax=ax, shrink=0.8)

    def update(frame: int):
        im.set_data(snapshots[frame])
        t_ns = frame * dt * 1e9
        time_text.set_text(f"{title} — t = {t_ns:.2f} ns")
        return [im, time_text]

    anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=interval, blit=False
    )

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        else:
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval)

    return anim


def plot_source_waveform(
    waveform: npt.NDArray[np.float64],
    dt: float,
    title: str = "Source Waveform",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a source temporal waveform.

    Parameters
    ----------
    waveform : ndarray
        1D array of source values at each timestep.
    dt : float
        Timestep [s].
    save_path : str, optional
        Save path for the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    t_ns = np.arange(len(waveform)) * dt * 1e9

    # Time domain
    ax1.plot(t_ns, waveform, "b-", linewidth=1.5)
    ax1.set_xlabel("Time [ns]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{title} — Time Domain")
    ax1.grid(True, alpha=0.3)

    # Frequency domain
    spectrum = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(len(waveform), d=dt) / 1e9  # GHz
    spectrum_db = 20 * np.log10(spectrum / spectrum.max() + 1e-30)

    ax2.plot(freqs, spectrum_db, "r-", linewidth=1.5)
    ax2.set_xlabel("Frequency [GHz]")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.set_title(f"{title} — Spectrum")
    ax2.set_ylim(-60, 5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
