"""
Fused CUDA Kernels for FDTD Field Updates
==========================================

Provides CuPy RawKernel implementations that fuse multiple array operations
into single GPU kernel launches, eliminating intermediate allocations.

These kernels are optional performance optimizations — the solver works
correctly with plain CuPy array operations (which mirror NumPy). The kernels
provide ~2-3x additional speedup on top of the basic GPU array path.

Usage
-----
These are automatically dispatched when the CuPy backend is active:

    from neurowave.core.backend import set_backend
    set_backend('cupy')
    # Solver now uses fused kernels automatically
"""

from __future__ import annotations

import math
from typing import Tuple

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def _get_block_grid(nx: int, ny: int, block_size: int = 16) -> Tuple[tuple, tuple]:
    """Compute CUDA grid/block dimensions for a 2D domain."""
    bx = block_size
    by = block_size
    gx = math.ceil(nx / bx)
    gy = math.ceil(ny / by)
    return (gx, gy), (bx, by)


def _get_block_grid_3d(nx: int, ny: int, nz: int, block_size: int = 8) -> Tuple[tuple, tuple]:
    """Compute CUDA grid/block dimensions for a 3D domain."""
    bx = block_size
    by = block_size
    bz = block_size
    gx = math.ceil(nx / bx)
    gy = math.ceil(ny / by)
    gz = math.ceil(nz / bz)
    return (gx, gy, gz), (bx, by, bz)


if HAS_CUPY:

    # ===========================================================================
    # 2D TMz Kernels
    # ===========================================================================

    _update_h_2d_tmz_src = r'''
    extern "C" __global__
    void update_h_2d_tmz(
        double* __restrict__ hx,
        double* __restrict__ hy,
        const double* __restrict__ ez,
        const double* __restrict__ da,
        const double* __restrict__ db,
        int nx, int ny,
        double inv_dx, double inv_dy
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        if (i < nx && j < ny) {
            int idx = i * ny + j;

            // Hx update: valid for j < ny-1
            if (j < ny - 1) {
                int idx_jp1 = i * ny + (j + 1);
                hx[idx] = da[idx] * hx[idx]
                         - db[idx] * inv_dy * (ez[idx_jp1] - ez[idx]);
            }

            // Hy update: valid for i < nx-1
            if (i < nx - 1) {
                int idx_ip1 = (i + 1) * ny + j;
                hy[idx] = da[idx] * hy[idx]
                         + db[idx] * inv_dx * (ez[idx_ip1] - ez[idx]);
            }
        }
    }
    '''

    _update_e_2d_tmz_src = r'''
    extern "C" __global__
    void update_e_2d_tmz(
        double* __restrict__ ez,
        const double* __restrict__ hx,
        const double* __restrict__ hy,
        const double* __restrict__ ca,
        const double* __restrict__ cb,
        int nx, int ny,
        double inv_dx, double inv_dy
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        // E-field update: valid for i >= 1, j >= 1
        if (i >= 1 && i < nx && j >= 1 && j < ny) {
            int idx = i * ny + j;
            int idx_im1 = (i - 1) * ny + j;
            int idx_jm1 = i * ny + (j - 1);

            double curl_h = (hy[idx] - hy[idx_im1]) * inv_dx
                          - (hx[idx] - hx[idx_jm1]) * inv_dy;

            ez[idx] = ca[idx] * ez[idx] + cb[idx] * curl_h;
        }
    }
    '''

    # ===========================================================================
    # 3D Kernels
    # ===========================================================================

    _update_h_3d_src = r'''
    extern "C" __global__
    void update_h_3d(
        double* __restrict__ hx,
        double* __restrict__ hy,
        double* __restrict__ hz,
        const double* __restrict__ ex,
        const double* __restrict__ ey,
        const double* __restrict__ ez,
        const double* __restrict__ da,
        const double* __restrict__ db,
        int nx, int ny, int nz,
        double inv_dx, double inv_dy, double inv_dz
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;

        if (i >= nx || j >= ny || k >= nz) return;

        int idx = (i * ny + j) * nz + k;
        double da_val = da[idx];
        double db_val = db[idx];

        // Hx: valid for j < ny-1, k < nz-1
        if (j < ny - 1 && k < nz - 1) {
            int idx_jp1 = (i * ny + (j + 1)) * nz + k;
            int idx_kp1 = (i * ny + j) * nz + (k + 1);
            hx[idx] = da_val * hx[idx] - db_val * (
                (ez[idx_jp1] - ez[idx]) * inv_dy -
                (ey[idx_kp1] - ey[idx]) * inv_dz
            );
        }

        // Hy: valid for i < nx-1, k < nz-1
        if (i < nx - 1 && k < nz - 1) {
            int idx_ip1 = ((i + 1) * ny + j) * nz + k;
            int idx_kp1 = (i * ny + j) * nz + (k + 1);
            hy[idx] = da_val * hy[idx] - db_val * (
                (ex[idx_kp1] - ex[idx]) * inv_dz -
                (ez[idx_ip1] - ez[idx]) * inv_dx
            );
        }

        // Hz: valid for i < nx-1, j < ny-1
        if (i < nx - 1 && j < ny - 1) {
            int idx_ip1 = ((i + 1) * ny + j) * nz + k;
            int idx_jp1 = (i * ny + (j + 1)) * nz + k;
            hz[idx] = da_val * hz[idx] - db_val * (
                (ey[idx_ip1] - ey[idx]) * inv_dx -
                (ex[idx_jp1] - ex[idx]) * inv_dy
            );
        }
    }
    '''

    _update_e_3d_src = r'''
    extern "C" __global__
    void update_e_3d(
        double* __restrict__ ex,
        double* __restrict__ ey,
        double* __restrict__ ez,
        const double* __restrict__ hx,
        const double* __restrict__ hy,
        const double* __restrict__ hz,
        const double* __restrict__ ca,
        const double* __restrict__ cb,
        int nx, int ny, int nz,
        double inv_dx, double inv_dy, double inv_dz
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        int k = blockDim.z * blockIdx.z + threadIdx.z;

        // E-field update: valid for i >= 1, j >= 1, k >= 1
        if (i < 1 || j < 1 || k < 1) return;
        if (i >= nx || j >= ny || k >= nz) return;

        int idx = (i * ny + j) * nz + k;
        int idx_im1 = ((i - 1) * ny + j) * nz + k;
        int idx_jm1 = (i * ny + (j - 1)) * nz + k;
        int idx_km1 = (i * ny + j) * nz + (k - 1);

        double ca_val = ca[idx];
        double cb_val = cb[idx];

        // Ex = Ca*Ex + Cb*(dHz/dy - dHy/dz)
        ex[idx] = ca_val * ex[idx] + cb_val * (
            (hz[idx] - hz[idx_jm1]) * inv_dy -
            (hy[idx] - hy[idx_km1]) * inv_dz
        );

        // Ey = Ca*Ey + Cb*(dHx/dz - dHz/dx)
        ey[idx] = ca_val * ey[idx] + cb_val * (
            (hx[idx] - hx[idx_km1]) * inv_dz -
            (hz[idx] - hz[idx_im1]) * inv_dx
        );

        // Ez = Ca*Ez + Cb*(dHy/dx - dHx/dy)
        ez[idx] = ca_val * ez[idx] + cb_val * (
            (hy[idx] - hy[idx_im1]) * inv_dx -
            (hx[idx] - hx[idx_jm1]) * inv_dy
        );
    }
    '''

    # Compile kernels
    _kernel_h_2d_tmz = cp.RawKernel(_update_h_2d_tmz_src, 'update_h_2d_tmz')
    _kernel_e_2d_tmz = cp.RawKernel(_update_e_2d_tmz_src, 'update_e_2d_tmz')
    _kernel_h_3d = cp.RawKernel(_update_h_3d_src, 'update_h_3d')
    _kernel_e_3d = cp.RawKernel(_update_e_3d_src, 'update_e_3d')


    def launch_update_h_2d_tmz(hx, hy, ez, da, db, nx, ny, dx, dy):
        """Launch fused H-field update kernel for 2D TMz."""
        grid, block = _get_block_grid(nx, ny)
        _kernel_h_2d_tmz(
            grid, block,
            (hx, hy, ez, da, db,
             np.int32(nx), np.int32(ny),
             np.float64(1.0 / dx), np.float64(1.0 / dy))
        )


    def launch_update_e_2d_tmz(ez, hx, hy, ca, cb, nx, ny, dx, dy):
        """Launch fused E-field update kernel for 2D TMz."""
        grid, block = _get_block_grid(nx, ny)
        _kernel_e_2d_tmz(
            grid, block,
            (ez, hx, hy, ca, cb,
             np.int32(nx), np.int32(ny),
             np.float64(1.0 / dx), np.float64(1.0 / dy))
        )


    def launch_update_h_3d(hx, hy, hz, ex, ey, ez, da, db, nx, ny, nz, dx, dy, dz):
        """Launch fused H-field update kernel for 3D."""
        grid, block = _get_block_grid_3d(nx, ny, nz)
        _kernel_h_3d(
            grid, block,
            (hx, hy, hz, ex, ey, ez, da, db,
             np.int32(nx), np.int32(ny), np.int32(nz),
             np.float64(1.0 / dx), np.float64(1.0 / dy), np.float64(1.0 / dz))
        )


    def launch_update_e_3d(ex, ey, ez, hx, hy, hz, ca, cb, nx, ny, nz, dx, dy, dz):
        """Launch fused E-field update kernel for 3D."""
        grid, block = _get_block_grid_3d(nx, ny, nz)
        _kernel_e_3d(
            grid, block,
            (ex, ey, ez, hx, hy, hz, ca, cb,
             np.int32(nx), np.int32(ny), np.int32(nz),
             np.float64(1.0 / dx), np.float64(1.0 / dy), np.float64(1.0 / dz))
        )

    # Need numpy for int32/float64 type coercion in kernel args
    import numpy as np

else:
    # Stubs when CuPy not available
    def launch_update_h_2d_tmz(*args, **kwargs):
        raise RuntimeError("CuPy not available")

    def launch_update_e_2d_tmz(*args, **kwargs):
        raise RuntimeError("CuPy not available")

    def launch_update_h_3d(*args, **kwargs):
        raise RuntimeError("CuPy not available")

    def launch_update_e_3d(*args, **kwargs):
        raise RuntimeError("CuPy not available")


def cuda_kernels_available() -> bool:
    """Check if fused CUDA kernels are available."""
    return HAS_CUPY
