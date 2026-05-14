"""
Backend Abstraction Layer
==========================

Unified array interface supporting NumPy (CPU) and CuPy (GPU).

This module provides a backend-agnostic array API that allows seamless
switching between CPU and GPU execution. The same FDTD code can run on
either backend without modification.

Supported Backends
------------------
- numpy: CPU execution (default, always available)
- cupy: NVIDIA GPU execution via CuPy
- jax: Google JAX (experimental, for TPU/GPU)
- torch: PyTorch tensors (for AI integration)

Usage
-----
```python
# Set backend globally
set_backend('cupy')

# Arrays are now on GPU
x = zeros((1000, 1000))  # CuPy array on GPU
y = ones_like(x)         # Also on GPU
z = x + y                # GPU computation

# Transfer to CPU if needed
x_cpu = to_numpy(x)
```

Author: NeuroWave Development Team
Date: 2026-05-13
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Any
from enum import Enum
import numpy as np

# Type alias for array types
ArrayType = Union[np.ndarray, Any]  # 'Any' for cupy.ndarray, torch.Tensor, etc.


class Backend(Enum):
    """Supported computational backends."""
    NUMPY = "numpy"
    CUPY = "cupy"
    TORCH = "torch"
    JAX = "jax"


class BackendManager:
    """Manages the active computational backend.

    This singleton class handles backend selection and provides
    a unified array API regardless of the underlying backend.

    Attributes
    ----------
    active : Backend
        Currently active backend.
    module : module
        The actual backend module (numpy, cupy, torch.tensor, etc.).
    """

    def __init__(self):
        self.active = Backend.NUMPY
        self.module = np
        self._available_backends = self._detect_available_backends()

    def _detect_available_backends(self) -> dict:
        """Detect which backends are installed."""
        available = {Backend.NUMPY: True}  # NumPy always available

        # Check for CuPy
        try:
            import cupy as cp
            available[Backend.CUPY] = True
        except ImportError:
            available[Backend.CUPY] = False

        # Check for PyTorch
        try:
            import torch
            available[Backend.TORCH] = True
        except ImportError:
            available[Backend.TORCH] = False

        # Check for JAX
        try:
            import jax.numpy as jnp
            available[Backend.JAX] = True
        except ImportError:
            available[Backend.JAX] = False

        return available

    def set_backend(self, backend: Union[str, Backend]) -> None:
        """Set the active computational backend.

        Parameters
        ----------
        backend : str or Backend
            Backend to use ('numpy', 'cupy', 'torch', or 'jax').

        Raises
        ------
        ValueError
            If backend is not installed or not supported.

        Examples
        --------
        >>> set_backend('cupy')
        >>> print(get_backend())
        Backend.CUPY
        """
        if isinstance(backend, str):
            backend = Backend(backend.lower())

        if backend not in self._available_backends:
            raise ValueError(f"Backend {backend.value} not recognized")

        if not self._available_backends[backend]:
            raise ImportError(
                f"Backend {backend.value} not available. "
                f"Install with: pip install {backend.value}"
            )

        self.active = backend

        # Load the appropriate module
        if backend == Backend.NUMPY:
            self.module = np
        elif backend == Backend.CUPY:
            import cupy as cp
            self.module = cp
        elif backend == Backend.TORCH:
            import torch
            self.module = torch
        elif backend == Backend.JAX:
            import jax.numpy as jnp
            self.module = jnp

    def get_backend(self) -> Backend:
        """Get the currently active backend."""
        return self.active

    def is_available(self, backend: Union[str, Backend]) -> bool:
        """Check if a backend is available."""
        if isinstance(backend, str):
            backend = Backend(backend.lower())
        return self._available_backends.get(backend, False)

    def to_numpy(self, arr: ArrayType) -> np.ndarray:
        """Convert array to NumPy (transfer from GPU if needed).

        Parameters
        ----------
        arr : array-like
            Array to convert (CuPy, PyTorch, JAX, or NumPy).

        Returns
        -------
        numpy_array : ndarray
            NumPy array on CPU.
        """
        if isinstance(arr, np.ndarray):
            return arr

        # CuPy → NumPy
        if self.is_available(Backend.CUPY):
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)

        # PyTorch → NumPy
        if self.is_available(Backend.TORCH):
            import torch
            if isinstance(arr, torch.Tensor):
                return arr.detach().cpu().numpy()

        # JAX → NumPy
        if self.is_available(Backend.JAX):
            import jax.numpy as jnp
            if isinstance(arr, jnp.ndarray):
                return np.array(arr)

        raise TypeError(f"Unknown array type: {type(arr)}")

    def to_backend(self, arr: np.ndarray) -> ArrayType:
        """Convert NumPy array to active backend (transfer to GPU if needed).

        Parameters
        ----------
        arr : ndarray
            NumPy array to convert.

        Returns
        -------
        backend_array : array-like
            Array in active backend format.
        """
        if self.active == Backend.NUMPY:
            return arr

        if self.active == Backend.CUPY:
            import cupy as cp
            return cp.asarray(arr)

        if self.active == Backend.TORCH:
            import torch
            return torch.from_numpy(arr)

        if self.active == Backend.JAX:
            import jax.numpy as jnp
            return jnp.array(arr)

        return arr


# Global backend manager
_backend_manager = BackendManager()


# ============================================================================
# Public API - Backend control
# ============================================================================

def set_backend(backend: Union[str, Backend]) -> None:
    """Set the active computational backend."""
    _backend_manager.set_backend(backend)


def get_backend() -> Backend:
    """Get the currently active backend."""
    return _backend_manager.get_backend()


def get_backend_module():
    """Get the active backend module (np, cp, torch, etc.)."""
    return _backend_manager.module


def is_backend_available(backend: Union[str, Backend]) -> bool:
    """Check if a backend is available."""
    return _backend_manager.is_available(backend)


def to_numpy(arr: ArrayType) -> np.ndarray:
    """Convert array to NumPy (GPU → CPU transfer if needed)."""
    return _backend_manager.to_numpy(arr)


def to_backend(arr: np.ndarray) -> ArrayType:
    """Convert NumPy array to active backend (CPU → GPU transfer if needed)."""
    return _backend_manager.to_backend(arr)


# ============================================================================
# Unified Array API - Backend-agnostic operations
# ============================================================================

def zeros(shape: Tuple[int, ...], dtype=np.float64) -> ArrayType:
    """Create array of zeros (on active backend)."""
    xp = _backend_manager.module
    return xp.zeros(shape, dtype=dtype)


def ones(shape: Tuple[int, ...], dtype=np.float64) -> ArrayType:
    """Create array of ones (on active backend)."""
    xp = _backend_manager.module
    return xp.ones(shape, dtype=dtype)


def empty(shape: Tuple[int, ...], dtype=np.float64) -> ArrayType:
    """Create uninitialized array (on active backend)."""
    xp = _backend_manager.module
    return xp.empty(shape, dtype=dtype)


def zeros_like(arr: ArrayType) -> ArrayType:
    """Create zeros with same shape/dtype as input."""
    xp = _backend_manager.module
    return xp.zeros_like(arr)


def ones_like(arr: ArrayType) -> ArrayType:
    """Create ones with same shape/dtype as input."""
    xp = _backend_manager.module
    return xp.ones_like(arr)


def arange(start, stop=None, step=1, dtype=None) -> ArrayType:
    """Create evenly spaced array."""
    xp = _backend_manager.module
    return xp.arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num=50, dtype=None) -> ArrayType:
    """Create linearly spaced array."""
    xp = _backend_manager.module
    return xp.linspace(start, stop, num, dtype=dtype)


def asarray(arr, dtype=None) -> ArrayType:
    """Convert to array on active backend."""
    xp = _backend_manager.module
    return xp.asarray(arr, dtype=dtype)


def meshgrid(*xi, indexing='xy'):
    """Create coordinate matrices from coordinate vectors."""
    xp = _backend_manager.module
    return xp.meshgrid(*xi, indexing=indexing)


# ============================================================================
# Math operations (automatically backend-aware)
# ============================================================================

def exp(arr: ArrayType) -> ArrayType:
    """Element-wise exponential."""
    xp = _backend_manager.module
    return xp.exp(arr)


def log(arr: ArrayType) -> ArrayType:
    """Element-wise natural logarithm."""
    xp = _backend_manager.module
    return xp.log(arr)


def sqrt(arr: ArrayType) -> ArrayType:
    """Element-wise square root."""
    xp = _backend_manager.module
    return xp.sqrt(arr)


def sin(arr: ArrayType) -> ArrayType:
    """Element-wise sine."""
    xp = _backend_manager.module
    return xp.sin(arr)


def cos(arr: ArrayType) -> ArrayType:
    """Element-wise cosine."""
    xp = _backend_manager.module
    return xp.cos(arr)


def abs(arr: ArrayType) -> ArrayType:
    """Element-wise absolute value."""
    xp = _backend_manager.module
    return xp.abs(arr)


def maximum(x1: ArrayType, x2: ArrayType) -> ArrayType:
    """Element-wise maximum."""
    xp = _backend_manager.module
    return xp.maximum(x1, x2)


def minimum(x1: ArrayType, x2: ArrayType) -> ArrayType:
    """Element-wise minimum."""
    xp = _backend_manager.module
    return xp.minimum(x1, x2)


# ============================================================================
# Summary
# ============================================================================

def print_backend_info():
    """Print information about available backends."""
    print("="*70)
    print("  NeuroWave Backend Information")
    print("="*70)
    print(f"  Active backend: {_backend_manager.active.value}")
    print(f"\n  Available backends:")
    for backend, available in _backend_manager._available_backends.items():
        status = "✓" if available else "✗"
        print(f"    {status} {backend.value}")

    if _backend_manager.active == Backend.CUPY:
        try:
            import cupy as cp
            print(f"\n  GPU Device:")
            print(f"    {cp.cuda.Device()}")
            print(f"    Memory: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB total")
        except:
            pass

    print("="*70)


# ============================================================================
# Additional Array API (needed for GPU-aware solvers)
# ============================================================================

def sum(arr: ArrayType) -> ArrayType:
    """Sum of all elements."""
    xp = _backend_manager.module
    return xp.sum(arr)


def where(condition, x, y):
    """Element-wise conditional selection."""
    xp = _backend_manager.module
    return xp.where(condition, x, y)


def isfinite(arr: ArrayType) -> ArrayType:
    """Element-wise finiteness check."""
    xp = _backend_manager.module
    return xp.isfinite(arr)


def max_val(arr: ArrayType):
    """Maximum value in array."""
    xp = _backend_manager.module
    return xp.max(arr)


def copy(arr: ArrayType) -> ArrayType:
    """Create a copy of the array on the same device."""
    if hasattr(arr, 'copy'):
        return arr.copy()
    xp = _backend_manager.module
    return xp.array(arr)


def to_scalar(val) -> float:
    """Convert a backend scalar (e.g., CuPy 0-d array) to Python float."""
    if hasattr(val, 'item'):
        return float(val.item())
    return float(val)


def get_array_module(arr: ArrayType):
    """Get the array module (numpy or cupy) for a given array.

    Works like cupy.get_array_module but without requiring cupy import.
    """
    if isinstance(arr, np.ndarray):
        return np
    if _backend_manager.is_available(Backend.CUPY):
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp
    return np


def is_gpu_active() -> bool:
    """Check if the current backend is GPU-based."""
    return _backend_manager.active in (Backend.CUPY,)
