# Installation

## From PyPI

```bash
pip install neurowave
```

### With GPU Support

```bash
pip install neurowave[gpu]
```

This installs CuPy for CUDA 12.x. Make sure you have an NVIDIA GPU with CUDA drivers installed.

### With All Optional Dependencies

```bash
pip install neurowave[all]
```

## From Source

```bash
git clone https://github.com/shahzaibshazoo/ceep-v1.git
cd ceep-v1
pip install -e ".[dev]"
```

## Google Colab

No local GPU? Use Colab's free T4:

```python
!git clone https://github.com/shahzaibshazoo/ceep-v1.git
%cd ceep-v1
!pip install -e . -q
!pip install cupy-cuda12x -q
```

Make sure to select **Runtime > Change runtime type > T4 GPU**.

## Verify Installation

```python
import neurowave
from neurowave.core.backend import set_backend, print_backend_info

# Check available backends
print_backend_info()

# Test GPU (if available)
try:
    set_backend('cupy')
    print("GPU backend ready!")
except ImportError:
    print("GPU not available, using CPU (numpy)")
    set_backend('numpy')
```

## Requirements

- Python >= 3.9
- NumPy >= 1.24
- Matplotlib >= 3.7
- CuPy >= 12.0 (optional, for GPU)
- NVIDIA GPU with CUDA 12.x drivers (optional, for GPU)
