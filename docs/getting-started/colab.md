# Running on Google Colab

Google Colab provides free T4 GPUs — perfect for running NeuroWave simulations without local hardware.

## Setup

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. Select **Runtime > Change runtime type > T4 GPU**
4. Run the setup cell:

```python
# Cell 1: Install
!git clone https://github.com/shahzaibshazoo/ceep-v1.git
%cd ceep-v1
!pip install -e . -q
!pip install cupy-cuda12x -q
print("Ready!")
```

## Run a Simulation

```python
# Cell 2: 16-Antenna Batched Simulation
import sys
sys.path.insert(0, 'src')
%run colab_cells/cell11_batched_antenna_array.py
```

## Available Colab Cells

Pre-built cells are in the `colab_cells/` directory:

| File | What it does |
|------|-------------|
| `cell1_setup.py` | Clone and install |
| `cell5_gpu_cpu_comparison.py` | Basic GPU vs CPU benchmark |
| `cell10_antenna_array_16x16.py` | Sequential 16-antenna (shows why batching matters) |
| `cell11_batched_antenna_array.py` | Batched 16-antenna (proper GPU usage) |

## Tips

- **Always restart runtime** after `git pull` — Python caches imported modules
- **Use batched solver** for antenna arrays — sequential per-antenna is slower than CPU on small grids
- **Check GPU memory** with `!nvidia-smi` if you hit OOM errors
- For grids larger than 1000x1000, the T4's 16GB is sufficient for single simulations but batched may need batching in chunks

## Troubleshooting

**CuPy not found:**
```python
!pip install cupy-cuda12x -q
```

**Module not found after git pull:**
```python
import sys
mods = [k for k in sys.modules if k.startswith('neurowave')]
for m in mods:
    del sys.modules[m]
```

**Out of memory:**
Reduce grid size or batch size. For 16 antennas on 500x500:
```python
# ~370MB GPU memory — fits on T4
solver = BatchedFDTD2D(nx=500, ny=500, ...)
```
