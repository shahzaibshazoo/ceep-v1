# Solvers API

## FDTD2D

Single 2D FDTD solver supporting TMz and TEz modes.

```python
from neurowave.solvers.fdtd_2d import FDTD2D
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | SimulationConfig | Grid + timestep configuration |
| `sources` | list | Excitation sources |
| `boundaries` | list | Boundary conditions (CPML, Mur) |
| `record_field` | str | Field to record ('Ez', 'Hx', etc.) |
| `record_interval` | int | Snapshot every N steps |
| `probe_points` | list of (int,int) | Points to monitor |

### Methods

- `solver.run(num_steps=None)` — Run simulation
- `solver.step()` — Advance one timestep
- `solver.get_field(component)` — Get field array

### Attributes

- `solver.field_snapshots` — List of recorded field arrays
- `solver.probe_data` — Dict of time-series at probe points
- `solver.grid` — The Grid2D object (access materials, fields)

---

## BatchedFDTD2D

Runs multiple simulations in parallel on GPU. Each batch element has a different source location but shares the same material geometry.

```python
from neurowave.solvers.fdtd_2d_batched import BatchedFDTD2D
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `nx, ny` | int | Grid dimensions |
| `dx` | float | Grid spacing (meters) |
| `total_steps` | int | Number of timesteps |
| `cpml_thickness` | int | Absorbing boundary thickness |
| `source_positions` | list of (int,int) | TX antenna locations |
| `probe_positions` | list of (int,int) | RX antenna locations |
| `frequency` | float | Source frequency (Hz) |

### Methods

- `solver.set_material_circle(cx, cy, r, eps_r, sigma_e)` — Add circular material
- `solver.run()` — Run on GPU, returns S-matrix dict
- `solver.run_cpu()` — Run on CPU (for validation)

### Returns

`s_matrix[tx_idx][rx_idx]` — NumPy array of shape `(total_steps,)`

---

## FDTD3D

Full 3D Yee grid solver.

```python
from neurowave.solvers.fdtd_3d import FDTD3D
```

Same interface as FDTD2D but requires `GridConfig` with `nz > 1`.

---

## DFTMonitor

Extract frequency-domain fields during simulation:

```python
from neurowave.solvers.dft import DFTMonitor

monitor = DFTMonitor(
    frequencies=[1e9, 2e9],
    component='Ez',
    region=(slice(50,150), slice(50,150))
)

solver = FDTD2D(config=config, sources=[...], dft_monitors=[monitor])
solver.run()

# Access frequency-domain data
ez_1ghz = monitor.get_fields(0)  # Complex array at 1 GHz
```

---

## MultistaticSParameters

Extract S-parameters from time-domain probe data:

```python
from neurowave.solvers.s_params import MultistaticSParameters

sparam = MultistaticSParameters(
    probe_data=solver.probe_data,
    dt=config.dt,
    source_waveform=source_time_series
)

S21_freq = sparam.get_s_parameter(tx=0, rx=1, frequencies=np.linspace(0.5e9, 2e9, 100))
```
