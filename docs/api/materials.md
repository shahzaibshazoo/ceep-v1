# Materials API

## Dispersive Poles

```python
from neurowave.materials.dispersive import (
    DebyePole,
    DrudePole,
    LorentzPole,
    ColeColePole,
)
```

### DebyePole

| Parameter | Type | Description |
|-----------|------|-------------|
| `delta_eps` | float | Permittivity change (eps_s - eps_inf) |
| `tau` | float | Relaxation time (seconds) |

### DrudePole

| Parameter | Type | Description |
|-----------|------|-------------|
| `omega_p` | float | Plasma frequency (rad/s) |
| `gamma` | float | Collision frequency (rad/s) |

### LorentzPole

| Parameter | Type | Description |
|-----------|------|-------------|
| `delta_eps` | float | Oscillator strength |
| `omega_0` | float | Resonance frequency (rad/s) |
| `delta` | float | Damping coefficient (rad/s) |

### ColeColePole

| Parameter | Type | Description |
|-----------|------|-------------|
| `delta_eps` | float | Permittivity change |
| `tau` | float | Relaxation time (seconds) |
| `alpha` | float | Distribution parameter (0 to 1) |

---

## TissueDatabase

```python
from neurowave.materials.tissue_database import TissueDatabase

db = TissueDatabase()
```

### Methods

- `db.get_tissue(name, frequency)` — Get TissueProperties at frequency
- `db.list_tissues()` — List all available tissue names
- `db.get_cole_cole_poles(name)` — Get raw Cole-Cole parameters

### TissueProperties

Returned by `get_tissue()`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `eps_r` | float | Relative permittivity at frequency |
| `sigma` | float | Effective conductivity (S/m) |
| `eps_inf` | float | High-frequency permittivity |
| `cole_cole_poles` | list | Cole-Cole pole parameters |

---

## Grid Material Methods

```python
# Rectangular region
grid.set_material_region(
    x_start, x_end, y_start, y_end,
    eps_r=1.0, mu_r=1.0, sigma_e=0.0, sigma_m=0.0,
    eps_inf=None, debye_poles=None
)

# Circular region
grid.set_material_circle(
    center_x, center_y, radius,
    eps_r=1.0, mu_r=1.0, sigma_e=0.0, sigma_m=0.0,
    eps_inf=None, debye_poles=None
)
```
