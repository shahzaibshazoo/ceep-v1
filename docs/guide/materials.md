# Materials & Tissues

## Simple Materials

Set permittivity and conductivity in rectangular or circular regions:

```python
# Rectangular region
solver.grid.set_material_region(
    x_start=80, x_end=120,
    y_start=80, y_end=120,
    eps_r=4.0,
    sigma_e=0.01
)

# Circular region
solver.grid.set_material_circle(
    center_x=100, center_y=100, radius=30,
    eps_r=9.0, sigma_e=0.1
)
```

## Dispersive Materials

For frequency-dependent materials, NeuroWave uses Auxiliary Differential Equations (ADE):

### Debye Model (Water, Biological Tissue)

```python
from neurowave.materials.dispersive import DebyePole

# Water at microwave frequencies
solver.grid.set_material_region(
    x_start=50, x_end=150, y_start=50, y_end=150,
    eps_r=80.0, eps_inf=4.0, sigma_e=0.0,
    debye_poles=[DebyePole(delta_eps=76.0, tau=9.4e-12)]
)
```

### Drude Model (Metals, Plasmas)

```python
from neurowave.materials.dispersive import DrudePole

# Gold at optical frequencies
solver.grid.set_material_region(
    ...,
    debye_poles=[DrudePole(omega_p=1.37e16, gamma=4.05e13)]
)
```

### Lorentz Model (Resonant Dielectrics)

```python
from neurowave.materials.dispersive import LorentzPole

# Optical resonance
solver.grid.set_material_region(
    ...,
    debye_poles=[LorentzPole(delta_eps=2.0, omega_0=4e15, delta=1e13)]
)
```

### Cole-Cole Model (Broadband Tissue Dispersion)

```python
from neurowave.materials.dispersive import ColeColePole

# Brain grey matter (broadband)
solver.grid.set_material_circle(
    center_x=100, center_y=100, radius=30,
    eps_r=50.0, eps_inf=4.0, sigma_e=0.7,
    debye_poles=[ColeColePole(delta_eps=46.0, tau=7.96e-12, alpha=0.1)]
)
```

## Tissue Database (Gabriel et al.)

The built-in database provides 4-term Cole-Cole parameters for 50+ biological tissues:

```python
from neurowave.materials.tissue_database import TissueDatabase

db = TissueDatabase()

# Get properties at a specific frequency
brain = db.get_tissue('brain_grey_matter', frequency=1e9)
print(f"eps_r={brain.eps_r:.1f}, sigma={brain.sigma:.2f} S/m")

blood = db.get_tissue('blood', frequency=1e9)
bone = db.get_tissue('bone_cortical', frequency=1e9)
skin = db.get_tissue('skin_dry', frequency=1e9)

# List all available tissues
for tissue_name in db.list_tissues():
    print(tissue_name)
```

### Available Tissues

Common tissues in the database:

| Tissue | eps_r @ 1 GHz | sigma @ 1 GHz |
|--------|---------------|---------------|
| Brain (grey matter) | ~52 | ~0.94 S/m |
| Brain (white matter) | ~39 | ~0.59 S/m |
| Blood | ~61 | ~1.58 S/m |
| Bone (cortical) | ~12 | ~0.16 S/m |
| Skin (dry) | ~41 | ~0.90 S/m |
| Muscle | ~55 | ~1.00 S/m |
| Fat | ~5.5 | ~0.05 S/m |
| CSF | ~69 | ~2.45 S/m |

## Multi-Pole Materials

Combine multiple poles for complex materials:

```python
from neurowave.materials.dispersive import DebyePole

# Two-pole Debye relaxation
solver.grid.set_material_circle(
    center_x=100, center_y=100, radius=20,
    eps_r=60.0, eps_inf=4.0, sigma_e=0.5,
    debye_poles=[
        DebyePole(delta_eps=40.0, tau=7.96e-12),  # Primary relaxation
        DebyePole(delta_eps=16.0, tau=15.9e-12),  # Secondary relaxation
    ]
)
```

The `DispersiveManager` supports up to 4 poles per region.
