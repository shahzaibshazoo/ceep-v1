# Datasets

This directory stores simulation datasets, validation data, and tissue models.

## Structure

```
datasets/
├── generated/        # Auto-generated simulation datasets (gitignored)
├── validation/       # Analytical solutions for validation
├── tissue_models/    # Biological tissue dielectric data
└── phantoms/         # Numerical phantom definitions
```

## Important Note

Large generated datasets should NOT be committed to git.
Use the `datasets/generated/` directory which is in `.gitignore`.
