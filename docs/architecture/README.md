# Architecture Documentation

This directory contains the software architecture documentation for NeuroWave.

## Contents (Planned)

| Document | Description |
|----------|-------------|
| `overview.md` | High-level architecture overview |
| `module_interactions.md` | Module dependency and interaction diagram |
| `data_flow.md` | Data flow through the simulation pipeline |
| `memory_layout.md` | GPU memory layout and optimization strategy |
| `backend_design.md` | Multi-backend abstraction design |
| `plugin_system.md` | Extensibility and plugin architecture |

## Design Principles

1. **Modularity** — Each component is self-contained and independently testable
2. **Backend Agnostic** — Core logic separated from compute backend
3. **Memory Efficient** — Structure-of-Arrays for GPU coalescing
4. **Type Safe** — Full type annotations throughout
5. **Documented** — Every public API has docstrings with equations
