# WaveForge 3D — Full Vector FDTD Extension

## Overview

WaveForge 3D extends the proven 2D TM-mode FDTD engine to a complete 3D electromagnetic solver. The 2D engine (Ex, Ey, Hz — 3 field updates per step) becomes a full Maxwell solver (Ex, Ey, Ez, Hx, Hy, Hz — 6 field updates per step, 12 curl derivatives total).

## Current State (2D Engine)

| Feature | Status |
|---------|--------|
| 2D TM FDTD (Ex, Ey, Hz) | Complete |
| Mur ABC (4 edges) | Complete |
| Per-cell materials (Ca/Cb) | Complete |
| Batch dimension support | Complete |
| GPU acceleration (PyTorch) | Complete |
| 10 simulation examples | Complete |
| Benchmarked: 1,481 Mcells/s (T4) | Complete |

## 3D Extension Phases

| Phase | Document | Scope |
|-------|----------|-------|
| 1 | [PHASE_1_MAXWELL_3D.md](PHASE_1_MAXWELL_3D.md) | Full 3D Maxwell's equations in discrete FDTD form |
| 2 | [PHASE_2_ENGINE_ARCHITECTURE.md](PHASE_2_ENGINE_ARCHITECTURE.md) | fdtd3d.py — the 3D time-stepper implementation |
| 3 | [PHASE_3_BOUNDARIES_PML.md](PHASE_3_BOUNDARIES_PML.md) | 3D Mur ABC (6 faces) + CPML absorbing boundaries |
| 4 | [PHASE_4_SOURCES_3D.md](PHASE_4_SOURCES_3D.md) | 3D sources: points, planes, TFSF, dipoles |
| 5 | [PHASE_5_MATERIALS_3D.md](PHASE_5_MATERIALS_3D.md) | 3D material system: volumes, dispersive, anisotropic |
| 6 | [PHASE_6_EXAMPLES_3D.md](PHASE_6_EXAMPLES_3D.md) | 3D simulation examples with physics validation |
| 7 | [PHASE_7_TESTING_VALIDATION.md](PHASE_7_TESTING_VALIDATION.md) | Unit tests, convergence, Mie scattering validation |
| 8 | [PHASE_8_PERFORMANCE_OPTIMIZATION.md](PHASE_8_PERFORMANCE_OPTIMIZATION.md) | GPU optimization, kernel fusion, memory bandwidth |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     WaveForge 3D                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ YeeGrid  │  │ FieldSet │  │ Sources  │  │Materials │   │
│  │ (3D)     │  │ 6 fields │  │ 3D types │  │ 3D vols  │   │
│  │ Nx×Ny×Nz │  │ Ex..Hz   │  │ TFSF     │  │ Ca/Cb/   │   │
│  │ CFL 3D   │  │ batch    │  │ Plane    │  │ Da/Db    │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │              │         │
│       └──────────────┼──────────────┼──────────────┘         │
│                      ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   FDTD3D Engine                       │   │
│  │                                                      │   │
│  │  for each step:                                      │   │
│  │    1. boundary.snapshot()     — save H on 6 faces    │   │
│  │    2. sources.step(fields,n)  — inject excitation    │   │
│  │    3. H-update (3 Faraday)    — Hx, Hy, Hz          │   │
│  │    4. boundary.apply()        — ABC/PML correction   │   │
│  │    5. E-update (3 Ampere)     — Ex, Ey, Ez          │   │
│  │    6. telemetry               — stability check      │   │
│  └──────────────────────────────────────────────────────┘   │
│                      ▼                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │               Boundary Conditions                    │   │
│  │  ┌────────────┐              ┌─────────────────┐     │   │
│  │  │ MurABC3D   │              │     CPML3D      │     │   │
│  │  │ 6 faces    │              │ 12 aux fields   │     │   │
│  │  │ -20dB      │              │ -60 to -80dB    │     │   │
│  │  │ fast/cheap │              │ gold standard   │     │   │
│  │  └────────────┘              └─────────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Differences: 2D vs 3D

| Aspect | 2D TM Mode | 3D Full Vector |
|--------|-----------|----------------|
| Field components | Ex, Ey, Hz (3) | Ex, Ey, Ez, Hx, Hy, Hz (6) |
| Curl derivatives | 4 per step | 12 per step |
| Field tensor shape | (Nx, Ny, 1) | (Nx, Ny, Nz) |
| Boundary faces | 4 edges | 6 faces |
| PML aux fields | 4 | 12 |
| Memory (128²/128³) | 0.8 MB | 50 MB |
| Memory (256²/256³) | 3.1 MB | 402 MB |
| Memory (512²/512³) | 12.6 MB | 3.2 GB |
| CFL denominator | √(1/Δx² + 1/Δy²) | √(1/Δx² + 1/Δy² + 1/Δz²) |
| Operations/cell/step | ~8 FLOP | ~24 FLOP |

## File Structure (Target)

```
src/core/
├── grid.py              # Already supports 3D (Nz parameter)
├── fields.py            # Already allocates 6 components
├── fdtd2d.py            # Existing 2D engine (unchanged)
├── fdtd3d.py            # NEW: 3D engine
├── boundaries.py        # Extend: MurABC3D, CPML3D classes
├── sources.py           # Extend: PlaneSource, TFSF
├── materials.py         # Extend: 3D geometry primitives
└── __init__.py
```

## GPU Memory Budget (T4 = 16 GB VRAM)

| Grid Size | Fields (6×FP32) | Materials | PML (D=10) | Total | Fits T4? |
|-----------|-----------------|-----------|------------|-------|----------|
| 64³      | 6.3 MB          | 4.2 MB    | 2.1 MB     | 13 MB | Yes |
| 128³     | 50 MB           | 34 MB     | 8.4 MB     | 92 MB | Yes |
| 256³     | 402 MB          | 268 MB    | 34 MB      | 704 MB | Yes |
| 512³     | 3.2 GB          | 2.1 GB    | 134 MB     | 5.5 GB | Yes |
| 768³     | 10.8 GB         | 7.2 GB    | 302 MB     | 18.3 GB | No |
| 512³ BF16| 1.6 GB          | 1.1 GB    | 67 MB      | 2.8 GB | Yes |

## Performance Targets

| Grid | Target Mcells/s | Steps/sec (128³) | Wall time for 1000 steps |
|------|-----------------|-------------------|--------------------------|
| 64³  | 500-800         | 1900-3050         | 0.3-0.5 s |
| 128³ | 200-400         | 95-190            | 5-10 s |
| 256³ | 100-200         | 6-12              | 83-167 s |
| 512³ | 50-100          | 0.4-0.7           | 24-42 min |

## Prerequisites

The existing 2D codebase already has infrastructure that 3D builds on:
- `YeeGrid` accepts `Nz` parameter and sets `is_3d = True` when `Nz > 1`
- `FieldSet` allocates all 6 field tensors with shape `(Nx, Ny, Nz)`
- `MaterialMap.build()` returns `(Nx, Ny, Nz)` coefficient tensors
- Ellipsis indexing in all operations handles batch dimensions transparently
- `torch.cuda.synchronize()` timing infrastructure already in place
