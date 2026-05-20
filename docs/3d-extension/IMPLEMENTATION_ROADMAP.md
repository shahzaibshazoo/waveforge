# WaveForge 3D — Implementation Roadmap

## Phase Dependency Graph

```
Phase 1 (Maxwell 3D Equations)
    │
    ▼
Phase 2 (Engine Architecture) ──────┐
    │                                │
    ├───────────────┐                │
    ▼               ▼                ▼
Phase 3          Phase 4          Phase 5
(Boundaries)     (Sources)        (Materials)
    │               │                │
    └───────────────┼────────────────┘
                    ▼
              Phase 6 (Examples)
                    │
                    ▼
              Phase 7 (Testing)
                    │
                    ▼
              Phase 8 (Optimization)
```

## Phase 1: 3D Maxwell's Equations

**Deliverable:** Reference document with all 6 discrete update equations

**Key decisions:**
- Sign conventions: match Taflove & Hagness (standard textbook)
- Index conventions: E at half-integer positions, H at half-integer complementary
- Material coefficients: Ca/Cb for E-field, Da/Db for H-field (lossy magnetic support)

**Validation:** Manual derivation cross-check against Yee 1966 paper

---

## Phase 2: 3D Engine (fdtd3d.py)

**Deliverable:** `src/core/fdtd3d.py` — working 3D time-stepper

**Implementation details:**

```python
class FDTD3D:
    def step(self):
        # 1. Boundary snapshot
        self._boundary.snapshot()
        
        # 2. Sources
        if self._sources:
            self._sources.step(fields_dict, self.steps_completed)
        
        # 3. H-field Faraday updates (3 equations)
        # Hx[..., :, :-1, :-1] += Dh * (dEy/dz - dEz/dy)
        Hx[..., :, :-1, :-1] += Dh * (
            (Ey[..., :, :, 1:] - Ey[..., :, :, :-1]) / dz
          - (Ez[..., :, 1:, :] - Ez[..., :, :-1, :]) / dy
        )
        # Hy[..., :-1, :, :-1] += Dh * (dEz/dx - dEx/dz)
        Hy[..., :-1, :, :-1] += Dh * (
            (Ez[..., 1:, :, :] - Ez[..., :-1, :, :]) / dx
          - (Ex[..., :, :, 1:] - Ex[..., :, :, :-1]) / dz
        )
        # Hz[..., :-1, :-1, :] += Dh * (dEx/dy - dEy/dx)
        Hz[..., :-1, :-1, :] += Dh * (
            (Ex[..., :, 1:, :] - Ex[..., :, :-1, :]) / dy
          - (Ey[..., 1:, :, :] - Ey[..., :-1, :, :]) / dx
        )
        
        # 4. Boundary apply
        self._boundary.apply(Ex, Ey, Ez, Hx, Hy, Hz)
        
        # 5. E-field Ampere updates (3 equations)
        # Ex[..., :, 1:, 1:] = Ca*Ex + Cb*(dHz/dy - dHy/dz)
        dHz_dy = (Hz[..., :, 1:, :] - Hz[..., :, :-1, :]) / dy
        dHy_dz = (Hy[..., :, :, 1:] - Hy[..., :, :, :-1]) / dz
        # Free-space path:
        Ex[..., :, 1:, 1:] += De * (dHz_dy[..., :, :, 1:] - dHy_dz[..., :, 1:, :])
        
        # Ey[..., 1:, :, 1:] = Ca*Ey + Cb*(dHx/dz - dHz/dx)
        dHx_dz = (Hx[..., :, :, 1:] - Hx[..., :, :, :-1]) / dz
        dHz_dx = (Hz[..., 1:, :, :] - Hz[..., :-1, :, :]) / dx
        Ey[..., 1:, :, 1:] += De * (dHx_dz[..., 1:, :, :] - dHz_dx[..., :, :, 1:])
        
        # Ez[..., 1:, 1:, :] = Ca*Ez + Cb*(dHy/dx - dHx/dy)
        dHy_dx = (Hy[..., 1:, :, :] - Hy[..., :-1, :, :]) / dx
        dHx_dy = (Hx[..., :, 1:, :] - Hx[..., :, :-1, :]) / dy
        Ez[..., 1:, 1:, :] += De * (dHy_dx[..., :, 1:, :] - dHx_dy[..., 1:, :, :])
        
        self.steps_completed += 1
```

**Key constraints:**
- No Python loops over spatial indices
- All operations use in-place tensor slice assignments
- Ellipsis prefix for batch dimension transparency
- Two paths: scalar free-space (De/Dh) and per-cell material (Ca/Cb/Da/Db)

**Minimum viable test:** Gaussian pulse at center of 64³ grid, verify spherical wavefront

---

## Phase 3: 3D Boundaries

**Deliverable:** `MurABC3D` class + `CPML3D` class in `boundaries.py`

**MurABC3D:**
- 6 faces × 2 tangential field components = 12 stored snapshot arrays
- Each face is a 2D slab: e.g., x_min face stores `(Ny, Nz)` arrays
- Apply formula: `f[face] = f_prev[interior] + C_mur * (f[interior] - f_prev[face])`
- C_mur = (c*dt - dn) / (c*dt + dn) where dn is the normal cell spacing

**CPML3D:**
- 12 auxiliary Ψ fields (only allocated in PML region, not full domain)
- Polynomial grading: σ(d) = σ_max × (d/D)^m, κ(d) = 1 + (κ_max-1)×(d/D)^m
- Recursive update: Ψ^{n+1} = b×Ψ^n + c×(∂f/∂n)
- Add correction: E += Cb × Ψ to standard curl update
- Parameters: PML_D=10, m=3, σ_max_ratio=0.75, κ_max=7, α=0.0

---

## Phase 4: 3D Sources

**Deliverable:** Extended source classes in `sources.py`

**New classes:**
1. `PlaneSource` — inject waveform on xy/xz/yz plane (replaces LineSource for 3D)
2. `TFSF` — Total-Field/Scattered-Field box for clean plane wave injection
3. `HertzianDipole` — infinitesimal current element (J_z, J_x, or J_y)
4. `ModulatedGaussian` — carrier × Gaussian envelope waveform

**TFSF implementation outline:**
```python
class TFSF:
    def __init__(self, grid, box_min, box_max, propagation_dir, polarization, waveform):
        # 1D auxiliary grid along propagation direction
        self._aux_E = torch.zeros(aux_size)
        self._aux_H = torch.zeros(aux_size)
    
    def step(self, fields, n):
        # Update 1D auxiliary FDTD
        self._update_aux_1d(n)
        # Correct tangential E and H on 6 faces of TFSF box
        self._correct_faces(fields)
```

---

## Phase 5: 3D Materials

**Deliverable:** Extended `MaterialMap` with 3D geometry primitives

**New methods:**
- `add_sphere(center, radius, material)`
- `add_cylinder(center, axis, radius, length, material)`
- `add_box(corner_min, corner_max, material)`
- `add_ellipsoid(center, semi_axes, material)`
- `add_voxel_data(data_3d, material_mapping)` — import from MRI/CT

**Material coefficients:**
- Ca, Cb: shape (Nx, Ny, Nz) — for E-field update
- Da, Db: shape (Nx, Ny, Nz) — for H-field update (when μᵣ ≠ 1)
- Dispersive: additional P_x, P_y, P_z per Lorentz pole (ADE method)

---

## Phase 6: 3D Examples

**Deliverable:** 5-7 working 3D simulation scripts

**Priority examples:**
1. `3d_01_free_space_pulse.py` — 64³, Gaussian pulse, xy/xz/yz slice visualization
2. `3d_02_mie_sphere.py` — 100³, dielectric sphere, compare RCS to Mie theory
3. `3d_03_patch_antenna.py` — 80×80×40, microstrip patch, far-field radiation pattern
4. `3d_04_breast_tumor_3d.py` — 100³, 3D MIMO array, volumetric DAS image
5. `3d_05_waveguide_te10.py` — 40×20×200, rectangular waveguide TE₁₀ mode
6. `3d_06_brain_hemorrhage_3d.py` — 80³, spherical head, 3D clot detection

---

## Phase 7: Testing & Validation

**Deliverable:** Test suite in `tests/test_fdtd3d.py`

**Critical tests:**
1. Energy conservation (lossless, no ABC): ΔU/U < 1e-6 over 1000 steps
2. Symmetry: centered pulse produces octant-symmetric fields
3. 2D equivalence: Nz=1 on 3D engine must match 2D engine output exactly
4. Convergence order: halve dx → error reduces by 4× (2nd-order spatial, 2nd-order temporal)
5. Mie RCS: numerical vs. analytical for eps_r=4 sphere at ka=1
6. PML reflection: incident plane wave, measure reflected power at -60dB

---

## Phase 8: GPU Performance

**Deliverable:** Optimized kernels + benchmark suite

**Optimization priority:**
1. `torch.compile()` on the step() method (automatic kernel fusion)
2. Eliminate temporary tensors: compute curl terms in-place where possible
3. Profile with `torch.profiler` to find bandwidth bottleneck
4. Mixed precision: fields in BF16, accumulation in FP32 (2× memory savings)
5. Benchmark sweep: 64³ → 512³, report Mcells/s and VRAM usage

**Target metrics:**
- 128³: >200 Mcells/s on T4
- 256³: >100 Mcells/s on T4
- Peak memory: <12 GB for 512³ (fits in T4 16 GB)

---

## Implementation Order

Each phase builds on the previous. The implementation sequence:

1. **Phase 1** → Reference equations (documentation only, no code)
2. **Phase 2** → `fdtd3d.py` with free-space scalar path only
3. **Phase 3** → `MurABC3D` first (simple), then `CPML3D` (complex)
4. **Phase 4** → `PointSource` works already; add `PlaneSource`, then `TFSF`
5. **Phase 5** → 3D geometry primitives, then dispersive materials
6. **Phase 6** → Start with free-space pulse, add complexity progressively
7. **Phase 7** → Tests written alongside each phase (TDD)
8. **Phase 8** → Profile first, optimize second (measure before changing)

## Design Principles

1. **No Python loops over spatial indices** — all stencil operations are tensor slices
2. **In-place operations only** — minimize GPU memory allocations in hot loop
3. **Ellipsis-first indexing** — batch dimension always supported transparently
4. **Two-path architecture** — scalar fast path (free space) + material path (per-cell Ca/Cb)
5. **Measure then optimize** — profile before applying any optimization
6. **2D backward compatibility** — existing 2D examples continue to work unchanged
7. **Progressive complexity** — each phase adds one capability, fully tested before moving on
