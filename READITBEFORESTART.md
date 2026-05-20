# WaveForge 3D — Master Context & Execution Guide

> **READ THIS ENTIRE FILE BEFORE DOING ANYTHING.**  
> This is your complete briefing. It tells you what exists, what to build, how to build it, and how to validate it.

---

## WHO YOU ARE

You are an elite computational electromagnetics engineer implementing a GPU-native 3D FDTD (Finite-Difference Time-Domain) electromagnetic simulator called **WaveForge**. You write production-quality PyTorch code that runs on NVIDIA GPUs. You understand Maxwell's equations, Yee grid staggering, CFL stability, absorbing boundary conditions, and GPU memory bandwidth optimization.

You work by **delegating to specialized agents** and **verifying their output**. You never write code without understanding the physics first. You never ship code without testing it.

---

## PROJECT STATE

### What Already Exists (2D Engine — COMPLETE, DO NOT MODIFY)

```
/home/zuu/GPU-MEEP/
├── src/core/
│   ├── grid.py          — YeeGrid class (supports Nz>1 for 3D already)
│   ├── fields.py        — FieldSet: 6 tensors (Ex,Ey,Ez,Hx,Hy,Hz) shape (Nx,Ny,Nz)
│   ├── fdtd2d.py        — 2D TM stepper (Ex,Ey,Hz only). DO NOT TOUCH.
│   ├── boundaries.py    — MurABC for 2D (4 edges, Hz only)
│   ├── sources.py       — GaussianPulse, RickerWavelet, PointSource, LineSource
│   ├── materials.py     — Material, MaterialMap, TISSUE_LIBRARY, Ca/Cb builder
│   └── __init__.py
├── src/visualization/
│   └── plot2d.py        — 2D field plotting
├── examples/            — 10 working 2D examples (01-10)
├── notebooks/           — Colab/Kaggle GPU benchmark notebooks
├── docs/3d-extension/   — 8 phase documents (YOUR REFERENCE)
├── tests/               — Existing 2D tests
└── READITBEFORESTART.md — THIS FILE
```

### 2D Engine Performance (Verified)
- Kaggle T4: **1,481 Mcells/s** at 1024×1024
- Colab T4: **350 Mcells/s** at 512×512
- Meep CPU: 16 Mcells/s → WaveForge is **92× faster**

### Backup Location
Full 2D backup at: `/home/zuu/GPU-MEEP-backup-2d/`

---

## WHAT YOU ARE BUILDING

### The 3D Extension

Extend the 2D TM-mode engine (3 field updates: Ex, Ey, Hz) to a **full 3D vector Maxwell solver** (6 field updates: Ex, Ey, Ez, Hx, Hy, Hz with 12 curl derivatives).

### The 6 Update Equations (MEMORIZE THESE)

**H-field (Faraday): H^{n+1/2} = H^{n-1/2} + (Δt/μ₀) × curl(E)**

```python
# Hx update: Hx += Dh * (dEy/dz - dEz/dy)
Hx[..., :, :-1, :-1] += Dh * (
    (Ey[..., :, :, 1:] - Ey[..., :, :, :-1]) / dz
  - (Ez[..., :, 1:, :] - Ez[..., :, :-1, :]) / dy
)

# Hy update: Hy += Dh * (dEz/dx - dEx/dz)
Hy[..., :-1, :, :-1] += Dh * (
    (Ez[..., 1:, :, :] - Ez[..., :-1, :, :]) / dx
  - (Ex[..., :, :, 1:] - Ex[..., :, :, :-1]) / dz
)

# Hz update: Hz += Dh * (dEx/dy - dEy/dx)
Hz[..., :-1, :-1, :] += Dh * (
    (Ex[..., :, 1:, :] - Ex[..., :, :-1, :]) / dy
  - (Ey[..., 1:, :, :] - Ey[..., :-1, :, :]) / dx
)
```

**E-field (Ampere): E^{n+1} = Ca×E^n + Cb × curl(H)**

```python
# Ex update: Ex = Ca*Ex + Cb*(dHz/dy - dHy/dz)
dHz_dy = (Hz[..., :, 1:, :] - Hz[..., :, :-1, :]) / dy
dHy_dz = (Hy[..., :, :, 1:] - Hy[..., :, :, :-1]) / dz
Ex[..., :, 1:, 1:] = Ca * Ex[..., :, 1:, 1:] + Cb * (dHz_dy[..., :, :, 1:] - dHy_dz[..., :, 1:, :])

# Ey update: Ey = Ca*Ey + Cb*(dHx/dz - dHz/dx)
dHx_dz = (Hx[..., :, :, 1:] - Hx[..., :, :, :-1]) / dz
dHz_dx = (Hz[..., 1:, :, :] - Hz[..., :-1, :, :]) / dx
Ey[..., 1:, :, 1:] = Ca * Ey[..., 1:, :, 1:] + Cb * (dHx_dz[..., 1:, :, :] - dHz_dx[..., :, :, 1:])

# Ez update: Ez = Ca*Ez + Cb*(dHy/dx - dHx/dy)
dHy_dx = (Hy[..., 1:, :, :] - Hy[..., :-1, :, :]) / dx
dHx_dy = (Hx[..., :, 1:, :] - Hx[..., :, :-1, :]) / dy
Ez[..., 1:, 1:, :] = Ca * Ez[..., 1:, 1:, :] + Cb * (dHy_dx[..., :, 1:, :] - dHx_dy[..., 1:, :, :])
```

### Free-Space Coefficients
```python
Dh = dt / MU0           # H-field coefficient (scalar)
De = dt / EPS0           # E-field coefficient (scalar, free-space only)
Ca = 1.0                 # free-space (no loss)
Cb = De                  # free-space
```

### Per-Cell Material Coefficients
```python
alpha = sigma * dt / (2 * EPS0 * eps_r)
Ca[i,j,k] = (1 - alpha) / (1 + alpha)      # decay factor [0,1]
Cb[i,j,k] = (dt / (EPS0 * eps_r)) / (1 + alpha)  # curl scaling
```

### CFL Stability (3D)
```python
dt_max = 1.0 / (C0 * math.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
dt = 0.99 * dt_max  # safety margin
```

---

## HOW TO IMPLEMENT (PHASE BY PHASE)

### Phase 2: Create `src/core/fdtd3d.py`

This is the FIRST code you write. Create the FDTD3D class:

```python
class FDTD3D:
    def __init__(self, grid, fields, boundary, sources=None, *, Ca=None, Cb=None, ...):
        ...
    
    def step(self):
        # 1. boundary.snapshot()
        # 2. sources.step(fields_dict, n)
        # 3. Hx, Hy, Hz Faraday updates (tensor slices, NO LOOPS)
        # 4. boundary.apply(...)
        # 5. Ex, Ey, Ez Ampere updates (two paths: scalar or Ca/Cb)
        # 6. self.steps_completed += 1
        # 7. stability check every n_check steps
    
    def run(self, n_steps, *, verbose=False):
        # torch.cuda.synchronize() before timing
        # step() loop
        # synchronize() after, compute mcells_per_second
```

**Critical rules:**
- ALL spatial operations are tensor slice assignments (no Python for-loops over i,j,k)
- Ellipsis `...` prefix on EVERY slice for batch dimension support
- In-place operations ONLY (`+=`, `-=`, or `tensor[slice] = expression`)
- Two code paths: scalar fast-path (Ca=None) and material path (Ca tensor)
- Timing uses `torch.cuda.synchronize()` — NEVER time individual steps

### Phase 3: Extend `src/core/boundaries.py`

Add `MurABC3D` class:
- 6 faces instead of 4 edges
- Store previous values of tangential E-field components on each face
- Apply: `E_face = E_prev_interior + C_mur * (E_interior - E_prev_face)`
- C_mur = (c*dt - dn) / (c*dt + dn) for each face normal

### Phase 4: Extend `src/core/sources.py`

Add:
- `PlaneSource(waveform, axis, position, component, grid, N_steps)` — injects on a plane
- Existing `PointSource` already works for 3D (uses k_idx)

### Phase 5: Extend `src/core/materials.py`

Add to `MaterialMap`:
- `add_sphere(center, radius, material)` — 3D sphere mask
- `add_cylinder(center, axis, radius, height, material)` — 3D cylinder
- `add_box(corner_min, corner_max, material)` — 3D rectangular region

### Phase 6: Create 3D examples in `examples/3d/`

Start with: `3d_01_free_space_pulse.py` — 64³ grid, Gaussian pulse at center, save 3 orthogonal slice PNGs.

### Phase 7: Tests in `tests/test_fdtd3d.py`

Critical tests:
1. Energy conservation (lossless, |ΔU/U| < 1e-5 over 1000 steps)
2. Symmetry (centered source → octant-symmetric fields)
3. 2D equivalence (Nz=1 on 3D engine must match 2D engine exactly)

### Phase 8: Optimization

Only after everything works correctly:
1. `torch.compile()` on step()
2. Profile with `torch.profiler`
3. Measure Mcells/s at 64³, 128³, 256³

---

## AGENT DEPLOYMENT STRATEGY

Use specialized agents for parallel work. Here's how to deploy them:

### Agent Roles

| Agent Type | Use For |
|------------|---------|
| `ruflo-core:coder` | Writing implementation code (fdtd3d.py, boundaries, sources) |
| `ruflo-testgen:tester` | Writing test files (test_fdtd3d.py, validation scripts) |
| `ruflo-core:reviewer` | Code review before merging (physics correctness, GPU patterns) |
| `ruflo-docs:docs-writer` | Documentation updates |
| `ruflo-swarm:architect` | Design decisions, API contracts |
| `general-purpose` | Research, exploration, debugging |

### Parallel Agent Pattern

When implementing a phase, deploy agents like this:

```
1. ARCHITECT agent: designs the interface/API (5 min)
2. Wait for architect output
3. CODER agents (2-3 in parallel):
   - Agent A: writes the main class
   - Agent B: writes helper functions
   - Agent C: writes the example script
4. Wait for all coders
5. TESTER agent: writes comprehensive tests
6. REVIEWER agent: reviews all code for physics errors
7. Fix any issues found by reviewer
8. Run tests to verify
```

### Agent Prompting Rules

When briefing an agent:
1. **Give it the equations** — paste the exact tensor slice notation
2. **Give it the file paths** — tell it exactly where to read and write
3. **Give it constraints** — "no Python loops", "use ellipsis indexing", "in-place only"
4. **Give it the test** — describe how to verify correctness
5. **Give it context** — explain what already exists (grid.py, fields.py interfaces)

### Example Agent Prompt (for fdtd3d.py implementation):

```
Implement src/core/fdtd3d.py. 

Read these files first:
- src/core/fdtd2d.py (follow this exact pattern)
- src/core/grid.py (YeeGrid interface)
- src/core/fields.py (FieldSet interface)

Create class FDTD3D that implements full 3D Maxwell updates.
The step() method must:
1. Call boundary.snapshot()
2. Call sources.step(fields_dict, n) 
3. Update Hx, Hy, Hz using Faraday (exact slices below)
4. Call boundary.apply(...)
5. Update Ex, Ey, Ez using Ampere (two paths: free-space scalar and material Ca/Cb)

[paste the 6 equations from above]

Rules:
- No Python for-loops over spatial indices
- All slices use ... prefix for batch dimension
- In-place tensor operations only
- Two paths: if Ca is None use scalar De, else use Ca/Cb tensors
- Stability check every n_check steps on all 6 components
- run() method with torch.cuda.synchronize() timing
```

---

## CRITICAL PHYSICS RULES (LEARNED FROM 2D DEVELOPMENT)

These are hard-won lessons. Violating any of these WILL produce wrong results:

1. **Faraday sign**: `Hz += Dh * (dEx/dy - dEy/dx)` — the order is Ex/dy MINUS Ey/dx. Getting this backwards makes waves implode instead of propagate.

2. **Staggered grid indexing**: H-fields are updated at INTERIOR cells only (`:-1` slices). E-fields start from index 1 (`1:` slices). The asymmetry is physical (Yee staggering).

3. **Material path must multiply FIRST**: `E = Ca*E + Cb*curl` not `E += Cb*curl`. The Ca factor decays the existing field (conductivity loss). Omitting it = lossless.

4. **CFL violation = explosion**: If dt is too large, fields go to infinity within ~50 steps. Always use `dt = 0.99 * dt_max`. NEVER increase dt to speed things up.

5. **Timing must use cuda.synchronize()**: Without it, perf_counter() only measures CPU dispatch time (~1μs), not actual GPU compute time (~1ms). Results will be 1000× too optimistic.

6. **Ellipsis EVERYWHERE**: Every tensor slice must start with `...` to support batch dimension `(B, Nx, Ny, Nz)`. This is non-negotiable.

7. **No torch.roll()**: It allocates a new tensor. Use slice indexing: `F[..., 1:, :, :] - F[..., :-1, :, :]` for differences.

8. **No tensor allocation in step()**: Pre-allocate everything in __init__. The step() hot loop must be allocation-free for GPU efficiency.

9. **Mur coefficient range**: C_mur = (c*dt - dx)/(c*dt + dx) must satisfy -1 < C_mur < 0. If it's positive, the ABC amplifies instead of absorbs.

10. **Source injection is ADDITIVE (soft source)**: `field[..., i, j, k] += waveform(t)`, never `= waveform(t)` (hard source creates artificial reflections).

---

## VALIDATION CHECKPOINTS

After each phase, verify with these tests:

### After Phase 2 (Engine):
```python
# Test: Gaussian pulse in 64³ free-space, check energy conservation
grid = YeeGrid(64, 64, 64, dx=1e-3, dy=1e-3, dz=1e-3, device='cuda')
fields = FieldSet(grid)
# ... setup source, boundary, run 200 steps
# Assert: total_energy at step 200 ≈ total_energy at step 50 (within 5%)
# Assert: no NaN/Inf
# Assert: field max < 1e6 (not diverging)
```

### After Phase 3 (Boundaries):
```python
# Test: pulse exits cleanly, no reflection
# Run 500 steps, measure field magnitude at center after pulse has left
# Assert: residual < 1e-3 * peak (for Mur ABC, ~-20dB)
```

### After Phase 5 (Materials):
```python
# Test: dielectric sphere creates visible scattering
# Assert: field pattern is NOT identical to free-space (material has effect)
# Assert: field inside sphere has lower amplitude (dielectric loading)
```

---

## FILE CREATION ORDER

Execute in this exact order:

```
1. src/core/fdtd3d.py              ← Phase 2 (THE CORE)
2. src/core/boundaries.py          ← Phase 3 (ADD MurABC3D class, keep existing 2D code)
3. src/core/sources.py             ← Phase 4 (ADD PlaneSource class, keep existing)
4. src/core/materials.py           ← Phase 5 (ADD sphere/cylinder/box methods, keep existing)
5. examples/3d/3d_01_free_space.py ← Phase 6 (first working 3D example)
6. tests/test_fdtd3d.py            ← Phase 7 (validation suite)
7. Optimize step() with profiling  ← Phase 8 (only after correctness proven)
```

---

## REFERENCE DOCUMENTATION

Read these for detailed specifications:

```
docs/3d-extension/
├── README.md                          ← Architecture overview + memory budgets
├── PHASE_1_MAXWELL_3D.md              ← All equations derived
├── PHASE_2_ENGINE_ARCHITECTURE.md     ← fdtd3d.py full specification
├── PHASE_3_BOUNDARIES_PML.md          ← MurABC3D + CPML3D design
├── PHASE_4_SOURCES_3D.md              ← PlaneSource, TFSF, dipoles
├── PHASE_5_MATERIALS_3D.md            ← 3D geometry + dispersive
├── PHASE_6_EXAMPLES_3D.md             ← 7 example specifications
├── PHASE_7_TESTING_VALIDATION.md      ← Test suite design
├── PHASE_8_PERFORMANCE_OPTIMIZATION.md ← GPU optimization guide
├── IMPLEMENTATION_ROADMAP.md          ← Phase dependency graph
└── PHASE_TRANSITIONS.md              ← Gate criteria (checklists)
```

---

## QUICK REFERENCE: EXISTING INTERFACES

### YeeGrid (grid.py)
```python
grid = YeeGrid(Nx, Ny, Nz=1, dx=1e-3, dy=1e-3, dz=None, device='cuda')
grid.Nx, grid.Ny, grid.Nz  # dimensions
grid.dx, grid.dy, grid.dz  # cell spacing (dz defaults to dx if Nz>1)
grid.dt                     # CFL-compliant time step
grid.device                 # torch device
grid.is_3d                  # True when Nz > 1
grid.shape                  # (Nx, Ny, Nz)
grid.cell_volume            # dx * dy * dz (for energy in Joules)
```

### FieldSet (fields.py)
```python
fields = FieldSet(grid)
fields.Ex  # shape (Nx, Ny, Nz) torch.float32 on grid.device
fields.Ey  # same
fields.Ez  # same
fields.Hx  # same
fields.Hy  # same
fields.Hz  # same
fields.total_energy()  # returns float (Joules)
fields.zero_()         # reset all to zero
```

### MaterialMap (materials.py)
```python
mm = MaterialMap(grid, default=Material('air', eps_r=1.0, sigma=0.0))
mm.add_circle(center, radius, material)       # 2D circle
mm.add_rectangle(x_range, y_range, material)  # 2D rect
# You will ADD: add_sphere, add_cylinder, add_box
Ca, Cb = mm.build()  # returns (Nx, Ny, Nz) tensors on grid.device
```

### Sources (sources.py)
```python
wv = RickerWavelet(amplitude=1.0, peak_freq=1e9)
src = PointSource(wv, i, j, k, component='Ez', grid=grid, N_steps=1000)
sources = SourceCollection([src])
sources.step({"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}, step_n)
```

---

## PERFORMANCE TARGETS

| Grid | Target Mcells/s (T4) | Max VRAM |
|------|---------------------|----------|
| 64³  | 500+ | 13 MB |
| 128³ | 200+ | 92 MB |
| 256³ | 100+ | 704 MB |
| 512³ | 50+  | 5.5 GB |

---

## DO NOT

- ❌ Modify `fdtd2d.py` or any existing 2D code
- ❌ Use Python for-loops over spatial indices (i, j, k)
- ❌ Use `torch.roll()` (allocates new tensor)
- ❌ Allocate tensors inside `step()` (pre-allocate in `__init__`)
- ❌ Time individual steps (use `run()` with `cuda.synchronize()`)
- ❌ Skip the ellipsis `...` in tensor slices
- ❌ Forget the material path (Ca*E + Cb*curl, not just E += De*curl)
- ❌ Use hard sources (`field = value`) instead of soft (`field += value`)
- ❌ Proceed to next phase without passing the gate criteria
- ❌ Optimize before correctness is proven

## DO

- ✅ Read `docs/3d-extension/PHASE_2_ENGINE_ARCHITECTURE.md` before writing fdtd3d.py
- ✅ Follow the EXACT same code patterns as fdtd2d.py
- ✅ Use agents in parallel for independent work
- ✅ Test energy conservation immediately after Phase 2
- ✅ Run existing 2D examples to verify no regression
- ✅ Profile before optimizing (Phase 8)
- ✅ Check all 6 components in stability checks
- ✅ Support batch dimension from day one

---

## START HERE

1. Read `docs/3d-extension/PHASE_2_ENGINE_ARCHITECTURE.md`
2. Read `src/core/fdtd2d.py` (your template)
3. Deploy a coder agent to implement `src/core/fdtd3d.py`
4. Deploy a tester agent to write `tests/test_fdtd3d.py`
5. Run tests, fix failures
6. Proceed to Phase 3

**Go.**
