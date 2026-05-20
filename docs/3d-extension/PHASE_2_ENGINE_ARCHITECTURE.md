# Phase 2: 3D Engine Architecture — fdtd3d.py

## Overview

This document specifies the design and implementation of the 3D FDTD time-stepper engine for WaveForge, extending the proven 2D architecture (`fdtd2d.py`) to full 3D volumetric simulations. The `FDTD3D` class maintains the core design principles of the 2D engine — vectorized tensor operations, zero Python loops in the hot path, and efficient GPU memory utilization — while scaling to handle all six electromagnetic field components in three spatial dimensions.

The primary goal is numerical correctness, GPU throughput (>100 Mcells/s on T4 for 128³ grids), and seamless integration with the existing WaveForge infrastructure (sources, boundaries, diagnostics).

---

## Design Philosophy: Consistency with FDTD2D

### Inheritance and API Surface

```python
class FDTD3D(FDTD2D):
    """3D full-vector FDTD time-stepper using Yee leapfrog integration."""
    pass
```

The `FDTD3D` class inherits the public interface from `FDTD2D`:

- **Construction:** `grid: YeeGrid`, `fields: FieldSet`, `boundary: MurABC`, `sources: Optional[SourceCollection]`
- **Core method:** `step()` — advance all fields by one time step in-place
- **Driver:** `run(n_steps, verbose=False)` — execute multiple steps with telemetry
- **Reset:** `reset()` — zero fields and telemetry
- **Properties:** `grid`, `fields`, `dt`, `time`, `throughput`

Differences appear only in:
1. The shape and count of field tensors (6 fields instead of 3)
2. The curl update equations (three H-components and three E-components)
3. Boundary condition application (6 faces instead of 4 edges)

### Tensor Layout Consistency

**2D layout (FDTD2D):**
```
Ex, Ey, Hz: shape (Nx, Ny, 1) or (B, Nx, Ny, 1)
```

**3D layout (FDTD3D):**
```
Ex, Ey, Ez, Hx, Hy, Hz: shape (Nx, Ny, Nz) or (B, Nx, Ny, Nz)
```

where Nz > 1 for true 3D. The leading ellipsis `...` absorbs optional batch dimension B, maintaining code consistency:

```python
Hz[..., :-1, :-1, :] += ...  # Works for both 2D and 3D
Hz[..., :-1, :-1] += ...      # Works only for 3D (no Z dimension)
```

### Material Path Unification

Like FDTD2D, FDTD3D supports two paths:

1. **Free-space fast path** (Ca = None, Cb = None):
   - Scalar `De` coefficient applied to all E-components
   - No tensor loads for material coefficients
   - Saves 8 × Nx × Ny × Nz × 4 bytes of DRAM bandwidth per step

2. **Material path** (Ca/Cb tensors provided):
   - Per-cell Ca and Cb tensors applied to all E-components
   - Optional Da/Db for magnetic loss (Phase 5)
   - Full flexibility for inhomogeneous materials

---

## Core Data Structures

### Grid and Field Initialization

The 3D simulation requires the same `YeeGrid` and `FieldSet` infrastructure as 2D, with Nz > 1:

```python
from src.core.grid import YeeGrid
from src.core.fields import FieldSet

# Create 3D Yee grid: 128 × 128 × 128 cells, 5 nm spacing
grid_3d = YeeGrid(
    Nx=128, Ny=128, Nz=128,
    dx=5e-9, dy=5e-9, dz=5e-9,
    num_batch=1,
    dtype=torch.float32,
    device="cuda:0"
)

# Allocate all 6 field components
fields_3d = FieldSet(grid_3d)
# fields_3d.Ex, .Ey, .Ez, .Hx, .Hy, .Hz all shape (1, 128, 128, 128)
```

### Boundary Condition Setup

The `MurABC` boundary handler operates on all six field faces (x_min, x_max, y_min, y_max, z_min, z_max):

```python
from src.core.boundaries import MurABC

boundary_3d = MurABC(
    grid=grid_3d,
    num_faces=6,  # x_min, x_max, y_min, y_max, z_min, z_max
    order=1       # First-order Mur ABC
)
```

All E-components are stored; H-components are snapshotted on entry. The boundary applies the Mur ABC to all six faces.

---

## Material Coefficient Tensors

### Precomputation

For inhomogeneous materials, precompute Ca and Cb once:

```python
# Example: assign two materials (free space + dielectric)
Ca = torch.ones((Nx, Ny, Nz), dtype=torch.float32, device=device)
Cb = torch.ones((Nx, Ny, Nz), dtype=torch.float32, device=device)

# Free space: Ca = 1, Cb = De (at time of construction)
# Dielectric (eps_r=4, sigma=0):
#   Ca = 1, Cb = De / eps_r
dielectric_mask = (grid.x > 1e-6) & (grid.x < 2e-6)
Cb[dielectric_mask] = De / 4.0

stepper = FDTD3D(
    grid=grid_3d,
    fields=fields_3d,
    boundary=boundary_3d,
    sources=None,
    Ca=Ca,
    Cb=Cb,
    stability_threshold=1e10,
    n_check=100
)
```

### Shape and Application

All material tensors are shape (Nx, Ny, Nz):

```python
# Material coefficients (tensors, not scalars)
self._Ca: Optional[torch.Tensor] = Ca  # Shape: (Nx, Ny, Nz)
self._Cb: Optional[torch.Tensor] = Cb  # Shape: (Nx, Ny, Nz)

# For magnetic materials (Phase 5):
self._Da: Optional[torch.Tensor] = Da  # Shape: (Nx, Ny, Nz)
self._Db: Optional[torch.Tensor] = Db  # Shape: (Nx, Ny, Nz)
```

---

## The Leapfrog Time-Stepping Sequence

### Overview

Each call to `step()` advances the fields by one time increment using the following order:

```
n ← steps_completed

1. boundary.snapshot()        → save all 3 H-components on 6 faces
2. sources.step(fields, n)    → inject soft sources into E-field
3. Hx update (Faraday)        → vectorized tensor slices
4. Hy update (Faraday)        → vectorized tensor slices
5. Hz update (Faraday)        → vectorized tensor slices
6. boundary.apply()           → ABC/PML on all 6 faces (H-fields only)
7. Ex update (Ampere)         → vectorized tensor slices
8. Ey update (Ampere)         → vectorized tensor slices
9. Ez update (Ampere)         → vectorized tensor slices
10. steps_completed += 1       → advance counter
11. stability check (if % n_check == 0)
```

This order ensures:
- Magnetic updates use the *current* E-field (half-step leapfrog assumption)
- Boundary snapshots preserve boundary-adjacent H-values before updating them
- Electric updates use the *new* H-field (leapfrog consistency)
- Stability threshold is checked periodically to catch divergence early

---

## Magnetic Field Updates (Faraday's Law)

All H-field updates are vectorized with no Python loops. Derivatives are computed via tensor slicing.

### Hx Update

**Equation:**
$$H_x^{n+1/2} = H_x^{n-1/2} + D_h \left(\frac{\partial E_z}{\partial y} - \frac{\partial E_y}{\partial z}\right)$$

**Tensor Implementation:**
```python
Dh = self._Dh  # dt / mu0
Ey = self._fields.Ey
Ez = self._fields.Ez
Hx = self._fields.Hx

dy = self._dy
dz = self._dz

# Hx lives at [i, j+1/2, k+1/2]
# ∂Ez/∂y: Ez[i, j+1, k+1/2] - Ez[i, j, k+1/2]
# ∂Ey/∂z: Ey[i, j+1/2, k+1] - Ey[i, j+1/2, k]

Hx[..., :, :-1, :-1] += Dh * (
    (Ez[..., :, 1:, :-1] - Ez[..., :, :-1, :-1]) / dy
    - (Ey[..., :, :-1, 1:] - Ey[..., :, :-1, :-1]) / dz
)
```

**Tensor slice breakdown:**
- `Hx[..., :, :-1, :-1]`: All H-components except the top-right boundary
- `Ez[..., :, 1:, :-1]` vs `Ez[..., :, :-1, :-1]`: Staggered y-derivative
- `Ey[..., :, :-1, 1:]` vs `Ey[..., :, :-1, :-1]`: Staggered z-derivative

### Hy Update

**Equation:**
$$H_y^{n+1/2} = H_y^{n-1/2} + D_h \left(\frac{\partial E_x}{\partial z} - \frac{\partial E_z}{\partial x}\right)$$

**Tensor Implementation:**
```python
Ex = self._fields.Ex

# Hy lives at [i+1/2, j, k+1/2]
# ∂Ex/∂z: Ex[i+1/2, j, k+1] - Ex[i+1/2, j, k]
# ∂Ez/∂x: Ez[i+1, j, k+1/2] - Ez[i, j, k+1/2]

dx = self._dx

Hy[..., :-1, :, :-1] += Dh * (
    (Ex[..., :-1, :, 1:] - Ex[..., :-1, :, :-1]) / dz
    - (Ez[..., 1:, :, :-1] - Ez[..., :-1, :, :-1]) / dx
)
```

### Hz Update

**Equation:**
$$H_z^{n+1/2} = H_z^{n-1/2} + D_h \left(\frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y}\right)$$

**Tensor Implementation:**
```python
# Hz lives at [i+1/2, j+1/2, k]
# ∂Ey/∂x: Ey[i+1, j+1/2, k] - Ey[i, j+1/2, k]
# ∂Ex/∂y: Ex[i+1/2, j+1, k] - Ex[i+1/2, j, k]

Hz[..., :-1, :-1, :] += Dh * (
    (Ey[..., 1:, :-1, :] - Ey[..., :-1, :-1, :]) / dx
    - (Ex[..., :-1, 1:, :] - Ex[..., :-1, :-1, :]) / dy
)
```

---

## Electric Field Updates (Ampere-Maxwell Law)

### Ex Update

**Equation (free space):**
$$E_x^{n+1} = E_x^n + D_e \left(\frac{\partial H_z}{\partial y} - \frac{\partial H_y}{\partial z}\right)$$

**Equation (with materials):**
$$E_x^{n+1} = C_a E_x^n + C_b \left(\frac{\partial H_z}{\partial y} - \frac{\partial H_y}{\partial z}\right)$$

**Tensor Implementation (free space):**
```python
De = self._De  # dt / eps0
Hx = self._fields.Hx
Hy = self._fields.Hy
Hz = self._fields.Hz
Ex = self._fields.Ex

# Ex lives at [i+1/2, j, k]
# ∂Hz/∂y: Hz[i+1/2, j+1/2, k] - Hz[i+1/2, j-1/2, k]
# ∂Hy/∂z: Hy[i+1/2, j, k+1/2] - Hy[i+1/2, j, k-1/2]

Ex[..., :, 1:, 1:] += De * (
    (Hz[..., :, 1:, :-1] - Hz[..., :, :-1, :-1]) / dy
    - (Hy[..., :, 1:, 1:] - Hy[..., :, 1:, :-1]) / dz
)
```

**Tensor Implementation (with materials):**
```python
if self._has_materials:
    dHz_dy = (Hz[..., :, 1:, :-1] - Hz[..., :, :-1, :-1]) / dy
    dHy_dz = (Hy[..., :, 1:, 1:] - Hy[..., :, 1:, :-1]) / dz
    
    Ex[..., :, 1:, 1:] = (
        self._Ca[..., :, 1:, 1:] * Ex[..., :, 1:, 1:]
        + self._Cb[..., :, 1:, 1:] * (dHz_dy - dHy_dz)
    )
```

### Ey Update

**Equation (free space):**
$$E_y^{n+1} = E_y^n + D_e \left(\frac{\partial H_x}{\partial z} - \frac{\partial H_z}{\partial x}\right)$$

**Tensor Implementation (free space):**
```python
# Ey lives at [i, j+1/2, k]
# ∂Hx/∂z: Hx[i, j+1/2, k+1/2] - Hx[i, j+1/2, k-1/2]
# ∂Hz/∂x: Hz[i+1/2, j+1/2, k] - Hz[i-1/2, j+1/2, k]

Ey[..., 1:, :, 1:] += De * (
    (Hx[..., 1:, :, 1:] - Hx[..., 1:, :, :-1]) / dz
    - (Hz[..., 1:, :, :-1] - Hz[..., :-1, :, :-1]) / dx
)
```

### Ez Update

**Equation (free space):**
$$E_z^{n+1} = E_z^n + D_e \left(\frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y}\right)$$

**Tensor Implementation (free space):**
```python
# Ez lives at [i, j, k+1/2]
# ∂Hy/∂x: Hy[i+1/2, j, k+1/2] - Hy[i-1/2, j, k+1/2]
# ∂Hx/∂y: Hx[i, j+1/2, k+1/2] - Hx[i, j-1/2, k+1/2]

Ez[..., 1:, 1:, :] += De * (
    (Hy[..., 1:, :, :] - Hy[..., :-1, :, :]) / dx
    - (Hx[..., :, 1:, :] - Hx[..., :, :-1, :]) / dy
)
```

---

## Vectorized Update Sequence in Code

The complete `step()` method follows this pattern:

```python
def step(self) -> None:
    """Advance all fields by one time step in-place."""
    n = self.steps_completed
    
    # Unpack local references — views, not copies
    Ex, Ey, Ez = self._fields.Ex, self._fields.Ey, self._fields.Ez
    Hx, Hy, Hz = self._fields.Hx, self._fields.Hy, self._fields.Hz
    Dh = self._Dh
    De = self._De
    dx, dy, dz = self._dx, self._dy, self._dz
    
    # Step 1: snapshot boundary H-fields
    self._boundary.snapshot()
    
    # Step 2: inject sources at step n
    if self._sources is not None:
        self._sources.step(
            {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz},
            n
        )
    
    # Steps 3-5: Faraday updates (all 3 H-components)
    Hx[..., :, :-1, :-1] += Dh * (
        (Ez[..., :, 1:, :-1] - Ez[..., :, :-1, :-1]) / dy
        - (Ey[..., :, :-1, 1:] - Ey[..., :, :-1, :-1]) / dz
    )
    
    Hy[..., :-1, :, :-1] += Dh * (
        (Ex[..., :-1, :, 1:] - Ex[..., :-1, :, :-1]) / dz
        - (Ez[..., 1:, :, :-1] - Ez[..., :-1, :, :-1]) / dx
    )
    
    Hz[..., :-1, :-1, :] += Dh * (
        (Ey[..., 1:, :-1, :] - Ey[..., :-1, :-1, :]) / dx
        - (Ex[..., :-1, 1:, :] - Ex[..., :-1, :-1, :]) / dy
    )
    
    # Step 6: apply boundary (ABC/PML on all 6 H-faces)
    self._boundary.apply({"Hx": Hx, "Hy": Hy, "Hz": Hz})
    
    # Steps 7-9: Ampere updates (all 3 E-components)
    if not self._has_materials:
        # Free-space fast path
        Ex[..., :, 1:, 1:] += De * (
            (Hz[..., :, 1:, :-1] - Hz[..., :, :-1, :-1]) / dy
            - (Hy[..., :, 1:, 1:] - Hy[..., :, 1:, :-1]) / dz
        )
        
        Ey[..., 1:, :, 1:] += De * (
            (Hx[..., 1:, :, 1:] - Hx[..., 1:, :, :-1]) / dz
            - (Hz[..., 1:, :, :-1] - Hz[..., :-1, :, :-1]) / dx
        )
        
        Ez[..., 1:, 1:, :] += De * (
            (Hy[..., 1:, :, :] - Hy[..., :-1, :, :]) / dx
            - (Hx[..., :, 1:, :] - Hx[..., :, :-1, :]) / dy
        )
    else:
        # Material path: per-cell Ca/Cb tensors
        dHz_dy = (Hz[..., :, 1:, :-1] - Hz[..., :, :-1, :-1]) / dy
        dHy_dz = (Hy[..., :, 1:, 1:] - Hy[..., :, 1:, :-1]) / dz
        Ex[..., :, 1:, 1:] = (
            self._Ca[..., :, 1:, 1:] * Ex[..., :, 1:, 1:]
            + self._Cb[..., :, 1:, 1:] * (dHz_dy - dHy_dz)
        )
        
        dHx_dz = (Hx[..., 1:, :, 1:] - Hx[..., 1:, :, :-1]) / dz
        dHz_dx = (Hz[..., 1:, :, :-1] - Hz[..., :-1, :, :-1]) / dx
        Ey[..., 1:, :, 1:] = (
            self._Ca[..., 1:, :, 1:] * Ey[..., 1:, :, 1:]
            + self._Cb[..., 1:, :, 1:] * (dHx_dz - dHz_dx)
        )
        
        dHy_dx = (Hy[..., 1:, :, :] - Hy[..., :-1, :, :]) / dx
        dHx_dy = (Hx[..., :, 1:, :] - Hx[..., :, :-1, :]) / dy
        Ez[..., 1:, 1:, :] = (
            self._Ca[..., 1:, 1:, :] * Ez[..., 1:, 1:, :]
            + self._Cb[..., 1:, 1:, :] * (dHy_dx - dHx_dy)
        )
    
    # Step 10: update step counter
    self.steps_completed += 1
    
    # Step 11: stability check
    if self.steps_completed % self._n_check == 0:
        self._check_stability()
```

---

## Stability Checking

The stability check monitors all six field components and raises `SimulationDivergedError` if any exceeds the threshold:

```python
def _check_stability(self) -> None:
    """Check all 6 field magnitudes for divergence."""
    self.n_stability_checks += 1
    
    for name, tensor in (
        ("Ex", self._fields.Ex),
        ("Ey", self._fields.Ey),
        ("Ez", self._fields.Ez),
        ("Hx", self._fields.Hx),
        ("Hy", self._fields.Hy),
        ("Hz", self._fields.Hz),
    ):
        val = float(tensor.abs().max().item())
        if val > self.last_field_max:
            self.last_field_max = val
        if val > self._threshold:
            raise SimulationDivergedError(
                f"Simulation diverged at step {self.steps_completed}: "
                f"{name} max = {val:.3e} exceeds threshold "
                f"{self._threshold:.3e}. "
                f"Check CFL condition (dt={self._grid.dt:.3e}) and "
                f"source amplitude."
            )
```

---

## GPU Memory Layout and Optimization

### Memory Requirements per Step

For a simulation domain of size Nx × Ny × Nz with single-precision (4-byte) floats:

**Active field tensors in step():**
- 6 fields (Ex, Ey, Ez, Hx, Hy, Hz): 6 × Nx × Ny × Nz × 4 bytes

**Material coefficients (optional):**
- Ca, Cb: 2 × Nx × Ny × Nz × 4 bytes (if not free-space)
- Da, Db: 2 × Nx × Ny × Nz × 4 bytes (Phase 5, if magnetic materials)

**Total resident memory:**

| Case | Size for 128³ | Size for 256³ | Size for 512³ |
|------|---------------|---------------|---------------|
| Fields only | 6 × 128³ × 4 B = 384 MB | 6 × 256³ × 4 B = 1.5 GB | 6 × 512³ × 4 B = 12 GB |
| + Ca/Cb | 8 × 128³ × 4 B = 512 MB | 8 × 256³ × 4 B = 2 GB | 8 × 512³ × 4 B = 16 GB |
| + Ca/Cb + Da/Db | 10 × 128³ × 4 B = 640 MB | 10 × 256³ × 4 B = 2.5 GB | 10 × 512³ × 4 B = 20 GB |

### GPU Fit Strategy

**NVIDIA T4 (16 GB VRAM):**
- 128³ + materials: Comfortably fits (512 MB used, 15.5 GB free)
- 256³ + materials: Fits with modest headroom (2 GB used, 14 GB free)
- 512³: Requires tiling or off-device storage

**NVIDIA A40 (48 GB VRAM):**
- 256³ + materials: Fits easily (2 GB used, 46 GB free)
- 512³ + materials: Fits comfortably (16 GB used, 32 GB free)

### Bandwidth Analysis

Per time step, the hot path performs:
- **6 H-field reads** (∂E/∂x, ∂E/∂y, ∂E/∂z for each H-component)
- **3 H-field writes**
- **6 E-field reads** (∂H/∂x, ∂H/∂y, ∂H/∂z for each E-component)
- **3 E-field writes**
- **0-10 material coefficient reads** (depending on path)

Approximate DRAM traffic:
- Free-space: 12 reads + 6 writes = 18 × Nx×Ny×Nz × 4 B per step = 72 × Nx×Ny×Nz bytes/step
- Material: 18 + 10 = 28 reads/writes per stencil

On modern GPUs with HBM (A40, A100), this translates to compute-limited performance for 3D stencils with 2 reads + 1 write per grid point.

---

## Boundary Condition Integration

The `MurABC` class is instantiated with 6 faces (x_min, x_max, y_min, y_max, z_min, z_max):

```python
# Construction
boundary_3d = MurABC(
    grid=grid_3d,
    num_faces=6,
    order=1
)

# Usage in step()
boundary_3d.snapshot()  # Save H-field boundary values
# ... H-updates ...
boundary_3d.apply({"Hx": Hx, "Hy": Hy, "Hz": Hz})  # Apply ABC
```

The snapshot captures all H-components on the 6 boundary planes, and apply() uses the stored boundary values to absorb outgoing waves via Mur's first-order ABC.

---

## Source Injection

Sources are injected into the E-field after snapshot() and before H-updates:

```python
if self._sources is not None:
    self._sources.step(
        {
            "Ex": Ex, "Ey": Ey, "Ez": Ez,
            "Hx": Hx, "Hy": Hy, "Hz": Hz
        },
        n  # step index
    )
```

**Supported source types (Phase 2+):**
- Point dipoles (Ex, Ey, Ez injections)
- Line sources (extended along one axis)
- Plane waves (via incident-field/total-field boundary)

---

## Telemetry and Performance Metrics

### Counters and Timers

```python
self.steps_completed: int        # Total steps executed
self.elapsed_time: float         # Wall-clock time in seconds
self.mcells_per_second: float    # Throughput metric
self.last_field_max: float       # Peak field magnitude observed
self.n_stability_checks: int     # Count of stability checks performed
```

### Throughput Calculation

Throughput is computed at regular intervals during `run()` using GPU synchronization:

```python
def run(self, n_steps: int, verbose: bool = False) -> None:
    # ... initialization ...
    for i in range(n_steps):
        self.step()
        
        if verbose and (i + 1) % report_interval == 0:
            if is_cuda:
                torch.cuda.synchronize()  # Block until GPU completes
            t_now = time.perf_counter()
            elapsed = t_now - t_start
            steps_done = self.steps_completed - start_steps
            
            # Throughput in Mcells/s (million cells per second)
            self.mcells_per_second = (
                steps_done * self._grid.Nx * self._grid.Ny * self._grid.Nz
            ) / elapsed / 1e6
```

**Target:** > 100 Mcells/s on NVIDIA T4 for 128³ grids.

### Example Output
```
  10%  step 100  field_max=1.234e+02  156.3 Mcells/s
  20%  step 200  field_max=1.456e+02  152.1 Mcells/s
  ...
  100% step 1000  field_max=1.789e+02  153.8 Mcells/s
```

---

## Validation and CFL Compliance

### CFL Condition Check

At construction, validate the time step:

```python
c0 = 3e8  # speed of light, m/s
dx, dy, dz = grid.dx, grid.dy, grid.dz
cfl_max = 1.0 / (c0 * (1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2)**0.5)
safety_factor = 0.99

dt_safe = safety_factor * cfl_max

if grid.dt > dt_safe:
    warnings.warn(
        f"CFL violation: dt={grid.dt:.3e} exceeds "
        f"safe limit {dt_safe:.3e} for grid spacing "
        f"({dx:.3e}, {dy:.3e}, {dz:.3e}). "
        f"Simulation may diverge."
    )
```

### Field Magnitude Inspection

During development and debugging, inspect field maxima:

```python
stepper = FDTD3D(...)
stepper.run(1000, verbose=True)
print(f"Peak field magnitude: {stepper.last_field_max:.3e}")
print(f"Throughput: {stepper.mcells_per_second:.1f} Mcells/s")
```

If peak field magnitude grows without bound, the simulation has likely diverged; check CFL, source amplitude, and material coefficients.

---

## Convergence and Numerical Dispersion

### Convergence Properties

The 3D FDTD method is second-order accurate in space and time:

$$\text{Error} = O(\Delta x^2 + \Delta y^2 + \Delta z^2 + \Delta t^2)$$

Grid refinement by a factor of 2 reduces error by ~4×. For 3D problems:
- 64³ grid: ~12 ms execution on T4
- 128³ grid: ~96 ms execution on T4
- 256³ grid: ~768 ms execution on T4

### Numerical Dispersion

Wave propagation at non-aligned angles experiences velocity dispersion:

$$v_{\text{numerical}} = c_0 \frac{\sin(k_x \Delta x / 2)}{(k_x \Delta x / 2)} \cdot \text{(similar for y, z)}$$

Minimize dispersion by keeping wavelength Λ > 10 × cell size:

$$\Delta x, \Delta y, \Delta z < \Lambda / 10$$

For 1 GHz in free space (Λ = 0.3 m), use ∆x < 3 cm. For optical frequencies (Λ = 1 μm), use ∆x < 100 nm.

---

## Relationship to FDTD2D

### Inheritance Chain

```
FDTD2D (2D TM only, 3 components)
  ↓
FDTD3D (3D full vector, 6 components, inherits interface)
```

### API Compatibility

Code written for FDTD2D requires minimal changes for FDTD3D:

**Before (2D):**
```python
grid_2d = YeeGrid(Nx=256, Ny=256, Nz=1, ...)
fields_2d = FieldSet(grid_2d)
stepper = FDTD2D(grid_2d, fields_2d, boundary_2d, sources_2d)
stepper.run(1000, verbose=True)
```

**After (3D):**
```python
grid_3d = YeeGrid(Nx=256, Ny=256, Nz=256, ...)  # Nz > 1
fields_3d = FieldSet(grid_3d)
stepper = FDTD3D(grid_3d, fields_3d, boundary_3d, sources_3d)
stepper.run(1000, verbose=True)  # Same API
```

The `step()`, `run()`, `reset()`, and property methods are identical in signature.

---

## Implementation Checklist

- [ ] Class definition: `FDTD3D` inheriting from `FDTD2D`
- [ ] Grid validation: Nx, Ny, Nz >= 4
- [ ] Boundary consistency: 6 faces (x_min, x_max, y_min, y_max, z_min, z_max)
- [ ] Material coefficient initialization: Ca, Cb shapes (Nx, Ny, Nz)
- [ ] H-field updates: 3 Faraday curl equations, all vectorized
- [ ] E-field updates: 3 Ampere curl equations, free-space and material paths
- [ ] Stability check: all 6 components monitored
- [ ] Throughput telemetry: Mcells/s calculation
- [ ] CFL validation and warning
- [ ] Integration tests: compare with FDTD2D results for Nz=1 case
- [ ] GPU memory profiling: peak VRAM for 128³, 256³, 512³ grids
- [ ] Performance benchmarks: > 100 Mcells/s target on T4

---

## References

- Taflove, A., & Hagness, S. C. (2005). Computational Electromagnetics: The Finite-Difference Time-Domain Method (3rd ed.). Artech House.
- Yee, K. S. (1966). "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media." IEEE Transactions on Antennas and Propagation, 14(3), 302-307.
- Sullivan, D. M. (2000). Electromagnetic Simulation Using the FDTD Method. IEEE Press.
- Giannopoulos, A. (2005). "Modelling ground penetrating radar by GprMax." Construction and Building Materials, 19(10), 755-762.
