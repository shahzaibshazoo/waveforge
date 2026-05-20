# Phase 3: 3D Boundary Conditions — Mur ABC & CPML

## Overview

This document specifies the design and implementation of absorbing boundary conditions (ABCs) for the 3D FDTD electromagnetic solver. Boundary conditions prevent unphysical reflections at the simulation domain edges and are critical for open-domain problems.

WaveForge 3D offers two boundary condition implementations:

1. **Mur ABC (First-Order)**: Simple, fast, memory-efficient. Reflection coefficient ~-20 dB.
2. **CPML (Convolutional Perfectly Matched Layer)**: Gold standard, higher accuracy. Reflection coefficient -60 to -80 dB.

Both implementations extend the proven 2D ABC infrastructure to handle six faces and all six field components.

---

## Background: 2D ABC to 3D

### Current 2D Implementation (Phase 5)

The existing 2D code implements **first-order Mur ABC** on 4 edges (x_min, x_max, y_min, y_max):

- Only Hz is damped (scalar boundary field)
- Stores previous interior Hz and previous edge Hz
- Updates edges with: `Hz_edge^{n+1} = Hz_prev_interior^n + C_mur * (Hz_new_interior^{n+1} - Hz_prev_edge^n)`
- Coefficient: `C_mur = (c*dt - dx) / (c*dt + dx)`

### CPML Scaffolding in 2D

The 2D `boundaries.py` file includes precomputed CPML coefficient arrays (b_E, c_E, b_H, c_H) with dual staggering to match E and H field positions, but full PML update is deferred.

### Extension to 3D

3D adds:
- 2 additional faces (z_min, z_max)
- 4 additional field components (Ez, Hx, Hy, and components of Ex, Ey, Ez on each face)
- More complex PML geometry: overlapping regions at edges and corners

---

## First-Order Mur ABC in 3D

### Theory

First-order Mur absorbing boundary condition is derived from the one-way wave equation:

$$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial n} = 0$$

where $n$ is the outward normal and $u$ is a field component.

Discretized on a face parallel to the y-z plane (x boundary):

$$u^{n+1}_{boundary} = u^n_{interior, prev} + C_{mur} (u^{n+1}_{interior} - u^n_{boundary})$$

where

$$C_{mur} = \frac{c \Delta t - \Delta x}{c \Delta t + \Delta x}$$

and $c$ is the local wave speed. In vacuum: $c = c_0 \approx 3 \times 10^8$ m/s.

### 3D Faces and Field Components

Each of the 6 faces requires damping of **2 tangential field components**:

| Face | Normal | Tangential E | Tangential H | Storage |
|------|--------|--------------|--------------|---------|
| x_min, x_max | ±x | Ey, Ez | Hy, Hz | 2 faces × 2 fields |
| y_min, y_max | ±y | Ex, Ez | Hx, Hz | 2 faces × 2 fields |
| z_min, z_max | ±z | Ex, Ey | Hx, Hy | 2 faces × 2 fields |

**Total field components per face:** 4 (2 E, 2 H)  
**Total faces:** 6  
**Total snapshot arrays:** 6 × 4 = 24

However, for efficiency, store only previous snapshots (not double-buffer):
- 12 arrays for previous tangential E on each face (2 E per face × 6 faces)
- 12 arrays for previous tangential H on each face (2 H per face × 6 faces)

This gives **24 snapshot tensors** in total (or 12 if storing only previous values).

### Data Structure

```python
class MurABC3D:
    """First-order Mur ABC for 3D FDTD."""
    
    def __init__(self, grid, fields):
        self.grid = grid
        self.c_mur = compute_mur_coefficient(grid)
        
        # Snapshot storage: previous field values on boundaries
        # Shape of each: (Nx_face, Ny_face, 1, 1) for batch dimension
        self.E_prev = {
            'x_min': {'Ey': zeros_like(fields.Ey[..., 0:1, :]),
                      'Ez': zeros_like(fields.Ez[..., 0:1, :])},
            'x_max': {'Ey': zeros_like(fields.Ey[..., -1:, :]),
                      'Ez': zeros_like(fields.Ez[..., -1:, :])},
            # ... similarly for y_min, y_max, z_min, z_max
        }
        
        self.H_prev = {
            'x_min': {'Hy': zeros_like(fields.Hy[..., 0:1, :]),
                      'Hz': zeros_like(fields.Hz[..., 0:1, :])},
            # ... similarly for other faces
        }
    
    def snapshot(self, fields):
        """Save previous boundary field values before update."""
        # Copy boundary values to previous storage
        # Ey at x_min boundary: x-index = 1 (just inside boundary)
        self.E_prev['x_min']['Ey'][...] = fields.Ey[..., 1:2, :]
        self.E_prev['x_min']['Ez'][...] = fields.Ez[..., 1:2, :]
        # ... etc for all faces and components
    
    def apply(self, fields):
        """Apply Mur ABC to boundary fields."""
        # x_min face
        fields.Ey[..., 0:1, :] = (
            self.E_prev['x_min']['Ey'] +
            self.c_mur * (fields.Ey[..., 1:2, :] - fields.Ey[..., 0:1, :])
        )
        fields.Ez[..., 0:1, :] = (
            self.E_prev['x_min']['Ez'] +
            self.c_mur * (fields.Ez[..., 1:2, :] - fields.Ez[..., 0:1, :])
        )
        # ... similarly for H, and for other faces
```

### ASCII Schematic: Mur ABC on One Face

```
            interior        boundary
            (active)        (ABC applied)
    
    y ↑      ╔═════╦═╗
      │      ║ Ey  ║E║  ← E_y interior at x=1
      │      ╚═════╩═╝
      │      ╔═════╦═╗
      │      ║ Ez  ║E║  ← E_z interior at x=1
      │      ╚═════╩═╝
      │      
      └──────→ x
    
    Boundary ABC:
    Ey[0, :, :] = Ey_prev[1, :, :] + c_mur * (Ey[1, :, :] - Ey[0, :, :])
    Ez[0, :, :] = Ez_prev[1, :, :] + c_mur * (Ez[1, :, :] - Ez[0, :, :])
```

### Memory Usage (Mur ABC Only)

For grid size Nx × Ny × Nz:
- 6 faces, 4 field components per face → 24 arrays
- Each array shape: ~(Nx_face) × (Ny_face) × 1 × 4 bytes (FP32)
- x-faces: 2 × (Ny × Nz) × 4 B each → 8 × Ny × Nz bytes
- y-faces: 2 × (Nx × Nz) × 4 B each → 8 × Nx × Nz bytes
- z-faces: 2 × (Nx × Ny) × 4 B each → 8 × Nx × Ny bytes

Total (approximate, neglecting tensor overhead):
$$M_{Mur} \approx 8 \times (2 Ny \times Nz + 2 Nx \times Nz + 2 Nx \times Ny) \text{ bytes}$$

For Nx = Ny = Nz = 256:
$$M_{Mur} \approx 8 \times (2 \times 256^2 \times 3) = 12.6 \text{ MB}$$

### Reflection Coefficient

First-order Mur ABC has frequency-dependent reflection:
- At normal incidence (θ = 0°): R ≈ -20 dB
- At oblique incidence (θ = 45°): R ≈ -10 dB

Use Mur ABC when:
- Memory is constrained
- Moderate accuracy sufficient
- Simulation near edges not critical

---

## CPML (Convolutional Perfectly Matched Layer)

### Theory: Stretched-Coordinate PML

PML absorbs waves by introducing complex-valued, frequency-dependent absorption into Maxwell's equations via stretched coordinates:

$$\tilde{x} = x + \frac{\sigma_x(x)}{j\omega\mu_0}, \quad \tilde{y} = y + \frac{\sigma_y(y)}{j\omega\mu_0}, \quad \tilde{z} = z + \frac{\sigma_z(z)}{j\omega\mu_0}$$

where σ(x) is spatially varying conductivity (increasing into the PML) and ω is angular frequency.

In the time domain, this maps to recursive auxiliary field equations. The convolutional approach eliminates frequency dependence by reformulating as time-domain recursions.

### PML Domain Structure

PML forms **thin absorbing slabs** around all 6 faces:

```
┌─────────────────────────────────────────────┐
│  z_max PML (thickness D_z)                  │
│  ┌───────────────────────────────────────┐  │
│  │  Corner: y_max ∩ z_max (overlap)      │  │
│  │  ┌─────────────────────────────────┐  │  │
│x_│  │  Interior (no absorption)       │  │  │
│m_│  │                                 │  │  │
│i_│  │                                 │  │  │
│n_│  │                                 │  │  │
│ _│  └─────────────────────────────────┘  │  │
│P_│  │  Corner: y_min ∩ z_max          │  │  │
│M_│  └───────────────────────────────────┘  │
│L_│                                         │
│  └─────────────────────────────────────────┘
│  z_min PML
└─────────────────────────────────────────────┘
  y_min         y_max
```

**Parameters:**
- D_x, D_y, D_z: PML thickness (typically 8-16 cells)
- σ_max: maximum conductivity at PML-interior boundary
- κ_max: maximum scaling factor (typically 1)
- α: damping factor (typically 0)
- m: polynomial grading order (typically 3-4)

### PML Coefficients

For each spatial direction k ∈ {x, y, z}, compute:

**Position in PML:** depth d ranges from 0 (at interior boundary) to D (at outer edge)

**Grading function** (polynomial of order m):

$$\sigma_k(d) = \sigma_{max} \left(\frac{d}{D_k}\right)^m$$

**Recursive coefficients** (computed once, stored as arrays):

$$b_k[n] = \exp\left(-\left(\frac{\sigma_k[n]}{\kappa_k[n]} + \alpha\right) \frac{\Delta t}{\epsilon_0}\right)$$

$$c_k[n] = \frac{\sigma_k[n]}{\sigma_k[n] \kappa_k[n] + \kappa_k[n]^2 \alpha} (b_k[n] - 1)$$

Typically: σ_max ≈ 3-4 m / (Z_0 D), κ_max = 1, α = 0 (or small).

### Auxiliary Fields

CPML modifies the curl updates with auxiliary fields that store curl-history:

For **E-field** at each grid point:

$$\Psi_{Ex,y}^{n+1} = b_y \Psi_{Ex,y}^n + c_y \frac{\partial H_z}{\partial y}$$

$$\Psi_{Ex,z}^{n+1} = b_z \Psi_{Ex,z}^n + c_z \frac{\partial H_y}{\partial z}$$

(and similarly for Ey and Ez)

For **H-field** at each grid point:

$$\Psi_{Hx,y}^{n+1} = b_y \Psi_{Hx,y}^n + c_y \frac{\partial E_z}{\partial y}$$

$$\Psi_{Hx,z}^{n+1} = b_z \Psi_{Hx,z}^n + c_z \frac{\partial E_y}{\partial z}$$

(and similarly for Hy and Hz)

**Total auxiliary fields:** 12 (6 for E + 6 for H)

### Modified Update Equations with CPML

The standard FDTD update becomes:

**Ex update (standard part):**
$$E_x^{n+1} = C_a E_x^n + C_b \left[\frac{\partial H_z}{\partial y} - \frac{\partial H_y}{\partial z}\right]$$

**With CPML correction:**
$$E_x^{n+1} = C_a E_x^n + C_b \left[\Psi_{Ex,y} + \Psi_{Ex,z}\right]$$

where the auxiliary fields already encode the spatially-varying absorption.

Similarly for all other field components.

### CPML Data Structure

```python
class CPML3D:
    """Convolutional PML for 3D FDTD."""
    
    def __init__(self, grid, D_x=12, D_y=12, D_z=12, sigma_max=None):
        self.grid = grid
        self.D = {'x': D_x, 'y': D_y, 'z': D_z}
        
        # Compute grading: σ(d) = σ_max * (d/D)^m
        self.sigma_max = sigma_max or self.compute_sigma_max()
        self.m = 3  # polynomial grading order
        
        # Precompute b and c coefficients in each direction
        # Shape: (D_x,) or (D_y,) or (D_z,)
        self.b = {}
        self.c = {}
        for direction in ['x', 'y', 'z']:
            d_vals = torch.arange(self.D[direction], device=grid.device)
            sigma = self.sigma_max * (d_vals / self.D[direction]) ** self.m
            
            # Frequency-independent recursive coefficients
            self.b[direction] = torch.exp(-(sigma / eps0) * grid.dt)
            self.c[direction] = (sigma / eps0) * (self.b[direction] - 1)
        
        # Auxiliary fields (only in PML regions)
        # Ψ_Ex_y: shape (Nx, Ny+D_y, Nz, *batch_dims)
        # But to save memory, only allocate in PML slabs
        self.psi_E = {}  # 'Ex_y', 'Ex_z', 'Ey_x', 'Ey_z', 'Ez_x', 'Ez_y'
        self.psi_H = {}  # 'Hx_y', 'Hx_z', 'Hy_x', 'Hy_z', 'Hz_x', 'Hz_y'
        
        self._allocate_pml_arrays(grid)
    
    def _allocate_pml_arrays(self, grid):
        """Allocate auxiliary field arrays only in PML regions."""
        # E-field auxiliary fields (one per direction pair)
        # Each only exists in the thin PML region perpendicular to that direction
        
        # Ψ_Ex_y: exists only in y-PML (y < D_y or y >= Ny - D_y)
        shape_Ex_y = (grid.Nx, 2*grid.D['y'], grid.Nz, *grid.batch_dims)
        self.psi_E['Ex_y'] = torch.zeros(shape_Ex_y, device=grid.device, dtype=torch.float32)
        
        # ... similarly for all 12 auxiliary fields
    
    def snapshot(self, fields):
        """No snapshot needed for CPML (uses current curl values)."""
        pass
    
    def step(self, fields, curl_cache):
        """Update auxiliary fields and add CPML correction."""
        # Update Ψ_Ex_y from ∂H_z/∂y
        dHz_dy = curl_cache['dHz_dy']  # shape (Nx, Ny, Nz, ...)
        
        # Apply only in y-PML region (y < D_y or y >= Ny - D_y)
        # Ψ^{n+1} = b * Ψ^n + c * curl
        self.psi_E['Ex_y'][..., :self.D['y'], :] = (
            self.b['y'] * self.psi_E['Ex_y'][..., :self.D['y'], :] +
            self.c['y'] * dHz_dy[..., :self.D['y'], :]
        )
        self.psi_E['Ex_y'][..., -self.D['y']:, :] = (
            self.b['y'] * self.psi_E['Ex_y'][..., -self.D['y']:, :] +
            self.c['y'] * dHz_dy[..., -self.D['y']:, :]
        )
        # ... similar recursions for all 12 auxiliary fields
    
    def apply(self, fields):
        """Add PML correction to field updates."""
        # Ex += Cb * (Ψ_Ex_y + Ψ_Ex_z)
        fields.Ex += Cb * (self.psi_E['Ex_y'] + self.psi_E['Ex_z'])
        # ... similarly for all 6 field components
```

### Memory Usage (CPML)

12 auxiliary arrays, each in PML slabs only:

- **E-field PML regions:** 6 arrays for E components, one per direction pair
  - Shape of each: ≈ (Nx, D_y, Nz) + permutations for y-PML, z-PML
  - Total: ≈ 2 × (Nx × D_y × Nz + Nx × Ny × D_z + D_x × Ny × Nz) × 4 B

- **H-field PML regions:** 6 arrays, same structure

For Nx = Ny = Nz = 256, D = 12:
$$M_{CPML} \approx 2 \times 2 \times (256 \times 12 \times 256 \times 3) \times 4 \text{ B} \approx 50 \text{ MB}$$

**Compare to interior:** 50 MB << 3.2 GB (field arrays), so CPML memory is negligible.

### Reflection Coefficient

CPML performance (typical):
- D = 8 cells: R ≈ -40 dB
- D = 12 cells: R ≈ -60 dB
- D = 16 cells: R ≈ -80 dB

Reflection is nearly frequency-independent and insensitive to incidence angle up to ~70°.

---

## Implementation Plan

### Phase 3A: MurABC3D (Week 1)

1. **Data structure:** Dict of face-keyed snapshot arrays (E_prev, H_prev)
2. **Initialization:** Allocate 12 E + 12 H snapshot tensors; compute c_mur
3. **snapshot() method:** Copy boundary E and H to previous storage
4. **apply() method:** Apply Mur formula on all 6 faces, all 4 components per face
5. **Interface compatibility:** Expose `snapshot()` and `apply()` matching CPML3D

**Testing:**
- Unit test: snapshot + apply maintains stability
- Convergence test: compare reflected wave amplitude to -20 dB theory

### Phase 3B: CPML3D (Week 2-3)

1. **Grading:** Implement σ(d) = σ_max (d/D)^m for m=3
2. **Coefficient precomputation:** b, c arrays in each direction
3. **Auxiliary field allocation:** 12 arrays, only in PML slabs
4. **Auxiliary field update:** Ψ^{n+1} = b Ψ^n + c (curl) for all 12
5. **FDTD integration:** Modify E and H updates to add Ψ correction
6. **Performance:** Measure overhead of auxiliary field recursions

**Testing:**
- Unit test: auxiliary field recursion stability
- Reflection test: -60 dB at D=12 cells
- Oblique incidence: R at 30°, 45°, 60°

### Phase 3C: Integration with FDTD3D

Modify `fdtd3d.py` step loop:

```python
def step(self, fields):
    # 1. Snapshot boundary values (H only for ABC)
    self.boundary.snapshot(fields)
    
    # 2. Inject sources
    self.sources.step(fields, self.n_step)
    
    # 3. Magnetic field update
    dE = self.compute_E_curls(fields)
    self.H_update(fields, dE)
    
    # 4. Apply ABC correction (Mur or CPML)
    self.boundary.apply(fields)
    
    # 5. Electric field update
    dH = self.compute_H_curls(fields)
    self.E_update(fields, dH)
    
    # 6. Telemetry
    if self.n_step % self.check_interval == 0:
        self.check_stability(fields)
```

---

## Edge and Corner Treatment in PML

### Overlapping PML Regions

At edges and corners, multiple PML regions overlap:

```
        z_max
      z_min PML
      (overlap)
         ╔══╦════╦══╗
         ║  ║    ║  ║  y_max PML edge:
         ║x_║ int║y_║  two PMLs meet
         ║m_║    ║m_║  → apply both σ_x and σ_y
         ║  ║    ║  ║
         ╚══╩════╩══╝
      z_min
      x_min        x_max
```

**2D edge (e.g., x_min ∩ y_min):**
- Apply σ_x along x, σ_y along y
- Auxiliary field for x-derivative: Ψ[..., 0:D_x, 0:D_y, :]
- Auxiliary field for y-derivative: Ψ[..., 0:D_x, 0:D_y, :]

**3D corner (e.g., x_min ∩ y_min ∩ z_min):**
- Apply σ_x, σ_y, σ_z simultaneously
- All three auxiliary fields contribute to each E and H component

Implementation: Auxiliary fields naturally handle overlaps because they exist on a full mesh. No special masking needed.

---

## Integration Checklist

- [ ] **MurABC3D class:** snapshot() and apply() methods
- [ ] **Mur ABC 24 snapshot arrays:** 2 E-components × 6 faces + 2 H-components × 6 faces
- [ ] **CPML3D class:** grading, b/c coefficients, auxiliary fields
- [ ] **CPML auxiliary allocation:** Only allocate in PML slabs to save memory
- [ ] **CPML step:** Update all 12 auxiliary fields before E and H updates
- [ ] **FDTD3D integration:** Call boundary.snapshot() and boundary.apply() at correct steps
- [ ] **Edge/corner handling:** Verify overlapping PML regions work correctly
- [ ] **Unit tests:** Stability, reflection coefficient, oblique incidence
- [ ] **Performance profiling:** CPML overhead on GPU (target: <10% per step)

---

## Configuration Example

```python
# Initialize grid
grid = YeeGrid(Nx=256, Ny=256, Nz=256, dx=1e-6)

# Choose boundary condition
boundary = MurABC3D(grid, fields)
# OR
boundary = CPML3D(grid, D_x=12, D_y=12, D_z=12, sigma_max=None)

# Simulate
sim = FDTD3D(grid, fields, sources, boundary, materials)
for n in range(n_steps):
    sim.step(fields)
    if n % 100 == 0:
        print(f"Step {n}, Energy = {fields.total_energy():.2e} J")
```

---

## Performance Notes

### GPU Memory Bandwidth

Both Mur and CPML are memory-bound on GPU:
- Mur ABC: 2× load (E_prev, H_prev) + 1× store (E_new, H_new) per boundary point
- CPML: 1× load (Ψ) + 1× load (curl) + 1× store (Ψ_new) per interior point in PML

CPML auxiliary field recursion can be fused with main curl computation for better cache locality.

### CPU vs GPU

- On CPU (PyTorch CPU backend): Mur ABC dominates boundary time due to synchronization overhead. CPML negligible.
- On GPU: Both are hidden by memory bandwidth saturation. Measure with `torch.cuda.synchronize()`.

---

## References

1. **Mur ABC:** Mur, G. (1981). "Absorbing Boundary Conditions for the Finite-Difference Approximation of the Time-Domain Electromagnetic-Field Equations." IEEE Trans. Electromagn. Compat., 23(4), 377-382.

2. **CPML:** Roden, J. A., & Gedney, S. D. (2000). "Convolutional PML (CPML): An Efficient FDTD Implementation of the CFS-PML for Arbitrary Media." Microwave Optical Technol. Lett., 27(5), 334-339.

3. **PML Stability:** Taflove, A., & Hagness, S. C. (2005). *Computational Electromagnetics: The Finite-Difference Time-Domain Method* (3rd ed.). Chapter 7.

4. **3D FDTD Boundaries:** Sullivan, D. M. (2000). *Electromagnetic Simulation Using the FDTD Method*. IEEE Press, Chapters 5-6.

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| MurABC3D design | DONE | 24 snapshot arrays, 6 faces, full field coverage |
| CPML3D design | DONE | 12 auxiliary fields, polynomial grading |
| GPU memory strategy | DONE | PML arrays only in absorbing regions |
| FDTD3D integration | PENDING | Awaits Phase 2 (FDTD3D engine) completion |
| Unit tests | PENDING | Awaits implementation |
| Performance validation | PENDING | After GPU backend ready |
