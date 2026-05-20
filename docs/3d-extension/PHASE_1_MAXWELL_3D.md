# Phase 1: 3D Maxwell's Equations — Full Vector FDTD

## Overview

This document specifies the mathematical foundation for extending WaveForge from 2D TM-mode FDTD (transverse magnetic with Ex, Ey, Hz components only) to full 3D vector FDTD with all six electromagnetic field components: Ex, Ey, Ez, Hx, Hy, Hz.

The 2D TM restriction limited simulations to out-of-plane magnetic fields and in-plane electric fields. Full 3D enables:
- Arbitrary field polarizations
- Volumetric structures (spheres, cubes, waveguides)
- 3D resonators and cavities
- General anisotropic materials
- Oblique incidence and scattering

---

## Maxwell's Equations in Continuous Form

### Faraday's Law (Magnetic Field Update)

$$\frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E}$$

Expanded to three scalar equations:

$$\frac{\partial B_x}{\partial t} = -\left(\frac{\partial E_z}{\partial y} - \frac{\partial E_y}{\partial z}\right)$$

$$\frac{\partial B_y}{\partial t} = -\left(\frac{\partial E_x}{\partial z} - \frac{\partial E_z}{\partial x}\right)$$

$$\frac{\partial B_z}{\partial t} = -\left(\frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y}\right)$$

With constitutive relation $\mathbf{B} = \mu_0 \mu_r \mathbf{H}$ in linear media:

$$\frac{\partial H_x}{\partial t} = -\frac{1}{\mu_0 \mu_r}\left(\frac{\partial E_z}{\partial y} - \frac{\partial E_y}{\partial z}\right)$$

$$\frac{\partial H_y}{\partial t} = -\frac{1}{\mu_0 \mu_r}\left(\frac{\partial E_x}{\partial z} - \frac{\partial E_z}{\partial x}\right)$$

$$\frac{\partial H_z}{\partial t} = -\frac{1}{\mu_0 \mu_r}\left(\frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y}\right)$$

### Ampere-Maxwell Law (Electric Field Update)

$$\frac{\partial \mathbf{D}}{\partial t} = \nabla \times \mathbf{H} - \mathbf{J}$$

With constitutive relation $\mathbf{D} = \epsilon_0 \epsilon_r \mathbf{E}$ and conductive loss $\mathbf{J} = \sigma \mathbf{E}$:

$$\frac{\partial E_x}{\partial t} = \frac{1}{\epsilon_0 \epsilon_r}\left(\frac{\partial H_z}{\partial y} - \frac{\partial H_y}{\partial z} - \sigma E_x\right)$$

$$\frac{\partial E_y}{\partial t} = \frac{1}{\epsilon_0 \epsilon_r}\left(\frac{\partial H_x}{\partial z} - \frac{\partial H_z}{\partial x} - \sigma E_y\right)$$

$$\frac{\partial E_z}{\partial t} = \frac{1}{\epsilon_0 \epsilon_r}\left(\frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y} - \sigma E_z\right)$$

---

## Yee Grid Staggering in 3D

The Yee cell is a unit cube with edges of length Δx, Δy, Δz. Field components are staggered at half-integer positions:

**Electric field components (face-centered):**
- Ex at positions [i+1/2, j, k]
- Ey at positions [i, j+1/2, k]
- Ez at positions [i, j, k+1/2]

**Magnetic field components (edge-centered):**
- Hx at positions [i, j+1/2, k+1/2]
- Hy at positions [i+1/2, j, k+1/2]
- Hz at positions [i+1/2, j+1/2, k]

### ASCII Yee Cell Diagram

```
        j+1
         *----------*
        /|         /|
       / |        / |
      *----------*  |
      |  |       |  |    k+1/2 (Hz plane)
      |  *-------|--*           
      | /        | /    k
      |/         |/
      *----------*
      i      i+1/2

    Vertices (nodes):  *
    Ex on x-faces:     ⊢→⊣ (perpendicular to j-k plane)
    Ey on y-faces:     ⊥ (perpendicular to i-k plane)
    Ez on z-faces:     ⊕ (perpendicular to i-j plane)
    
    Hx loops in y-z plane at [i, j+1/2, k+1/2]
    Hy loops in x-z plane at [i+1/2, j, k+1/2]
    Hz loops in x-y plane at [i+1/2, j+1/2, k]
```

---

## Discrete FDTD Updates on the Yee Grid

Time stepping uses half-steps: magnetic fields at n±1/2, electric fields at n.

### Magnetic Field Update (Faraday's Law)

Derivatives computed at staggered positions. Each magnetic component uses centered differences of surrounding electric components.

**Hx Update:**
$$H_x^{n+1/2}[i, j+\tfrac{1}{2}, k+\tfrac{1}{2}] = H_x^{n-1/2}[i, j+\tfrac{1}{2}, k+\tfrac{1}{2}] + \frac{\Delta t}{\mu_0 \mu_r[i, j+\tfrac{1}{2}, k+\tfrac{1}{2}]} \left[\frac{E_z[i, j+1, k+\tfrac{1}{2}] - E_z[i, j, k+\tfrac{1}{2}]}{\Delta y} - \frac{E_y[i, j+\tfrac{1}{2}, k+1] - E_y[i, j+\tfrac{1}{2}, k]}{\Delta z}\right]$$

**Hy Update:**
$$H_y^{n+1/2}[i+\tfrac{1}{2}, j, k+\tfrac{1}{2}] = H_y^{n-1/2}[i+\tfrac{1}{2}, j, k+\tfrac{1}{2}] + \frac{\Delta t}{\mu_0 \mu_r[i+\tfrac{1}{2}, j, k+\tfrac{1}{2}]} \left[\frac{E_x[i+\tfrac{1}{2}, j, k+1] - E_x[i+\tfrac{1}{2}, j, k]}{\Delta z} - \frac{E_z[i+1, j, k+\tfrac{1}{2}] - E_z[i, j, k+\tfrac{1}{2}]}{\Delta x}\right]$$

**Hz Update:**
$$H_z^{n+1/2}[i+\tfrac{1}{2}, j+\tfrac{1}{2}, k] = H_z^{n-1/2}[i+\tfrac{1}{2}, j+\tfrac{1}{2}, k] + \frac{\Delta t}{\mu_0 \mu_r[i+\tfrac{1}{2}, j+\tfrac{1}{2}, k]} \left[\frac{E_y[i+1, j+\tfrac{1}{2}, k] - E_y[i, j+\tfrac{1}{2}, k]}{\Delta x} - \frac{E_x[i+\tfrac{1}{2}, j+1, k] - E_x[i+\tfrac{1}{2}, j, k]}{\Delta y}\right]$$

### Electric Field Update (Ampere-Maxwell Law)

With loss coefficient $\sigma$, define:

$$C_a[i, j, k] = \frac{1 - \frac{\sigma \Delta t}{2\epsilon_0 \epsilon_r[i,j,k]}}{1 + \frac{\sigma \Delta t}{2\epsilon_0 \epsilon_r[i,j,k]}}$$

$$C_b[i, j, k] = \frac{\Delta t}{\epsilon_0 \epsilon_r[i,j,k]} \cdot \frac{1}{1 + \frac{\sigma \Delta t}{2\epsilon_0 \epsilon_r[i,j,k]}}$$

**Ex Update:**
$$E_x^{n+1}[i+\tfrac{1}{2}, j, k] = C_a[i+\tfrac{1}{2}, j, k] \cdot E_x^n[i+\tfrac{1}{2}, j, k] + C_b[i+\tfrac{1}{2}, j, k] \left[\frac{H_z[i+\tfrac{1}{2}, j+\tfrac{1}{2}, k] - H_z[i+\tfrac{1}{2}, j-\tfrac{1}{2}, k]}{\Delta y} - \frac{H_y[i+\tfrac{1}{2}, j, k+\tfrac{1}{2}] - H_y[i+\tfrac{1}{2}, j, k-\tfrac{1}{2}]}{\Delta z}\right]$$

**Ey Update:**
$$E_y^{n+1}[i, j+\tfrac{1}{2}, k] = C_a[i, j+\tfrac{1}{2}, k] \cdot E_y^n[i, j+\tfrac{1}{2}, k] + C_b[i, j+\tfrac{1}{2}, k] \left[\frac{H_x[i, j+\tfrac{1}{2}, k+\tfrac{1}{2}] - H_x[i, j+\tfrac{1}{2}, k-\tfrac{1}{2}]}{\Delta z} - \frac{H_z[i+\tfrac{1}{2}, j+\tfrac{1}{2}, k] - H_z[i-\tfrac{1}{2}, j+\tfrac{1}{2}, k]}{\Delta x}\right]$$

**Ez Update:**
$$E_z^{n+1}[i, j, k+\tfrac{1}{2}] = C_a[i, j, k+\tfrac{1}{2}] \cdot E_z^n[i, j, k+\tfrac{1}{2}] + C_b[i, j, k+\tfrac{1}{2}] \left[\frac{H_y[i+\tfrac{1}{2}, j, k+\tfrac{1}{2}] - H_y[i-\tfrac{1}{2}, j, k+\tfrac{1}{2}]}{\Delta x} - \frac{H_x[i, j+\tfrac{1}{2}, k+\tfrac{1}{2}] - H_x[i, j-\tfrac{1}{2}, k+\tfrac{1}{2}]}{\Delta y}\right]$$

---

## CFL Stability Condition for 3D

The Courant-Friedrichs-Lewy (CFL) condition ensures numerical stability. For 3D with uniform grid spacing, the maximum stable time step is:

$$\Delta t \leq \frac{1}{c_0 \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}}$$

where $c_0 = 1/\sqrt{\mu_0 \epsilon_0}$ is the speed of light in vacuum.

**For isotropic uniform grids** (Δx = Δy = Δz = Δ):

$$\Delta t \leq \frac{\Delta}{c_0 \sqrt{3}}$$

**Safety factor:** In practice, use:

$$\Delta t = 0.99 \times \frac{\Delta}{c_0 \sqrt{3}}$$

This ensures numerical stability with margin for rounding errors. Violation of this condition leads to exponential growth of numerical noise.

---

## Material Coefficients

Material properties vary spatially. At each grid point, precompute frequency-independent coefficients:

### Electric Field Coefficients

For permittivity $\epsilon_r$ and conductivity $\sigma$:

$$C_a[i, j, k] = \frac{1 - \frac{\sigma \Delta t}{2\epsilon_0 \epsilon_r}}{1 + \frac{\sigma \Delta t}{2\epsilon_0 \epsilon_r}}$$

$$C_b[i, j, k] = \frac{\Delta t}{\epsilon_0 \epsilon_r} \cdot \frac{1}{1 + \frac{\sigma \Delta t}{2\epsilon_0 \epsilon_r}}$$

### Magnetic Field Coefficients

For permeability $\mu_r$ and magnetic loss $\sigma_m$:

$$D_a[i, j, k] = \frac{1 - \frac{\sigma_m \Delta t}{2\mu_0 \mu_r}}{1 + \frac{\sigma_m \Delta t}{2\mu_0 \mu_r}}$$

$$D_b[i, j, k] = \frac{\Delta t}{\mu_0 \mu_r} \cdot \frac{1}{1 + \frac{\sigma_m \Delta t}{2\mu_0 \mu_r}}$$

### Numerical Values

- $\epsilon_0 = 8.854 \times 10^{-12}$ F/m
- $\mu_0 = 4\pi \times 10^{-7}$ H/m
- $c_0 = 2.998 \times 10^8$ m/s

---

## 2D TM vs 3D: Component and Derivative Count

### 2D TM Mode (Current WaveForge)

**Active components:** Ex, Ey, Hz (3 fields)

**Update equations:** 3
- Hx = 0, Hy = 0 (out-of-plane, not computed)
- Ez = 0 (perpendicular to plane, not computed)

**Curl terms per update:**
- Hz: ∂Ex/∂y, ∂Ey/∂x (2 derivatives)
- Ex: ∂Hz/∂y (1 derivative)
- Ey: -∂Hz/∂x (1 derivative)

**Total spatial derivatives:** 6 (per time step, per grid point)

### 3D Full Vector

**Active components:** Ex, Ey, Ez, Hx, Hy, Hz (6 fields)

**Update equations:** 6 (3 magnetic + 3 electric)

**Curl terms per update:**
- Hx: ∂Ey/∂z, ∂Ez/∂y (2 derivatives)
- Hy: ∂Ez/∂x, ∂Ex/∂z (2 derivatives)
- Hz: ∂Ex/∂y, ∂Ey/∂x (2 derivatives)
- Ex: ∂Hz/∂y, ∂Hy/∂z (2 derivatives)
- Ey: ∂Hx/∂z, ∂Hz/∂x (2 derivatives)
- Ez: ∂Hy/∂x, ∂Hx/∂y (2 derivatives)

**Total spatial derivatives:** 12 (per time step, per grid point)

**Scaling comparison:**

| Aspect | 2D TM | 3D | Ratio |
|--------|-------|----|----|
| Field components | 3 | 6 | 2× |
| Field arrays in memory | 3 | 6 | 2× |
| Material coefficient arrays | 2 (Ca, Cb) | 4 (Ca, Cb, Da, Db) | 2× |
| Derivatives per point per step | 6 | 12 | 2× |
| Time-to-solution for N³ grid | O(3N³) | O(6N³) | 2× |
| Arithmetic intensity | Bandwidth-bound | Slightly compute-bound | - |

---

## Memory Requirements

For a simulation domain of size Nx × Ny × Nz with single-precision (4-byte) floats:

### Field Arrays

- 6 field components (Ex, Ey, Ez, Hx, Hy, Hz): **6 × Nx × Ny × Nz × 4 bytes**
- Double buffering (current + next): **12 × Nx × Ny × Nz × 4 bytes** (if in-place updates not used)

### Material Coefficient Arrays

Precomputed once per simulation:
- Ca: **1 × Nx × Ny × Nz × 4 bytes** (electric permittivity & loss)
- Cb: **1 × Nx × Ny × Nz × 4 bytes**
- Da: **1 × Nx × Ny × Nz × 4 bytes** (magnetic permeability & loss)
- Db: **1 × Nx × Ny × Nz × 4 bytes**

Total material coefficients: **4 × Nx × Ny × Nz × 4 bytes**

### Total for Typical Case

For Nx = Ny = Nz = 256:
- Field arrays (double buffer): 12 × 256³ × 4 B = 3.2 GB
- Material coefficients: 4 × 256³ × 4 B = 1.06 GB
- **Total: ~4.3 GB per device**

For Nx = Ny = Nz = 512:
- Field arrays (double buffer): 12 × 512³ × 4 B = 25.6 GB
- Material coefficients: 4 × 512³ × 4 B = 8.6 GB
- **Total: ~34 GB per device**

### GPU Memory Fit Strategy

On 40 GB GPUs (A100, H100):
- 256³ domain: fully resident
- 512³ domain: requires tiling in one dimension or reduced time steps between I/O
- 768³ domain: not practical without multi-GPU distribution

---

## Forward Finite Differences on Staggered Grid

All spatial derivatives use forward differences (higher-order schemes like central differences would require staggered access patterns incompatible with the Yee cell).

### Ex-component derivatives at [i+1/2, j, k]:

**∂Hz/∂y:**
$$\frac{\partial H_z}{\partial y}\bigg|_{i+1/2, j, k} \approx \frac{H_z[i+\tfrac{1}{2}, j+\tfrac{1}{2}, k] - H_z[i+\tfrac{1}{2}, j-\tfrac{1}{2}, k]}{\Delta y}$$

**∂Hy/∂z:**
$$\frac{\partial H_y}{\partial z}\bigg|_{i+1/2, j, k} \approx \frac{H_y[i+\tfrac{1}{2}, j, k+\tfrac{1}{2}] - H_y[i+\tfrac{1}{2}, j, k-\tfrac{1}{2}]}{\Delta z}$$

### Ey-component derivatives at [i, j+1/2, k]:

**∂Hx/∂z:**
$$\frac{\partial H_x}{\partial z}\bigg|_{i, j+1/2, k} \approx \frac{H_x[i, j+\tfrac{1}{2}, k+\tfrac{1}{2}] - H_x[i, j+\tfrac{1}{2}, k-\tfrac{1}{2}]}{\Delta z}$$

**∂Hz/∂x:**
$$\frac{\partial H_z}{\partial x}\bigg|_{i, j+1/2, k} \approx \frac{H_z[i+\tfrac{1}{2}, j+\tfrac{1}{2}, k] - H_z[i-\tfrac{1}{2}, j+\tfrac{1}{2}, k]}{\Delta x}$$

### Ez-component derivatives at [i, j, k+1/2]:

**∂Hy/∂x:**
$$\frac{\partial H_y}{\partial x}\bigg|_{i, j, k+1/2} \approx \frac{H_y[i+\tfrac{1}{2}, j, k+\tfrac{1}{2}] - H_y[i-\tfrac{1}{2}, j, k+\tfrac{1}{2}]}{\Delta x}$$

**∂Hx/∂y:**
$$\frac{\partial H_x}{\partial y}\bigg|_{i, j, k+1/2} \approx \frac{H_x[i, j+\tfrac{1}{2}, k+\tfrac{1}{2}] - H_x[i, j-\tfrac{1}{2}, k+\tfrac{1}{2}]}{\Delta y}$$

### Hx-component derivatives at [i, j+1/2, k+1/2]:

**∂Ez/∂y:**
$$\frac{\partial E_z}{\partial y}\bigg|_{i, j+1/2, k+1/2} \approx \frac{E_z[i, j+1, k+\tfrac{1}{2}] - E_z[i, j, k+\tfrac{1}{2}]}{\Delta y}$$

**∂Ey/∂z:**
$$\frac{\partial E_y}{\partial z}\bigg|_{i, j+1/2, k+1/2} \approx \frac{E_y[i, j+\tfrac{1}{2}, k+1] - E_y[i, j+\tfrac{1}{2}, k]}{\Delta z}$$

### Hy-component derivatives at [i+1/2, j, k+1/2]:

**∂Ex/∂z:**
$$\frac{\partial E_x}{\partial z}\bigg|_{i+1/2, j, k+1/2} \approx \frac{E_x[i+\tfrac{1}{2}, j, k+1] - E_x[i+\tfrac{1}{2}, j, k]}{\Delta z}$$

**∂Ez/∂x:**
$$\frac{\partial E_z}{\partial x}\bigg|_{i+1/2, j, k+1/2} \approx \frac{E_z[i+1, j, k+\tfrac{1}{2}] - E_z[i, j, k+\tfrac{1}{2}]}{\Delta x}$$

### Hz-component derivatives at [i+1/2, j+1/2, k]:

**∂Ey/∂x:**
$$\frac{\partial E_y}{\partial x}\bigg|_{i+1/2, j+1/2, k} \approx \frac{E_y[i+1, j+\tfrac{1}{2}, k] - E_y[i, j+\tfrac{1}{2}, k]}{\Delta x}$$

**∂Ex/∂y:**
$$\frac{\partial E_x}{\partial y}\bigg|_{i+1/2, j+1/2, k} \approx \frac{E_x[i+\tfrac{1}{2}, j+1, k] - E_x[i+\tfrac{1}{2}, j, k]}{\Delta y}$$

---

## Implementation Checklist

- [ ] Memory layout: 6 field arrays with Yee staggering
- [ ] CFL computation: Δt = 0.99 × Δ / (c₀√3)
- [ ] Material coefficient precomputation: Ca, Cb, Da, Db arrays
- [ ] Magnetic field kernel: 3 updates (Hx, Hy, Hz) from 6 E-derivatives
- [ ] Electric field kernel: 3 updates (Ex, Ey, Ez) from 6 H-derivatives
- [ ] Boundary conditions: PML or periodic (Phase 2)
- [ ] Source injection: point dipole or plane wave (Phase 2)
- [ ] Field output: binary dumps or real-time visualization (Phase 2)

---

## References

- Yee, K. S. (1966). "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media." IEEE Transactions on Antennas and Propagation, 14(3), 302-307.
- Taflove, A., & Hagness, S. C. (2005). Computational Electromagnetics: The Finite-Difference Time-Domain Method (3rd ed.). Artech House.
- Sullivan, D. M. (2000). Electromagnetic Simulation Using the FDTD Method. IEEE Press.
