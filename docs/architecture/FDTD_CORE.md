# GPU-MEEP: FDTD Core Engine Specification

> Finite-Difference Time-Domain Implementation Details
> Physics, Numerics, CUDA Mapping, and Boundary Conditions

---

# FDTD_CORE: Sections 1-2

## 1. Yee Grid

### 1.1 Staggered Grid Structure

The Yee cell places electric and magnetic field components at spatially offset positions within a unit cell. For cell index `(i,j,k)`:

```
        Ez(i,j,k+½)
        |
        |       Hy(i+½,j,k+½)
        |      /
        +-----/--------+
       /|    /         /|
      / |   /         / |
     /  |  /         /  |
    +---|--/---------+   |
    |   | /  Hx(i,j+½,k+½)
    |   |/           |   |
Ey(i,j+½,k)         |   + (i+1,j,k)
    |   |         Ey(i+1,j+½,k)
    |   |            |  /
    |   +------------|--
    |  / Ez(i,j,k+½) | /
    | /    Hx(i,j+½,k+½)
    |/               |/
    +---Ex(i+½,j,k)--+
  (i,j,k)          (i+1,j,k)

    Hz(i+½,j+½,k) is at the center of the top/bottom face.
```

### 1.2 Component Positions (3D Staggering)

| Component | Position              |
|-----------|-----------------------|
| `Ex`      | `(i+½, j,   k  )`    |
| `Ey`      | `(i,   j+½, k  )`    |
| `Ez`      | `(i,   j,   k+½)`    |
| `Hx`      | `(i,   j+½, k+½)`    |
| `Hy`      | `(i+½, j,   k+½)`    |
| `Hz`      | `(i+½, j+½, k  )`    |

E-field components live on cell edges; H-field components live on cell face centers.

### 1.3 Grid Indexing Convention

Cell `(i,j,k)` with `i in [0, Nx-1]`, `j in [0, Ny-1]`, `k in [0, Nz-1]` owns:

- **E-components** on its three "low" edges: `Ex(i+½,j,k)`, `Ey(i,j+½,k)`, `Ez(i,j,k+½)`
- **H-components** on its three "low" faces: `Hx(i,j+½,k+½)`, `Hy(i+½,j,k+½)`, `Hz(i+½,j+½,k)`

The H-field update for `Hx[i,j,k]` uses `Ey[i,j,k]`, `Ey[i,j,k+1]`, `Ez[i,j,k]`, `Ez[i,j+1,k]`. Boundary conditions determine behavior at `Nx-1`, `Ny-1`, `Nz-1`.

### 1.4 Tensor Storage Layout

Each component is stored as a `torch.Tensor` of shape `(Nx, Ny, Nz)` with `dtype=torch.float32` (or `float64`). The array index `[i,j,k]` maps to the physical position defined by the staggering table above.

```python
Ex: Tensor[Nx, Ny, Nz]  # value at physical position ((i+0.5)*dx, j*dy, k*dz)
Ey: Tensor[Nx, Ny, Nz]  # value at physical position (i*dx, (j+0.5)*dy, k*dz)
Ez: Tensor[Nx, Ny, Nz]  # value at physical position (i*dx, j*dy, (k+0.5)*dz)
Hx: Tensor[Nx, Ny, Nz]  # value at physical position (i*dx, (j+0.5)*dy, (k+0.5)*dz)
Hy: Tensor[Nx, Ny, Nz]  # value at physical position ((i+0.5)*dx, j*dy, (k+0.5)*dz)
Hz: Tensor[Nx, Ny, Nz]  # value at physical position ((i+0.5)*dx, (j+0.5)*dy, k*dz)
```

Memory layout: contiguous along the last axis (k) for coalesced GPU memory access in z-oriented kernels. For custom CUDA kernels, consider `(Nz, Ny, Nx)` layout for x-stride-1 access patterns common in Cartesian sweeps.

### 1.5 Non-Uniform Grid Support

For non-uniform grids, replace scalar `dx, dy, dz` with 1D arrays:

```python
dx: Tensor[Nx]   # dx[i] = x_{i+1} - x_i
dy: Tensor[Ny]   # dy[j] = y_{j+1} - y_j
dz: Tensor[Nz]   # dz[k] = z_{k+1} - z_k
```

Inverse spacing arrays (precomputed for kernel efficiency):

```python
inv_dx: Tensor[Nx]   # 1.0 / dx[i]
inv_dy: Tensor[Ny]   # 1.0 / dy[j]
inv_dz: Tensor[Nz]   # 1.0 / dz[k]
```

These are broadcast across the full 3D grid during updates. For uniform grids, a scalar `float` is stored and broadcast is trivial.

### 1.6 Grid Metadata

```python
@dataclass
class GridMetadata:
    Nx: int; Ny: int; Nz: int          # cell counts per axis
    dx: Tensor | float                   # spacing (array or scalar)
    dy: Tensor | float
    dz: Tensor | float
    origin: Tuple[float, float, float]   # physical coord of cell (0,0,0) corner
    Lx: float; Ly: float; Lz: float     # physical domain size
    dt: float                            # time step
    x: Tensor  # coordinate array, shape (Nx,) — cell edge positions
    y: Tensor  # shape (Ny,)
    z: Tensor  # shape (Nz,)
```

---

## 2. Maxwell's Equations in FDTD Form

### 2.1 Continuous Curl Equations

Faraday's law:

$$\frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E}$$

Ampere's law with sources:

$$\frac{\partial \mathbf{D}}{\partial t} = \nabla \times \mathbf{H} - \mathbf{J}$$

### 2.2 Constitutive Relations

$$\mathbf{D} = \varepsilon \mathbf{E}, \quad \mathbf{B} = \mu \mathbf{H}$$

With conductivity (lossy media, electric loss `sigma`, magnetic loss `sigma*`):

$$\mathbf{J} = \sigma \mathbf{E}, \quad \mathbf{M} = \sigma^* \mathbf{H}$$

Substituting:

$$\mu \frac{\partial \mathbf{H}}{\partial t} = -\nabla \times \mathbf{E} - \sigma^* \mathbf{H}$$

$$\varepsilon \frac{\partial \mathbf{E}}{\partial t} = \nabla \times \mathbf{H} - \sigma \mathbf{E}$$

### 2.3 Finite-Difference Discretization of Curl Components

Spatial derivatives use central differences on the staggered grid. The six scalar equations:

**Faraday (H-field update):**

$$\frac{\partial H_x}{\partial t} = \frac{1}{\mu}\left(\frac{\partial E_y}{\partial z} - \frac{\partial E_z}{\partial y}\right) - \frac{\sigma^*}{\mu} H_x$$

$$\frac{\partial H_y}{\partial t} = \frac{1}{\mu}\left(\frac{\partial E_z}{\partial x} - \frac{\partial E_x}{\partial z}\right) - \frac{\sigma^*}{\mu} H_y$$

$$\frac{\partial H_z}{\partial t} = \frac{1}{\mu}\left(\frac{\partial E_x}{\partial y} - \frac{\partial E_y}{\partial x}\right) - \frac{\sigma^*}{\mu} H_z$$

**Ampere (E-field update):**

$$\frac{\partial E_x}{\partial t} = \frac{1}{\varepsilon}\left(\frac{\partial H_z}{\partial y} - \frac{\partial H_y}{\partial z}\right) - \frac{\sigma}{\varepsilon} E_x$$

$$\frac{\partial E_y}{\partial t} = \frac{1}{\varepsilon}\left(\frac{\partial H_x}{\partial z} - \frac{\partial H_z}{\partial x}\right) - \frac{\sigma}{\varepsilon} E_y$$

$$\frac{\partial E_z}{\partial t} = \frac{1}{\varepsilon}\left(\frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y}\right) - \frac{\sigma}{\varepsilon} E_z$$

### 2.4 Time Discretization (Leapfrog Scheme)

H-fields are evaluated at half-integer time steps `n+½`, E-fields at integer steps `n+1`:

$$\mathbf{H}^{n+1/2} \leftarrow \mathbf{H}^{n-1/2}, \mathbf{E}^{n}$$

$$\mathbf{E}^{n+1} \leftarrow \mathbf{E}^{n}, \mathbf{H}^{n+1/2}$$

This gives second-order accuracy in time and space simultaneously.

### 2.5 Full Update Equations (All 6 Components)

**H-field updates** (from time `n-½` to `n+½`):

$$H_x^{n+1/2}[i,j,k] = D_a \cdot H_x^{n-1/2}[i,j,k] + D_b \left(\frac{E_y[i,j,k+1] - E_y[i,j,k]}{\Delta z[k]} - \frac{E_z[i,j+1,k] - E_z[i,j,k]}{\Delta y[j]}\right)$$

$$H_y^{n+1/2}[i,j,k] = D_a \cdot H_y^{n-1/2}[i,j,k] + D_b \left(\frac{E_z[i+1,j,k] - E_z[i,j,k]}{\Delta x[i]} - \frac{E_x[i,j,k+1] - E_x[i,j,k]}{\Delta z[k]}\right)$$

$$H_z^{n+1/2}[i,j,k] = D_a \cdot H_z^{n-1/2}[i,j,k] + D_b \left(\frac{E_x[i,j+1,k] - E_x[i,j,k]}{\Delta y[j]} - \frac{E_y[i+1,j,k] - E_y[i,j,k]}{\Delta x[i]}\right)$$

**E-field updates** (from time `n` to `n+1`):

$$E_x^{n+1}[i,j,k] = C_a \cdot E_x^{n}[i,j,k] + C_b \left(\frac{H_z[i,j,k] - H_z[i,j-1,k]}{\Delta y[j]} - \frac{H_y[i,j,k] - H_y[i,j,k-1]}{\Delta z[k]}\right)$$

$$E_y^{n+1}[i,j,k] = C_a \cdot E_y^{n}[i,j,k] + C_b \left(\frac{H_x[i,j,k] - H_x[i,j,k-1]}{\Delta z[k]} - \frac{H_z[i,j,k] - H_z[i-1,j,k]}{\Delta x[i]}\right)$$

$$E_z^{n+1}[i,j,k] = C_a \cdot E_z^{n}[i,j,k] + C_b \left(\frac{H_y[i,j,k] - H_y[i-1,j,k]}{\Delta x[i]} - \frac{H_x[i,j,k] - H_x[i,j-1,k]}{\Delta y[j]}\right)$$

### 2.6 Material Coefficients (Ca, Cb, Da, Db)

**Electric update coefficients** (per grid point, precomputed):

$$C_a[i,j,k] = \frac{1 - \frac{\sigma \Delta t}{2\varepsilon}}{1 + \frac{\sigma \Delta t}{2\varepsilon}}$$

$$C_b[i,j,k] = \frac{\frac{\Delta t}{\varepsilon}}{1 + \frac{\sigma \Delta t}{2\varepsilon}}$$

**Magnetic update coefficients**:

$$D_a[i,j,k] = \frac{1 - \frac{\sigma^* \Delta t}{2\mu}}{1 + \frac{\sigma^* \Delta t}{2\mu}}$$

$$D_b[i,j,k] = \frac{\frac{\Delta t}{\mu}}{1 + \frac{\sigma^* \Delta t}{2\mu}}$$

For lossless media: `sigma = 0, sigma* = 0` yields `Ca = 1, Cb = dt/eps, Da = 1, Db = dt/mu`.

### 2.7 Lossy Media Incorporation

The coefficients above derive from semi-implicit time averaging of the conductive loss term. Taking Ampere's law at time step `n+½`:

$$\varepsilon \frac{E_x^{n+1} - E_x^{n}}{\Delta t} = (\nabla \times \mathbf{H})_x^{n+1/2} - \sigma \frac{E_x^{n+1} + E_x^{n}}{2}$$

Rearranging for `E_x^{n+1}`:

$$\left(1 + \frac{\sigma \Delta t}{2\varepsilon}\right) E_x^{n+1} = \left(1 - \frac{\sigma \Delta t}{2\varepsilon}\right) E_x^{n} + \frac{\Delta t}{\varepsilon} (\nabla \times \mathbf{H})_x^{n+1/2}$$

This yields the `Ca`, `Cb` definitions above. The scheme is unconditionally stable for any `sigma >= 0` (given CFL is satisfied for the propagation part). The magnetic loss `sigma*` is treated identically for the H-update.

**Storage**: `Ca`, `Cb` are `Tensor[Nx, Ny, Nz]` each (6 coefficient tensors total for full anisotropic loss). For homogeneous regions, use scalar broadcast. For heterogeneous media, these are materialized as full 3D arrays co-located with their respective field components.

**CFL stability condition** (uniform grid, lossless):

$$\Delta t \leq \frac{1}{c \sqrt{\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2} + \frac{1}{\Delta z^2}}}$$

where `c = 1/sqrt(mu_0 * eps_0)` is the speed of light in vacuum.

---

# FDTD_CORE: Sections 3 & 4

## 3. GPU-Optimized Update Equations

### 3.1 Tensor-Form Update Equations

The six FDTD update equations for linear, isotropic media with conductivity:

```
H^{n+1/2} = Da * H^{n-1/2} + Db * curl(E^n)
E^{n+1}   = Ca * E^n       + Cb * curl(H^{n+1/2})
```

Coefficient tensors (precomputed from material grids):

```python
# Magnetic coefficients — shape (Nx, Ny, Nz) per component
Da = torch.ones(Nx, Ny, Nz, device='cuda', dtype=torch.float32)   # 1.0 for lossless
Db = dt / (mu * dl)  # scalar or per-cell tensor

# Electric coefficients — shape (Nx, Ny, Nz) per component
Ca = (1 - sigma*dt/(2*eps)) / (1 + sigma*dt/(2*eps))  # (Nx,Ny,Nz)
Cb = (dt/(eps*dl)) / (1 + sigma*dt/(2*eps))            # (Nx,Ny,Nz)
```

### 3.2 Curl via Shifted Indexing

The curl operator on a Yee grid reduces to finite differences between adjacent cells. Implemented as slice operations (NOT `torch.roll`, which copies the entire tensor and cannot exclude boundaries):

```python
# Partial derivatives via forward differences (Yee staggering)
# dFz/dy: F_z(i, j+1, k) - F_z(i, j, k)
dHz_dy = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy   # shape: (Nx, Ny-1, Nz)

# dFy/dz: F_y(i, j, k+1) - F_y(i, j, k)
dHy_dz = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz   # shape: (Nx, Ny, Nz-1)

# dFx/dz: F_x(i, j, k+1) - F_x(i, j, k)
dHx_dz = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz

# dFz/dx: F_z(i+1, j, k) - F_z(i, j, k)
dHz_dx = (Hz[1:, :, :] - Hz[:-1, :, :]) / dx

# dFy/dx: F_y(i+1, j, k) - F_y(i, j, k)
dHy_dx = (Hy[1:, :, :] - Hy[:-1, :, :]) / dx

# dFx/dy: F_x(i, j+1, k) - F_x(i, j, k)
dHx_dy = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy
```

Boundary handling: slice indexing naturally produces tensors one cell shorter along the differentiation axis. Two strategies:

1. **Interior-only update** — update `Ex[1:-1, 1:-1, 1:-1]`, leave boundaries to PML/BC kernel.
2. **Padded storage** — allocate `(Nx+1, Ny+1, Nz+1)`, update interior `(Nx, Ny, Nz)` without size mismatch.

We use strategy (1): the update region is `[1:Nx, 1:Ny, 1:Nz]` and boundary conditions are applied in a separate pass.

### 3.3 Complete Timestep — PyTorch Pseudocode

```python
def timestep(Ex, Ey, Ez, Hx, Hy, Hz,
             Ca_ex, Cb_ex, Ca_ey, Cb_ey, Ca_ez, Cb_ez,
             Da_hx, Db_hx, Da_hy, Db_hy, Da_hz, Db_hz,
             dx, dy, dz):
    """One full FDTD leapfrog timestep. All tensors shape (Nx,Ny,Nz), device=cuda."""

    # --- H-field update (half-step) ---
    # curl_E components (forward differences on E)
    dEz_dy = (Ez[:, 1:, :] - Ez[:, :-1, :]) / dy
    dEy_dz = (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz
    dEx_dz = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz
    dEz_dx = (Ez[1:, :, :] - Ez[:-1, :, :]) / dx
    dEy_dx = (Ey[1:, :, :] - Ey[:-1, :, :]) / dx
    dEx_dy = (Ex[:, 1:, :] - Ex[:, :-1, :]) / dy

    # Update interior region: Hx = Da*Hx + Db*(dEz/dy - dEy/dz)
    s = (slice(None), slice(None, -1), slice(None, -1))
    Hx[s] = Da_hx[s] * Hx[s] + Db_hx[s] * (dEz_dy[:, :, :-1] - dEy_dz[:, :-1, :])

    s = (slice(None, -1), slice(None), slice(None, -1))
    Hy[s] = Da_hy[s] * Hy[s] + Db_hy[s] * (dEx_dz[:-1, :, :] - dEz_dx[:, :, :-1])

    s = (slice(None, -1), slice(None, -1), slice(None))
    Hz[s] = Da_hz[s] * Hz[s] + Db_hz[s] * (dEy_dx[:, :-1, :] - dEx_dy[:-1, :, :])

    # --- E-field update (full step) ---
    # curl_H components (backward differences on H, offset by half-cell)
    dHz_dy = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy
    dHy_dz = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz
    dHx_dz = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz
    dHz_dx = (Hz[1:, :, :] - Hz[:-1, :, :]) / dx
    dHy_dx = (Hy[1:, :, :] - Hy[:-1, :, :]) / dx
    dHx_dy = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy

    s = (slice(None), slice(1, None), slice(1, None))
    Ex[s] = Ca_ex[s] * Ex[s] + Cb_ex[s] * (dHz_dy[:, :, 1:] - dHy_dz[:, 1:, :])

    s = (slice(1, None), slice(None), slice(1, None))
    Ey[s] = Ca_ey[s] * Ey[s] + Cb_ey[s] * (dHx_dz[1:, :, :] - dHz_dx[:, :, 1:])

    s = (slice(1, None), slice(1, None), slice(None))
    Ez[s] = Ca_ez[s] * Ez[s] + Cb_ez[s] * (dHy_dx[:, 1:, :] - dHx_dy[1:, :, :])

    return Ex, Ey, Ez, Hx, Hy, Hz
```

### 3.4 In-Place vs Out-of-Place Updates

| Approach | Pros | Cons |
|----------|------|------|
| In-place (`Hx[s] = ...`) | Zero allocation, ~2x memory savings | Breaks `autograd` graph; no gradient through update |
| Out-of-place (`Hx_new = ...`) | Autograd-compatible; adjoint via backprop | 2x memory (old + new); GC pressure |

**Recommendation**: Use in-place for forward simulation (production). Use out-of-place with `torch.no_grad()` disabled only when computing adjoint sensitivities via backpropagation through time (BPTT). For most inverse design, use the adjoint method (run forward in-place, store checkpoints, recompute backward).

### 3.5 Kernel Fusion

Native PyTorch launches separate CUDA kernels for each arithmetic op. The E-field update `Ex = Ca*Ex + Cb*(dHz_dy - dHy_dz)` alone triggers: 1 subtract, 1 multiply (Cb), 1 multiply (Ca), 1 add = 4 kernel launches + 4 intermediate tensors.

Fusion strategies:

1. **`torch.compile` (inductor backend)** — fuses elementwise chains automatically. Achieves ~80% of handwritten CUDA for structured grids.
2. **Custom Triton kernel** — full control, single kernel for curl+multiply+accumulate:

```python
@triton.jit
def fused_e_update_kernel(
    Ex_ptr, Ca_ptr, Cb_ptr, Hz_ptr, Hy_ptr,
    Ny: tl.constexpr, Nz: tl.constexpr, dy_inv, dz_inv,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # Compute linear index -> (i, j, k)
    k = offs % Nz
    j = (offs // Nz) % Ny
    # Load Hz[i,j,k] and Hz[i,j-1,k]; Hy[i,j,k] and Hy[i,j,k-1]
    hz_here = tl.load(Hz_ptr + offs)
    hz_prev = tl.load(Hz_ptr + offs - Nz)       # j-1 stride
    hy_here = tl.load(Hy_ptr + offs)
    hy_prev = tl.load(Hy_ptr + offs - 1)        # k-1 stride
    curl = (hz_here - hz_prev) * dy_inv - (hy_here - hy_prev) * dz_inv
    ca = tl.load(Ca_ptr + offs)
    cb = tl.load(Cb_ptr + offs)
    ex = tl.load(Ex_ptr + offs)
    tl.store(Ex_ptr + offs, ca * ex + cb * curl)
```

This reduces 4 kernel launches to 1, eliminates intermediate allocations, and achieves peak memory bandwidth utilization (~85% HBM bandwidth on A100).

---

## 4. Tensor Memory Layout

### 4.1 Field Storage: Structure of Arrays (SoA)

Six independent tensors, each `(Nx, Ny, Nz)`, `dtype=float32`:

```python
Ex = torch.zeros(Nx, Ny, Nz, device='cuda', dtype=torch.float32)
Ey = torch.zeros(Nx, Ny, Nz, device='cuda', dtype=torch.float32)
Ez = torch.zeros(Nx, Ny, Nz, device='cuda', dtype=torch.float32)
Hx = torch.zeros(Nx, Ny, Nz, device='cuda', dtype=torch.float32)
Hy = torch.zeros(Nx, Ny, Nz, device='cuda', dtype=torch.float32)
Hz = torch.zeros(Nx, Ny, Nz, device='cuda', dtype=torch.float32)
```

### 4.2 Staggered Grid: Metadata, Not Storage

All six components share identical storage shape `(Nx, Ny, Nz)`. The half-cell offset is encoded in the finite-difference stencil direction, not in array dimensions. Physical position mapping:

| Component | Grid Position |
|-----------|--------------|
| Ex(i,j,k) | (i+1/2, j, k) |
| Ey(i,j,k) | (i, j+1/2, k) |
| Ez(i,j,k) | (i, j, k+1/2) |
| Hx(i,j,k) | (i, j+1/2, k+1/2) |
| Hy(i,j,k) | (i+1/2, j, k+1/2) |
| Hz(i,j,k) | (i+1/2, j+1/2, k) |

Uniform storage eliminates indexing complexity in GPU kernels.

### 4.3 Memory Stride Analysis

PyTorch default: C-contiguous (row-major). For tensor shape `(Nx, Ny, Nz)`:

```
stride = (Ny*Nz, Nz, 1)         # in elements
stride = (Ny*Nz*4, Nz*4, 4)     # in bytes (float32)
```

Accessing `F[i, j, k]` and `F[i, j, k+1]` are adjacent in memory (stride-1). A warp of 32 threads reading consecutive `k` indices achieves **perfectly coalesced** 128-byte cache-line transactions.

Accessing `F[i, j, k]` and `F[i, j+1, k]` has stride `Nz` elements — coalesced only if each thread in a warp takes a different `k`. Accessing `F[i, j, k]` and `F[i+1, j, k]` has stride `Ny*Nz` — worst case for coalescing.

**Implication**: The innermost loop (z-axis) maps to threadIdx.x. Grid decomposition: `blockDim = (1, 1, 128)` or similar z-dominant threading.

### 4.4 Warp-Aligned Padding

Pad `Nz` to the nearest multiple of 32 to ensure every warp's memory access is aligned:

```python
def pad_to_warp(N, warp_size=32):
    return ((N + warp_size - 1) // warp_size) * warp_size

Nz_padded = pad_to_warp(Nz)  # e.g., 100 -> 128
# Allocate padded, operate on [:, :, :Nz]
Ex = torch.zeros(Nx, Ny, Nz_padded, device='cuda', dtype=torch.float32)
```

### 4.5 Coefficient Tensor Strategies

| Strategy | Memory | Use Case |
|----------|--------|----------|
| Full `(Nx,Ny,Nz)` per component | 6 * Nx*Ny*Nz * 4B | Arbitrary inhomogeneous media |
| Material index `uint8 (Nx,Ny,Nz)` + LUT | Nx*Ny*Nz + 256*4B | Up to 256 materials |
| Uniform (scalar) | 4B | Homogeneous background |
| Slab `(Nx,1,1)` | Nx*4B | Layered media |

For mixed media, the index-lookup approach trades one extra indirection for ~6x memory reduction on coefficients.

### 4.6 Memory Estimation

Per-field memory: `Nx * Ny * Nz * 4` bytes. Total for 6 fields + 12 coefficient tensors (Ca, Cb per component):

```
M_total = 18 * Nx * Ny * Nz * 4 bytes
```

| Grid (Nx,Ny,Nz) | Cells | 6 Fields | +12 Coefficients | Total |
|------------------|-------|----------|------------------|-------|
| 128^3 | 2.1M | 48 MB | 96 MB | 144 MB |
| 256^3 | 16.8M | 384 MB | 768 MB | 1.15 GB |
| 512^3 | 134M | 3.07 GB | 6.14 GB | 9.2 GB |
| 512x512x1024 | 268M | 6.14 GB | 12.3 GB | 18.4 GB |

At 512^3 with full coefficients, a single A100 (80GB) can fit the simulation. An RTX 4090 (24GB) tops out near 300^3 with full coefficients, or 512^3 with index-lookup coefficients (~2.5 GB for indices + fields).

### 4.7 Single Stacked Tensor vs Six Separate Tensors

**Option A: Six separate tensors** `Ex, Ey, Ez, Hx, Hy, Hz` each `(Nx,Ny,Nz)`

- Pros: Each component independently contiguous; natural for slice-based curl; no stride overhead from leading dimension; simpler kernel indexing.
- Cons: Six `cudaMalloc` calls; scattered pointers; slightly more boilerplate.

**Option B: Single tensor** `F` of shape `(6, Nx, Ny, Nz)`

- Pros: Single allocation; easy to checkpoint (`torch.save(F, ...)`); batch-friendly for `torch.compile`.
- Cons: Leading dimension stride = `Nx*Ny*Nz*4` bytes — accessing different components in the same kernel requires large stride jumps, polluting L2 cache. `F[0]` and `F[1]` are `Nx*Ny*Nz*4` bytes apart — no locality benefit when updating Ex using Hy, Hz.

**Decision**: Use six separate tensors (Option A). The FDTD update never needs Ex and Ey in the same cache line. Separate allocations maximize L2 hit rate per kernel and avoid false sharing across SM partitions.
---

# Section 5: CUDA Execution Mapping

## 5.1 Thread-to-Cell Mapping

Each FDTD cell maps to exactly one CUDA thread. The 3D grid is decomposed into thread blocks of shape `(8, 8, 8)` = 512 threads/block.

```
Block dimensions:  blockDim = dim3(8, 8, 8)   → 512 threads
Warps per block:   512 / 32 = 16 warps
```

Thread indexing within a block:

```c
int ix = blockIdx.x * 8 + threadIdx.x;
int iy = blockIdx.y * 8 + threadIdx.y;
int iz = blockIdx.z * 8 + threadIdx.z;
int idx = ix * Ny * Nz + iy * Nz + iz;  // Z-fastest (column-major in Z)
```

## 5.2 Grid Launch Configuration

| Problem Size | Grid Blocks | Total Threads | Notes |
|---|---|---|---|
| 256³ | (32, 32, 32) = 32,768 | 16,777,216 | Fits in L2 partially |
| 512³ | (64, 64, 64) = 262,144 | 134,217,728 | Full GPU saturation |
| 768³ | (96, 96, 96) = 884,736 | 452,984,832 | Memory-bandwidth bound |

For non-power-of-2 domains, blocks at boundaries require bounds checking:

```c
if (ix >= Nx || iy >= Ny || iz >= Nz) return;
```

## 5.3 Warp Execution and Memory Coalescing

With `blockDim = (8, 8, 8)`, warp lane assignment follows the linearized threadIdx order:

```
Warp 0: threadIdx.z ∈ [0,7], threadIdx.y ∈ [0,3], threadIdx.x = 0
         → lanes 0-31 cover iz=0..7, iy=0..3 for fixed ix=0
```

Since Z is the fastest-varying dimension in memory layout and warp lanes vary in Z first, 8 consecutive lanes access 8 consecutive `float` addresses (32 bytes). A full warp spans 4 rows in Y × 8 cells in Z = 32 cells at consecutive Z-addresses within each Y-row.

**Coalescing analysis for E-field update (reading Hz neighbors):**

```
Hz[ix][iy][iz]     → address A
Hz[ix][iy-1][iz]   → address A - Nz*sizeof(float)   // stride = Nz floats away
```

For Nz=256: Y-neighbor is 1024 bytes away (different cache line). Z-neighbor is 4 bytes away (same cache line). This is the fundamental stencil bandwidth cost.

**Cache lines touched per warp (E_x update, reading H_y and H_z):**
- `Hz[ix][iy][iz]` and `Hz[ix][iy-1][iz]`: 2 cache lines (coalesced in Z, Y-stride miss)
- `Hy[ix][iy][iz]` and `Hy[ix][iy][iz-1]`: 1-2 cache lines (Z-adjacent, mostly same line)
- Total: ~4-6 L1 cache line requests per warp per field component read.

## 5.4 Register Usage Per Thread

For the E-field update kernel `E_x^{n+1} = Ca * E_x^n + Cb * (dHz/dy - dHy/dz)`:

| Register Purpose | Count |
|---|---|
| H-field neighbors: Hz(iy), Hz(iy-1), Hy(iz), Hy(iz-1) | 4 |
| Coefficients: Ca, Cb | 2 |
| Current field value: Ex | 1 |
| Finite differences (temporaries) | 2 |
| Index computation (ix, iy, iz, linear idx) | 4 |
| Address intermediates | 2 |
| **Total** | **~15 registers** |

At 15 regs/thread × 512 threads/block = 7,680 registers/block. SM 8.0 (A100) has 65,536 registers per SM → supports 8 concurrent blocks (limited by other resources to ~4-5 in practice).

## 5.5 Kernel Launch Overhead and CUDA Graphs

Single kernel launch overhead: ~5-10 μs. Per timestep with 3 separate launches (H, E, PML): ~15-30 μs overhead.

For a 256³ grid, kernel execution time ≈ 50-100 μs. Launch overhead is 15-30% of compute — unacceptable.

**CUDA Graph amortization:**

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    H_update_kernel(...); E_update_kernel(...); PML_kernel(...)
for t in range(num_steps):
    g.replay()  # ~3 μs total overhead (single command buffer submission)
```

## 5.6 Stream Scheduling

```
Stream 0 (compute):  [H-update] → [E-update] → [PML-update]
                         ↓ event
Stream 1 (I/O):      [DtoH field snapshot]  (overlaps next H-update)
```

Dependencies enforced via `cudaStreamWaitEvent`. H→E→PML are serialized (data dependency). Field output is asynchronous on a secondary stream.

**Fused kernel alternative:** Combine E-update + PML into a single kernel (branch on PML region flag per cell). Eliminates one launch and one global memory round-trip for E-field. Measured speedup: 15-25% for PML-heavy simulations.

## 5.7 Occupancy Analysis (SM 8.0 / A100)

```
Registers/thread:    15  → max 4,369 threads/SM (register-limited)
Shared mem/block:    0 bytes (pure register stencil)
Threads/block:       512
Max blocks/SM:       min(2048/512, 65536/(15×512), 32) = min(4, 8, 32) = 4
Active threads/SM:   4 × 512 = 2048
Occupancy:           2048 / 2048 = 100%
```

With 15 regs/thread, we achieve full occupancy. If register count grows to 32 (complex dispersive materials), occupancy drops to 50% — still acceptable for memory-bound kernels.

## 5.8 Performance Ladder: PyTorch vs Custom CUDA vs Triton

| Approach | 512³ E-update time | Bandwidth util. | Dev effort |
|---|---|---|---|
| PyTorch tensor ops (`E += Cb * (roll(Hz,-1,1) - Hz)`) | ~8.2 ms | 25-35% | Low |
| Triton kernel (explicit stencil, tiled) | ~2.1 ms | 65-75% | Medium |
| Custom CUDA kernel (hand-tuned, vectorized loads) | ~1.4 ms | 80-90% | High |
| Theoretical roofline (1555 GB/s, 6 reads + 1 write) | ~1.1 ms | 100% | — |

PyTorch penalty sources: (1) `roll()` allocates temporaries, (2) separate kernel per arithmetic op, (3) no stencil-aware caching. Triton closes most of the gap via fused loads and shared memory tiling.

---

# Section 6: CFL Stability Condition

## 6.1 Derivation for 3D FDTD

The Yee scheme yields an explicit update; stability requires the numerical domain of dependence contains the physical one (CFL criterion). Von Neumann analysis with ansatz `E ~ exp(i(kx·x + ky·y + kz·z - ωt))`:

```
[sin(ωΔt/2) / (cΔt/2)]² = [sin(kx·Δx/2) / (Δx/2)]² 
                          + [sin(ky·Δy/2) / (Δy/2)]²
                          + [sin(kz·Δz/2) / (Δz/2)]²
```

LHS max (when `sin(ωΔt/2) = 1`): `(2/(cΔt))²`. RHS max (all sines = 1): `(2/Δx)² + (2/Δy)² + (2/Δz)²`. Stability requires LHS_max >= RHS_max:

```
(2/(cΔt))² ≥ (2/Δx)² + (2/Δy)² + (2/Δz)²
```

Solving for Δt:

```
          1
Δt ≤ ─────────────────────────────
      c × √(1/Δx² + 1/Δy² + 1/Δz²)
```

## 6.2 Uniform Grid Simplification

For Δx = Δy = Δz = h:

```
Δt_max = h / (c × √3)     [3D]
Δt_max = h / (c × √2)     [2D]
Δt_max = h / c             [1D]
```

## 6.3 Courant Number

Define the Courant number `S = cΔt/Δx`. The CFL condition becomes:

```
S ≤ 1/√(d)    where d = spatial dimensionality
```

| Dimension | S_max | Numerical value |
|---|---|---|
| 1D | 1.0 | 1.0 |
| 2D | 1/√2 | 0.7071 |
| 3D | 1/√3 | 0.5774 |

**Practical choice: S = 0.5** — provides ~13% safety margin below the 3D limit and accommodates numerical perturbations from PML, dispersive media, and sub-cell averaging.

## 6.4 Material Impact on CFL

The phase velocity in a medium is `v = c₀ / √(εᵣ μᵣ)`. The CFL condition uses the *maximum* wave speed in the domain. For vacuum regions, `c = c₀ ≈ 3×10⁸ m/s`.

If the entire domain has `εᵣ ≥ ε_min > 1`:

```
Δt_max = (Δx × √(ε_min × μ_min)) / (c₀ × √3)
```

However, PML regions typically operate at `εᵣ = 1`, so CFL is almost always governed by `c₀` regardless of material content.

## 6.5 CFL Enforcement in Code

```python
def compute_stable_dt(dx, dy, dz, courant=0.5, c=2.998e8):
    """Compute maximum stable timestep satisfying CFL condition."""
    inv_sum = 1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2
    dt_max = 1.0 / (c * math.sqrt(inv_sum))
    dt = courant * dt_max  # Apply safety factor (courant < 1/√3)
    return dt

# Validation at runtime:
def validate_cfl(dt, dx, dy, dz, c=2.998e8):
    inv_sum = 1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2
    dt_limit = 1.0 / (c * math.sqrt(inv_sum))
    if dt > dt_limit:
        raise ValueError(
            f"CFL VIOLATED: dt={dt:.3e} > dt_max={dt_limit:.3e}. "
            f"Simulation will be numerically unstable."
        )
```

## 6.6 CFL Violation Consequences

When `S > 1/√3`, the amplification factor `|G| > 1` for at least one spatial frequency. Fields grow exponentially:

```
|E(t)| ~ |G|^n × |E(0)|     where n = t/Δt
```

For `S = 0.6` (3% over limit): `|G| ≈ 1.003` → fields double in ~231 steps. For `S = 0.7` (21% over limit): `|G| ≈ 1.05` → fields double in ~14 steps. Instability manifests as checkerboard patterns at the Nyquist frequency, originating at material boundaries or PML interfaces.

## 6.7 Non-Uniform Grid CFL

For graded meshes with cell sizes `{Δx_i, Δy_j, Δz_k}`:

```
Δt ≤ 1 / (c × √(1/Δx_min² + 1/Δy_min² + 1/Δz_min²))
```

The *minimum* cell size in each dimension governs the global timestep. This is the primary drawback of explicit FDTD on non-uniform grids: one small cell constrains the entire simulation.

## 6.8 Resolution and Timestep Table

Assuming uniform grid, free-space (`c = c₀`), S = 0.5, wavelength resolution = 20 cells/λ:

| Resolution (cells/λ) | Δx (nm) @ λ=1550nm | Δt (fs) | Steps for 100 fs | Min λ resolved (nm) |
|---|---|---|---|---|
| 10 | 155.0 | 0.1495 | 669 | 1550 |
| 20 | 77.5 | 0.0747 | 1339 | 775 |
| 30 | 51.7 | 0.0498 | 2008 | 517 |
| 40 | 38.75 | 0.0374 | 2674 | 387.5 |
| 60 | 25.83 | 0.0249 | 4016 | 258.3 |

**Computation:** `Δt = S × Δx / (c₀ × √3) = 0.5 × Δx / (2.998×10⁸ × 1.732)`

Practical accuracy requires 20+ cells/λ for < 1% phase error over long propagation distances.

---

# Section 7: Numerical Stability and Dispersion

## 7.1 Numerical Dispersion Relation

Substituting plane-wave trial solutions E = E₀ exp[j(kx·iΔx + ky·jΔy + kz·kΔz - ω·nΔt)] into the Yee update equations yields the **FDTD numerical dispersion relation**:

```
[1/(cΔt) · sin(ωΔt/2)]² = [1/Δx · sin(kx·Δx/2)]² + [1/Δy · sin(ky·Δy/2)]² + [1/Δz · sin(kz·Δz/2)]²
```

Compare with the continuous dispersion relation (ω/c)² = kx² + ky² + kz². The discrete form reduces to continuous only as Δx, Δy, Δz, Δt → 0. The **numerical phase velocity** v_p,num differs from c:

```
v_p,num / c = (ω·Δx) / (2c · arcsin[ cΔt/Δx · sin(ωΔt/2) / sin(kΔx/2) ])
```

### Anisotropic Dispersion Artifact

Phase velocity error depends on propagation direction relative to grid axes. For uniform cubic grid (Δx = Δy = Δz = δ) at Courant limit S = cΔt/δ = 1/√3 (3D):
- Along grid axis (θ=0°): maximum phase error
- Along body diagonal (θ=54.7°): minimum phase error

Mitigation:
1. **Grid resolution rule**: Δx ≤ λ_min / 10 (minimum 10 cells per shortest wavelength)
2. **High-accuracy rule**: 20 cells/wavelength → < 1% cumulative phase error
3. **Operate at Courant limit**: S = S_max minimizes dispersion for axis-aligned propagation

Phase error accumulates over distance L: Δφ = (L/λ) · 2π · (1 - v_p,num/c). At 10 cells/λ, Courant limit: |1 - v_p,num/c| ≈ 0.8% → Δφ ≈ 0.05 rad/wavelength.

## 7.2 Floating-Point Precision Impact

### FP32 vs FP64 Error Budget

| Property         | FP32           | FP64           | BF16          |
|-----------------|----------------|----------------|---------------|
| Mantissa bits   | 23             | 52             | 7             |
| Machine epsilon | 1.19 × 10⁻⁷   | 2.22 × 10⁻¹⁶  | 3.91 × 10⁻³  |
| Relative error  | ~10⁻⁷/op      | ~10⁻¹⁶/op     | ~10⁻²/op     |

### Roundoff Accumulation

Each update introduces relative error ε_mach. After N_steps:
- Uncorrelated (random walk): ΔE_rms/E ~ ε_mach · √N_steps
- Coherent (resonant structures): ΔE_worst/E ~ ε_mach · N_steps

### Energy Drift Analysis

| N_steps | FP32 drift (random) | FP32 drift (coherent) | FP64 drift (coherent) |
|---------|---------------------|-----------------------|-----------------------|
| 10⁴    | 10⁻⁵               | 10⁻³                 | 10⁻¹²                |
| 10⁵    | 3×10⁻⁵             | 10⁻²                 | 10⁻¹¹                |
| 10⁶    | 10⁻⁴               | 10⁻¹                 | 10⁻¹⁰                |

### When FP64 Is Mandatory

- Resonant cavities with Q > 10⁴ (ring-down requires > 10⁵ steps)
- High-Q photonic crystal cavities (Q ~ 10⁶, coherent accumulation)
- Long waveguide propagation (> 1000λ path length)
- Adjoint sensitivity analysis over > 10⁴ steps

### BF16 Danger

With ε_mach ≈ 3.9 × 10⁻³, energy drift reaches O(1) after:
```
N_critical = 1/ε_mach² ≈ 65,000 steps (random)
N_critical = 1/ε_mach ≈ 256 steps (coherent/worst-case)
```
BF16 is **unstable for > 100 steps** without periodic correction to higher precision.

## 7.3 Stability Diagnostics

### Energy Conservation Monitor

```python
def compute_em_energy(E, H, eps, mu):
    """All tensors on GPU, shape [3, Nx, Ny, Nz]."""
    W_e = 0.5 * torch.sum(eps * E**2)
    W_m = 0.5 * torch.sum(mu * H**2)
    return W_e + W_m
```

- **Lossless**: dW/dt = 0 (constant to within roundoff)
- **Lossy (σ > 0)**: dW/dt ≤ 0 (monotonically decreasing)
- **Diverging energy**: CFL violation or implementation error

### Divergence Check

∇·(εE) = ρ_free must hold at every time step. Numerically:
```
div_E[i,j,k] = (eps_x[i]*Ex[i] - eps_x[i-1]*Ex[i-1])/Δx
             + (eps_y[j]*Ey[j] - eps_y[j-1]*Ey[j-1])/Δy
             + (eps_z[k]*Ez[k] - eps_z[k-1]*Ez[k-1])/Δz
```
If max|div_E| grows exponentially → instability. Check every 50-100 steps.

### Field Magnitude Monitoring

```python
def check_stability(E, H, threshold=1e6):
    if torch.max(torch.abs(E)) > threshold or torch.max(torch.abs(H)) > threshold:
        raise RuntimeError("Field divergence detected — CFL violation or source error")
```

### NaN/Inf Detection

Execute every N steps (N=10 debug, N=100 production). Cost: ~2μs per `any()` reduction.
```python
def nan_check(E, H, step):
    if torch.isnan(E).any() or torch.isnan(H).any() or torch.isinf(E).any() or torch.isinf(H).any():
        raise RuntimeError(f"NaN/Inf detected at step {step}")
```

## 7.4 Lossy Media Stability

### Conductive Media (σ > 0)

Semi-implicit E-field update with conductivity:
```
Eⁿ⁺¹ = C_a · Eⁿ + C_b · (∇×H)ⁿ⁺¹/²
C_a = (1 - σΔt/2ε) / (1 + σΔt/2ε)    → |C_a| < 1, always stabilizing
C_b = (Δt/ε) / (1 + σΔt/2ε)
```

### Gain Media (σ < 0)

Negative conductivity yields |C_a| > 1, amplifying fields. Stability requires:
```
Δt < 2ε / (|σ| · (1 + S/S_max))
```
In practice: reduce Courant number by factor (1 - |σ|Δt/2ε).

### Dispersive Media (ADE Formulation)

**Debye model**: dP/dt + P/τ = ε₀(ε_s - ε_∞)/τ · E. Discretized:
```
Pⁿ⁺¹ = [(1 - Δt/2τ)/(1 + Δt/2τ)] · Pⁿ + [ε₀(ε_s - ε_∞)Δt/τ/(1 + Δt/2τ)] · Eⁿ⁺¹/²
```
ADE formulation is unconditionally stable provided base-grid CFL is satisfied.

**Drude model** near plasma frequency: ε(ω) = ε_∞ - ω_p²/(ω² + jγω). When ω → ω_p: Re(ε) → 0, λ_eff → ∞. Required handling:
- Subcell averaging of permittivity near ε = 0 crossings
- Adaptive Courant number: reduce S when min(Re(ε)) < 0.1·ε₀
- Monitor auxiliary current J_Drude for unbounded growth

## 7.5 Mixed Precision Stability Protocol

### BF16 Field Updates (Short Runs)

Stable for < 1000 steps with Courant number S < 0.4·S_max:
```python
E = E.bfloat16()
H = H.bfloat16()
C_b = (dt / eps).bfloat16()  # S < 0.4 * S_max provides rounding headroom
```
Reduced Courant ensures effective amplification per step from rounding stays below unity.

### Kahan Compensated Summation (DFT Monitors)

For frequency-domain accumulation over N_steps >> 1000, Kahan summation reduces error from O(N·ε) to O(ε):
```python
class KahanDFTAccumulator:
    def __init__(self, shape, freqs, device='cuda'):
        self.real = torch.zeros(len(freqs), *shape, dtype=torch.float32, device=device)
        self.imag = torch.zeros_like(self.real)
        self.comp_r = torch.zeros_like(self.real)
        self.comp_i = torch.zeros_like(self.real)

    def accumulate(self, field, step, dt, freqs):
        for i, f in enumerate(freqs):
            phase = 2 * math.pi * f * step * dt
            y = field * math.cos(phase) - self.comp_r[i]
            t = self.real[i] + y
            self.comp_r[i] = (t - self.real[i]) - y
            self.real[i] = t
            y = -field * math.sin(phase) - self.comp_i[i]
            t = self.imag[i] + y
            self.comp_i[i] = (t - self.imag[i]) - y
            self.imag[i] = t
```

### Periodic FP32 Correction

For BF16 runs exceeding 100 steps, cast to FP32 every 100 steps:
```python
def precision_correction(E_bf16, H_bf16, correction_interval=100, step=0):
    if step % correction_interval == 0:
        E_f32, H_f32 = E_bf16.float(), H_bf16.float()
        E_f32, H_f32 = fdtd_step_fp32(E_f32, H_f32)
        return E_f32.bfloat16(), H_f32.bfloat16()
    return fdtd_step_bf16(E_bf16, H_bf16)
```
Bounds accumulated BF16 error to ~100 × ε_BF16 ≈ 0.39 (recoverable range).

### Gradient Computation: Always FP32

Adjoint/backpropagation through FDTD requires FP32 minimum:
```python
# NEVER: loss.backward() with BF16 fields — gradient underflow guaranteed
# CORRECT:
E_f32 = E.float().requires_grad_(True)
H_f32 = H.float().requires_grad_(True)
loss.backward()  # gradients in FP32
```
Gradients are O(10⁻⁴) to O(10⁻⁸) smaller than fields. BF16 minimum normal is 2⁻¹²⁶ with only 7-bit precision — gradients below ~10⁻² are effectively zero in BF16.

---

# Section 8: PML (Perfectly Matched Layer) Implementation

## 8.1 PML Theory

The PML terminates the finite computational domain by introducing an artificial absorbing
medium impedance-matched to free space for all frequencies and angles of incidence.

**Stretched-coordinate formulation.** Each spatial derivative is replaced:
```
∂/∂x  →  (1/s_x) × ∂/∂x    where  s_x = κ_x + σ_x / (α_x + jωε₀)
```
- `κ_x ≥ 1`: real stretching (absorbs evanescent waves)
- `σ_x ≥ 0`: conductivity loss (absorbs propagating waves)
- `α_x ≥ 0`: CFS parameter (stabilizes low-frequency/dc response)

**CFS-PML.** Standard PML (α=0) suffers late-time instability from dc/evanescent modes.
Setting α > 0 shifts the pole from ω=0, providing stable absorption at all frequencies
and eliminating late-time linear growth artifacts.

**CPML.** The 1/s_x stretching produces a time-domain convolution. CPML implements this
via recursive (IIR) auxiliary "psi" variables. The inverse stretching expands as:
```
1/s_x = 1/κ_x + (σ_x/κ_x) / (σ_x·κ_x + κ_x²·α_x + jωε₀·κ_x²)
```
The convolution kernel decays exponentially, enabling one-pole recursive approximation.

---

## 8.2 CPML Update Equations

### 8.2.1 Auxiliary Psi Variables (12 total)

E-field psi terms (6):

| Field | Psi | Replaces | Field | Psi | Replaces |
|-------|-----|----------|-------|-----|----------|
| Ex | psi_Exy | ∂Hz/∂y | Ex | psi_Exz | ∂Hy/∂z |
| Ey | psi_Eyx | ∂Hz/∂x | Ey | psi_Eyz | ∂Hx/∂z |
| Ez | psi_Ezx | ∂Hy/∂x | Ez | psi_Ezy | ∂Hx/∂y |

H-field psi terms (6):

| Field | Psi | Replaces | Field | Psi | Replaces |
|-------|-----|----------|-------|-----|----------|
| Hx | psi_Hxy | ∂Ez/∂y | Hx | psi_Hxz | ∂Ey/∂z |
| Hy | psi_Hyx | ∂Ez/∂x | Hy | psi_Hyz | ∂Ex/∂z |
| Hz | psi_Hzx | ∂Ey/∂x | Hz | psi_Hzy | ∂Ex/∂y |

### 8.2.2 CPML Coefficients

For a given PML axis (y-direction shown):
```
b_y = exp( -(σ_y/κ_y + α_y) × Δt/ε₀ )
c_y = (σ_y / (σ_y·κ_y + κ_y²·α_y)) × (b_y - 1)
```
When σ_y = 0: c_y = 0, b_y = exp(-α_y·Δt/ε₀).

### 8.2.3 Recursive Update and Field Correction

Generic form:
```
psi_Exy^{n+1}[i,j,k] = b_y[j] × psi_Exy^n[i,j,k] + c_y[j] × (Hz[i,j,k] - Hz[i,j-1,k])/Δy
```

Modified E-field update in PML (Ex example):
```
Ex^{n+1} = Ex^n + Cb × ( (1/κ_y)(Hz[i,j,k]-Hz[i,j-1,k])/Δy
                        - (1/κ_z)(Hy[i,j,k]-Hy[i,j,k-1])/Δz )
                 + Cb × ( psi_Exy^{n+1} - psi_Exz^{n+1} )
```
where Cb = Δt/(ε₀·ε_r).

### 8.2.4 Complete E-field CPML Equations

```
# Ex:
psi_Exy^{n+1} = b_y × psi_Exy^n + c_y × (Hz[i,j,k] - Hz[i,j-1,k])/Δy
psi_Exz^{n+1} = b_z × psi_Exz^n + c_z × (Hy[i,j,k] - Hy[i,j,k-1])/Δz
Ex^{n+1} += Cb_x × (psi_Exy^{n+1} - psi_Exz^{n+1})

# Ey:
psi_Eyz^{n+1} = b_z × psi_Eyz^n + c_z × (Hx[i,j,k] - Hx[i,j,k-1])/Δz
psi_Eyx^{n+1} = b_x × psi_Eyx^n + c_x × (Hz[i,j,k] - Hz[i-1,j,k])/Δx
Ey^{n+1} += Cb_y × (psi_Eyz^{n+1} - psi_Eyx^{n+1})

# Ez:
psi_Ezx^{n+1} = b_x × psi_Ezx^n + c_x × (Hy[i,j,k] - Hy[i-1,j,k])/Δx
psi_Ezy^{n+1} = b_y × psi_Ezy^n + c_y × (Hx[i,j,k] - Hx[i,j-1,k])/Δy
Ez^{n+1} += Cb_z × (psi_Ezx^{n+1} - psi_Ezy^{n+1})
```

### 8.2.5 Complete H-field CPML Equations

```
# Hx:
psi_Hxy^{n+1} = b_y × psi_Hxy^n + c_y × (Ez[i,j+1,k] - Ez[i,j,k])/Δy
psi_Hxz^{n+1} = b_z × psi_Hxz^n + c_z × (Ey[i,j,k+1] - Ey[i,j,k])/Δz
Hx^{n+1} += Db_x × (psi_Hxz^{n+1} - psi_Hxy^{n+1})

# Hy:
psi_Hyx^{n+1} = b_x × psi_Hyx^n + c_x × (Ez[i+1,j,k] - Ez[i,j,k])/Δx
psi_Hyz^{n+1} = b_z × psi_Hyz^n + c_z × (Ex[i,j,k+1] - Ex[i,j,k])/Δz
Hy^{n+1} += Db_y × (psi_Hyx^{n+1} - psi_Hyz^{n+1})

# Hz:
psi_Hzx^{n+1} = b_x × psi_Hzx^n + c_x × (Ey[i+1,j,k] - Ey[i,j,k])/Δx
psi_Hzy^{n+1} = b_y × psi_Hzy^n + c_y × (Ex[i,j+1,k] - Ex[i,j,k])/Δy
Hz^{n+1} += Db_z × (psi_Hzy^{n+1} - psi_Hzx^{n+1})
```
where Db = Δt/(μ₀·μ_r).

---

## 8.3 PML Grading Profiles

### 8.3.1 Polynomial Grading

Parameters graded from inner interface (d=0) to outer boundary (d=D):
```
σ(d) = σ_max × (d/D)^m          m = 3 or 4
κ(d) = 1 + (κ_max - 1) × (d/D)^m
α(d) = α_max × (1 - d/D)        linear decrease into PML
```

### 8.3.2 Optimal σ_max

```
σ_opt = -(m+1) × ln(R(0)) / (2η₀D)
```
For -40 dB one-way reflection target:
```
σ_opt = (m+1) / (150π·Δx) × c₀
```
Numerically (m=3, Δx=10nm): σ_opt ~ 1.13e9 S/m.

### 8.3.3 Recommended Ranges

| Parameter | Range | Notes |
|-----------|-------|-------|
| κ_max | 5-15 | Larger for evanescent-heavy problems |
| α_max | 0.02-0.05 S/m | Prevents late-time growth |
| m | 3-4 | Polynomial order for σ and κ |

### 8.3.4 Reflection vs. PML Thickness

| D (cells) | Theoretical R (dB) | Practical R (dB)* |
|-----------|--------------------|--------------------|
| 5         | -35                | -25 to -30         |
| 8         | -55                | -40 to -50         |
| 10        | -70                | -50 to -60         |
| 15        | -105               | -65 to -80         |
| 20        | -140               | -80 to -100        |

*Practical values include discretization error and fp32 arithmetic. Default: D=10.

---

## 8.4 GPU Implementation

### 8.4.1 Memory Layout

PML tensors allocated only for 6 boundary slabs:
```
Face ±x: shape (D, Ny, Nz)    Face ±y: shape (Nx, D, Nz)    Face ±z: shape (Nx, Ny, D)
```
Each face stores 4 psi tensors (2 E-field psi + 2 H-field psi for that axis).

### 8.4.2 Memory Budget

```
Total: 12 psi arrays × D × N² × 4 bytes (float32)
N=512, D=10: 12 × 10 × 512 × 512 × 4B = 125.8 MB
```
Coefficient arrays (b, c, 1/κ): 1D vectors of length D per axis -- negligible.

### 8.4.3 Kernel Strategy

**Option A -- Separate PML kernels:**
```python
def update_E_pml_xfaces(Ex, Ey, Ez, Hx, Hy, Hz, psi_x, b_x, c_x, kappa_x):
    # Map local p ∈ [0,D) to global: -x face → i=p, +x face → i=Nx-D+p
    psi_Eyx[p,j,k] = b_x[p] * psi_Eyx[p,j,k] + c_x[p] * dHz_dx
    psi_Ezx[p,j,k] = b_x[p] * psi_Ezx[p,j,k] + c_x[p] * dHy_dx
    Ey[i_global,j,k] += Cb * psi_Eyx[p,j,k]
    Ez[i_global,j,k] += Cb * psi_Ezx[p,j,k]
# Launch: grid=(ceil(D/4), ceil(Ny/8), ceil(Nz/8)), block=(4,8,8)=256 threads
```

**Option B -- Fused with main update kernel:**
```python
# Inside main E-field kernel:
if i < D or i >= Nx - D:
    p = i if i < D else i - (Nx - D)
    psi_Eyx[p,j,k] = b_x[p] * psi_Eyx[p,j,k] + c_x[p] * dHz_dx
    Ey[i,j,k] += Cb * psi_Eyx[p,j,k]
```
Fused avoids redundant global memory loads but introduces warp divergence at
PML/interior boundary. Profiling: 5-8% speedup on A100 for grids >= 256^3.

### 8.4.4 Index Mapping

```python
def pml_to_global(face: str, p: int, N: int, D: int) -> int:
    return p if face == 'lo' else N - D + p

# κ-modified finite difference in PML:
dHz_dx = (Hz[i,j,k] - Hz[i-1,j,k]) / (kappa_x[p] * dx)
```

---

## 8.5 PML Performance Impact

### 8.5.1 Cell Count Overhead

```
PML cells = 6·D·N² - 12·D²·N + 8·D³  (inclusion-exclusion)
```

| Grid | D | PML cells | Total | PML % |
|------|---|-----------|-------|-------|
| 128³ | 10 | 0.89M | 2.10M | 42% |
| 256³ | 10 | 3.77M | 16.78M | 22% |
| 512³ | 10 | 15.4M | 134.2M | 11.5% |
| 512³ | 8 | 12.4M | 134.2M | 9.2% |
| 1024³| 10 | 62.1M | 1073.7M | 5.8% |

### 8.5.2 Time Overhead

PML adds 2 FMA + 1 add per psi update (+4 FLOP per E/H component per PML cell).

| Grid | D | PML % | Time overhead | Kernel type |
|------|---|-------|---------------|-------------|
| 256³ | 10 | 22% | 18-22% | Separate |
| 256³ | 10 | 22% | 15-18% | Fused |
| 512³ | 10 | 11.5% | 13-15% | Separate |
| 512³ | 10 | 11.5% | 11-13% | Fused |
| 512³ | 8 | 9.2% | 9-11% | Fused |
| 1024³| 10 | 5.8% | 7-9% | Fused |

### 8.5.3 Optimization Strategies

1. **Fused PML kernel**: eliminates redundant global loads of H-field values already
   fetched for standard E-update. Saves 2 global loads per PML cell.
2. **fp16 psi storage**: halves PML memory; verified <0.1 dB reflection degradation for D>=8.
3. **Async streams**: overlap psi loads/stores with interior cell computation.
4. **Symmetry reduction**: omit PML on symmetry planes, up to 50% PML overhead reduction.
