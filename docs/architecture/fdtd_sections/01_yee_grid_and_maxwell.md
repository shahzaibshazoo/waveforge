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
