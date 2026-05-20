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