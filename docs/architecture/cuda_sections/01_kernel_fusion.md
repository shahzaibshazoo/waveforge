# Section 1: Kernel Fusion Strategy

## 1.1 Why Fusion Matters for FDTD

FDTD operates at 0.16 FLOP/byte arithmetic intensity — firmly memory-bound. Each separate kernel launch:
- Reads field data from DRAM (HBM)
- Performs trivial arithmetic
- Writes results back to DRAM

If two sequential kernels read the same data, a fused kernel reads it ONCE, keeps it in registers, computes both results, writes once. This eliminates redundant DRAM round-trips — the primary bottleneck.

**Bandwidth savings from fusion:**
```
Separate kernels:  kernel_A reads X, writes Y; kernel_B reads Y, writes Z
  → Total traffic: read(X) + write(Y) + read(Y) + write(Z) = 4 × N bytes

Fused kernel: reads X, computes Y in registers, computes Z, writes Z
  → Total traffic: read(X) + write(Z) = 2 × N bytes

Savings: 50% bandwidth reduction → up to 2× speedup (memory-bound regime)
```

## 1.2 Fusion Candidates

| Fusion Pair | Bandwidth Saved | Estimated Speedup | Recommended |
|-------------|----------------|-------------------|-------------|
| E-update + PML (boundary) | 6 field re-reads | ~15% on boundary blocks | Yes |
| Curl + coeff multiply + accumulate | 2 intermediate writes | ~25% per component | Yes |
| Material lookup + field update | 1 extra material read | ~8% | Yes (if materials non-uniform) |
| DFT across N frequencies | (N-1) field re-reads | ~40% for N=8 | Yes |
| Source inject + field update | None (sparse vs dense) | Negative (divergence) | **No** |
| H-update + E-update (full step) | 6 writes + 6 reads | ~30% | Possible but complex |

## 1.3 Fusion Implementation Approaches

### torch.compile (Inductor Backend)

```python
@torch.compile(mode='max-autotune')
def update_E_fused(Ex, Ey, Ez, Hx, Hy, Hz, Ca_x, Cb_x, dt_dy, dt_dz):
    curl_x = (Hz[:, 1:, :] - Hz[:, :-1, :]) * dt_dy - (Hy[:, :, 1:] - Hy[:, :, :-1]) * dt_dz
    Ex[:, 1:, 1:] = Ca_x[:, 1:, 1:] * Ex[:, 1:, 1:] + Cb_x[:, 1:, 1:] * curl_x
    return Ex
```

**Limitations:** Inductor handles element-wise chains well but struggles with stencil shifts (roll/slice). It may not fuse the slice operations with the arithmetic.

### Triton JIT (Recommended Primary Path)

```python
@triton.jit
def fused_E_update_kernel(
    Ex_ptr, Hz_ptr, Hy_ptr, Ca_ptr, Cb_ptr,
    Nx, Ny, Nz, stride_y, stride_z,
    dt_dy, dt_dz,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Load stencil neighbors (fused reads)
    hz_ijk = tl.load(Hz_ptr + offsets)
    hz_jm1 = tl.load(Hz_ptr + offsets - stride_y)
    hy_ijk = tl.load(Hy_ptr + offsets)
    hy_km1 = tl.load(Hy_ptr + offsets - stride_z)
    
    # Fused curl + coefficient + accumulate
    ca = tl.load(Ca_ptr + offsets)
    cb = tl.load(Cb_ptr + offsets)
    ex = tl.load(Ex_ptr + offsets)
    
    curl = (hz_ijk - hz_jm1) * dt_dy - (hy_ijk - hy_km1) * dt_dz
    ex_new = ca * ex + cb * curl
    
    tl.store(Ex_ptr + offsets, ex_new)
```

**Advantages:** Full control over memory access pattern. Automatic block-size tuning. No intermediate tensor allocation.

### Custom CUDA Kernels (Maximum Performance)

Reserved for when Triton underperforms by >10% (measured). Required for:
- Shared memory tiling with explicit halo management
- Fused PML with block-level predication
- Warp-level primitives (`__shfl_sync` for neighbor exchange)

### Decision Tree

```
Is the operation element-wise or simple reduction?
  → Yes: torch.compile (zero effort)
  → No: Is it a stencil/neighbor pattern?
      → Yes: Triton (good control, fast iteration)
      → Triton measured <90% of roofline?
          → Yes: Custom CUDA (maximum performance)
          → No: Stay with Triton
```

## 1.4 Fused E-Field + PML Kernel Design

```cuda
__global__ void update_E_with_PML(
    float* Ex, float* Hz, float* Hy,
    float* Ca, float* Cb,
    float* psi_Exy, float* psi_Exz,
    float* b_y, float* c_y, float* b_z, float* c_z,
    int Nx, int Ny, int Nz, int pml_depth,
    float dt_dy, float dt_dz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= Nx || j >= Ny || k >= Nz) return;
    int idx = i * Ny * Nz + j * Nz + k;
    
    float dHz_dy = (Hz[idx] - Hz[idx - Nz]) * dt_dy;
    float dHy_dz = (Hy[idx] - Hy[idx - 1]) * dt_dz;
    
    // Standard E-field update
    float curl = dHz_dy - dHy_dz;
    Ex[idx] = Ca[idx] * Ex[idx] + Cb[idx] * curl;
    
    // PML correction (only for boundary cells)
    bool in_pml_y = (j < pml_depth) || (j >= Ny - pml_depth);
    bool in_pml_z = (k < pml_depth) || (k >= Nz - pml_depth);
    
    if (in_pml_y) {
        int pml_j = (j < pml_depth) ? j : (Ny - 1 - j);
        int psi_idx = i * pml_depth * Nz + pml_j * Nz + k;
        psi_Exy[psi_idx] = b_y[pml_j] * psi_Exy[psi_idx] + c_y[pml_j] * dHz_dy;
        Ex[idx] += Cb[idx] * psi_Exy[psi_idx];
    }
    if (in_pml_z) {
        int pml_k = (k < pml_depth) ? k : (Nz - 1 - k);
        int psi_idx = i * Ny * pml_depth + j * pml_depth + pml_k;
        psi_Exz[psi_idx] = b_z[pml_k] * psi_Exz[psi_idx] + c_z[pml_k] * dHy_dz;
        Ex[idx] += Cb[idx] * psi_Exz[psi_idx];
    }
}
```

**Block-level optimization:** For interior thread blocks (no thread has `in_pml_*` true), the branches are uniformly NOT taken → zero divergence, zero PML overhead. Only boundary blocks (~11% of total) execute PML logic.

## 1.5 Fusion Anti-Patterns

| Anti-Pattern | Problem | Alternative |
|-------------|---------|-------------|
| Fuse across streams | Breaks async overlap (I/O, detect) | Keep on separate streams |
| Fuse sparse + dense | 99% threads idle on sparse path → warp divergence | Separate kernels |
| Fuse until register spill | Spills to local memory (DRAM speed) → slower than 2 kernels | Limit fusion depth |
| Fuse incompatible block sizes | Optimal block differs per operation | Profile separately first |
| Fuse DFT + field update | DFT only at detector cells (sparse) | Separate gather + DFT kernel |

**Rule of thumb:** Fuse operations that touch the SAME cells with the SAME access pattern. Don't fuse operations with fundamentally different parallelism structures.
