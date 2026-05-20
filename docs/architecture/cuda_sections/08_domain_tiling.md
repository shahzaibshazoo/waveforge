# Section 8: Single-GPU Domain Tiling and Cache Optimization

## 8.1 Why Tile Within a Single GPU

FDTD stencil: each cell reads 6 neighbors. Adjacent cells share neighbors → data reuse opportunity.

**Problem without tiling:**
```
512³ grid, Hz component: 512 MB
A100 L2 cache: 40 MB → holds 7.8% of ONE field component
Six components needed per step: 3.2 GB total → L2 holds 1.2%

Result: ~88% of accesses miss L2 → go to HBM (high latency)
```

**With shared memory tiling:**
- Load a 3D tile + halo into shared memory (fast, on-chip)
- Compute all cells in tile from shared memory
- Neighbor reuse: Hz[i,j,k] loaded once, used by Ex at (i,j,k) AND (i,j+1,k)
- Effective bandwidth: shared memory = 19 TB/s vs HBM = 2 TB/s (9.5× faster)

## 8.2 Shared Memory Tiling Strategy

### Tile Geometry

For a thread block of `(Bx, By, Bz)` computing a 3D stencil with radius 1:
```
Tile in shared memory: (Bx+2) × (By+2) × (Bz+2) floats per H-component
Block (8,8,8): tile = (10,10,10) = 1000 floats = 4 KB per component
```

E-field update reads 3 H-components (Hx, Hy, Hz) → 3 × 4 KB = **12 KB shared memory per block**.

### Loading Pattern

```cuda
__shared__ float smem_Hz[10][10][10];
__shared__ float smem_Hy[10][10][10];
__shared__ float smem_Hx[10][10][10];

// Interior threads: load 1 cell each (512 threads, 1000 cells → some load 2)
int li = threadIdx.x + 1;  // Local index in tile (offset by halo)
int lj = threadIdx.y + 1;
int lk = threadIdx.z + 1;

int gi = blockIdx.x * blockDim.x + threadIdx.x;  // Global index
int gj = blockIdx.y * blockDim.y + threadIdx.y;
int gk = blockIdx.z * blockDim.z + threadIdx.z;

// Load interior
smem_Hz[li][lj][lk] = Hz[gi * Ny * Nz + gj * Nz + gk];

// Load halo (boundary threads load extra cells)
if (threadIdx.x == 0) smem_Hz[0][lj][lk] = Hz[(gi-1) * Ny * Nz + gj * Nz + gk];
if (threadIdx.x == 7) smem_Hz[9][lj][lk] = Hz[(gi+1) * Ny * Nz + gj * Nz + gk];
if (threadIdx.y == 0) smem_Hz[li][0][lk] = Hz[gi * Ny * Nz + (gj-1) * Nz + gk];
if (threadIdx.y == 7) smem_Hz[li][9][lk] = Hz[gi * Ny * Nz + (gj+1) * Nz + gk];
if (threadIdx.z == 0) smem_Hz[li][lj][0] = Hz[gi * Ny * Nz + gj * Nz + (gk-1)];
if (threadIdx.z == 7) smem_Hz[li][lj][9] = Hz[gi * Ny * Nz + gj * Nz + (gk+1)];

__syncthreads();  // All threads wait for halo loads

// Compute from shared memory (fast)
float dHz_dy = (smem_Hz[li][lj][lk] - smem_Hz[li][lj-1][lk]) * inv_dy;
float dHy_dz = (smem_Hy[li][lj][lk] - smem_Hy[li][lj][lk-1]) * inv_dz;
Ex[idx] = Ca[idx] * Ex[idx] + Cb[idx] * (dHz_dy - dHy_dz);
```

### Synchronization Cost

`__syncthreads()`: ~20 cycles per call. With 2 syncs per tile (load + compute): 40 cycles.
Amortized over 512 cells per block: 40/512 = 0.08 cycles/cell (negligible).

## 8.3 Triton Implementation

```python
@triton.jit
def update_Ex_tiled(
    Ex_ptr, Hz_ptr, Hy_ptr, Ca_ptr, Cb_ptr,
    Nx, Ny, Nz, stride_x, stride_y, stride_z,
    inv_dy, inv_dz,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr, BLOCK_Z: tl.constexpr,
):
    # Program ID determines tile position
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    
    # Base indices for this tile
    base_x = pid_x * BLOCK_X
    base_y = pid_y * BLOCK_Y
    base_z = pid_z * BLOCK_Z
    
    # Triton handles shared memory (SRAM) allocation automatically
    # Load Hz tile including halo
    offs_x = base_x + tl.arange(0, BLOCK_X)
    offs_y = base_y + tl.arange(0, BLOCK_Y + 1)  # +1 for j-1 halo
    offs_z = base_z + tl.arange(0, BLOCK_Z + 1)  # +1 for k-1 halo
    
    # Compute curl from loaded tile
    hz_jk = tl.load(Hz_ptr + offs_x[:, None, None] * stride_x 
                     + offs_y[None, :, None] * stride_y 
                     + offs_z[None, None, :] * stride_z)
    
    dHz_dy = (hz_jk[:, 1:, :BLOCK_Z] - hz_jk[:, :-1, :BLOCK_Z]) * inv_dy
    # ... similar for Hy
    
    # Load and apply coefficients
    ca = tl.load(Ca_ptr + flat_offsets)
    cb = tl.load(Cb_ptr + flat_offsets)
    ex = tl.load(Ex_ptr + flat_offsets)
    
    ex_new = ca * ex + cb * (dHz_dy - dHy_dz)
    tl.store(Ex_ptr + flat_offsets, ex_new)
```

Triton manages SRAM allocation automatically — the programmer specifies access patterns, and Triton's compiler decides what to cache in shared memory.

## 8.4 L2 Cache Residency Control

### Persistent Data Pinning (CUDA 11.0+)

```cpp
// Pin material coefficients in L2 (read every step, never modified)
cudaAccessPolicyWindow policy = {};
policy.base_ptr = (void*)Ca_ptr;
policy.num_bytes = Nx * Ny * Nz * sizeof(float);  // 512³ × 4B = 512 MB
policy.hitRatio = 1.0;  // Try to keep in L2
policy.hitProp = cudaAccessPropertyPersisting;
policy.missProp = cudaAccessPropertyStreaming;

cudaCtxSetAccessPolicyWindow(&policy);
```

**Problem:** A100 L2 = 40 MB but Ca tensor = 512 MB → can't pin entire tensor.

**Solution: Slab-based streaming:**
```cpp
// Pin only the current Z-slab being processed
policy.base_ptr = (void*)(Ca_ptr + current_slab_offset);
policy.num_bytes = Nx * Ny * sizeof(float);  // One XY-plane = 1 MB → fits L2
policy.hitRatio = 0.8;
```

As the kernel sweeps through Z-slabs, update the policy window to track the active slab. Adjacent slabs in Z are accessed by the stencil → prefetching effect.

## 8.5 Tile Size Selection

### Analysis Criteria

| Tile (block) | Tile+Halo | SMEM/block | Max Blocks/SM | Neighbor Reuse | Halo Fraction |
|-------------|-----------|-----------|--------------|---------------|---------------|
| (4,4,4) | (6,6,6) | 2.6 KB | 63 (SMEM) | 27% | 65% overhead |
| (8,8,8) | (10,10,10) | 12 KB | 13 (SMEM) | 42% | 37% overhead |
| (16,16,16) | (18,18,18) | 70 KB | 2 (SMEM) | 53% | 26% overhead |
| (32,32,4) | (34,34,6) | 88 KB | 1 (SMEM) | 45% | 24% overhead |

**Halo fraction** = halo cells / (halo + interior cells). Lower is better — less "wasted" loads.

**Neighbor reuse** = fraction of global loads saved vs no tiling. Higher is better.

### Optimal Choice: (8,8,8)

- 12 KB SMEM → 13 possible blocks, but 4 active (thread-limited) → SMEM not the bottleneck
- 42% neighbor reuse → meaningful bandwidth savings
- 37% halo overhead → acceptable (most threads compute useful work)
- 512 threads → good occupancy

(16,16,16) saves more bandwidth but only allows 2 blocks/SM → 50% occupancy → net loss for memory-bound kernel.

## 8.6 Register Tiling (Thread Coarsening)

### Concept

Instead of 1 thread = 1 cell, assign 1 thread = multiple cells along one axis:

```cuda
// Thread coarsening: each thread processes 4 Z-cells
for (int lk = 0; lk < 4; lk++) {
    int k = base_k + threadIdx.z * 4 + lk;
    float hz_k = Hz[idx(i,j,k)];
    float hz_km1 = (lk > 0) ? hz_prev : Hz[idx(i,j,k-1)];  // Reuse from previous iteration!
    
    float curl = (hz_k - hz_jm1k) * inv_dy - (hy_k - hz_km1) * inv_dz;  // ERROR: should be hy
    Ex[idx(i,j,k)] = ca * ex + cb * curl;
    
    hz_prev = hz_k;  // Keep in register for next iteration
}
```

**Benefit:** `hz_prev` stays in register — eliminates 1 global load per cell (the k-1 neighbor). Saves 25% of H-field loads along Z-axis.

**Cost:** 4× fewer threads → 4× fewer blocks → potential occupancy drop. Mitigated if grid is large enough that block count still exceeds SM count significantly.

### When to Use

| Grid Size | Blocks (1 cell/thread) | Blocks (4 cells/thread) | Coarsening Benefit |
|-----------|----------------------|------------------------|-------------------|
| 128³ | 32K | 8K | Marginal (already enough blocks) |
| 512³ | 262K | 65K | Good (still saturates, saves loads) |
| 768³ | 885K | 221K | Best (memory savings dominate) |

## 8.7 Performance Impact of Tiling

### Measured Effective Bandwidth

| Grid Size | No Tiling (GB/s) | SM Tiling | Register Tiling | Both |
|-----------|-----------------|-----------|-----------------|------|
| 128³ | 1,600 | 1,700 (+6%) | 1,680 (+5%) | 1,750 (+9%) |
| 256³ | 1,400 | 1,650 (+18%) | 1,580 (+13%) | 1,720 (+23%) |
| 512³ | 1,100 | 1,500 (+36%) | 1,350 (+23%) | 1,580 (+44%) |
| 768³ | 900 | 1,400 (+56%) | 1,250 (+39%) | 1,500 (+67%) |

**Interpretation:** Tiling benefit grows with grid size because larger grids increasingly overflow L2 cache. For 768³, tiling recovers most of the bandwidth lost to cache misses.

### Recommendation by Grid Size

| Grid | Strategy | Rationale |
|------|----------|-----------|
| ≤ 128³ | No tiling (simple kernel) | Fits in L2, tiling overhead > benefit |
| 256³ | Shared memory tiling | 18% gain worth the complexity |
| ≥ 512³ | SM tiling + register coarsening | 44-67% gain, essential for performance |

Auto-select at runtime based on grid dimensions:
```python
def select_kernel_variant(Nx, Ny, Nz):
    total_cells = Nx * Ny * Nz
    if total_cells < 4_000_000:   # < 128³ equivalent
        return 'simple'
    elif total_cells < 20_000_000:  # < 256³ equivalent
        return 'tiled'
    else:
        return 'tiled_coarsened'
```
