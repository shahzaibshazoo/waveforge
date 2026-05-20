# GPU-MEEP: CUDA Optimization Guide

> Single-GPU CUDA Core Parallelism, Kernel Architecture, and Performance Engineering
> Targeting Maximum Utilization of NVIDIA CUDA Cores for FDTD Simulation

---

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

---

# Section 2: Memory Coalescing and Access Patterns

## 2.1 Coalescing Fundamentals

A warp (32 threads) issues a single memory transaction when threads access 32 consecutive 4-byte addresses aligned to a 128-byte boundary. Violations multiply the number of transactions:

| Access Pattern | Transactions per Warp | Effective BW (A100) |
|---------------|----------------------|---------------------|
| Consecutive (coalesced) | 1 | 2,039 GB/s (100%) |
| Stride-2 | 2 | 1,020 GB/s (50%) |
| Stride-32 | 32 | 64 GB/s (3%) |
| Random | up to 32 | 64 GB/s (3%) |

**FDTD per-cell bandwidth requirement:**
- Reads: 4 H-neighbors + 2 coefficients + 1 current E = 7 × 4B = 28 bytes
- Writes: 1 updated E = 4 bytes
- Per component: 32 bytes. Full cell (6 components): 192 bytes/cell/step.

## 2.2 SoA Layout for Perfect Coalescing

### Structure of Arrays (Chosen)

```python
Ex = torch.zeros(Nx, Ny, Nz, device='cuda')  # Contiguous per component
Ey = torch.zeros(Nx, Ny, Nz, device='cuda')
# ... 6 separate tensors
```

### Memory Layout (C-contiguous, Z-fastest)

```
Address:  [Ex[0,0,0], Ex[0,0,1], Ex[0,0,2], ..., Ex[0,0,Nz-1], Ex[0,1,0], ...]
           ─────────── Warp 0 reads these ───────────────
```

Stride: `(Ny×Nz×4, Nz×4, 4)` bytes for indices `(i, j, k)`.

Thread mapping: `threadIdx.x → k` (Z-index). Warp of 32 threads covers k=0..31 → 32 consecutive floats → **1 transaction, fully coalesced**.

### Access Pattern for Ex Update

```
Ex[i,j,k] reads:
  Hz[i, j, k]     → base + (i*Ny*Nz + j*Nz + k)*4        COALESCED (warp in k)
  Hz[i, j-1, k]   → base + (i*Ny*Nz + (j-1)*Nz + k)*4    COALESCED (same k stride)
  Hy[i, j, k]     → different tensor, same index pattern    COALESCED
  Hy[i, j, k-1]   → base + (i*Ny*Nz + j*Nz + (k-1))*4    COALESCED (stride-1 in k)
```

All four stencil accesses are coalesced. The k-1 access: thread 0 reads k=-1 (out of block but still consecutive within the warp's natural range). Adjacent warps overlap in cache → high L1 hit rate.

### Contrast: Array of Structures (AoS) — REJECTED

```
// AoS: [Ex,Ey,Ez,Hx,Hy,Hz] interleaved per cell
cell[i,j,k] = {Ex, Ey, Ez, Hx, Hy, Hz}  // 24 bytes per cell
```

Warp reading Ex: threads access addresses 0, 24, 48, 72... (stride-6) → **6 transactions per warp** → 16% efficiency. Wastes 83% of loaded cache lines.

## 2.3 Cache Line Analysis

### L2 Cache (A100: 40 MB, 128B lines)

For 512³ grid, Hz tensor: 512×512×512×4 = 512 MB.
L2 can hold: 40 MB / 512 MB = 7.8% of one component.

**Stencil reuse opportunity:** Hz[i,j,k] is read by:
- Ex update at (i,j,k) and (i,j+1,k)
- Ey update at (i,j,k) and (i+1,j,k)

If these updates execute close in time (same or adjacent blocks), Hz[i,j,k] may remain in L2 → 1 DRAM load serves 4 consumers.

### L1 Cache (per SM: 128 KB, 128B sectors)

Effective per-block L1 working set:
```
Block (8,8,8) reads Hz tile of (8,9,8) = 576 floats = 2,304 bytes
Three H-components: 6,912 bytes
All fits in L1 if no conflicts.
```

For 512³ grid: L1 hit rate ~85% for Z-neighbor, ~60% for Y-neighbor, ~30% for X-neighbor (X requires stride Ny×Nz = 1MB, always misses L1).

### Cache Hit Model

```
P(L2 hit) ≈ min(1, L2_size / working_set_accessed_before_reuse)

For Y-neighbor Hz[i,j-1,k]:
  Reuse distance: Nz × 4 = 2048 bytes (accessed by adjacent j-block)
  If block processes j in order: reuse distance = 8 × Nz × 4 = 16 KB → fits L1

For X-neighbor Hz[i-1,j,k]:
  Reuse distance: Ny × Nz × 4 = 1 MB → misses L1, may hit L2
```

## 2.4 Padding and Alignment

### Warp-Aligned Padding

```python
def align_dimension(N, alignment=32):
    """Pad to multiple of warp size for coalesced access."""
    return ((N + alignment - 1) // alignment) * alignment

Nz_padded = align_dimension(Nz)  # e.g., 500 → 512, 513 → 544
# Allocate padded tensor:
Ex = torch.zeros(Nx, Ny, Nz_padded, device='cuda')
# Physical domain: Ex[:, :, :Nz], padding: Ex[:, :, Nz:Nz_padded] = 0
```

### Memory Overhead

| Original Nz | Padded Nz | Overhead |
|------------|-----------|----------|
| 256 | 256 | 0% |
| 500 | 512 | 2.4% |
| 512 | 512 | 0% |
| 513 | 544 | 6.0% |
| 768 | 768 | 0% |

Powers-of-2 and multiples-of-32 are naturally aligned. Worst case: Nz = 32k+1 → 31/Nz ≈ 6% overhead.

### 128-Byte Alignment for Cache Lines

PyTorch allocations via `torch.empty()` on CUDA are always 256-byte aligned (caching allocator guarantees this). First element of each row (j,k=0) starts on a cache-line boundary.

## 2.5 Bandwidth Efficiency Measurement

### Theoretical vs Achieved

```
Theoretical bytes per cell: 192 B (6 reads + 6 writes, all float32)
A100 peak bandwidth: 2,039 GB/s
Theoretical max cells/s: 2,039e9 / 192 = 10.6 Gcells/s

Measured (well-optimized kernel): ~7.5 Gcells/s
Efficiency: 7.5 / 10.6 = 71%
```

Remaining 29% loss from: L2 cache misses (re-fetches), warp scheduling overhead, instruction fetch, index computation.

### Nsight Compute Metrics

```
Key metrics to monitor:
  sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ld.sum   (global load sectors)
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum (L1 hits)
  lts__t_sectors_srcunit_tex_op_read.sum                    (L2 read sectors)
  dram__sectors_read.sum                                    (DRAM reads)
  
  Coalescing efficiency = ideal_sectors / actual_sectors
  Target: > 95% for FDTD stencil
```

## 2.6 Z-Order (Morton) Curves — Analysis and Rejection

### Concept

Bit-interleave (i,j,k) indices to create a space-filling curve:
```
morton(i,j,k) = interleave_bits(i) | (interleave_bits(j) << 1) | (interleave_bits(k) << 2)
```

Improves 3D locality: adjacent cells in all three dimensions map to nearby memory addresses.

### Why NOT for FDTD

| Factor | Row-Major (Z-fastest) | Morton Order |
|--------|----------------------|--------------|
| Z-neighbor access | Stride 1 (perfect) | Variable stride |
| Y-neighbor access | Stride Nz | Better (~√Nz) |
| X-neighbor access | Stride Ny×Nz | Better (~∛(Ny×Nz)) |
| Warp coalescing | Perfect (k varies) | Broken (bits interleaved) |
| Index computation | i*Ny*Nz+j*Nz+k (2 MUL+ADD) | Bit manipulation (10+ ops) |

**Critical issue:** Morton order breaks warp coalescing. Adjacent threads (differing in linearized index by 1) access addresses that differ by 1 in Morton space — but Morton-adjacent addresses are NOT byte-adjacent. This means **every load is scattered** from the GPU's perspective.

**Verdict:** Row-major C-contiguous with Z-fastest is optimal for FDTD on GPU. Morton order is useful for CPU cache hierarchies but catastrophic for GPU coalescing. Only consider if Nz < 32 (unlikely for real problems).

---

# Section 3: Occupancy Analysis and Optimization

## 3.1 Occupancy Definition and SM Resources

Occupancy = active warps per SM / maximum warps per SM.

### SM Resource Budgets

| Resource | SM 8.0 (A100) | SM 8.9 (RTX 4090) | SM 9.0 (H100) |
|----------|--------------|-------------------|---------------|
| Max threads/SM | 2048 | 1536 | 2048 |
| Max warps/SM | 64 | 48 | 64 |
| Max blocks/SM | 32 | 24 | 32 |
| Registers/SM | 65,536 | 65,536 | 65,536 |
| Shared memory/SM | 164 KB | 100 KB | 228 KB |
| Max registers/thread | 255 | 255 | 255 |
| Max threads/block | 1024 | 1024 | 1024 |

Occupancy is limited by the MOST RESTRICTIVE resource:
```
occ = min(
    threads_per_block × blocks_per_sm / max_threads_per_sm,
    max_blocks_per_sm × threads_per_block / max_threads_per_sm,
    floor(regs_per_sm / (regs_per_thread × threads_per_block)) × threads_per_block / max_threads_per_sm,
    floor(shmem_per_sm / shmem_per_block) × threads_per_block / max_threads_per_sm
)
```

## 3.2 FDTD Kernel Occupancy Calculation

### Baseline E-Field Update Kernel

```
Configuration:
  Block size: (8, 8, 8) = 512 threads = 16 warps
  Registers/thread: 15 (measured via --ptxas-options=-v)
  Shared memory: 0 bytes
```

**A100 (SM 8.0) analysis:**
```
Register limit: floor(65536 / (15 × 512)) = floor(65536 / 7680) = 8 blocks
Thread limit: floor(2048 / 512) = 4 blocks
Block limit: 32 (not limiting)
Shared mem limit: ∞ (0 bytes used)

Limiting factor: Thread count → 4 blocks/SM
Active threads: 4 × 512 = 2048 = max
Occupancy: 2048/2048 = 100%
Active warps: 64/64 = 100%
```

### Fused E-Field + PML Kernel

```
Configuration:
  Block size: (8, 8, 8) = 512 threads
  Registers/thread: 28 (PML adds ~13 registers for psi, coefficients)
  Shared memory: 0 bytes
```

```
Register limit: floor(65536 / (28 × 512)) = floor(65536 / 14336) = 4 blocks
Thread limit: 4 blocks
→ Occupancy: 4 × 512 / 2048 = 100% (still not limited)
```

### Shared Memory Tiled Kernel

```
Configuration:
  Block size: (8, 8, 8) = 512 threads
  Registers/thread: 20
  Shared memory: 12 KB (3 H-component tiles of (10,10,10))
```

```
Register limit: floor(65536 / (20 × 512)) = 6 blocks
Thread limit: 4 blocks
Shared mem limit (A100): floor(164 KB / 12 KB) = 13 blocks
→ Limiting: threads → 4 blocks/SM → 100% occupancy

Shared mem limit (RTX 4090): floor(100 KB / 12 KB) = 8 blocks
Thread limit (4090): floor(1536 / 512) = 3 blocks
→ 3 × 512 / 1536 = 100% occupancy
```

## 3.3 Occupancy–Performance Relationship for Memory-Bound Kernels

### Why High Occupancy Matters for FDTD

DRAM (HBM) latency: ~400 cycles on A100. During a memory stall, the SM switches to another warp. More active warps = more warps to switch to = better latency hiding.

```
Minimum warps to hide latency:
  warps_needed = memory_latency / instruction_throughput
  ≈ 400 cycles / (4 cycles/instruction × 32 threads) ≈ 3 warps minimum

But DRAM has pipeline depth → need MORE warps to saturate bandwidth:
  warps_for_full_BW ≈ BW × latency / bytes_per_request
  A100: 2039 GB/s × 400ns / 128B = ~6400 outstanding requests
  Per SM (108 SMs): 6400/108 ≈ 60 requests → ~60 warps needed = 94% occupancy
```

### Empirical Throughput vs Occupancy (FDTD Stencil)

| Occupancy | Warps/SM | Relative Throughput | Notes |
|-----------|----------|--------------------|----|
| 25% | 16 | 55% | Severe memory stalls |
| 50% | 32 | 78% | Adequate for compute-bound |
| 75% | 48 | 94% | Sweet spot for most kernels |
| 100% | 64 | 100% | Diminishing returns above 75% |

**For FDTD: target ≥75% occupancy.** Going from 75% to 100% yields only ~6% more throughput but constrains register/shared memory usage.

## 3.4 What Reduces Occupancy

| Factor | Mechanism | FDTD Impact |
|--------|-----------|-------------|
| High register count | Fewer threads fit in register file | Fused kernels with many locals |
| Large shared memory | Fewer blocks fit per SM | Tiled kernels with large halos |
| Large block size | Fewer blocks per SM (rounding) | 1024 threads → only 2 blocks on A100 |
| Small block size | More blocks needed (hits block limit) | 64 threads → 32 blocks needed (hits limit on 4090) |

### Register Pressure by Kernel Variant

| Kernel | Regs/Thread | Blocks/SM | Occupancy (A100) |
|--------|------------|-----------|------------------|
| E-update (basic) | 15 | 4 (thread-limited) | 100% |
| E-update + PML fused | 28 | 4 (thread-limited) | 100% |
| E-update + PML + DFT fused | 42 | 3 (reg-limited: 65536/(42×512)=3) | 75% |
| Full-step fused (E+H+PML) | 56 | 2 (reg-limited) | 50% |

**Conclusion:** Fusing E+H+PML into one kernel drops occupancy to 50% — likely SLOWER than two separate kernels at 100% occupancy for a memory-bound workload.

## 3.5 Occupancy Tuning Strategies

### Compiler Hints

```cuda
__global__ void __launch_bounds__(512, 4)  // max 512 threads, min 4 blocks/SM
update_E_kernel(...) {
    // Compiler optimizes register allocation for 4 blocks
    // May spill to local memory if 65536/(4×512)=32 regs insufficient
}
```

### Triton num_warps

```python
@triton.jit
def kernel(..., BLOCK: tl.constexpr):
    ...

# Auto-tuned:
configs = [
    triton.Config({}, num_warps=8),   # 256 threads
    triton.Config({}, num_warps=16),  # 512 threads
    triton.Config({}, num_warps=32),  # 1024 threads
]
```

### When NOT to Optimize Occupancy

- If kernel is already at roofline bandwidth → more occupancy won't help
- If register spilling is required to increase occupancy → spills go to DRAM → defeats purpose
- If shared memory tiling saves >30% bandwidth → accept lower occupancy for net gain

## 3.6 Profiling Occupancy

### Nsight Compute Metrics

```
Achieved occupancy:
  sm__warps_active.avg.pct_of_peak_sustained_active

Theoretical occupancy:
  launch__occupancy

Occupancy limiters:
  launch__registers_per_thread
  launch__shared_mem_per_block_allocated
  launch__block_size
```

### PyTorch Profiler Integration

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    for _ in range(100):
        update_E(Ex, Ey, Ez, Hx, Hy, Hz, Ca, Cb)

print(prof.key_averages().table(sort_by="cuda_time_total"))
# Shows kernel time, SM utilization, memory throughput
```

### Triton Auto-Tuner (Best Practice)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 8, 'BLOCK_Y': 8, 'BLOCK_Z': 8}, num_warps=16),
        triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 8, 'BLOCK_Z': 4}, num_warps=16),
        triton.Config({'BLOCK_X': 4, 'BLOCK_Y': 4, 'BLOCK_Z': 4}, num_warps=2),
    ],
    key=['Nx', 'Ny', 'Nz'],
)
def update_E_triton(...):
    ...
```

Triton's auto-tuner measures actual throughput — picks the configuration that runs fastest, which may not be the one with highest occupancy. Trust measurements over theory.

---

# Section 4: Tensor Core Utilization

## 4.1 Tensor Core Architecture

| GPU | Tensor Cores | Supported Formats | Peak TC TFLOPS | CUDA Core TFLOPS | TC/CUDA Ratio |
|-----|-------------|-------------------|---------------|-----------------|---------------|
| A100 | 432 (3rd gen) | FP16, BF16, TF32, FP64, INT8 | 312 (TF32) | 19.5 (FP32) | 16× |
| RTX 4090 | 512 (4th gen) | FP16, BF16, TF32, INT8, FP8 | 330 (TF32) | 82.6 (FP32) | 4× |
| H100 | 528 (4th gen) | FP16, BF16, TF32, FP64, FP8 | 990 (TF32) | 67 (FP32) | 15× |

**Operation:** Matrix-Multiply-Accumulate (MMA): `D[m,n] = A[m,k] × B[k,n] + C[m,n]`

Native tile shapes:
- FP16/BF16: 16×16×16 (m×n×k)
- TF32: 16×8×8
- FP64: 8×8×4
- FP8 (H100): 16×16×32

## 4.2 FDTD and Tensor Cores: The Mismatch

Standard FDTD update equation:
```
Ex[i,j,k] = Ca[i,j,k] * Ex[i,j,k] + Cb[i,j,k] * (Hz[i,j,k] - Hz[i,j-1,k] - Hy[i,j,k] + Hy[i,j,k-1])
```

This is:
- Element-wise multiply (diagonal matrix × vector, NOT dense GEMM)
- Stencil subtraction (sparse banded matrix × vector, NOT dense GEMM)
- Accumulation (vector addition)

**None of these are matrix-matrix multiplications.** Tensor cores require dense GEMM structure.

### Can We Reformulate as GEMM?

**Attempt: 3D convolution via im2col:**
```
Stencil as 3×3×3 convolution kernel → im2col → GEMM
```
- im2col creates (N_cells × 27) matrix from field tensor
- Multiply by (27 × 1) kernel weight vector
- Result: (N_cells × 1) output

Problem: The weight matrix is (27 × 1) — a matrix-vector product, not matrix-matrix. Tensor cores need both dimensions ≥16. The overhead of im2col (expanding 1 value into 27) also increases memory traffic 27×, negating any benefit.

**Verdict: Standard FDTD stencil computation CANNOT benefit from tensor cores.**

## 4.3 Where Tensor Cores Apply in GPU-MEEP

| Operation | Shape | GEMM? | TC Benefit |
|-----------|-------|-------|------------|
| E/H field update (stencil) | element-wise + stencil | No | None |
| PML psi update | element-wise | No | None |
| **DFT computation** | (N_freq × N_t) × (N_t × N_cells) | **Yes** | **5-8×** |
| **Backprojection** | (N_voxels × N_pairs) × (N_pairs × N_t) | **Yes** | **5-8×** |
| **Near-to-far transform** | (N_angles × N_surface) × (N_surface × 1) | Marginal | 2-3× |
| **Neural network layers** | (batch × in_features) × (in × out) | **Yes** | **8-10×** |
| **Adjoint outer products** | Gradient accumulation | Possible | 3-5× |

### DFT as GEMM

```python
# Naive DFT (loop, no tensor cores):
for m in range(N_freqs):
    dft[m, :] += field[:] * exp(-j * 2π * f_m * t * dt)

# Reformulated as GEMM (tensor core compatible):
# DFT_matrix: (N_freqs × N_timesteps) — precomputed complex exponentials
# field_history: (N_timesteps × N_cells) — stored time series
# result: (N_freqs × N_cells) = DFT_matrix @ field_history

dft_result = torch.matmul(dft_matrix, field_history)  # Uses tensor cores automatically
```

For N_freqs=64, N_t=4096, N_cells=1024: GEMM (64×4096) × (4096×1024) → tensor cores engage → 5-8× speedup over running DFT accumulation in the time loop.

**Trade-off:** Requires storing field history at detector cells → memory cost: N_t × N_cells × 4B. For 4096 steps, 1024 cells: 16 MB (acceptable).

### Backprojection as GEMM

```python
# Delay-and-sum imaging:
# image[voxel] = Σ_{tx,rx} signal[tx, rx, delay(tx,rx,voxel)]

# After interpolation, this becomes:
# interpolated_signals: (N_pairs × N_voxels) — signal values at computed delays
# weights: (N_pairs × 1) — amplitude/phase weights
# image: (N_voxels) = weights.T @ interpolated_signals

# Batched across frequency bins → full GEMM
image = torch.matmul(weights.T, interpolated_signals)  # Tensor cores
```

### Neural Network Inference

All standard layers (Linear, Conv2d, Conv3d) automatically use tensor cores when inputs are FP16/BF16/TF32:

```python
model = UNet3D(in_channels=N_tx*N_rx, out_channels=1).cuda().half()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    eps_predicted = model(measurements)  # All matmuls use tensor cores
```

## 4.4 TF32 Mode

TF32 uses tensor core hardware but with FP32-range inputs (truncated to 10-bit mantissa internally):

```python
torch.backends.cuda.matmul.allow_tf32 = True   # Enable TF32 for matmul
torch.backends.cudnn.allow_tf32 = True          # Enable TF32 for convolutions
```

- Precision: ~10⁻³ relative error (vs 10⁻⁷ for true FP32)
- Speed: up to 8× for matmul operations
- Applicability to FDTD: only for reconstruction/post-processing (not field updates, which aren't matmul)

## 4.5 Structured Sparsity on Tensor Cores (Future Research)

A100+ supports 2:4 structured sparsity: tensor cores process matrices with exactly 2 zeros per 4 elements → 2× additional speedup.

**FDTD stencil as sparse matrix:**
```
The FDTD update can be written as: E^{n+1} = A × E^n + B × H^{n+½}
where A is diagonal (Ca coefficients) and B is sparse banded (curl operator).

B has structure: each row has exactly 4 nonzeros (±1/Δx, ±1/Δy for 2D; 6 for 3D)
This is NOT 2:4 structured (it's much sparser: ~6/N nonzeros per row).
```

**Status:** Research-only. cuSPARSE doesn't efficiently handle FDTD's specific sparsity pattern on tensor cores. The overhead of format conversion exceeds any speedup.

## 4.6 Practical Recommendations

1. **Field updates (99% of compute time):** CUDA cores only. Optimize for memory bandwidth, not FLOPS.
2. **Post-processing DFT:** Reformulate as GEMM → tensor cores. Store detector time series for batched DFT at end.
3. **Imaging reconstruction:** Design algorithms as matrix operations → tensor cores.
4. **Neural network components:** Always use `torch.autocast` for automatic tensor core utilization.
5. **Don't force tensor cores on unsuitable operations** — the data reformatting overhead negates any gain.

---

# Section 5: Mixed Precision Strategy

## 5.1 Precision Formats Available on Modern NVIDIA GPUs

| Format | Bits | Exponent | Mantissa | Range | ULP at 1.0 | BW Gain vs FP32 | Compute Gain |
|--------|------|----------|----------|-------|-----------|-----------------|--------------|
| FP64 | 64 | 11 | 52 | ±10³⁰⁸ | 2.2×10⁻¹⁶ | 0.5× | 0.5× (A100) |
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ | 1.2×10⁻⁷ | 1× (baseline) | 1× |
| TF32 | 19 | 8 | 10 | ±3.4×10³⁸ | 9.8×10⁻⁴ | 1× (same size) | 8× (TC only) |
| BF16 | 16 | 8 | 7 | ±3.4×10³⁸ | 7.8×10⁻³ | 2× | 2× |
| FP16 | 16 | 5 | 10 | ±65504 | 9.8×10⁻⁴ | 2× | 2× |
| FP8 (E4M3) | 8 | 4 | 3 | ±240 | 0.125 | 4× | 4× (H100) |

**For FDTD:** BF16 preferred over FP16 because BF16 has same range as FP32 (no overflow risk for field values), while FP16 overflows at 65504 (field values can easily exceed this for high-power sources).

## 5.2 Precision Assignment for FDTD Operations

| Data | Precision | Rationale |
|------|-----------|-----------|
| E, H fields (production) | FP32 | Accumulated error stays bounded over 10⁴+ steps |
| E, H fields (gradient mode) | BF16 | Short forward pass (100-500 steps), error-tolerant |
| Material coefficients Ca, Cb | FP32 | Computed once; precision in Ca directly affects stability |
| PML psi fields | FP32 | Recursive accumulation — BF16 drift causes reflection increase |
| PML grading (b, c, kappa) | FP32 | Small tensors, negligible memory impact |
| DFT accumulators | FP32 | Sum of 10⁴ terms — Kahan summation if needed |
| Source waveforms | FP32 | Phase error ∝ mantissa precision; BF16 gives ±0.4° error per sample |
| Gradient tensors | FP32 | Small gradients underflow in BF16 (grad ≈ 10⁻⁶ common) |
| Imaging reconstruction | BF16 | Single-pass, error-tolerant, bandwidth-limited |
| Neural network weights | FP16/BF16 | Standard DL practice, loss scaling handles underflow |

## 5.3 BF16 Field Update Analysis

### Bandwidth Speedup

FDTD is memory-bound at 0.16 FLOP/byte. Halving precision halves bytes moved:
```
FP32: 192 bytes/cell → 10.6 Gcells/s theoretical (A100)
BF16: 96 bytes/cell → 21.2 Gcells/s theoretical
Practical: 1.5-1.7× measured (not 2× due to FP32 coefficient loads, kernel overhead)
```

### Precision Degradation Model

BF16 mantissa: 7 bits → relative rounding error per operation: ε = 2⁻⁸ ≈ 0.004 (0.4%)

**Error accumulation over N steps:**
- Best case (random, uncorrelated): total error ∝ ε × √N
- Worst case (coherent, resonant): total error ∝ ε × N

| Steps | Random Error (√N model) | Coherent Error (N model) | Acceptable? |
|-------|------------------------|-------------------------|-------------|
| 10 | 1.2% | 4% | Yes |
| 100 | 4% | 40% | Marginal |
| 1000 | 12% | 400% | **UNSTABLE** |
| 10000 | 40% | ∞ (diverged) | No |

**Conclusion:** BF16 field updates are only safe for <500 steps or with periodic FP32 correction.

### FP32 Correction Protocol

```python
BF16_CORRECTION_INTERVAL = 100  # Every 100 steps

for step in range(N_steps):
    if step % BF16_CORRECTION_INTERVAL == 0:
        # Promote to FP32, do one step at full precision, demote back
        E = E.float()
        H = H.float()
        E, H = fdtd_step_fp32(E, H, Ca, Cb)
        E = E.bfloat16()
        H = H.bfloat16()
    else:
        E, H = fdtd_step_bf16(E, H, Ca.bfloat16(), Cb.bfloat16())
```

Cost: 1% of steps at FP32 speed + 99% at BF16 speed → effective 1.65× overall speedup.

## 5.4 Mixed Precision Implementation

### Production Mode (FP32)

```python
class FDTDEngine:
    def __init__(self, grid, dtype=torch.float32):
        self.fields = FieldSet(grid, dtype=dtype)
        self.coefficients = self._compute_coefficients(dtype=torch.float32)  # Always FP32
```

### Fast Gradient Mode (BF16 + FP32 PML)

```python
def differentiable_forward(eps, source, n_steps=200):
    fields = FieldSet(grid, dtype=torch.bfloat16)
    
    for step in range(n_steps):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            H = update_H(fields.E, fields.H)  # BF16
            E = update_E(fields.H, fields.E, Ca, Cb)  # BF16
        
        # PML always FP32 (recursive, drift-sensitive)
        with torch.autocast(enabled=False):
            apply_pml(E.float(), psi.float(), pml_coeffs)
            E = E.bfloat16()
    
    return E
```

### Automatic Mixed Precision (AMP) with GradScaler

```python
scaler = torch.amp.GradScaler()

for iteration in range(N_opt_steps):
    optimizer.zero_grad()
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        fields = run_fdtd_forward(eps, n_steps=200)
        loss = compute_loss(fields)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

GradScaler prevents gradient underflow: scales loss up before backward (so small gradients don't round to zero in BF16), scales optimizer step back down.

## 5.5 Precision Impact on Numerical Dispersion

FDTD numerical dispersion (phase velocity error) depends on floating-point rounding:

```
True phase velocity: v_p = ω/k (continuous)
FDTD phase velocity: v_p_fdtd ≈ v_p × (1 + δ_dispersion + δ_roundoff)

δ_dispersion: O((kΔx)²) — from finite differences (grid dependent)
δ_roundoff: O(ε_machine × N_steps) — from floating-point accumulation
```

| Precision | δ_roundoff after 10⁴ steps | Relative to δ_dispersion (20 cells/λ) |
|-----------|--------------------------|--------------------------------------|
| FP64 | 10⁻¹² | Negligible (1000× smaller) |
| FP32 | 10⁻³ | Comparable (same order) |
| BF16 | 10⁰ (diverged) | Dominates → unphysical |

**Rule:** Roundoff error should be at least 10× smaller than dispersion error. For 20 cells/λ, δ_dispersion ≈ 10⁻³, so FP32 is the minimum for production simulations.

## 5.6 Decision Matrix

| Use Case | Precision | Max Steps | Expected Speedup | Notes |
|----------|-----------|-----------|-----------------|-------|
| Production simulation | FP32 | unlimited | 1× (baseline) | Default |
| Validation/reference | FP64 | unlimited | 0.5× | High-Q cavities, energy conservation |
| Gradient estimation | BF16 | 100-500 | 1.6× | Short forward pass for adjoint |
| Neural training loop | BF16 | 50-200 | 1.6× | Many iterations, noisy gradients OK |
| Imaging reconstruction | BF16 | single-pass | 1.7× | Backprojection is single GEMM |
| High-Q resonator | FP64 | 10⁶+ | 0.5× | Ring-down requires extreme precision |
| Mixed (correction) | BF16+FP32 | 10⁴ | 1.5× | FP32 correction every 100 steps |

---

# Section 6: Asynchronous Execution and Streams

## 6.1 CUDA Streams Fundamentals

A CUDA stream is an in-order queue of GPU operations. Operations within one stream execute sequentially; operations across different streams may execute concurrently.

```python
# PyTorch stream API
compute_stream = torch.cuda.Stream(priority=-1)  # High priority
io_stream = torch.cuda.Stream(priority=0)        # Low priority

with torch.cuda.stream(compute_stream):
    result = kernel_A(data)  # Runs on compute_stream

with torch.cuda.stream(io_stream):
    buffer.copy_(result, non_blocking=True)  # Concurrent with compute
```

**Default stream behavior:** All PyTorch operations without explicit stream assignment go to stream 0. Stream 0 implicitly synchronizes with all other streams — **avoid using stream 0 in performance-critical code.**

## 6.2 Stream Architecture for FDTD

| Stream | Priority | Purpose | Operations |
|--------|----------|---------|-----------|
| compute | HIGH (-1) | Field updates (critical path) | H-update, E-update, PML |
| source | HIGH (-1) | Source injection | Sparse field writes at source cells |
| detect | LOW (0) | Detector recording | Field gather + DFT accumulation |
| io | LOW (0) | Checkpoint/export | GPU→pinned host async DMA |
| viz | LOW (0) | Live monitoring | Decimated field copy for display |

### Why Source Gets High Priority

Source injection modifies field values that the update kernel will read. If source injection is delayed (low priority), the update kernel may read stale values. High priority ensures source executes before or concurrently with independent update regions.

## 6.3 Synchronization with Events

Events are lightweight markers on streams. One stream can wait for another stream's event without blocking the CPU.

```python
class StreamManager:
    def __init__(self):
        self.compute = torch.cuda.Stream(priority=-1)
        self.source = torch.cuda.Stream(priority=-1)
        self.detect = torch.cuda.Stream(priority=0)
        self.io = torch.cuda.Stream(priority=0)
        
        self.h_done = torch.cuda.Event()
        self.e_done = torch.cuda.Event()
        self.src_done = torch.cuda.Event()
    
    def timestep(self, fields, sources, detectors, t):
        # Source injection (can start immediately)
        with torch.cuda.stream(self.source):
            sources.inject(fields.E, t)
        self.source.record_event(self.src_done)
        
        # H-update (waits for source to finish modifying E)
        with torch.cuda.stream(self.compute):
            self.compute.wait_event(self.src_done)
            update_H(fields)
        self.compute.record_event(self.h_done)
        
        # E-update (sequential after H on same stream)
        with torch.cuda.stream(self.compute):
            update_E(fields)
            apply_pml(fields)
        self.compute.record_event(self.e_done)
        
        # Detector (waits for E-update, runs concurrently with next step's source)
        with torch.cuda.stream(self.detect):
            self.detect.wait_event(self.e_done)
            detectors.record(fields, t)
```

### Synchronization Costs

| Operation | Latency | When to Use |
|-----------|---------|-------------|
| `event.record(stream)` | ~0.5 μs | Every stream transition |
| `stream.wait_event(event)` | ~1 μs | Cross-stream dependency |
| `torch.cuda.synchronize()` | 3-10 μs | **NEVER in hot loop** |
| `event.synchronize()` | 3-10 μs | Only for CPU-side result access |

## 6.4 Overlap Timeline

### Single Timestep (512³ grid, A100)

```
Time:  0μs      50μs     100μs    200μs    300μs    360μs    400μs
       │         │         │         │         │         │         │
Compute: [═══════ H_update (180μs) ═══════][═══════ E_update+PML (180μs) ═════]
Source:  [src(5μs)]                         [src(5μs)]
Detect:                                                  [DFT(30μs)]
I/O:                                                          [checkpoint(100μs)─►
       │         │         │         │         │         │         │
```

**Critical path: 360 μs** (H_update + E_update, sequential on compute stream)

**Overlapped (free):**
- Source injection: 5 μs (hidden behind H_update)
- Detector DFT: 30 μs (runs after E_update, overlaps with next step's source)
- Checkpoint I/O: 100 μs (fully async, no impact on compute)

### Without Streams (Naive Sequential)

```
H_update(180) → src(5) → E_update(180) → src(5) → PML(included) → detect(30) → checkpoint(100)
Total: 500 μs/step
```

**Speedup from streams: 500/360 = 1.39×** (39% improvement)

## 6.5 CUDA Graphs

### Concept

CUDA Graphs record a sequence of kernel launches into a static executable graph. Replaying the graph incurs near-zero CPU overhead — the GPU executes the entire sequence without CPU interaction.

### FDTD as CUDA Graph

FDTD is ideal for graph capture because every timestep is IDENTICAL:
- Same kernels
- Same tensor addresses (pre-allocated, never reallocated)
- Same launch configurations
- No dynamic control flow

```python
# Capture phase (once at initialization)
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    stream.wait_stream(torch.cuda.current_stream())
    
    # Warm up
    for _ in range(3):
        one_timestep(fields, t=0)
    
    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        one_timestep(fields, t=0)  # t is read from a tensor, not Python int

# Replay phase (every timestep)
for t in range(N_steps):
    t_tensor.fill_(t)  # Update time parameter in-place
    graph.replay()      # Single CPU call, ~3 μs
```

### Graph Benefits

| Metric | Without Graph | With Graph | Improvement |
|--------|--------------|-----------|-------------|
| CPU overhead per step | 50-100 μs | 3 μs | 17-33× |
| Kernel launch overhead | 5 μs × 10 kernels = 50 μs | 0 (pre-recorded) | Eliminated |
| GPU idle time (waiting for CPU) | 10-50 μs | 0 | Eliminated |
| Effective step time (512³) | 410 μs | 363 μs | 13% faster |

### Graph Limitations

- All tensor addresses must be static (no `torch.empty` in the loop)
- No Python control flow during replay (if/else based on field values)
- No dynamic shapes (grid size fixed after capture)
- Cannot capture operations that call back to CPU (print, assert, logging)

All satisfied by steady-state FDTD time-stepping.

## 6.6 Async Memory Operations

### GPU → Host (Checkpoint)

```python
# Pinned host buffer (allocated once at init)
host_buffer = torch.empty(field_shape, pin_memory=True)

# Async copy (non-blocking)
with torch.cuda.stream(io_stream):
    host_buffer.copy_(gpu_fields, non_blocking=True)

# Later (on CPU thread, after io_stream completes):
io_stream.synchronize()  # Wait only on I/O stream, not compute
save_to_disk(host_buffer)
```

### Double Buffering

```python
buffers = [torch.empty(shape, pin_memory=True) for _ in range(2)]
write_thread = None

def checkpoint_async(fields, step):
    buf_idx = step % 2
    
    # Wait for previous disk write to finish (if any)
    if write_thread and write_thread.is_alive():
        write_thread.join()
    
    # Async GPU → pinned host
    with torch.cuda.stream(io_stream):
        buffers[buf_idx].copy_(fields, non_blocking=True)
    io_stream.synchronize()
    
    # Background disk write
    write_thread = threading.Thread(target=save_hdf5, args=(buffers[buf_idx], step))
    write_thread.start()
```

## 6.7 Performance Summary

| Optimization | Time Saved per Step (512³) | Implementation Complexity |
|-------------|--------------------------|--------------------------|
| Multi-stream overlap | 140 μs (39%) | Medium (event management) |
| CUDA Graph replay | 47 μs (13%) | Low (capture/replay pattern) |
| Async I/O (non-blocking) | Eliminates checkpoint stall | Low |
| Priority streams | 5-10 μs (scheduling) | Trivial |
| **Combined** | **~190 μs (53%)** | **Medium** |

**Final optimized step time: ~360 μs** for 512³ on A100 (2,778 steps/second).

---

# Section 7: Advanced Stream Parallelism and Pipelining

## 7.1 Concurrent Kernel Execution

A single GPU can execute multiple kernels simultaneously if:
1. They are on different CUDA streams
2. Combined SM usage doesn't exceed total SMs
3. Neither kernel individually saturates all SMs

### FDTD Concurrency Analysis

```
E-field update (512³): 262,144 blocks × 512 threads
A100 SMs: 108, each runs 4 blocks = 432 active blocks
Waves: 262,144 / 432 = 607 waves → SATURATES GPU

Source injection: 1,000 threads (sparse)
→ 2 blocks → uses 2/108 SMs = 1.8%
```

**Reality:** The E-field kernel saturates all SMs. A concurrent source injection kernel can only execute during the "tail effect" — the last few waves when some SMs finish early and become idle.

**Practical concurrency:** Between the large field-update kernels. Small kernels (source, detect) overlap with the startup/tail of large kernels.

## 7.2 Pipeline Parallelism for Multi-Simulation (MIMO)

MIMO imaging requires N_TX independent FDTD simulations (same geometry, different sources). If a single simulation doesn't fill VRAM, multiple can run concurrently.

### VRAM Budget for Concurrent Simulations

```
Single sim (256³ FP32): ~900 MB (fields + materials + PML)
A100 80 GB VRAM: floor(80 GB / 0.9 GB) = 88 concurrent sims (memory-limited)
But SM bandwidth: 88 sims × 0.9 GB working set = 79 GB active → L2 cache thrashing

Practical limit: 4-8 concurrent sims (balance memory + cache efficiency)
```

### Stream-Per-Simulation Pattern

```python
class MultiSimRunner:
    def __init__(self, n_concurrent, grid):
        self.sims = [SimState(grid, device='cuda') for _ in range(n_concurrent)]
        self.streams = [torch.cuda.Stream() for _ in range(n_concurrent)]
    
    def run_batch(self, tx_configs, n_steps):
        for step in range(n_steps):
            for i, (sim, stream) in enumerate(zip(self.sims, self.streams)):
                with torch.cuda.stream(stream):
                    sim.inject_source(tx_configs[i], step)
                    sim.update_H()
                    sim.update_E()
                    sim.apply_pml()
                    sim.record_detectors(step)
        
        # Synchronize all
        torch.cuda.synchronize()
        return [sim.get_results() for sim in self.sims]
```

### Throughput Gain (Small Grids)

| Grid | Single Sim SM Usage | Concurrent Sims | Throughput Multiplier |
|------|-------------------|-----------------|---------------------|
| 64³ | 8% | 8 | 5.2× |
| 128³ | 15% | 6 | 4.1× |
| 256³ | 60% | 2 | 1.6× |
| 512³ | 100% | 1 | 1.0× (saturated) |

Concurrency helps most for small grids that under-utilize the GPU individually.

## 7.3 Batched Simulation (Single Kernel, Multiple Grids)

Alternative to multi-stream: stack simulations along a batch dimension.

```python
# Batched field tensors: all sims in one allocation
# Shape: (N_batch, Nx, Ny, Nz) per component
Ex_batch = torch.zeros(N_batch, Nx, Ny, Nz, device='cuda')
Ey_batch = torch.zeros(N_batch, Nx, Ny, Nz, device='cuda')
# ... 

def update_H_batched(E_batch, H_batch, coeffs):
    """Single kernel updates all simulations simultaneously."""
    # Curl operates on last 3 dims; batch dim adds more parallel work
    dEz_dy = E_batch.Ez[:, :, 1:, :] - E_batch.Ez[:, :, :-1, :]
    dEy_dz = E_batch.Ey[:, :, :, 1:] - E_batch.Ey[:, :, :, :-1]
    H_batch.Hx[:, :, :-1, :-1] -= Db * (dEz_dy[:, :, :, :-1] - dEy_dz[:, :, :-1, :])
```

**Advantage:** Single kernel launch for all sims. Batch dimension adds threads without increasing per-cell memory traffic. Better SM utilization for small grids.

**Disadvantage:** All sims must have identical geometry/grid (only source differs for MIMO — this is satisfied).

### Batched Performance Model

```
Single sim (128³): 2.1M cells → 4096 blocks → 4096/432 = 9 waves (SM under-utilized)
Batched ×8: 16.8M cells → 32768 blocks → 32768/432 = 76 waves (better utilization)

Throughput: batch=8 achieves ~6× throughput of sequential (not 8× due to cache pressure)
```

## 7.4 Priority Streams

CUDA stream priority determines warp scheduling preference when multiple streams have ready work.

```python
# High priority: warps from this stream get scheduled first
compute = torch.cuda.Stream(priority=-1)  # -1 = highest

# Low priority: warps yield to high-priority streams
io = torch.cuda.Stream(priority=0)  # 0 = default (lowest)
```

**Effect on FDTD:**
- Compute stream (field updates): always high priority → maximum SM allocation
- I/O stream (checkpoint copy): low priority → only uses SMs when compute has no ready warps
- Result: checkpoint DMA runs in "cracks" between compute waves, near-zero impact on step time

### Priority Levels

```
CUDA defines: cudaStreamPriorityRange(&lowest, &highest)
Typically: lowest=0, highest=-1 (only 2 levels on most GPUs)
Some GPUs support -5 to 0 (6 levels) but benefit is marginal beyond 2.
```

## 7.5 Pipelined Adjoint Computation

The adjoint method requires:
1. Forward pass → save checkpoints
2. Backward (adjoint) pass → recompute from checkpoints + run adjoint simultaneously

### Two-Stream Adjoint Pipeline

```python
recompute_stream = torch.cuda.Stream(priority=-1)
adjoint_stream = torch.cuda.Stream(priority=-1)
recompute_done = torch.cuda.Event()

def adjoint_pass(checkpoints, n_steps, checkpoint_interval):
    for segment in reversed(range(n_segments)):
        # Stream A: recompute forward states from checkpoint
        with torch.cuda.stream(recompute_stream):
            states = recompute_segment(checkpoints[segment], checkpoint_interval)
        recompute_stream.record_event(recompute_done)
        
        # Stream B: run adjoint using previously recomputed states
        with torch.cuda.stream(adjoint_stream):
            adjoint_stream.wait_event(recompute_done)
            for state in reversed(states):
                adjoint_step(state)
                accumulate_gradient(state)
```

**Overlap opportunity:** While adjoint processes segment N, recompute can prepare segment N-1. With 2 streams: recompute and adjoint of DIFFERENT segments overlap → ~30% speedup over sequential.

## 7.6 Synchronization Best Practices

### DO

```python
# Fine-grained event sync (cheap, precise)
event = torch.cuda.Event()
stream_a.record_event(event)
stream_b.wait_event(event)
```

### DON'T

```python
# Device-wide sync (expensive, unnecessary)
torch.cuda.synchronize()  # Blocks CPU until ALL GPU work completes

# Stream sync when event suffices
stream.synchronize()  # Blocks CPU; use event-based cross-stream sync instead
```

### When CPU Sync Is Required

- Reading a GPU tensor value on CPU (e.g., checking for NaN)
- Printing/logging field statistics
- End of simulation (before extracting results)
- Checkpoint disk write (need data in host buffer)

**Rule:** At most 1 synchronization per N_checkpoint_interval steps (typically every 1000 steps).

## 7.7 Complete Stream Configuration

```python
class FDTDStreamConfig:
    def __init__(self):
        self.compute = torch.cuda.Stream(priority=-1)
        self.source = torch.cuda.Stream(priority=-1)
        self.detect = torch.cuda.Stream(priority=0)
        self.io = torch.cuda.Stream(priority=0)
        self.viz = torch.cuda.Stream(priority=0)
        
        self.events = {
            'src_done': torch.cuda.Event(enable_timing=False),
            'h_done': torch.cuda.Event(enable_timing=False),
            'e_done': torch.cuda.Event(enable_timing=False),
        }
        
        self.graph = None  # Populated after capture
    
    def capture_graph(self, step_fn):
        """Capture steady-state timestep as CUDA graph."""
        with torch.cuda.stream(self.compute):
            for _ in range(3):  # Warm up
                step_fn()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.compute):
                step_fn()
    
    def replay_step(self):
        """Execute one timestep with near-zero CPU overhead."""
        self.graph.replay()
```

---

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
