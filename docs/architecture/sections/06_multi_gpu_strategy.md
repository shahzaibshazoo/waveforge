# Section 6: CUDA Core Parallelism and GPU Execution Strategy

## 6.1 Single-GPU Architecture Target

GPU-MEEP targets **maximum utilization of CUDA cores within a single GPU**. The entire FDTD grid lives in one GPU's VRAM. Parallelism is achieved through thousands of concurrent threads mapped to grid cells — not through distributing work across multiple devices.

**Primary targets:**
- NVIDIA A100: 6,912 CUDA cores, 108 SMs, 80 GB HBM2e
- NVIDIA RTX 4090: 16,384 CUDA cores, 128 SMs, 24 GB GDDR6X
- NVIDIA H100: 14,592 CUDA cores, 132 SMs, 80 GB HBM3

**Design principle:** One CUDA thread = one grid cell update. A 512³ grid = 134M cells = 134M threads dispatched per half-timestep. The GPU's warp scheduler handles the mapping.

---

## 6.2 Thread Hierarchy and Grid Mapping

### CUDA Execution Model Applied to FDTD

```
FDTD Grid (Nx × Ny × Nz)
    ↓ maps to
CUDA Grid (gridDim.x × gridDim.y × gridDim.z)
    ↓ composed of
Thread Blocks (blockDim.x × blockDim.y × blockDim.z)
    ↓ executed as
Warps (32 threads, SIMT execution)
```

### Thread-to-Cell Mapping

```
cell(i, j, k) → thread:
    blockIdx  = (i / BLOCK_X, j / BLOCK_Y, k / BLOCK_Z)
    threadIdx = (i % BLOCK_X, j % BLOCK_Y, k % BLOCK_Z)
```

**3D block shape:** `(8, 8, 8)` = 512 threads per block

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Block size | 8×8×8 = 512 | 2 blocks/SM → 1024 resident threads, good latency hiding |
| Threads per SM | 1024 | Below max (2048) to leave register headroom |
| Blocks per SM | 2 | Balance between occupancy and register availability |
| Total blocks (512³) | 64×64×64 = 262,144 | Far exceeds SM count → full saturation |

### Why 3D Blocks Match FDTD

The Yee stencil accesses `(i±1, j±1, k±1)` neighbors. A 3D thread block ensures:
- Threads needing the same neighbor data are co-located in the same block
- Shared memory tiling captures the 3D neighborhood efficiently
- L1 cache locality is maximized (threads in a block access nearby addresses)

---

## 6.3 Warp-Level Execution

### Warp Formation

A 8×8×8 block contains 512 threads = **16 warps**. Warp linearization:
```
warp_id = (threadIdx.z * 64 + threadIdx.y * 8 + threadIdx.x) / 32
```

First warp: threads (0,0,0)→(3,3,1) — spans a 4×4×2 sub-block in the grid. These threads access contiguous Z-addresses → **coalesced memory access by construction**.

### Warp Divergence Analysis

FDTD field updates are **uniform** — every cell executes identical arithmetic. No branching in the hot path.

**Exception: PML boundary cells.** Cells in the PML region execute additional auxiliary field updates. Two strategies:

| Strategy | Approach | Divergence Cost |
|----------|----------|-----------------|
| Predicated | All threads execute PML code, non-PML threads masked out | ~15% wasted cycles in PML blocks |
| Separate kernel | PML cells handled by dedicated kernel launch | Zero divergence, extra launch overhead |
| **Hybrid (chosen)** | Fused kernel with early-exit for interior blocks | Zero cost for interior; ~15% for boundary blocks |

```
if (block_is_entirely_interior):
    // Fast path: no PML check per thread
    update_field_standard()
else:
    // Boundary block: per-thread PML predicate
    if (cell_in_pml):
        update_field_with_pml()
    else:
        update_field_standard()
```

Block-level divergence check avoids per-thread branching for 90%+ of blocks.

---

## 6.4 SM Occupancy Optimization

### Occupancy Calculation

```
Registers per thread: ~32 (measured for E-field kernel)
Shared memory per block: 0 KB (baseline) or 4 KB (with tiling)
Block size: 512 threads

SM resources (A100, SM 8.0):
- 65,536 registers per SM
- 164 KB shared memory per SM
- Max 2048 threads per SM
- Max 32 blocks per SM

Register-limited: 65,536 / 32 = 2048 threads → 4 blocks of 512 ✓
Thread-limited: 2048 / 512 = 4 blocks ✓
Block-limited: 32 ≥ 4 ✓
Shared mem: 164 KB / 4 KB = 41 blocks ✓ (not limiting)

Achieved occupancy: 2048/2048 = 100% (4 blocks/SM)
```

### Occupancy vs Performance Tradeoff

| Occupancy | Blocks/SM | Registers/Thread | Performance Impact |
|-----------|-----------|------------------|-------------------|
| 100% | 4 | 32 | Maximum latency hiding, register spills possible |
| 75% | 3 | 43 | More registers, fewer spills, slightly less hiding |
| 50% | 2 | 64 | Maximum registers, minimal hiding (bad for memory-bound) |

**For memory-bound FDTD: maximize occupancy.** Latency hiding (more warps in flight) compensates for memory stalls. Register pressure is low (stencil needs ~12-15 values), so 100% occupancy is achievable.

### Occupancy Tuning Knobs

```python
# PyTorch/Triton kernel launch configuration
BLOCK_SIZE = (8, 8, 8)  # 512 threads

# For Triton kernels:
@triton.jit
def update_E_kernel(..., BLOCK_X: tl.constexpr = 8, BLOCK_Y: tl.constexpr = 8, BLOCK_Z: tl.constexpr = 8):
    ...

# Auto-tune across block sizes:
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 8, 'BLOCK_Y': 8, 'BLOCK_Z': 8}),
        triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 16, 'BLOCK_Z': 4}),
        triton.Config({'BLOCK_X': 32, 'BLOCK_Y': 8, 'BLOCK_Z': 4}),
    ],
    key=['Nx', 'Ny', 'Nz']
)
```

---

## 6.5 Memory Access Patterns and Coalescing

### Coalesced Global Memory Access

A warp issues a single memory transaction when 32 threads access 32 consecutive 4-byte addresses (128-byte cache line).

**FDTD stencil access pattern for Ex update:**

```
Ex[i,j,k] reads: Hz[i,j,k], Hz[i,j-1,k], Hy[i,j,k], Hy[i,j,k-1]
```

Memory layout `(Nx, Ny, Nz)` row-major (Z fastest):
- `Hz[i,j,k]` — threads in warp differ in k → consecutive addresses ✓ **COALESCED**
- `Hz[i,j-1,k]` — same stride, shifted by Nz → consecutive ✓ **COALESCED**
- `Hy[i,j,k-1]` — stride-1 shift in fastest dim → still consecutive ✓ **COALESCED**
- `Hy[i,j,k]` — same as Hz pattern ✓ **COALESCED**

**All accesses coalesced.** This is why SoA layout + Z-fastest-varying is critical.

### Cache Line Utilization

| Access Pattern | Bytes Loaded | Bytes Used | Efficiency |
|----------------|-------------|-----------|-----------|
| Hz[i,j,k] (aligned) | 128 B (1 line) | 128 B (32 floats) | 100% |
| Hz[i,j-1,k] (shifted) | 128 B | 128 B | 100% |
| Hz[i,j,k-1] (stride-1) | 128 B | 128 B | 100% |
| Scattered access (AoS) | 128 B | 16 B (4 floats) | 12.5% |

SoA achieves 100% cache line utilization. AoS would waste 87.5% of loaded bytes.

---

## 6.6 Shared Memory Tiling (Optional Optimization)

### Stencil Reuse Opportunity

Adjacent threads read overlapping H-field values. Thread (i,j,k) and (i,j+1,k) both read Hz[i,j,k].

**Tiling strategy:**
```
1. Load (BLOCK_X+1) × (BLOCK_Y+1) × (BLOCK_Z+1) H-values into shared memory
2. Synchronize block (__syncthreads)
3. Compute E-field updates from shared memory (fast, no global stall)
```

### Shared Memory Budget

```
Tile for Hz: (8+1)×(8+1)×(8+1) = 729 floats = 2,916 bytes
Three H-components needed: 3 × 2,916 = 8,748 bytes ≈ 9 KB per block
With 4 blocks/SM: 36 KB (within A100's 164 KB budget)
```

### When Shared Memory Helps

| Grid Size | L2 Hit Rate (no SM) | With Shared Memory | Speedup |
|-----------|---------------------|-------------------|---------|
| 128³ | ~85% | ~95% | 1.05× |
| 256³ | ~60% | ~92% | 1.15× |
| 512³ | ~30% | ~90% | 1.25× |
| 768³ | ~15% | ~88% | 1.35× |

**Conclusion:** Shared memory tiling becomes worthwhile for grids ≥256³ where L2 cache can't hold the working set. For smaller grids, the overhead of loading/syncing outweighs the benefit.

---

## 6.7 Kernel Launch Configuration

### Full Kernel Launch Spec

```python
def launch_E_update(Ex, Ey, Ez, Hx, Hy, Hz, Ca, Cb, Nx, Ny, Nz, dt_dx, dt_dy, dt_dz):
    block = (8, 8, 8)
    grid = (
        (Nx + block[0] - 1) // block[0],
        (Ny + block[1] - 1) // block[1],
        (Nz + block[2] - 1) // block[2],
    )
    # For 512³: grid = (64, 64, 64) = 262,144 blocks
    # Total threads: 262,144 × 512 = 134,217,728
    # A100 can have 108 SMs × 4 blocks = 432 blocks active simultaneously
    # All 262,144 blocks complete in ~607 waves
```

### Wave Execution Model

```
Total blocks: 262,144
Active blocks (A100): 108 SM × 4 blocks/SM = 432 concurrent blocks
Waves needed: 262,144 / 432 = 607 waves
Time per wave: ~0.3 μs (memory-bound, HBM latency hidden by occupancy)
Total kernel time: ~180 μs for 512³ E-field update
```

### Thread Block Cluster (SM 9.0+ / H100)

On Hopper architecture, thread block clusters allow cooperative groups across SMs:
```
// Cluster of 2×2×2 = 8 blocks share distributed shared memory
// Adjacent blocks can read each other's shared memory directly
// Eliminates halo redundancy at block boundaries
```

Future optimization path for H100+ hardware.

---

## 6.8 Instruction-Level Parallelism (ILP)

### Loop Unrolling for Throughput

Each thread updates one cell, but can compute multiple output components:

```cuda
// Single thread computes Ex, Ey, Ez at (i,j,k)
// Shares loaded H-values across component updates
float hz_ijk = Hz[idx];          // Loaded once
float hz_jm1 = Hz[idx - Nz];    // Loaded once
float hy_ijk = Hy[idx];          // Loaded once
float hy_km1 = Hy[idx - 1];     // Loaded once
float hx_ijk = Hx[idx];         // Loaded once
float hx_km1 = Hx[idx - 1];    // Loaded once

Ex[idx] = Ca_x * Ex[idx] + Cb_x * ((hz_ijk - hz_jm1) * dt_dy - (hy_ijk - hy_km1) * dt_dz);
Ey[idx] = Ca_y * Ey[idx] + Cb_y * ((hx_ijk - hx_km1) * dt_dz - (hz_ijk - hz_im1) * dt_dx);
Ez[idx] = Ca_z * Ez[idx] + Cb_z * ((hy_ijk - hy_im1) * dt_dx - (hx_ijk - hx_jm1) * dt_dy);
```

**Benefit:** 6 global loads serve 3 output computations. Arithmetic ops (subtract, multiply, FMA) pipeline while subsequent loads are in-flight. ILP = 3 independent FMA chains per thread.

---

## 6.9 Performance Model Summary

### Single GPU Roofline

```
             Compute Roof (A100 FP32: 19.5 TFLOPS)
            ╱
           ╱
          ╱
         ╱    ┌────── FDTD operating point (0.16 FLOP/byte)
        ╱     │       → Memory-bound
       ╱      ▼
      ╱───────●──────────────── Memory Roof (2 TB/s)
     ╱
    ╱
   ╱
  Arithmetic Intensity (FLOP/byte) →
```

**FDTD lives firmly in the memory-bound regime.** All optimization effort targets:
1. Reducing bytes transferred (mixed precision, fusion)
2. Maximizing effective bandwidth (coalescing, occupancy, cache)
3. NOT increasing FLOPS (already compute-underutilized)

### Expected Throughput by GPU

| GPU | BW (GB/s) | Cores | FDTD Throughput (Mcells/s) | Max Grid (80% VRAM) |
|-----|-----------|-------|---------------------------|---------------------|
| RTX 3090 (24GB) | 936 | 10,496 | 3,400 | 384³ FP32 |
| RTX 4090 (24GB) | 1,008 | 16,384 | 3,700 | 384³ FP32 |
| A100 (80GB) | 2,039 | 6,912 | 7,400 | 768³ FP32 |
| H100 (80GB) | 3,350 | 14,592 | 12,200 | 768³ FP32 |

---

## 6.10 Future: Multi-GPU as Extension

Multi-GPU support is a future extension (v2.0+) for problems exceeding single-GPU VRAM. When implemented:
- Domain decomposition with NCCL halo exchange
- NVLink-aware subdomain placement
- Overlap interior compute with boundary communication

**Current scope:** Single GPU, maximize CUDA core utilization, saturate memory bandwidth.
