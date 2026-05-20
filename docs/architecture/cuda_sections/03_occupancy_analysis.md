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
