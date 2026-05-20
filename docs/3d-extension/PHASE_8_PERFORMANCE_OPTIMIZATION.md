# Phase 8: GPU Performance Optimization

## Performance Analysis

### Memory Bandwidth Bottleneck

3D FDTD is fundamentally memory-bound. Each cell update requires:

**Reads**:
- 3 E-field components (at current and adjacent cells): ~12 float32 values = 48 bytes
- 3 H-field components: ~12 values = 48 bytes
- Ca, Cb, Da, Db tensors: 4 values = 16 bytes
- Total per cell: ~112 bytes

**Writes**:
- 3 updated E-field components: 12 bytes
- 3 updated H-field components: 12 bytes
- Total per cell: ~24 bytes

**Net**: ~136 bytes loaded, ~24 bytes stored per cell per timestep. Effective bandwidth requirement is read-dominated.

### Arithmetic Intensity

```
Operations per cell update: ~200 FLOPs (6 curl computations, 6 multiplications, 6 additions)
Data per operation: 136 bytes / 200 FLOP ≈ 0.68 bytes/FLOP
or equivalently: ~1.5 FLOP/byte (very low arithmetic intensity)
```

**Consequence**: Even simple optimizations (e.g., kernel fusion) yield significant speedup because arithmetic intensity is limited, not computation.

### Theoretical Peak Performance

On NVIDIA T4 GPU:

```
Peak memory bandwidth: 320 GB/s (typical)
Peak compute: 65 TFLOPS (FP32)

If limited by memory (arithmetic intensity 0.3 FLOP/byte):
Max throughput = 320 GB/s × 0.3 FLOP/byte ÷ 200 FLOP/cell
               = 480 Mcells/s (theoretical maximum)
```

**Realistic target** accounting for cache misses, divergence, latency: **300-500 Mcells/s** on T4 for 128³ grid.

### Grid Size Scaling

For grid size N³:

```
Memory footprint: 4 (Ca, Cb, Da, Db) × N³ × 4 bytes = 16N³ bytes
At 16 GB T4 memory: max N³ ≈ 1 billion cells → N ≈ 1000

Time per 1000 steps on 256³ grid (16.8M cells):
16.8M cells × 1000 steps ÷ (400 Mcells/s) ≈ 42 seconds
```

## Optimization Strategies

### 1. Fused Kernels

**Problem**: Separate kernels for E-field x/y/z components and H-field x/y/z components = 6 kernel launches. Each launch incurs overhead and redundant memory loads.

**Solution**: Fuse all E-field updates into a single kernel, all H-field updates into a single kernel.

**Code Example** (conceptual):

```cuda
__global__ void fused_e_field_update(
    float *E_x, float *E_y, float *E_z,
    float *H_x, float *H_y, float *H_z,
    float *Ca, float *Cb,
    int Nx, int Ny, int Nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= 1 || i >= Nx-1 || j >= 1 || j >= Ny-1 || k >= 1 || k >= Nz-1)
        return;
    
    int idx = i * Ny * Nz + j * Nz + k;
    
    // Compute all three curl components simultaneously
    float dH_z_dy = (H_z[idx] - H_z[(i)*(Ny)*(Nz) + (j-1)*(Nz) + k]) / dy;
    float dH_y_dz = (H_y[idx] - H_y[(i)*(Ny)*(Nz) + (j)*(Nz) + (k-1)]) / dz;
    float dH_x_dz = (H_x[idx] - H_x[(i)*(Ny)*(Nz) + (j)*(Nz) + (k-1)]) / dz;
    float dH_z_dx = (H_z[idx] - H_z[(i-1)*(Ny)*(Nz) + (j)*(Nz) + k]) / dx;
    float dH_y_dx = (H_y[idx] - H_y[(i-1)*(Ny)*(Nz) + (j)*(Nz) + k]) / dx;
    float dH_x_dy = (H_x[idx] - H_x[(i)*(Ny)*(Nz) + (j-1)*(Nz) + k]) / dy;
    
    E_x[idx] = Ca[idx] * E_x[idx] + Cb[idx] * (dH_z_dy - dH_y_dz);
    E_y[idx] = Ca[idx] * E_y[idx] + Cb[idx] * (dH_x_dz - dH_z_dx);
    E_z[idx] = Ca[idx] * E_z[idx] + Cb[idx] * (dH_y_dx - dH_x_dy);
}
```

**Speedup**: Reduces kernel launches from 6 to 2, eliminates redundant global memory loads. Typical gain: 20-30%.

### 2. Register Tiling

**Problem**: Each H-field read fetches from global memory; if H values reused across multiple E-field computations, we waste bandwidth.

**Solution**: Tile computation so that each thread block loads H-field data into registers, reuses it for multiple E-cell updates.

**Code Sketch** (conceptual):

```cuda
__global__ void h_field_tiled_update(
    float *E_x, float *E_y, float *E_z,
    float *H_x, float *H_y, float *H_z,
    float *Ca, float *Cb,
    int Nx, int Ny, int Nz
) {
    // Block: 16×16×4 threads
    // Each thread computes E at 2 locations using shared H values
    
    __shared__ float H_x_tile[18][18][6];  // +2 halo
    __shared__ float H_y_tile[18][18][6];
    __shared__ float H_z_tile[18][18][6];
    
    // Load H into shared memory (with halo for stencil)
    for (int load_iter = 0; load_iter < 2; load_iter++) {
        int ti = threadIdx.x;
        int tj = threadIdx.y;
        int tk = threadIdx.z + load_iter * 4;
        
        if (ti < 18 && tj < 18 && tk < 6) {
            int gi = blockIdx.x * 16 + ti - 1;
            int gj = blockIdx.y * 16 + tj - 1;
            int gk = blockIdx.z * 4 + tk - 1;
            
            if (gi >= 0 && gi < Nx && gj >= 0 && gj < Ny && gk >= 0 && gk < Nz) {
                H_x_tile[ti][tj][tk] = H_x[gi*Ny*Nz + gj*Nz + gk];
                H_y_tile[ti][tj][tk] = H_y[gi*Ny*Nz + gj*Nz + gk];
                H_z_tile[ti][tj][tk] = H_z[gi*Ny*Nz + gj*Nz + gk];
            }
        }
    }
    
    __syncthreads();
    
    // Compute E-field using local tile data
    int ti = threadIdx.x + 1;
    int tj = threadIdx.y + 1;
    
    for (int tk = threadIdx.z + 1; tk < 5; tk++) {
        int gi = blockIdx.x * 16 + ti;
        int gj = blockIdx.y * 16 + tj;
        int gk = blockIdx.z * 4 + tk;
        
        float dH_z_dy = (H_z_tile[ti][tj][tk] - H_z_tile[ti][tj-1][tk]);
        float dH_y_dz = (H_y_tile[ti][tj][tk] - H_y_tile[ti][tj][tk-1]);
        // ... compute E components
        
        E_x[gi*Ny*Nz + gj*Nz + gk] = Ca[gi*Ny*Nz + gj*Nz + gk] * E_x[...] 
                                     + Cb[gi*Ny*Nz + gj*Nz + gk] * (dH_z_dy - dH_y_dz);
    }
}
```

**Speedup**: Reduces global memory bandwidth by ~40% (reuses loaded H values). Typical gain: 25-40%.

### 3. Shared Memory Optimization

**Problem**: Each thread computes only 1 cell; stencil requires neighboring values, causing non-coalesced global memory access.

**Solution**: Use shared memory to load a tile of H-field data, perform stencil computation with coalesced local access.

**Block Structure**:
- Block size: 16×16×4 threads (1024 threads, 32KB shared memory)
- Load 18×18×6 H-tile (halo for stencil, ~3.1 KB per component, total ~9 KB)
- Each thread computes ~4 E-cells

**Speedup**: Improves memory access pattern from scattered to coalesced. Typical gain: 15-20%.

### 4. Memory Layout Optimization

#### Current (SoA - Structure of Arrays):

```
E_x[Nx*Ny*Nz], E_y[Nx*Ny*Nz], E_z[Nx*Ny*Nz]
H_x[Nx*Ny*Nz], H_y[Nx*Ny*Nz], H_z[Nx*Ny*Nz]
```

Access pattern for E_x update: needs E_x, H_y, H_z. Loads from 3 separate arrays.

#### Alternative (AoS - Array of Structures):

```
struct Cell {
    float E_x, E_y, E_z;
    float H_x, H_y, H_z;
}

Cell[Nx*Ny*Nz]
```

Access pattern: all 6 components colocated in memory. Better cache locality.

#### Hybrid (Tiled SoA):

```
// 4×4×4 tiles of SoA
E_tile[Nx/4, Ny/4, Nz/4, 4, 4, 4, 3]  // 3 for x/y/z components
H_tile[Nx/4, Ny/4, Nz/4, 4, 4, 4, 3]
```

Improves cache locality while maintaining coalesced memory access.

**Recommendation**: Hybrid SoA/AoS with 4×4×4 tiles. Gain: 10-20%.

### 5. Mixed Precision (BF16 Storage, FP32 Accumulation)

**Strategy**:
1. Store E, H fields in BF16 (16-bit brain float) — reduces memory footprint by 2×
2. Load BF16 → upconvert to FP32 for computation
3. Accumulate in FP32, downconvert to BF16 for storage

**Trade-off**:
- **Pro**: 2× reduction in memory bandwidth (vs FP32), 1× reduction in memory footprint
- **Con**: BF16 has 8-bit mantissa → ~0.4% relative error per operation. Acceptable for stable FDTD (dissipative).

**Code Sketch**:

```cuda
// Load BF16, convert to FP32
float E_x_local = __bfloat162float(E_x_bf16[idx]);

// Compute in FP32
float E_x_new = Ca[idx] * E_x_local + Cb[idx] * curl_term;

// Store as BF16
E_x_bf16[idx] = __float2bfloat16_rn(E_x_new);
```

**Speedup**: 1.5-2× from reduced memory bandwidth (also reduces power by ~30%). Typical gain: 40-60% (limited by compute, not memory).

**Caveat**: BF16 not ideal for multi-GPU communication or precision-critical calibrations. Use for production simulations, not validation.

### 6. Computation-Communication Overlap (Multi-GPU Future)

For multi-GPU configurations, hide communication latency:

```cuda
// GPU 0: Compute interior cells
launch_interior_kernel(E, H, interior_bounds);

// GPU 0/1: Overlap boundary computation with GPU-GPU communication
cudaMemcpyAsync(GPU_1_buffer, GPU_0_boundary, size, cudaMemcpyDeviceToDevice);

// GPU 0: Compute boundary cells while transfer in-flight
launch_boundary_kernel(E, H, boundary_bounds);

// Wait for communication to complete
cudaStreamSynchronize();
```

**Speedup**: 10-20% for 2-GPU setup (more for 4+ GPU).

## PyTorch-Specific Optimizations

WaveForge is built on PyTorch. Leverage PyTorch optimizations:

### 1. torch.compile() with Inductor Backend

**Purpose**: Auto-fuse kernels, eliminate tensor allocations, JIT-compile hot paths.

**Usage**:

```python
from waveforge import Simulation3D

sim = Simulation3D(...)

# Compile hot loop
@torch.compile(backend='inductor')
def step_compiled(E, H, Ca, Cb, Da, Db):
    E_new = torch.zeros_like(E)
    H_new = torch.zeros_like(H)
    
    # E-field update
    E_new[..., 2] = Ca * E[..., 2] + Cb * (
        (H[1:, :, :, 1] - H[:-1, :, :, 1]) / dx -
        (H[:, 1:, :, 0] - H[:, :-1, :, 0]) / dy
    )
    # ... more updates
    
    return E_new, H_new

# Warm up (compiles JIT on first call)
E, H = step_compiled(E, H, Ca, Cb, Da, Db)

# Subsequent calls use compiled kernel
for step in range(1000):
    E, H = step_compiled(E, H, Ca, Cb, Da, Db)
```

**Speedup**: 20-50% from kernel fusion and elimination of Python overhead.

### 2. Avoid Temporary Allocations in Hot Loop

**Bad**:
```python
for step in range(1000):
    E_temp = torch.zeros_like(E)  # allocation
    E_temp[:] = E + C * H  # assign
    E = E_temp  # reassign
```

**Good**:
```python
E_new = torch.zeros_like(E)
for step in range(1000):
    E_new[:] = E + C * H  # in-place assignment
    E, E_new = E_new, E  # swap (no allocation)
```

**Speedup**: 5-15% from reduced allocation overhead.

### 3. In-Place Operations

**Pattern**:
```python
# In-place multiply-add: E += Ca * E + Cb * curl
E.mul_(Ca).add_(Cb * curl)

# NOT: E = E * Ca + Cb * curl  (creates temporary)
```

**Speedup**: 10-20% from reduced temporary tensors.

### 4. Disable Gradient Tracking

**Essential** for inference-only simulations:

```python
import torch

with torch.no_grad():
    for step in range(1000):
        E, H = sim.step()
```

or

```python
torch.set_grad_enabled(False)
```

**Speedup**: 30-50% (eliminates autograd graph construction).

## Profiling and Bottleneck Identification

### PyTorch Profiler

```python
import torch

sim = Simulation3D(grid_size=(128, 128, 128))

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./trace'),
    record_shapes=True
) as prof:
    for step in range(100):
        sim.step()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Output**:
```
----- cuda_time_total -----
Name                        cuda_time_total  cuda_time_per_step
fused_e_update              45.2 ms          0.452 ms
fused_h_update              43.8 ms          0.438 ms
memory_copy                  8.5 ms          0.085 ms
source_injection             1.2 ms          0.012 ms
```

### NVIDIA nsys

```bash
nsys profile -t cuda python run_simulation.py
nsys stats report.nsys-rep
```

Identifies:
- Kernel launch latency
- Memory bandwidth utilization
- PCIe communication (if multi-GPU)
- Thermal throttling

### NVIDIA nvprof (Legacy)

```bash
nvprof python run_simulation.py
```

## Benchmark Suite

### Grid Scaling Test

```python
def benchmark_grid_scaling():
    """
    Measure throughput vs grid size.
    """
    grid_sizes = [64, 128, 256, 512]
    results = {}
    
    for grid in grid_sizes:
        sim = Simulation3D(
            domain_size=(grid/1000, grid/1000, grid/1000),
            grid_resolution=(grid, grid, grid)
        )
        
        # Warm up
        for _ in range(10):
            sim.step()
        
        # Timed run
        n_steps = 500
        torch.cuda.synchronize()
        t_start = time.time()
        
        for _ in range(n_steps):
            sim.step()
        
        torch.cuda.synchronize()
        t_end = time.time()
        
        total_cells = grid**3 * n_steps
        throughput = total_cells / (t_end - t_start) / 1e6  # Mcells/s
        
        results[grid] = {
            'throughput_mcells_per_s': throughput,
            'time_per_step_ms': (t_end - t_start) / n_steps * 1000,
            'memory_gb': sim.get_memory_usage() / 1e9
        }
        
        print(f"Grid {grid}³: {throughput:.1f} Mcells/s, "
              f"{results[grid]['memory_gb']:.2f} GB")
    
    return results
```

**Expected Results** (T4 GPU):
- 64³: ~800 Mcells/s
- 128³: ~500 Mcells/s
- 256³: ~350 Mcells/s
- 512³: ~150 Mcells/s (approaching memory limit, thermal throttling)

### Precision Impact Test

```python
def benchmark_precision():
    """
    Compare FP32 vs BF16 throughput and accuracy.
    """
    grid = 256
    n_steps = 500
    
    # FP32 baseline
    sim_fp32 = Simulation3D(grid_size=(grid, grid, grid), precision='float32')
    t_start = time.time()
    for _ in range(n_steps):
        sim_fp32.step()
    t_fp32 = time.time() - t_start
    
    # BF16
    sim_bf16 = Simulation3D(grid_size=(grid, grid, grid), precision='bfloat16')
    t_start = time.time()
    for _ in range(n_steps):
        sim_bf16.step()
    t_bf16 = time.time() - t_start
    
    speedup = t_fp32 / t_bf16
    print(f"BF16 speedup: {speedup:.2f}x")
```

**Expected**: 1.5-2.0x speedup, <1% difference in energy conservation.

## Performance Targets by Application

| Use Case | Grid | Target Throughput | Typical Time |
|----------|------|---|---|
| Prototyping | 64³ | 500-800 Mcells/s | ~10 ms/step |
| Standard | 128³ | 300-500 Mcells/s | ~20 ms/step |
| High-res imaging | 256³ | 200-350 Mcells/s | ~100 ms/step |
| Ultra-high-res | 512³ | 100-150 Mcells/s | ~800 ms/step |

## Optimization Roadmap

**Phase 8.1**: Kernel fusion (2-4 week effort, 20-30% gain)
**Phase 8.2**: Register tiling + shared memory (3-5 weeks, 30-40% gain)
**Phase 8.3**: Mixed precision (BF16) support (2-3 weeks, 40-60% gain)
**Phase 8.4**: torch.compile() integration (1-2 weeks, 20-50% gain)
**Phase 8.5**: Multi-GPU communication overlap (4-6 weeks, 10-20% per GPU)

**Combined target**: 5-10x speedup over naive implementation.

