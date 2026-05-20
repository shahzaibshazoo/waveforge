# Section 4: GPU Execution Model

## 4.1 Why FDTD Maps to GPU

The FDTD algorithm exhibits **massive data parallelism**: each cell's field update depends only on its immediate neighbors (6-point stencil). For an N³ grid, N³ independent updates execute per half-timestep.

### Arithmetic Intensity Analysis

Per cell E-field update:
- **Reads:** 4 H-field neighbors + 2 material coefficients + 1 current E value = 7 floats = 28 bytes
- **Writes:** 1 updated E value = 4 bytes
- **FLOPs:** 2 subtractions (curl) + 2 multiplications (coefficient scaling) + 1 addition = ~5 FLOPs
- **Per component:** 5 FLOPs / 32 bytes = **0.16 FLOP/byte**

Full cell (6 components): ~30 FLOPs, ~192 bytes transferred.

**Conclusion:** FDTD is **memory-bandwidth bound**. The A100's 2 TB/s HBM bandwidth is the ceiling, not its 19.5 TFLOPS FP32 compute. All optimization must target memory throughput.

### Theoretical Peak Performance

```
A100 80GB: 2,039 GB/s HBM bandwidth
Bytes per cell per step: 192 B (6 components, read+write, FP32)
Max cells/s = 2,039e9 / 192 = 10.6 Gcells/s (theoretical)
Achievable (70% efficiency): ~7.4 Gcells/s = 7,400 Mcells/s
```

With mixed precision (FP16): 96 bytes/cell → **14.8 Gcells/s theoretical**.

---

## 4.2 Kernel Architecture

### E-Field Update Kernel

```
__global__ void update_E(
    float* Ex, float* Ey, float* Ez,           // Output (read-modify-write)
    const float* Hx, const float* Hy, const float* Hz,  // Input (read-only)
    const float* Ca, const float* Cb,          // Material coefficients
    int Nx, int Ny, int Nz, float dt_dx, float dt_dy, float dt_dz
)
```

**Grid mapping:**
- 1 CUDA thread = 1 grid cell
- Thread block: `(8, 8, 8)` = 512 threads (good occupancy, fits 3D locality)
- Grid dims: `(ceil(Nx/8), ceil(Ny/8), ceil(Nz/8))`

**Update equation (Ex component):**
```
Ex[i,j,k] = Ca[i,j,k] * Ex[i,j,k]
           + Cb[i,j,k] * ( (Hz[i,j,k] - Hz[i,j-1,k]) * dt_dy
                          - (Hy[i,j,k] - Hy[i,j,k-1]) * dt_dz )
```

### H-Field Update Kernel

Identical structure to E-update but reads E neighbors and writes H. Operates at half-timestep offset.

### PML Kernel (CPML Formulation)

```
psi_Exy[i,j,k] = b_y[j] * psi_Exy[i,j,k]
                + c_y[j] * (Hz[i,j,k] - Hz[i,j-1,k])
Ex[i,j,k] += Cb[i,j,k] * psi_Exy[i,j,k]
```

**PML-specific considerations:**
- Only executes on boundary cells (PML_depth × surface_area)
- Can be fused with E-field kernel using predicated execution (branch on cell position)
- Separate kernel preferred when PML region << interior (avoid branch divergence in bulk)

### Source Injection Kernel

```
__global__ void inject_source(
    float* field_component,
    const int* indices,     // Sparse cell indices
    const float* amplitudes, // Per-cell amplitude
    float waveform_value,   // Current time sample
    int N_source_cells
)
```

Launched with N_source_cells threads. Scatter pattern — low occupancy but overlaps with bulk updates.

### Detector Kernel (DFT Accumulation)

```
// For each monitor frequency f_m:
dft_real[m, cell] += field[cell] * cos(2π * f_m * t * dt)
dft_imag[m, cell] += field[cell] * sin(2π * f_m * t * dt)
```

Gather + FMA pattern. Launched on detector cells only.

---

## 4.3 CUDA Stream Strategy

### Stream Assignment

| Stream | Purpose | Sync Requirements |
|--------|---------|-------------------|
| 0 (default) | Field update kernels (H→E→H→...) | Sequential within stream |
| 1 | Source injection | Event sync before field update reads source cells |
| 2 | Detector recording | Event after field update completes |
| 3 | Halo exchange (multi-GPU) | Event after boundary region update |
| 4 | Async I/O (checkpoint) | No sync needed (double-buffered) |

### Timeline for One Timestep

```
         ┌────────────────────── Timestep n ──────────────────────┐
Stream 0: [═══ H_update_interior ═══][wait_halo][H_boundary][═══ E_update ═══]
Stream 1: [src_H]                              [src_E]
Stream 2:                                      [detect_H]         [detect_E]
Stream 3: [══ halo_send ══][halo_recv]
Stream 4:                                                   [checkpoint_async]
```

### CUDA Graph Capture

For steady-state time-stepping (no dynamic branches):
```python
with torch.cuda.graph(graph):
    step_body()  # Entire timestep captured as static graph

for t in range(N_steps):
    graph.replay()  # Near-zero CPU overhead per step
```

CUDA Graphs eliminate kernel launch overhead (~5-10 μs per launch × ~10 kernels = 50-100 μs saved/step).

---

## 4.4 Kernel Fusion Opportunities

### Beneficial Fusions

| Fusion | Benefit | Condition |
|--------|---------|-----------|
| E_update + PML_E | 1 kernel launch, shared H reads | Always (PML cells are subset) |
| H_update + PML_H | Same as above | Always |
| Field_update + material_lookup | Avoid extra global load | When materials fit in shared memory |
| DFT_accumulate across frequencies | Single field read, multiple accumulates | N_freqs ≤ register budget |

### When NOT to Fuse

- Source injection + field update: Source is sparse (1% of cells), field update is dense. Fusing adds branch divergence to 99% of threads.
- Checkpoint copy + field update: Copy is on different stream for overlap. Fusing serializes them.

---

## 4.5 Occupancy Analysis

### Thread Block Sizing

| Block Shape | Threads | Registers/Thread (est.) | Shared Mem | Occupancy (SM 8.0) |
|-------------|---------|------------------------|------------|---------------------|
| (8,8,8) | 512 | 32 | 0 | 100% (2 blocks/SM) |
| (16,16,4) | 1024 | 32 | 0 | 50% (1 block/SM) |
| (8,8,4) | 256 | 40 | 2 KB | 100% (4 blocks/SM) |
| (32,8,4) | 1024 | 28 | 4 KB | 50% (1 block/SM) |

**Selected default: `(8, 8, 8)` = 512 threads.**

Rationale:
- 3D locality matches 3D stencil access pattern
- 2 blocks/SM = 1024 threads → good latency hiding
- 32 registers per thread × 512 threads = 16K registers (SM has 64K)
- Leaves headroom for compiler register spilling

### Register Pressure

Per-thread state for E-field update:
- 4 H neighbor values: 4 registers
- 2 material coefficients: 2 registers
- 1 current E value: 1 register
- Index computation: 3-4 registers
- Temporaries: 2-3 registers
- **Total: ~12-13 registers** (well within budget)

### Shared Memory Optimization

For stencil operations, neighboring threads read overlapping H values. Shared memory tiling:

```
Tile size: (8+2) × (8+2) × (8+2) = 1000 floats = 4 KB per H component
3 H components needed per E-component update: 12 KB
```

**Trade-off:** Shared memory reduces global memory reads by ~30% (halo reuse) but limits blocks/SM. Beneficial for large grids where cache misses dominate; less impactful on small grids fitting in L2.

---

## 4.6 PyTorch Integration

### Custom Autograd Function

```python
class FDTDStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, E, H, materials, dt, dx):
        # Launch CUDA kernels (custom or Triton)
        H_new = update_H_kernel(E, H, materials, dt, dx)
        E_new = update_E_kernel(H_new, E, materials, dt, dx)
        ctx.save_for_backward(E, H, materials)
        return E_new, H_new

    @staticmethod
    def backward(ctx, grad_E, grad_H):
        # Adjoint update (time-reversed)
        E, H, materials = ctx.saved_tensors
        grad_materials = adjoint_kernel(grad_E, grad_H, E, H)
        return grad_E_prev, grad_H_prev, grad_materials, None, None
```

### Triton Kernel Path (Rapid Prototyping)

```python
@triton.jit
def update_Ex_kernel(
    Ex_ptr, Hz_ptr, Hy_ptr, Ca_ptr, Cb_ptr,
    Nx, Ny, Nz, dt_dy, dt_dz,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr, BLOCK_Z: tl.constexpr
):
    # Triton handles block/grid mapping, bounds checking, memory coalescing
    pid = tl.program_id(0)
    # ... stencil computation in Triton DSL
```

Triton advantages: auto-tuning block sizes, no CUDA boilerplate, integrates with torch.compile.

### torch.compile Compatibility

All operations use standard PyTorch tensor ops where possible:
```python
def update_H_pytorch(E, H, coeff, dt_dx):
    curl_E_x = (E.Ez.roll(-1, 1) - E.Ez) * dt_dx - (E.Ey.roll(-1, 2) - E.Ey) * dt_dx
    H.Hx -= coeff.Db * curl_E_x
```

`torch.compile(mode='max-autotune')` fuses these into efficient kernels automatically. Custom CUDA kernels used only where torch.compile underperforms (measured >10% gap).
