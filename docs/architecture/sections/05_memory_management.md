# Section 5: Memory Management and Tensor Layout

## 5.1 VRAM Budget Analysis

### Per-Component Memory Formula

```
mem_per_component = Nx × Ny × Nz × sizeof(dtype)
```

### Full Simulation Memory Model

| Category | Formula | Notes |
|----------|---------|-------|
| E-fields (3 components) | 3 × N³ × 4B | Ex, Ey, Ez (FP32) |
| H-fields (3 components) | 3 × N³ × 4B | Hx, Hy, Hz (FP32) |
| Material coefficients | 6 × N³ × 4B | Ca, Cb per component (or 2 if uniform) |
| PML psi fields | 12 × D × S × 4B | 6 faces × 2 psi per face, D=depth, S=surface |
| PML coefficients | 6 × D × 4B | Negligible |
| Source buffers | N_src × N_t × 4B | Usually small |
| Detector DFT | N_freq × N_det × 8B | Complex64 |
| **Total (dominant)** | **(12 + N_mat) × N³ × 4B** | |

### VRAM Requirements Table (FP32)

| Grid Size | Cells | Fields (6) | +Materials (6) | +PML (D=10) | Total Est. |
|-----------|-------|-----------|----------------|-------------|------------|
| 128³ | 2.1M | 50 MB | 100 MB | +12 MB | **~120 MB** |
| 256³ | 16.8M | 403 MB | 806 MB | +48 MB | **~900 MB** |
| 512³ | 134M | 3.2 GB | 6.4 GB | +190 MB | **~7.0 GB** |
| 768³ | 453M | 10.8 GB | 21.6 GB | +430 MB | **~23 GB** |
| 1024³ | 1.07B | 25.6 GB | 51.2 GB | +770 MB | **~54 GB** |

### Mixed Precision (BF16 fields, FP32 materials)

| Grid Size | Fields (BF16) | Materials (FP32) | Total Est. |
|-----------|---------------|------------------|------------|
| 512³ | 1.6 GB | 3.2 GB | **~5.2 GB** |
| 768³ | 5.4 GB | 10.8 GB | **~17 GB** |
| 1024³ | 12.8 GB | 25.6 GB | **~40 GB** |

---

## 5.2 Tensor Memory Layout

### Structure of Arrays (SoA) Design

Each field component is a separate contiguous 3D tensor:

```python
fields = {
    'Ex': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
    'Ey': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
    'Ez': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
    'Hx': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
    'Hy': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
    'Hz': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
}
```

**Why SoA over AoS:**
- E-field update reads only H components → SoA loads exactly what's needed
- Coalesced access: adjacent threads read adjacent memory (contiguous in Z)
- AoS (interleaved Ex,Ey,Ez,Hx,Hy,Hz per cell) wastes 50% bandwidth loading unused components

### Memory Ordering

PyTorch default: **row-major (C-contiguous)**, last dimension varies fastest.

For tensor shape `(Nx, Ny, Nz)`:
- Stride: `(Ny*Nz, Nz, 1)`
- Adjacent threads (threadIdx.x mapped to Z) access contiguous memory ✓
- **Thread block (8,8,8):** innermost 8 threads read consecutive Z addresses = 32-byte aligned (perfect coalescing)

### Padding Strategy

```python
def pad_to_warp(N, warp_size=32):
    return ((N + warp_size - 1) // warp_size) * warp_size

Nz_padded = pad_to_warp(Nz)  # Ensures last dimension is multiple of 32
```

Padding the fastest-varying dimension to multiples of 32 ensures:
- Full warp coalescing (no partial transactions)
- 128-byte cache line alignment
- Negligible memory overhead (~3% worst case for Nz=33→64)

---

## 5.3 Memory Pool Architecture

### Pre-Allocation Policy

```python
class TensorPool:
    def __init__(self, grid_shape, device, dtype=torch.float32):
        self.fields = self._allocate_fields(grid_shape, device, dtype)
        self.pml = self._allocate_pml(grid_shape, pml_depth, device, dtype)
        self.materials = self._allocate_materials(grid_shape, device, dtype)
        self.scratch = self._allocate_scratch(grid_shape, device, dtype)
        # All allocations happen here. ZERO allocations during time-stepping.
```

### Design Rules

1. **No runtime allocation in hot loop.** All tensors pre-allocated at `initialize()`. Violation → assertion failure in debug mode.
2. **Scratch buffers** for temporary computations reuse same memory across steps.
3. **Double buffering** for checkpoint: buffer A writes to disk while buffer B captures next checkpoint.
4. **Pinned host memory** (`torch.cuda.HostAllocator`) for async GPU→CPU transfers during checkpointing.

### CUDA Memory Allocator Integration

```python
torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use 95% of VRAM
torch.cuda.memory.empty_cache()  # Defragment before simulation
```

PyTorch's caching allocator handles sub-allocation. We configure:
- Large initial pool (avoid fragmentation)
- `max_split_size_mb=512` to prevent excessive fragmentation
- Explicit `torch.cuda.synchronize()` before measuring peak memory

---

## 5.4 Mixed Precision Strategy

### Precision Assignment by Data Type

| Data | Precision | Rationale |
|------|-----------|-----------|
| E, H fields | BF16 or FP32 | BF16 for speed; FP32 for accuracy-critical runs |
| Material coefficients (Ca, Cb) | FP32 | Computed once, precision matters for stability |
| PML psi fields | FP32 | Accumulation over many steps; BF16 drifts |
| PML grading (b, c, kappa) | FP32 | Small tensors, precision-sensitive |
| DFT accumulators | FP32 or FP64 | Long accumulation (N_steps terms); FP32 OK with Kahan summation |
| Source waveforms | FP32 | Small, precision-sensitive for phase accuracy |
| Gradient tensors (adjoint) | FP32 | Gradient underflow risk with FP16 |

### BF16 Field Update Precision Impact

Numerical dispersion error in FDTD: `δ(kΔx)` depends on floating-point rounding.

- FP32 (24-bit mantissa): relative error ~10⁻⁷ per step, accumulates to ~10⁻⁴ over 10³ steps
- BF16 (8-bit mantissa): relative error ~10⁻², accumulates to ~1.0 over 10³ steps → **UNSTABLE for long runs**
- **Mitigation:** Kahan compensated summation in BF16, or periodic FP32 correction steps

**Recommendation:** Use BF16 for field components only when:
- Simulation is short (<1000 steps) OR
- Periodic FP32 renormalization applied (every 100 steps) OR
- Application is gradient computation (short forward pass + adjoint)

### AMP (Automatic Mixed Precision) Integration

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    H_new = update_H(E, H, coeffs)  # Runs in BF16
    E_new = update_E(H_new, E, coeffs)  # Runs in BF16

# PML always in FP32:
with torch.autocast(enabled=False):
    apply_pml(E_new, psi, pml_coeffs.float())
```

---

## 5.5 Memory Bandwidth Optimization

### Achievable Bandwidth

| GPU | Peak BW | Achievable (80%) | Cells/s (FP32, 192B/cell) |
|-----|---------|-------------------|---------------------------|
| V100 | 900 GB/s | 720 GB/s | 3.75 Gcells/s |
| A100 | 2,039 GB/s | 1,631 GB/s | 8.5 Gcells/s |
| H100 | 3,350 GB/s | 2,680 GB/s | 14.0 Gcells/s |

### Access Pattern Optimization

**Coalesced Access:** Threads in a warp access consecutive 4-byte addresses. The stencil `Hz[i,j,k] - Hz[i,j-1,k]` requires:
- `Hz[i,j,k]`: coalesced (threads differ in k)
- `Hz[i,j-1,k]`: coalesced (same stride pattern, shifted by Nz)

Both are coalesced. The problematic pattern is `Hz[i,j,k-1]` (stride-1 neighbor in fastest dimension) — also coalesced since threads at k and k-1 are both within the same cache line for block size 8.

### L2 Cache Residency Control (SM 8.0+)

```cpp
cudaAccessPolicyWindow policy;
policy.base_ptr = (void*)Hz_ptr;
policy.num_bytes = Nx * Ny * Nz * sizeof(float);
policy.hitRatio = 0.6;  // 60% of L2 reserved for H-fields
policy.hitProp = cudaAccessPropertyPersisting;
policy.missProp = cudaAccessPropertyStreaming;
cudaCtxSetAccessPolicyWindow(&policy);
```

For 512³: 6 field tensors = 3.2 GB. A100 L2 = 40 MB → caches ~1.2% of fields. Residency control pins most-reused data (e.g., the z-1 plane being read by many warps).

### Register Blocking

Keep stencil values in registers across E-field components:
```
// Hz[i,j,k] needed by both Ex and Ey updates
register float hz_ijk = Hz[idx];
Ex_new = Ca_x * Ex + Cb_x * (hz_ijk - hz_ijm1k) * dt_dy - ...
Ey_new = Ca_y * Ey - Cb_y * (hz_ijk - hz_im1jk) * dt_dx + ...
```

Avoids redundant global memory loads when updating multiple components per thread.
