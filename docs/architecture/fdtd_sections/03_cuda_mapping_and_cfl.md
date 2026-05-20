# Section 5: CUDA Execution Mapping

## 5.1 Thread-to-Cell Mapping

Each FDTD cell maps to exactly one CUDA thread. The 3D grid is decomposed into thread blocks of shape `(8, 8, 8)` = 512 threads/block.

```
Block dimensions:  blockDim = dim3(8, 8, 8)   → 512 threads
Warps per block:   512 / 32 = 16 warps
```

Thread indexing within a block:

```c
int ix = blockIdx.x * 8 + threadIdx.x;
int iy = blockIdx.y * 8 + threadIdx.y;
int iz = blockIdx.z * 8 + threadIdx.z;
int idx = ix * Ny * Nz + iy * Nz + iz;  // Z-fastest (column-major in Z)
```

## 5.2 Grid Launch Configuration

| Problem Size | Grid Blocks | Total Threads | Notes |
|---|---|---|---|
| 256³ | (32, 32, 32) = 32,768 | 16,777,216 | Fits in L2 partially |
| 512³ | (64, 64, 64) = 262,144 | 134,217,728 | Full GPU saturation |
| 768³ | (96, 96, 96) = 884,736 | 452,984,832 | Memory-bandwidth bound |

For non-power-of-2 domains, blocks at boundaries require bounds checking:

```c
if (ix >= Nx || iy >= Ny || iz >= Nz) return;
```

## 5.3 Warp Execution and Memory Coalescing

With `blockDim = (8, 8, 8)`, warp lane assignment follows the linearized threadIdx order:

```
Warp 0: threadIdx.z ∈ [0,7], threadIdx.y ∈ [0,3], threadIdx.x = 0
         → lanes 0-31 cover iz=0..7, iy=0..3 for fixed ix=0
```

Since Z is the fastest-varying dimension in memory layout and warp lanes vary in Z first, 8 consecutive lanes access 8 consecutive `float` addresses (32 bytes). A full warp spans 4 rows in Y × 8 cells in Z = 32 cells at consecutive Z-addresses within each Y-row.

**Coalescing analysis for E-field update (reading Hz neighbors):**

```
Hz[ix][iy][iz]     → address A
Hz[ix][iy-1][iz]   → address A - Nz*sizeof(float)   // stride = Nz floats away
```

For Nz=256: Y-neighbor is 1024 bytes away (different cache line). Z-neighbor is 4 bytes away (same cache line). This is the fundamental stencil bandwidth cost.

**Cache lines touched per warp (E_x update, reading H_y and H_z):**
- `Hz[ix][iy][iz]` and `Hz[ix][iy-1][iz]`: 2 cache lines (coalesced in Z, Y-stride miss)
- `Hy[ix][iy][iz]` and `Hy[ix][iy][iz-1]`: 1-2 cache lines (Z-adjacent, mostly same line)
- Total: ~4-6 L1 cache line requests per warp per field component read.

## 5.4 Register Usage Per Thread

For the E-field update kernel `E_x^{n+1} = Ca * E_x^n + Cb * (dHz/dy - dHy/dz)`:

| Register Purpose | Count |
|---|---|
| H-field neighbors: Hz(iy), Hz(iy-1), Hy(iz), Hy(iz-1) | 4 |
| Coefficients: Ca, Cb | 2 |
| Current field value: Ex | 1 |
| Finite differences (temporaries) | 2 |
| Index computation (ix, iy, iz, linear idx) | 4 |
| Address intermediates | 2 |
| **Total** | **~15 registers** |

At 15 regs/thread × 512 threads/block = 7,680 registers/block. SM 8.0 (A100) has 65,536 registers per SM → supports 8 concurrent blocks (limited by other resources to ~4-5 in practice).

## 5.5 Kernel Launch Overhead and CUDA Graphs

Single kernel launch overhead: ~5-10 μs. Per timestep with 3 separate launches (H, E, PML): ~15-30 μs overhead.

For a 256³ grid, kernel execution time ≈ 50-100 μs. Launch overhead is 15-30% of compute — unacceptable.

**CUDA Graph amortization:**

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    H_update_kernel(...); E_update_kernel(...); PML_kernel(...)
for t in range(num_steps):
    g.replay()  # ~3 μs total overhead (single command buffer submission)
```

## 5.6 Stream Scheduling

```
Stream 0 (compute):  [H-update] → [E-update] → [PML-update]
                         ↓ event
Stream 1 (I/O):      [DtoH field snapshot]  (overlaps next H-update)
```

Dependencies enforced via `cudaStreamWaitEvent`. H→E→PML are serialized (data dependency). Field output is asynchronous on a secondary stream.

**Fused kernel alternative:** Combine E-update + PML into a single kernel (branch on PML region flag per cell). Eliminates one launch and one global memory round-trip for E-field. Measured speedup: 15-25% for PML-heavy simulations.

## 5.7 Occupancy Analysis (SM 8.0 / A100)

```
Registers/thread:    15  → max 4,369 threads/SM (register-limited)
Shared mem/block:    0 bytes (pure register stencil)
Threads/block:       512
Max blocks/SM:       min(2048/512, 65536/(15×512), 32) = min(4, 8, 32) = 4
Active threads/SM:   4 × 512 = 2048
Occupancy:           2048 / 2048 = 100%
```

With 15 regs/thread, we achieve full occupancy. If register count grows to 32 (complex dispersive materials), occupancy drops to 50% — still acceptable for memory-bound kernels.

## 5.8 Performance Ladder: PyTorch vs Custom CUDA vs Triton

| Approach | 512³ E-update time | Bandwidth util. | Dev effort |
|---|---|---|---|
| PyTorch tensor ops (`E += Cb * (roll(Hz,-1,1) - Hz)`) | ~8.2 ms | 25-35% | Low |
| Triton kernel (explicit stencil, tiled) | ~2.1 ms | 65-75% | Medium |
| Custom CUDA kernel (hand-tuned, vectorized loads) | ~1.4 ms | 80-90% | High |
| Theoretical roofline (1555 GB/s, 6 reads + 1 write) | ~1.1 ms | 100% | — |

PyTorch penalty sources: (1) `roll()` allocates temporaries, (2) separate kernel per arithmetic op, (3) no stencil-aware caching. Triton closes most of the gap via fused loads and shared memory tiling.

---

# Section 6: CFL Stability Condition

## 6.1 Derivation for 3D FDTD

The Yee scheme yields an explicit update; stability requires the numerical domain of dependence contains the physical one (CFL criterion). Von Neumann analysis with ansatz `E ~ exp(i(kx·x + ky·y + kz·z - ωt))`:

```
[sin(ωΔt/2) / (cΔt/2)]² = [sin(kx·Δx/2) / (Δx/2)]² 
                          + [sin(ky·Δy/2) / (Δy/2)]²
                          + [sin(kz·Δz/2) / (Δz/2)]²
```

LHS max (when `sin(ωΔt/2) = 1`): `(2/(cΔt))²`. RHS max (all sines = 1): `(2/Δx)² + (2/Δy)² + (2/Δz)²`. Stability requires LHS_max >= RHS_max:

```
(2/(cΔt))² ≥ (2/Δx)² + (2/Δy)² + (2/Δz)²
```

Solving for Δt:

```
          1
Δt ≤ ─────────────────────────────
      c × √(1/Δx² + 1/Δy² + 1/Δz²)
```

## 6.2 Uniform Grid Simplification

For Δx = Δy = Δz = h:

```
Δt_max = h / (c × √3)     [3D]
Δt_max = h / (c × √2)     [2D]
Δt_max = h / c             [1D]
```

## 6.3 Courant Number

Define the Courant number `S = cΔt/Δx`. The CFL condition becomes:

```
S ≤ 1/√(d)    where d = spatial dimensionality
```

| Dimension | S_max | Numerical value |
|---|---|---|
| 1D | 1.0 | 1.0 |
| 2D | 1/√2 | 0.7071 |
| 3D | 1/√3 | 0.5774 |

**Practical choice: S = 0.5** — provides ~13% safety margin below the 3D limit and accommodates numerical perturbations from PML, dispersive media, and sub-cell averaging.

## 6.4 Material Impact on CFL

The phase velocity in a medium is `v = c₀ / √(εᵣ μᵣ)`. The CFL condition uses the *maximum* wave speed in the domain. For vacuum regions, `c = c₀ ≈ 3×10⁸ m/s`.

If the entire domain has `εᵣ ≥ ε_min > 1`:

```
Δt_max = (Δx × √(ε_min × μ_min)) / (c₀ × √3)
```

However, PML regions typically operate at `εᵣ = 1`, so CFL is almost always governed by `c₀` regardless of material content.

## 6.5 CFL Enforcement in Code

```python
def compute_stable_dt(dx, dy, dz, courant=0.5, c=2.998e8):
    """Compute maximum stable timestep satisfying CFL condition."""
    inv_sum = 1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2
    dt_max = 1.0 / (c * math.sqrt(inv_sum))
    dt = courant * dt_max  # Apply safety factor (courant < 1/√3)
    return dt

# Validation at runtime:
def validate_cfl(dt, dx, dy, dz, c=2.998e8):
    inv_sum = 1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2
    dt_limit = 1.0 / (c * math.sqrt(inv_sum))
    if dt > dt_limit:
        raise ValueError(
            f"CFL VIOLATED: dt={dt:.3e} > dt_max={dt_limit:.3e}. "
            f"Simulation will be numerically unstable."
        )
```

## 6.6 CFL Violation Consequences

When `S > 1/√3`, the amplification factor `|G| > 1` for at least one spatial frequency. Fields grow exponentially:

```
|E(t)| ~ |G|^n × |E(0)|     where n = t/Δt
```

For `S = 0.6` (3% over limit): `|G| ≈ 1.003` → fields double in ~231 steps. For `S = 0.7` (21% over limit): `|G| ≈ 1.05` → fields double in ~14 steps. Instability manifests as checkerboard patterns at the Nyquist frequency, originating at material boundaries or PML interfaces.

## 6.7 Non-Uniform Grid CFL

For graded meshes with cell sizes `{Δx_i, Δy_j, Δz_k}`:

```
Δt ≤ 1 / (c × √(1/Δx_min² + 1/Δy_min² + 1/Δz_min²))
```

The *minimum* cell size in each dimension governs the global timestep. This is the primary drawback of explicit FDTD on non-uniform grids: one small cell constrains the entire simulation.

## 6.8 Resolution and Timestep Table

Assuming uniform grid, free-space (`c = c₀`), S = 0.5, wavelength resolution = 20 cells/λ:

| Resolution (cells/λ) | Δx (nm) @ λ=1550nm | Δt (fs) | Steps for 100 fs | Min λ resolved (nm) |
|---|---|---|---|---|
| 10 | 155.0 | 0.1495 | 669 | 1550 |
| 20 | 77.5 | 0.0747 | 1339 | 775 |
| 30 | 51.7 | 0.0498 | 2008 | 517 |
| 40 | 38.75 | 0.0374 | 2674 | 387.5 |
| 60 | 25.83 | 0.0249 | 4016 | 258.3 |

**Computation:** `Δt = S × Δx / (c₀ × √3) = 0.5 × Δx / (2.998×10⁸ × 1.732)`

Practical accuracy requires 20+ cells/λ for < 1% phase error over long propagation distances.
