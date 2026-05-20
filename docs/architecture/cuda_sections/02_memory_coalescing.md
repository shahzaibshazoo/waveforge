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
