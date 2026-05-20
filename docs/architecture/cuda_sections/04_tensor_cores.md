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
