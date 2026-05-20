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
