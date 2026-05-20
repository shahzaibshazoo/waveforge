# Section 7: Numerical Stability and Dispersion

## 7.1 Numerical Dispersion Relation

Substituting plane-wave trial solutions E = E₀ exp[j(kx·iΔx + ky·jΔy + kz·kΔz - ω·nΔt)] into the Yee update equations yields the **FDTD numerical dispersion relation**:

```
[1/(cΔt) · sin(ωΔt/2)]² = [1/Δx · sin(kx·Δx/2)]² + [1/Δy · sin(ky·Δy/2)]² + [1/Δz · sin(kz·Δz/2)]²
```

Compare with the continuous dispersion relation (ω/c)² = kx² + ky² + kz². The discrete form reduces to continuous only as Δx, Δy, Δz, Δt → 0. The **numerical phase velocity** v_p,num differs from c:

```
v_p,num / c = (ω·Δx) / (2c · arcsin[ cΔt/Δx · sin(ωΔt/2) / sin(kΔx/2) ])
```

### Anisotropic Dispersion Artifact

Phase velocity error depends on propagation direction relative to grid axes. For uniform cubic grid (Δx = Δy = Δz = δ) at Courant limit S = cΔt/δ = 1/√3 (3D):
- Along grid axis (θ=0°): maximum phase error
- Along body diagonal (θ=54.7°): minimum phase error

Mitigation:
1. **Grid resolution rule**: Δx ≤ λ_min / 10 (minimum 10 cells per shortest wavelength)
2. **High-accuracy rule**: 20 cells/wavelength → < 1% cumulative phase error
3. **Operate at Courant limit**: S = S_max minimizes dispersion for axis-aligned propagation

Phase error accumulates over distance L: Δφ = (L/λ) · 2π · (1 - v_p,num/c). At 10 cells/λ, Courant limit: |1 - v_p,num/c| ≈ 0.8% → Δφ ≈ 0.05 rad/wavelength.

## 7.2 Floating-Point Precision Impact

### FP32 vs FP64 Error Budget

| Property         | FP32           | FP64           | BF16          |
|-----------------|----------------|----------------|---------------|
| Mantissa bits   | 23             | 52             | 7             |
| Machine epsilon | 1.19 × 10⁻⁷   | 2.22 × 10⁻¹⁶  | 3.91 × 10⁻³  |
| Relative error  | ~10⁻⁷/op      | ~10⁻¹⁶/op     | ~10⁻²/op     |

### Roundoff Accumulation

Each update introduces relative error ε_mach. After N_steps:
- Uncorrelated (random walk): ΔE_rms/E ~ ε_mach · √N_steps
- Coherent (resonant structures): ΔE_worst/E ~ ε_mach · N_steps

### Energy Drift Analysis

| N_steps | FP32 drift (random) | FP32 drift (coherent) | FP64 drift (coherent) |
|---------|---------------------|-----------------------|-----------------------|
| 10⁴    | 10⁻⁵               | 10⁻³                 | 10⁻¹²                |
| 10⁵    | 3×10⁻⁵             | 10⁻²                 | 10⁻¹¹                |
| 10⁶    | 10⁻⁴               | 10⁻¹                 | 10⁻¹⁰                |

### When FP64 Is Mandatory

- Resonant cavities with Q > 10⁴ (ring-down requires > 10⁵ steps)
- High-Q photonic crystal cavities (Q ~ 10⁶, coherent accumulation)
- Long waveguide propagation (> 1000λ path length)
- Adjoint sensitivity analysis over > 10⁴ steps

### BF16 Danger

With ε_mach ≈ 3.9 × 10⁻³, energy drift reaches O(1) after:
```
N_critical = 1/ε_mach² ≈ 65,000 steps (random)
N_critical = 1/ε_mach ≈ 256 steps (coherent/worst-case)
```
BF16 is **unstable for > 100 steps** without periodic correction to higher precision.

## 7.3 Stability Diagnostics

### Energy Conservation Monitor

```python
def compute_em_energy(E, H, eps, mu):
    """All tensors on GPU, shape [3, Nx, Ny, Nz]."""
    W_e = 0.5 * torch.sum(eps * E**2)
    W_m = 0.5 * torch.sum(mu * H**2)
    return W_e + W_m
```

- **Lossless**: dW/dt = 0 (constant to within roundoff)
- **Lossy (σ > 0)**: dW/dt ≤ 0 (monotonically decreasing)
- **Diverging energy**: CFL violation or implementation error

### Divergence Check

∇·(εE) = ρ_free must hold at every time step. Numerically:
```
div_E[i,j,k] = (eps_x[i]*Ex[i] - eps_x[i-1]*Ex[i-1])/Δx
             + (eps_y[j]*Ey[j] - eps_y[j-1]*Ey[j-1])/Δy
             + (eps_z[k]*Ez[k] - eps_z[k-1]*Ez[k-1])/Δz
```
If max|div_E| grows exponentially → instability. Check every 50-100 steps.

### Field Magnitude Monitoring

```python
def check_stability(E, H, threshold=1e6):
    if torch.max(torch.abs(E)) > threshold or torch.max(torch.abs(H)) > threshold:
        raise RuntimeError("Field divergence detected — CFL violation or source error")
```

### NaN/Inf Detection

Execute every N steps (N=10 debug, N=100 production). Cost: ~2μs per `any()` reduction.
```python
def nan_check(E, H, step):
    if torch.isnan(E).any() or torch.isnan(H).any() or torch.isinf(E).any() or torch.isinf(H).any():
        raise RuntimeError(f"NaN/Inf detected at step {step}")
```

## 7.4 Lossy Media Stability

### Conductive Media (σ > 0)

Semi-implicit E-field update with conductivity:
```
Eⁿ⁺¹ = C_a · Eⁿ + C_b · (∇×H)ⁿ⁺¹/²
C_a = (1 - σΔt/2ε) / (1 + σΔt/2ε)    → |C_a| < 1, always stabilizing
C_b = (Δt/ε) / (1 + σΔt/2ε)
```

### Gain Media (σ < 0)

Negative conductivity yields |C_a| > 1, amplifying fields. Stability requires:
```
Δt < 2ε / (|σ| · (1 + S/S_max))
```
In practice: reduce Courant number by factor (1 - |σ|Δt/2ε).

### Dispersive Media (ADE Formulation)

**Debye model**: dP/dt + P/τ = ε₀(ε_s - ε_∞)/τ · E. Discretized:
```
Pⁿ⁺¹ = [(1 - Δt/2τ)/(1 + Δt/2τ)] · Pⁿ + [ε₀(ε_s - ε_∞)Δt/τ/(1 + Δt/2τ)] · Eⁿ⁺¹/²
```
ADE formulation is unconditionally stable provided base-grid CFL is satisfied.

**Drude model** near plasma frequency: ε(ω) = ε_∞ - ω_p²/(ω² + jγω). When ω → ω_p: Re(ε) → 0, λ_eff → ∞. Required handling:
- Subcell averaging of permittivity near ε = 0 crossings
- Adaptive Courant number: reduce S when min(Re(ε)) < 0.1·ε₀
- Monitor auxiliary current J_Drude for unbounded growth

## 7.5 Mixed Precision Stability Protocol

### BF16 Field Updates (Short Runs)

Stable for < 1000 steps with Courant number S < 0.4·S_max:
```python
E = E.bfloat16()
H = H.bfloat16()
C_b = (dt / eps).bfloat16()  # S < 0.4 * S_max provides rounding headroom
```
Reduced Courant ensures effective amplification per step from rounding stays below unity.

### Kahan Compensated Summation (DFT Monitors)

For frequency-domain accumulation over N_steps >> 1000, Kahan summation reduces error from O(N·ε) to O(ε):
```python
class KahanDFTAccumulator:
    def __init__(self, shape, freqs, device='cuda'):
        self.real = torch.zeros(len(freqs), *shape, dtype=torch.float32, device=device)
        self.imag = torch.zeros_like(self.real)
        self.comp_r = torch.zeros_like(self.real)
        self.comp_i = torch.zeros_like(self.real)

    def accumulate(self, field, step, dt, freqs):
        for i, f in enumerate(freqs):
            phase = 2 * math.pi * f * step * dt
            y = field * math.cos(phase) - self.comp_r[i]
            t = self.real[i] + y
            self.comp_r[i] = (t - self.real[i]) - y
            self.real[i] = t
            y = -field * math.sin(phase) - self.comp_i[i]
            t = self.imag[i] + y
            self.comp_i[i] = (t - self.imag[i]) - y
            self.imag[i] = t
```

### Periodic FP32 Correction

For BF16 runs exceeding 100 steps, cast to FP32 every 100 steps:
```python
def precision_correction(E_bf16, H_bf16, correction_interval=100, step=0):
    if step % correction_interval == 0:
        E_f32, H_f32 = E_bf16.float(), H_bf16.float()
        E_f32, H_f32 = fdtd_step_fp32(E_f32, H_f32)
        return E_f32.bfloat16(), H_f32.bfloat16()
    return fdtd_step_bf16(E_bf16, H_bf16)
```
Bounds accumulated BF16 error to ~100 × ε_BF16 ≈ 0.39 (recoverable range).

### Gradient Computation: Always FP32

Adjoint/backpropagation through FDTD requires FP32 minimum:
```python
# NEVER: loss.backward() with BF16 fields — gradient underflow guaranteed
# CORRECT:
E_f32 = E.float().requires_grad_(True)
H_f32 = H.float().requires_grad_(True)
loss.backward()  # gradients in FP32
```
Gradients are O(10⁻⁴) to O(10⁻⁸) smaller than fields. BF16 minimum normal is 2⁻¹²⁶ with only 7-bit precision — gradients below ~10⁻² are effectively zero in BF16.
