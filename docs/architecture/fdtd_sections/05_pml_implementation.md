# Section 8: PML (Perfectly Matched Layer) Implementation

## 8.1 PML Theory

The PML terminates the finite computational domain by introducing an artificial absorbing
medium impedance-matched to free space for all frequencies and angles of incidence.

**Stretched-coordinate formulation.** Each spatial derivative is replaced:
```
∂/∂x  →  (1/s_x) × ∂/∂x    where  s_x = κ_x + σ_x / (α_x + jωε₀)
```
- `κ_x ≥ 1`: real stretching (absorbs evanescent waves)
- `σ_x ≥ 0`: conductivity loss (absorbs propagating waves)
- `α_x ≥ 0`: CFS parameter (stabilizes low-frequency/dc response)

**CFS-PML.** Standard PML (α=0) suffers late-time instability from dc/evanescent modes.
Setting α > 0 shifts the pole from ω=0, providing stable absorption at all frequencies
and eliminating late-time linear growth artifacts.

**CPML.** The 1/s_x stretching produces a time-domain convolution. CPML implements this
via recursive (IIR) auxiliary "psi" variables. The inverse stretching expands as:
```
1/s_x = 1/κ_x + (σ_x/κ_x) / (σ_x·κ_x + κ_x²·α_x + jωε₀·κ_x²)
```
The convolution kernel decays exponentially, enabling one-pole recursive approximation.

---

## 8.2 CPML Update Equations

### 8.2.1 Auxiliary Psi Variables (12 total)

E-field psi terms (6):

| Field | Psi | Replaces | Field | Psi | Replaces |
|-------|-----|----------|-------|-----|----------|
| Ex | psi_Exy | ∂Hz/∂y | Ex | psi_Exz | ∂Hy/∂z |
| Ey | psi_Eyx | ∂Hz/∂x | Ey | psi_Eyz | ∂Hx/∂z |
| Ez | psi_Ezx | ∂Hy/∂x | Ez | psi_Ezy | ∂Hx/∂y |

H-field psi terms (6):

| Field | Psi | Replaces | Field | Psi | Replaces |
|-------|-----|----------|-------|-----|----------|
| Hx | psi_Hxy | ∂Ez/∂y | Hx | psi_Hxz | ∂Ey/∂z |
| Hy | psi_Hyx | ∂Ez/∂x | Hy | psi_Hyz | ∂Ex/∂z |
| Hz | psi_Hzx | ∂Ey/∂x | Hz | psi_Hzy | ∂Ex/∂y |

### 8.2.2 CPML Coefficients

For a given PML axis (y-direction shown):
```
b_y = exp( -(σ_y/κ_y + α_y) × Δt/ε₀ )
c_y = (σ_y / (σ_y·κ_y + κ_y²·α_y)) × (b_y - 1)
```
When σ_y = 0: c_y = 0, b_y = exp(-α_y·Δt/ε₀).

### 8.2.3 Recursive Update and Field Correction

Generic form:
```
psi_Exy^{n+1}[i,j,k] = b_y[j] × psi_Exy^n[i,j,k] + c_y[j] × (Hz[i,j,k] - Hz[i,j-1,k])/Δy
```

Modified E-field update in PML (Ex example):
```
Ex^{n+1} = Ex^n + Cb × ( (1/κ_y)(Hz[i,j,k]-Hz[i,j-1,k])/Δy
                        - (1/κ_z)(Hy[i,j,k]-Hy[i,j,k-1])/Δz )
                 + Cb × ( psi_Exy^{n+1} - psi_Exz^{n+1} )
```
where Cb = Δt/(ε₀·ε_r).

### 8.2.4 Complete E-field CPML Equations

```
# Ex:
psi_Exy^{n+1} = b_y × psi_Exy^n + c_y × (Hz[i,j,k] - Hz[i,j-1,k])/Δy
psi_Exz^{n+1} = b_z × psi_Exz^n + c_z × (Hy[i,j,k] - Hy[i,j,k-1])/Δz
Ex^{n+1} += Cb_x × (psi_Exy^{n+1} - psi_Exz^{n+1})

# Ey:
psi_Eyz^{n+1} = b_z × psi_Eyz^n + c_z × (Hx[i,j,k] - Hx[i,j,k-1])/Δz
psi_Eyx^{n+1} = b_x × psi_Eyx^n + c_x × (Hz[i,j,k] - Hz[i-1,j,k])/Δx
Ey^{n+1} += Cb_y × (psi_Eyz^{n+1} - psi_Eyx^{n+1})

# Ez:
psi_Ezx^{n+1} = b_x × psi_Ezx^n + c_x × (Hy[i,j,k] - Hy[i-1,j,k])/Δx
psi_Ezy^{n+1} = b_y × psi_Ezy^n + c_y × (Hx[i,j,k] - Hx[i,j-1,k])/Δy
Ez^{n+1} += Cb_z × (psi_Ezx^{n+1} - psi_Ezy^{n+1})
```

### 8.2.5 Complete H-field CPML Equations

```
# Hx:
psi_Hxy^{n+1} = b_y × psi_Hxy^n + c_y × (Ez[i,j+1,k] - Ez[i,j,k])/Δy
psi_Hxz^{n+1} = b_z × psi_Hxz^n + c_z × (Ey[i,j,k+1] - Ey[i,j,k])/Δz
Hx^{n+1} += Db_x × (psi_Hxz^{n+1} - psi_Hxy^{n+1})

# Hy:
psi_Hyx^{n+1} = b_x × psi_Hyx^n + c_x × (Ez[i+1,j,k] - Ez[i,j,k])/Δx
psi_Hyz^{n+1} = b_z × psi_Hyz^n + c_z × (Ex[i,j,k+1] - Ex[i,j,k])/Δz
Hy^{n+1} += Db_y × (psi_Hyx^{n+1} - psi_Hyz^{n+1})

# Hz:
psi_Hzx^{n+1} = b_x × psi_Hzx^n + c_x × (Ey[i+1,j,k] - Ey[i,j,k])/Δx
psi_Hzy^{n+1} = b_y × psi_Hzy^n + c_y × (Ex[i,j+1,k] - Ex[i,j,k])/Δy
Hz^{n+1} += Db_z × (psi_Hzy^{n+1} - psi_Hzx^{n+1})
```
where Db = Δt/(μ₀·μ_r).

---

## 8.3 PML Grading Profiles

### 8.3.1 Polynomial Grading

Parameters graded from inner interface (d=0) to outer boundary (d=D):
```
σ(d) = σ_max × (d/D)^m          m = 3 or 4
κ(d) = 1 + (κ_max - 1) × (d/D)^m
α(d) = α_max × (1 - d/D)        linear decrease into PML
```

### 8.3.2 Optimal σ_max

```
σ_opt = -(m+1) × ln(R(0)) / (2η₀D)
```
For -40 dB one-way reflection target:
```
σ_opt = (m+1) / (150π·Δx) × c₀
```
Numerically (m=3, Δx=10nm): σ_opt ~ 1.13e9 S/m.

### 8.3.3 Recommended Ranges

| Parameter | Range | Notes |
|-----------|-------|-------|
| κ_max | 5-15 | Larger for evanescent-heavy problems |
| α_max | 0.02-0.05 S/m | Prevents late-time growth |
| m | 3-4 | Polynomial order for σ and κ |

### 8.3.4 Reflection vs. PML Thickness

| D (cells) | Theoretical R (dB) | Practical R (dB)* |
|-----------|--------------------|--------------------|
| 5         | -35                | -25 to -30         |
| 8         | -55                | -40 to -50         |
| 10        | -70                | -50 to -60         |
| 15        | -105               | -65 to -80         |
| 20        | -140               | -80 to -100        |

*Practical values include discretization error and fp32 arithmetic. Default: D=10.

---

## 8.4 GPU Implementation

### 8.4.1 Memory Layout

PML tensors allocated only for 6 boundary slabs:
```
Face ±x: shape (D, Ny, Nz)    Face ±y: shape (Nx, D, Nz)    Face ±z: shape (Nx, Ny, D)
```
Each face stores 4 psi tensors (2 E-field psi + 2 H-field psi for that axis).

### 8.4.2 Memory Budget

```
Total: 12 psi arrays × D × N² × 4 bytes (float32)
N=512, D=10: 12 × 10 × 512 × 512 × 4B = 125.8 MB
```
Coefficient arrays (b, c, 1/κ): 1D vectors of length D per axis -- negligible.

### 8.4.3 Kernel Strategy

**Option A -- Separate PML kernels:**
```python
def update_E_pml_xfaces(Ex, Ey, Ez, Hx, Hy, Hz, psi_x, b_x, c_x, kappa_x):
    # Map local p ∈ [0,D) to global: -x face → i=p, +x face → i=Nx-D+p
    psi_Eyx[p,j,k] = b_x[p] * psi_Eyx[p,j,k] + c_x[p] * dHz_dx
    psi_Ezx[p,j,k] = b_x[p] * psi_Ezx[p,j,k] + c_x[p] * dHy_dx
    Ey[i_global,j,k] += Cb * psi_Eyx[p,j,k]
    Ez[i_global,j,k] += Cb * psi_Ezx[p,j,k]
# Launch: grid=(ceil(D/4), ceil(Ny/8), ceil(Nz/8)), block=(4,8,8)=256 threads
```

**Option B -- Fused with main update kernel:**
```python
# Inside main E-field kernel:
if i < D or i >= Nx - D:
    p = i if i < D else i - (Nx - D)
    psi_Eyx[p,j,k] = b_x[p] * psi_Eyx[p,j,k] + c_x[p] * dHz_dx
    Ey[i,j,k] += Cb * psi_Eyx[p,j,k]
```
Fused avoids redundant global memory loads but introduces warp divergence at
PML/interior boundary. Profiling: 5-8% speedup on A100 for grids >= 256^3.

### 8.4.4 Index Mapping

```python
def pml_to_global(face: str, p: int, N: int, D: int) -> int:
    return p if face == 'lo' else N - D + p

# κ-modified finite difference in PML:
dHz_dx = (Hz[i,j,k] - Hz[i-1,j,k]) / (kappa_x[p] * dx)
```

---

## 8.5 PML Performance Impact

### 8.5.1 Cell Count Overhead

```
PML cells = 6·D·N² - 12·D²·N + 8·D³  (inclusion-exclusion)
```

| Grid | D | PML cells | Total | PML % |
|------|---|-----------|-------|-------|
| 128³ | 10 | 0.89M | 2.10M | 42% |
| 256³ | 10 | 3.77M | 16.78M | 22% |
| 512³ | 10 | 15.4M | 134.2M | 11.5% |
| 512³ | 8 | 12.4M | 134.2M | 9.2% |
| 1024³| 10 | 62.1M | 1073.7M | 5.8% |

### 8.5.2 Time Overhead

PML adds 2 FMA + 1 add per psi update (+4 FLOP per E/H component per PML cell).

| Grid | D | PML % | Time overhead | Kernel type |
|------|---|-------|---------------|-------------|
| 256³ | 10 | 22% | 18-22% | Separate |
| 256³ | 10 | 22% | 15-18% | Fused |
| 512³ | 10 | 11.5% | 13-15% | Separate |
| 512³ | 10 | 11.5% | 11-13% | Fused |
| 512³ | 8 | 9.2% | 9-11% | Fused |
| 1024³| 10 | 5.8% | 7-9% | Fused |

### 8.5.3 Optimization Strategies

1. **Fused PML kernel**: eliminates redundant global loads of H-field values already
   fetched for standard E-update. Saves 2 global loads per PML cell.
2. **fp16 psi storage**: halves PML memory; verified <0.1 dB reflection degradation for D>=8.
3. **Async streams**: overlap psi loads/stores with interior cell computation.
4. **Symmetry reduction**: omit PML on symmetry planes, up to 50% PML overhead reduction.
