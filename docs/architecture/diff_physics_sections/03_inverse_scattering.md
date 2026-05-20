# Section 3: Inverse Scattering

## 3.1 Problem Formulation

### Forward Problem
Given material distribution ε(x,y,z), compute scattered fields via FDTD:
```
ε(x,y,z) → FDTD simulation → E_scattered(r_rx, t) for all TX/RX pairs
```

### Inverse Problem
Given measured scattered fields, reconstruct material distribution:
```
minimize  J(ε) = ½ Σ_{tx} ||E_sim(ε, tx) - E_meas(tx)||² + R(ε)
   ε

where R(ε) = regularization term
```

**Properties:**
- Nonlinear: E_sim depends nonlinearly on ε (multiple scattering, resonances)
- Ill-posed: many ε distributions can produce similar scattered fields
- Non-convex: multiple local minima (especially for large objects, high frequencies)
- High-dimensional: ε has N³ unknowns (millions for typical grids)

## 3.2 Linearized Approximations

### Born Approximation (Single Scattering)

```
E_scattered ≈ ∫ G(r_rx, r') × Δε(r') × E_incident(r') dr'
```

Discretized: `E_s = G × diag(E_inc) × Δε` → linear system, solvable via least-squares.

**Valid when:** |Δε| << ε_background AND object size << λ (weak scatterer).

### Rytov Approximation

```
ln(E_total / E_incident) ≈ ∫ G(r, r') × Δε(r') × k² dr'
```

Better for smooth, large objects. Still linear in Δε.

### Distorted Born Iterative (DBI)

```
for iteration in range(N_iter):
    E_total = FDTD(ε_current)                    # Full forward solve
    G = compute_Green_function(ε_current)         # Linearize around current
    Δε = solve_linear(G, E_meas - E_sim)         # Linear inversion step
    ε_current += Δε
```

Handles stronger scatterers by re-linearizing. Converges for moderate contrast (ε_r < 5).

### When Linearization Fails

| Condition | Born/Rytov | DBI | Full Nonlinear |
|-----------|-----------|-----|---------------|
| ε_r contrast < 1.5 | Works | Works | Overkill |
| ε_r contrast 1.5-5 | Fails | Works (slow) | Recommended |
| ε_r contrast > 5 | Fails | May diverge | Required |
| Object > 3λ | Fails | Slow | Required |
| Multiple scattering | Fails | Partial | Handles correctly |

## 3.3 Full Nonlinear Inversion via Differentiable FDTD

### Optimization Loop

```python
# Initialize permittivity (uniform background or prior estimate)
eps = torch.ones(Nx, Ny, Nz, device='cuda', requires_grad=True) * eps_background
optimizer = torch.optim.Adam([eps], lr=1e-3)

for iteration in range(N_iterations):
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device='cuda')
    
    for tx_idx in range(N_tx):
        # Forward simulation (differentiable)
        fields = fdtd_forward(eps, source=tx_configs[tx_idx], n_steps=N_t)
        
        # Extract simulated receiver data
        E_sim = fields.at_receivers(rx_positions)  # (N_rx, N_t)
        
        # Data misfit
        E_meas = measurements[tx_idx]  # (N_rx, N_t)
        misfit = torch.sum((E_sim - E_meas)**2)
        total_loss += misfit
    
    # Regularization
    total_loss += lambda_tv * total_variation(eps)
    
    # Adjoint gradient computation
    total_loss.backward()  # Calls adjoint FDTD internally
    
    # Update material
    optimizer.step()
    
    # Enforce physical constraints
    with torch.no_grad():
        eps.clamp_(1.0, 80.0)  # ε_r bounds
```

### Gradient Computation

`total_loss.backward()` triggers the adjoint method for each TX simulation:
- Adjoint runs time-reversed FDTD with adjoint sources at receiver locations
- Produces ∂L/∂ε: gradient of loss with respect to permittivity at every cell
- All N_TX adjoint passes are independent → can be parallelized (batched or sequential)

## 3.4 Regularization Strategies

### Total Variation (TV) — Edge-Preserving

```python
def total_variation_3d(eps):
    """Anisotropic TV: sum of absolute differences along each axis."""
    dx = torch.abs(eps[1:, :, :] - eps[:-1, :, :]).sum()
    dy = torch.abs(eps[:, 1:, :] - eps[:, :-1, :]).sum()
    dz = torch.abs(eps[:, :, 1:] - eps[:, :, :-1]).sum()
    return dx + dy + dz

# Smooth approximation (differentiable):
def total_variation_smooth(eps, beta=1e-3):
    """Huber-like smooth TV for gradient-based optimization."""
    dx = eps[1:, :, :] - eps[:-1, :, :]
    dy = eps[:, 1:, :] - eps[:, :-1, :]
    dz = eps[:, :, 1:] - eps[:, :, :-1]
    return (torch.sqrt(dx**2 + beta**2) - beta).sum() + \
           (torch.sqrt(dy**2 + beta**2) - beta).sum() + \
           (torch.sqrt(dz**2 + beta**2) - beta).sum()
```

### Tikhonov (L2) — Smooth Solution

```python
def tikhonov(eps, eps_prior):
    return torch.sum((eps - eps_prior)**2)
```

### Physics-Based Constraints

```python
with torch.no_grad():
    eps.clamp_(1.0, 80.0)       # Permittivity bounds (vacuum to water)
    sigma.clamp_(0.0, 10.0)     # Conductivity non-negative
    # Enforce symmetry (if known):
    eps[:] = 0.5 * (eps + eps.flip(0))  # Mirror symmetry in x
```

### Regularization Weight Schedule

```python
# Start strong (smooth), decrease over iterations (allow detail)
lambda_tv = lambda_tv_init * (0.95 ** iteration)
```

## 3.5 Multi-Frequency Inversion

### Frequency Hopping Strategy

```
Iteration 1-20:   f = 1 GHz (λ = 30 cm, resolves features > 3 cm)
Iteration 21-50:  f = 2 GHz (λ = 15 cm, resolves features > 1.5 cm)
Iteration 51-100: f = 4 GHz (λ = 7.5 cm, resolves features > 0.75 cm)
Iteration 101+:   f = 1-4 GHz simultaneous (multi-frequency data fit)
```

**Rationale:** Low-frequency objective landscape is more convex (fewer wavelengths across object → fewer local minima). Starting from low frequency provides a good basin of attraction for high-frequency refinement.

### Broadband Alternative

Use broadband pulse (covers 1-4 GHz in single simulation), extract frequency components via DFT:
```python
# Single broadband FDTD simulation
E_time = fdtd_forward(eps, source=broadband_pulse)  # (N_rx, N_t)

# Extract frequency components
freqs = [1e9, 2e9, 3e9, 4e9]
E_freq = []
for f in freqs:
    dft_kernel = torch.exp(-2j * pi * f * t_array * dt)  # (N_t,)
    E_freq.append(torch.matmul(E_time, dft_kernel))  # (N_rx,)

# Multi-frequency loss
loss = sum(mse(E_freq_sim[f], E_freq_meas[f]) for f in freqs)
```

Advantage: one simulation provides all frequencies. Cost: longer time series needed (to resolve lowest frequency).

## 3.6 Convergence and Performance

### Convergence Characteristics

| Object Type | Iterations to Converge | Final Misfit | SSIM |
|------------|----------------------|-------------|------|
| Single cylinder (ε=3) | 30-50 | 10⁻⁴ | 0.98 |
| Multiple cylinders | 80-150 | 10⁻³ | 0.92 |
| Realistic breast phantom | 200-500 | 10⁻² | 0.85 |
| Through-wall (concrete + targets) | 300-800 | 10⁻² | 0.80 |

### Computational Cost

```
Per iteration:
  N_tx forward simulations: 32 × 5s = 160s (parallelizable)
  N_tx adjoint simulations: 32 × 5s = 160s (parallelizable)
  Optimizer step: <1s
  Total per iteration: 320s (sequential) or 80s (4 concurrent sims)

Full reconstruction (200 iterations):
  Sequential: 200 × 320s = 17.8 hours
  4× concurrent: 200 × 80s = 4.4 hours
  On A100 (larger grid fits in BF16): ~3 hours
```

### Acceleration Strategies

| Strategy | Speedup | Quality Impact |
|----------|---------|---------------|
| BF16 forward (short adjoint) | 1.5× | Negligible (100-step forward) |
| Coarse-to-fine grid | 3-5× | Improves convergence (multi-scale) |
| Mini-batch TX (8/32 per iter) | 4× | Noisier gradients, more iterations needed |
| Neural initialization | 10-50× | Reduces iterations from 200 to 20 |
| Frequency hopping | 2× | Better convergence rate |

## 3.7 Comparison with Classical Methods

| Method | Multiple Scattering | Memory | Iterations | GPU-Native | Accuracy |
|--------|-------------------|--------|-----------|------------|----------|
| Born/Rytov | No | Low | 1 (direct) | Yes (GEMM) | Low (weak scatterers only) |
| Distorted Born (DBI) | Partial | Medium | 10-30 | Partially | Medium |
| Contrast Source (CSI) | Yes | Medium | 50-200 | Partially | High |
| **Full nonlinear (GPU-MEEP)** | **Yes** | **High** | **50-500** | **Yes (native)** | **Highest** |
| Newton-based (Gauss-Newton) | Yes | Very high | 10-30 | Difficult | High |

GPU-MEEP's advantage: the forward model (FDTD) and gradient computation (adjoint) are both GPU-native, eliminating the CPU bottleneck that plagues classical iterative methods.
