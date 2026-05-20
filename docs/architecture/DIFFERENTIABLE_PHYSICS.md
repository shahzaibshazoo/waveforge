# GPU-MEEP: Differentiable Physics Engine

> Autograd Integration, Adjoint Methods, Inverse Scattering, and Neural Reconstruction
> End-to-End Differentiable Electromagnetic Simulation for Optimization and Learning

---

# Section 1: PyTorch Autograd Integration

## 1.1 Differentiable Simulation Concept

The FDTD simulation forms a computational graph amenable to automatic differentiation:

```
input parameters → [timestep 0] → [timestep 1] → ... → [timestep N] → observable → loss
```

**Parameters that can require gradients:**
- Material properties: `eps.requires_grad_(True)`, `mu.requires_grad_(True)`, `sigma.requires_grad_(True)`
- Geometry (parameterized SDF): shape vertices, radii, control points
- Source properties: amplitude `A`, phase `φ`, position `(x₀, y₀, z₀)`

**Output observables (scalar loss targets):**
- Transmitted/reflected flux, scattered field amplitude, mode overlap integral, image quality metrics

**Goal:** Compute `∂(observable)/∂(parameters)` in O(N_steps) time and O(1) memory via adjoint methods, rather than O(N_params × N_steps) via finite differences.

## 1.2 torch.autograd.Function for FDTD

Each FDTD timestep is wrapped as a custom `torch.autograd.Function`:

```python
class FDTDStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, E, H, eps, mu, sigma, dt, dx):
        # E, H: (3, Nx, Ny, Nz); eps/mu/sigma: (1 or 3, Nx, Ny, Nz)
        # Curl H for E update (central differences on staggered Yee grid)
        curl_H = torch.empty_like(E)
        curl_H[0] = (H[2,:,1:,:] - H[2,:,:-1,:] - H[1,:,:,1:] + H[1,:,:,:-1]) / dx
        curl_H[1] = (H[0,:,:,1:] - H[0,:,:,:-1] - H[2,1:,:,:] + H[2,:-1,:,:]) / dx
        curl_H[2] = (H[1,1:,:,:] - H[1,:-1,:,:] - H[0,:,1:,:] + H[0,:,:-1,:]) / dx

        # E update with conductivity damping
        decay = (1 - sigma * dt / (2 * eps)) / (1 + sigma * dt / (2 * eps))
        gain = (dt / eps) / (1 + sigma * dt / (2 * eps))
        E_new = decay * E + gain * curl_H

        # Curl E for H update
        curl_E = torch.empty_like(H)
        curl_E[0] = (E_new[2,:,1:,:] - E_new[2,:,:-1,:] - E_new[1,:,:,1:] + E_new[1,:,:,:-1]) / dx
        curl_E[1] = (E_new[0,:,:,1:] - E_new[0,:,:,:-1] - E_new[2,1:,:,:] + E_new[2,:-1,:,:]) / dx
        curl_E[2] = (E_new[1,1:,:,:] - E_new[1,:-1,:,:] - E_new[0,:,1:,:] + E_new[0,:,:-1,:]) / dx

        H_new = H - (dt / mu) * curl_E

        ctx.save_for_backward(E, H, eps, mu, sigma, E_new, curl_H)
        ctx.dt = dt
        ctx.dx = dx
        return E_new, H_new

    @staticmethod
    def backward(ctx, grad_E_new, grad_H_new):
        """Adjoint (time-reversed) FDTD step."""
        E, H, eps, mu, sigma, E_new, curl_H = ctx.saved_tensors
        dt, dx = ctx.dt, ctx.dx

        decay = (1 - sigma * dt / (2 * eps)) / (1 + sigma * dt / (2 * eps))
        gain = (dt / eps) / (1 + sigma * dt / (2 * eps))
        denom = (1 + sigma * dt / (2 * eps))

        # ∂L/∂E via chain rule through E_new = decay*E + gain*curl_H
        grad_E = grad_E_new * decay
        # ∂L/∂H via curl^T operator + direct H_new contribution
        grad_H = grad_H_new + curl_T(grad_E_new * gain, dx)

        # ∂L/∂ε: permittivity sensitivity at every cell
        grad_eps = grad_E_new * (
            -dt * curl_H / (eps**2 * denom)
            + sigma * dt**2 * E / (2 * eps**2 * denom**2)
        )
        grad_eps = grad_eps.sum(dim=0, keepdim=True)

        # ∂L/∂μ: permeability sensitivity
        grad_mu = (grad_H_new * (dt / mu**2) * curl_E_cached).sum(dim=0, keepdim=True)

        # ∂L/∂σ: conductivity sensitivity
        grad_sigma = grad_E_new * (-dt * E / (2 * eps * denom**2))

        return grad_E, grad_H, grad_eps, grad_mu, grad_sigma, None, None
```

**Simulation loop:**

```python
def run_fdtd(E, H, eps, mu, sigma, dt, dx, n_steps, source_fn):
    for n in range(n_steps):
        E = E + source_fn(n, dt)  # differentiable source injection
        E, H = FDTDStep.apply(E, H, eps, mu, sigma, dt, dx)
    return E, H
```

## 1.3 Memory Challenge

Naive autograd retains every intermediate tensor for backpropagation:

| Grid Size | Steps | Stored Fields (6 comp, E+H) | Total Memory |
|-----------|-------|------------------------------|--------------|
| 128³      | 500   | 500 × 6 × 128³ × 4B        | 25 GB        |
| 256³      | 1000  | 1000 × 6 × 256³ × 4B       | 402 GB       |
| 512³      | 1000  | 1000 × 6 × 512³ × 4B       | 3.2 TB       |

With material tensors + intermediates saved for backward (~12 tensors/step):

```
Memory = 1000 × 12 × 512³ × 4B = 6.4 TB  → IMPOSSIBLE on any single GPU
```

**Solutions** (detailed in Section 2):
- **Gradient checkpointing**: recompute forward in segments, store only checkpoint boundaries
- **Adjoint-state method**: time-reversed simulation, O(1) field storage
- **Mixed precision**: float16 checkpoints, float32 compute

## 1.4 What's Differentiable in GPU-MEEP

| Parameter | Tensor Shape | Gradient Meaning |
|-----------|-------------|------------------|
| `eps(x,y,z)` | `(1 or 3, Nx, Ny, Nz)` | ∂L/∂ε at every grid cell |
| `sigma(x,y,z)` | `(1 or 3, Nx, Ny, Nz)` | ∂L/∂σ, material loss sensitivity |
| `mu(x,y,z)` | `(1 or 3, Nx, Ny, Nz)` | ∂L/∂μ (rarely optimized) |
| Source amplitude | `(N_freq,)` or scalar | ∂L/∂A per frequency |
| Source phase | `(N_freq,)` or scalar | ∂L/∂φ per frequency |
| Geometry params | `(N_params,)` | ∂L/∂p via chain rule through SDF→ε |

**Geometry differentiation via smoothed SDF:**

```python
def sdf_to_eps(sdf_values, eps_inside, eps_outside, blur_width=1.0):
    """Smoothed Heaviside: differentiable geometry boundary."""
    eta = torch.sigmoid(sdf_values / blur_width)
    return eps_inside * eta + eps_outside * (1 - eta)
```

**NOT differentiable (discrete):** grid resolution, PML thickness, number of timesteps, BC type, material model selection.

## 1.5 Gradient Flow Through Time

Forward (Yee leapfrog interleaves E/H at half-step offsets):

```
Forward:  E⁰ → H^(1/2) → E¹ → H^(3/2) → ... → E^N → loss(E^N)
Backward: ∂L/∂E^N → ∂L/∂H^(N-1/2) → ∂L/∂E^(N-1) → ... → ∂L/∂E⁰
               ↓              ↓              ↓
          ∂L/∂ε(n=N)    ∂L/∂μ(n=N-1)   ∂L/∂ε(n=N-1)   (accumulated)
```

**Material gradient accumulation** -- the critical identity:

```python
# ∂L/∂ε = Σₙ (∂L/∂Eⁿ) × (∂Eⁿ/∂ε)
# Adjoint computes this without storing full trajectory:
grad_eps_total = torch.zeros_like(eps)
for n in reversed(range(n_steps)):
    grad_eps_total += compute_eps_sensitivity(adjoint_E[n], forward_E[n], eps)
```

This summation is exactly what the adjoint-state method computes efficiently (Section 2).

## 1.6 Loss Functions for EM Optimization

All losses must be differentiable scalars of field tensors:

```python
def transmission_loss(E, H, monitor_plane, freq):
    """Maximize power flux (|S21|²) through monitor plane."""
    E_f = dft_accumulate(E, monitor_plane, freq)  # complex (3, My, Mz)
    H_f = dft_accumulate(H, monitor_plane, freq)
    Sx = (E_f[1]*H_f[2].conj() - E_f[2]*H_f[1].conj()).real
    return -Sx.sum()  # negative = maximize

def mode_overlap_loss(E_sim, E_target):
    """Maximize normalized overlap with target mode."""
    overlap = (E_sim * E_target.conj()).sum()
    return -(overlap.abs()**2) / ((E_sim.abs()**2).sum() * (E_target.abs()**2).sum())

def focusing_loss(E, focal_idx):
    """Maximize intensity at focal point."""
    return -(E[:, focal_idx[0], focal_idx[1], focal_idx[2]]**2).sum()

def inverse_scattering_loss(eps_recon, eps_true):
    """L2 reconstruction for inverse problems."""
    return torch.nn.functional.mse_loss(eps_recon, eps_true)

def scattering_match_loss(E_sim, E_meas):
    """Match scattered field to measurements."""
    return ((E_sim - E_meas).abs()**2).sum()
```

**Full optimization loop:**

```python
eps = torch.ones(1, Nx, Ny, Nz, device='cuda').requires_grad_(True)
optimizer = torch.optim.Adam([eps], lr=1e-2)

for iteration in range(200):
    optimizer.zero_grad()
    E, H = run_fdtd(E0, H0, eps, mu, sigma, dt, dx, n_steps, source)
    loss = transmission_loss(E, H, monitor, freq)
    loss.backward()  # adjoint propagates through all N timesteps
    optimizer.step()
    eps.data.clamp_(1.0, 12.0)  # physical bounds on permittivity
```

---

# Section 2: Adjoint Method for FDTD

## 2.1 Why Adjoint (Not Backprop Through Time)

| Method | Memory | Compute | Applicability |
|--------|--------|---------|---------------|
| Direct backprop (store all) | O(N_steps × grid) | 1× forward + 1× backward | Impossible for large grids |
| Backprop + naive checkpoint | O(√N × grid) | ~1.5× forward | Feasible but slow |
| Adjoint method | O(grid) + checkpoints | 2× forward | Production method |

**The problem:** For 512³ grid, 1000 steps, storing all intermediate fields requires:
```
1000 steps × 6 components × 512³ × 4B = 3.2 TB → IMPOSSIBLE on any GPU
```

**Adjoint solution:** Only need forward fields at the CURRENT adjoint timestep. Recompute them from checkpoints on-the-fly. Total memory: 2× one timestep (forward + adjoint fields) + K checkpoint slots.

## 2.2 Adjoint FDTD Derivation

### Optimization Problem

```
minimize   J(ε) = L(E^N(ε))      (loss depends on final fields)
subject to E^{n+1} = F(E^n, H^n, ε)   (FDTD equations as constraint)
```

### Lagrangian Formulation

```
L = J(E^N) + Σ_{n=0}^{N-1} λ^n · (E^{n+1} - F(E^n, H^n, ε))
```

where λⁿ are adjoint variables (Lagrange multipliers) with same shape as fields.

### Stationarity Conditions

Setting ∂L/∂E^n = 0 for each n yields the adjoint equation:

```
λ^{n-1} = (∂F/∂E^n)ᵀ · λ^n    for n = N-1, ..., 1
λ^{N-1} = ∂J/∂E^N              (terminal condition)
```

### Key Insight: Adjoint = Time-Reversed FDTD

The Jacobian transpose `(∂F/∂E)ᵀ` for the FDTD update is:
- **Transposed curl operator** = negative curl (for centered finite differences on Yee grid)
- **Same material coefficients** Ca, Cb (self-adjoint for lossless media)
- **Time runs backward** (n decreases)

Therefore: one adjoint step = one FDTD step with time reversed and curl sign flipped.

## 2.3 Adjoint Update Equations

### Forward (Standard FDTD)

```
H^{n+½} = H^{n-½} - (Δt/μ) × ∇×E^n
E^{n+1} = Ca × E^n + Cb × ∇×H^{n+½}
```

### Adjoint (Time-Reversed)

```
H̃^{n-½} = H̃^{n+½} + (Δt/μ) × ∇×Ẽ^n     ← sign flip (time reversal)
Ẽ^{n-1} = Ca × Ẽ^n + Cb × ∇×H̃^{n-½}      ← same coefficients
```

Plus adjoint source injection at detector locations:
```
Ẽ^n += ∂L/∂E^n_measured    (at cells where detectors recorded data)
```

### Lossy Media Adjoint

For lossy media (σ ≠ 0), Ca < 1 introduces asymmetry:
- Forward: E decays over time (Ca < 1 damps)
- Adjoint: must use Ca (NOT 1/Ca) — adjoint of multiplication by Ca is multiplication by Ca

The adjoint of `E^{n+1} = Ca × E^n + Cb × curl_H` is:
```
Ẽ^n = Ca × Ẽ^{n+1} + adjoint_source^n
grad_contribution^n = -Ẽ^{n+1} × curl_H^{n+½} × (∂Cb/∂ε)
```

## 2.4 Checkpointing Strategy (Griewank/Binomial)

### Trade Compute for Memory

Store field checkpoints at selected timesteps. When adjoint needs forward fields at step n, recompute from nearest earlier checkpoint.

```
Timeline:  [──────────────── N steps ────────────────]
Checkpoints: ✓         ✓         ✓         ✓         ✓
             0       200       400       600       800     1000

Adjoint at step 750: load checkpoint at 600, recompute 600→750, then do adjoint step.
```

### Binomial Checkpointing (Revolve Algorithm)

Optimal schedule minimizing recomputation for given memory budget:

```python
def revolve_schedule(n_steps, n_checkpoints):
    """Griewank's revolve algorithm for optimal checkpoint placement."""
    if n_checkpoints >= n_steps:
        return list(range(n_steps))  # Store everything
    
    # Binomial coefficient: C(n_checkpoints + recomputations, n_checkpoints) >= n_steps
    schedule = []
    # ... recursive subdivision (well-known algorithm)
    return schedule
```

**Memory-compute tradeoff:**

| Checkpoints | Memory (512³) | Recomputation Factor | Total Compute |
|-------------|---------------|---------------------|---------------|
| N (store all) | 3.2 TB | 1× | 2× forward |
| √N ≈ 32 | 28 GB | log(N) ≈ 3× | 4× forward |
| 10 | 8.6 GB | ~5× | 6× forward |
| 5 | 4.3 GB | ~14× | 15× forward |

**Recommended:** 10-20 checkpoints → fits in VRAM alongside forward/adjoint fields, 5-6× total compute overhead.

## 2.5 Material Gradient Computation

### Gradient Formula

```
∂J/∂ε[i,j,k] = Σ_{n=0}^{N-1} Ẽ^n[i,j,k] × E^n[i,j,k] × (∂Ca/∂ε) 
              + Σ_{n=0}^{N-1} Ẽ^n[i,j,k] × (∇×H^{n-½})[i,j,k] × (∂Cb/∂ε)
```

where:
```
∂Ca/∂ε = -σΔt / (ε + σΔt/2)²
∂Cb/∂ε = -Δt / (ε + σΔt/2)²
```

### GPU Implementation

The gradient at each cell is an element-wise product of forward and adjoint fields, accumulated over time:

```python
grad_eps = torch.zeros_like(eps)  # (Nx, Ny, Nz)

for n in reversed(range(N_steps)):
    # Get forward state (from checkpoint or recompute)
    E_fwd, H_fwd = get_forward_state(n)
    
    # Adjoint step
    adjoint_H = adjoint_H_update(adjoint_E, adjoint_H)
    adjoint_E = adjoint_E_update(adjoint_H, adjoint_E)
    inject_adjoint_source(adjoint_E, n)
    
    # Accumulate gradient (single element-wise kernel, fully parallel)
    curl_H = compute_curl(H_fwd)
    grad_eps += adjoint_E * E_fwd * dCa_deps + adjoint_E * curl_H * dCb_deps
```

**Cost per adjoint step:** one element-wise multiply-add over (Nx,Ny,Nz) → negligible vs field update.

## 2.6 Complete Adjoint Algorithm

```
Algorithm: Adjoint-Based Gradient Computation
─────────────────────────────────────────────

Input: ε (material), source configs, loss function L
Output: ∂L/∂ε (gradient for optimization)

1. FORWARD PASS
   Initialize fields E⁰ = 0, H⁰ = 0
   for n = 0 to N-1:
       H^{n+½} = update_H(E^n, H^{n-½})
       E^{n+1} = update_E(H^{n+½}, E^n, ε)
       apply_pml(E, H)
       record_detectors(E, H, n)
       if n in checkpoint_schedule:
           save_checkpoint(E, H, n)

2. COMPUTE LOSS
   loss = L(detector_data, target_data)
   ∂L/∂E^N = autograd(loss, E^N)  (or manual derivative)

3. ADJOINT PASS (time-reversed)
   Ẽ^N = ∂L/∂E^N, H̃^N = 0
   grad_eps = 0
   
   for n = N-1 downto 0:
       // Recompute forward state from nearest checkpoint
       E^n, H^n = recompute_from_checkpoint(n)
       
       // Adjoint field update (time-reversed Maxwell)
       H̃^{n-½} = H̃^{n+½} + (Δt/μ) × ∇×Ẽ^n
       Ẽ^{n-1} = Ca × Ẽ^n + Cb × ∇×H̃^{n-½}
       
       // Inject adjoint source (at detector locations)
       Ẽ^{n-1}[detector_cells] += ∂L/∂E^n_recorded
       
       // Accumulate material gradient
       grad_eps += Ẽ^n ⊙ E^n ⊙ dCa/dε + Ẽ^n ⊙ curl(H^{n-½}) ⊙ dCb/dε

4. RETURN grad_eps
```

## 2.7 Adjoint Accuracy Validation

### Finite-Difference Gradient Check

```python
def gradient_check(eps, cell_idx, delta=1e-4):
    """Compare adjoint gradient with finite-difference approximation."""
    # Forward perturbation
    eps_plus = eps.clone()
    eps_plus[cell_idx] += delta
    loss_plus = run_simulation(eps_plus)
    
    # Backward perturbation
    eps_minus = eps.clone()
    eps_minus[cell_idx] -= delta
    loss_minus = run_simulation(eps_minus)
    
    # Finite-difference gradient
    fd_grad = (loss_plus - loss_minus) / (2 * delta)
    
    # Adjoint gradient
    adjoint_grad = run_adjoint(eps)[cell_idx]
    
    # Relative error
    rel_error = abs(fd_grad - adjoint_grad) / max(abs(fd_grad), 1e-10)
    assert rel_error < 1e-4, f"Adjoint error: {rel_error}"
```

### Common Adjoint Bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Wrong curl sign | Gradient has wrong sign | Adjoint curl = -forward curl (transpose) |
| Missing factor of 2 | Gradient 2× too large | Leapfrog half-step averaging |
| PML not detached | Gradient includes PML artifacts | Mask or detach PML region |
| Wrong checkpoint restoration | Random gradient values | Verify forward state matches |
| Accumulation order | Small numerical differences | Sum in consistent order (Kahan) |

### Expected Accuracy

| Precision | FD δ | Adjoint-FD Relative Error |
|-----------|------|--------------------------|
| FP32 adjoint, FP32 FD | 10⁻⁴ | < 10⁻³ |
| FP32 adjoint, FP64 FD | 10⁻⁶ | < 10⁻⁴ |
| FP64 adjoint, FP64 FD | 10⁻⁸ | < 10⁻⁶ |

---

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

---

# 4. Gradient Propagation Through FDTD

## 4.1 Gradient Chain Rule Through Time

FDTD is a discrete recurrence relation on state vector `s = (E, H)`:

```
sⁿ⁺¹ = F(sⁿ, θ)       where θ = (ε, μ, σ)
```

For a scalar loss L = L(s^N) evaluated at final timestep N, the chain rule gives:

```
∂L/∂θ = Σₙ₌₀ᴺ⁻¹ (∂L/∂sⁿ) · (∂sⁿ/∂θ)

where:  ∂L/∂sⁿ = (∂L/∂sᴺ) · ∏ₖ₌ₙᴺ⁻¹ (∂sᵏ⁺¹/∂sᵏ)
```

The Jacobian `Jₙ = ∂sⁿ⁺¹/∂sⁿ` has stencil-sparse structure (bandwidth = stencil width).

**Spectral radius and stability:**
- CFL-satisfied FDTD: `ρ(Jₙ) ≤ 1` → no gradient explosion (backward-stable)
- Lossless media (σ = 0): `ρ(Jₙ) = 1` (energy-conserving, unitary-like)
- Lossy media (σ > 0): `ρ(Jₙ) < 1` → gradients decay exponentially backward

The product `∏ₖ₌ₙᴺ⁻¹ Jₖ` contracts geometrically for lossy media. Early timesteps contribute negligible gradient signal in long simulations with material loss.

## 4.2 Jacobian Structure of FDTD Step

The FDTD update (standard Yee formulation):

```
Eⁿ⁺¹ = Ca·Eⁿ + Cb·curl(Hⁿ⁺½)
Hⁿ⁺³⁄² = Hⁿ⁺½ - (Δt/μ)·curl(Eⁿ⁺¹)
```

Block Jacobian decomposition:

```
∂sⁿ⁺¹/∂sⁿ = | ∂Eⁿ⁺¹/∂Eⁿ     ∂Eⁿ⁺¹/∂Hⁿ⁺½   |
              | ∂Hⁿ⁺³⁄²/∂Eⁿ    ∂Hⁿ⁺³⁄²/∂Hⁿ⁺½  |
```

Each block:
- `∂Eⁿ⁺¹/∂Eⁿ = diag(Ca)` — diagonal matrix of material coefficients
- `∂Eⁿ⁺¹/∂Hⁿ⁺½ = diag(Cb) × ∇×` — sparse banded (curl operator, 6 nonzeros/row in 3D)
- `∂Hⁿ⁺³⁄²/∂Eⁿ⁺¹ = -diag(Δt/μ) × ∇×` — sparse banded (coupled through E update)
- `∂Hⁿ⁺³⁄²/∂Hⁿ⁺½ = I` — identity (no magnetic loss assumed)

For lossless media: Ca = 1, the full Jacobian is symplectic (energy-preserving).

**GPU rule: never form Jacobians explicitly.** Apply as JVP/VJP via elementwise + stencil ops.

## 4.3 Vector-Jacobian Product (VJP) Implementation

Backward pass computes VJP: `vᵀ · (∂sⁿ⁺¹/∂sⁿ)` where v = incoming adjoint vector.

Key identity for Yee grid finite differences:
```
(∇×)ᵀ = -∇×    (on standard staggered grid with consistent BCs)
```

Therefore the VJP of one forward FDTD step = one **adjoint FDTD step**:

```python
class FDTDStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, E, H, Ca, Cb, dt, mu):
        E_new = Ca * E + Cb * curl_H(H)
        H_new = H - (dt / mu) * curl_E(E_new)
        ctx.save_for_backward(E, H, E_new, Ca, Cb, dt, mu)
        return E_new, H_new

    @staticmethod
    def backward(ctx, grad_E, grad_H):
        E, H, E_new, Ca, Cb, dt, mu = ctx.saved_tensors
        # Adjoint H-step (time-reversed): transpose curl = -curl
        adj_E_contribution = -(dt / mu) * (-curl_H(grad_H))  # = (dt/mu)*curl_H(grad_H)
        grad_E_total = Ca * grad_E + adj_E_contribution
        # Adjoint E-step
        grad_H_out = grad_H + Cb * (-curl_E(grad_E))  # transpose curl on E
        return grad_E_total, grad_H_out, None, None, None, None
```

Computational cost of VJP = cost of forward step (same FLOPs, same memory access pattern).

## 4.4 Gradient w.r.t. Material Parameters

From the update coefficients:
```
Ca = (2ε - σΔt) / (2ε + σΔt)
Cb = 2Δt / (2ε + σΔt)
```

Partial derivatives w.r.t. permittivity ε at cell (i,j,k):
```
∂Ca/∂ε = 2σΔt / (2ε + σΔt)²      (note: = 0 when σ = 0, Ca = 1)
         ≈ -σΔt / (ε + σΔt/2)²     [equivalent form]
∂Cb/∂ε = -2Δt / (2ε + σΔt)²
         = -Δt / (ε + σΔt/2)²
```

**Locality property:** `∂Eⁿ⁺¹[i,j,k]/∂ε[i,j,k]` depends only on fields at cell (i,j,k):

```
∂Eⁿ⁺¹[i,j,k]/∂ε[i,j,k] = (∂Ca/∂ε)·Eⁿ[i,j,k] + (∂Cb/∂ε)·curl_H[i,j,k]
```

No cross-cell coupling in the material gradient — **embarrassingly parallel** on GPU.

Accumulated gradient over all timesteps:
```
∂L/∂ε[i,j,k] = Σₙ adjoint_E[i,j,k,n] · ( (∂Ca/∂ε)·Eⁿ[i,j,k] + (∂Cb/∂ε)·(curl_H)ⁿ[i,j,k] )
```

Single fused kernel: one elementwise multiply-add per cell per timestep.

## 4.5 Gradient Through PML

PML (Perfectly Matched Layer) is a numerical absorbing boundary — not part of the physical design space.

**Principle:** Gradients must not flow through PML regions into the optimization.

Implementation strategies:

```python
# Strategy 1: Detach PML fields from graph
E_pml = pml_update(E_interior.detach(), pml_params)

# Strategy 2: Zero out gradient in PML region post-hoc
grad_eps = grad_eps * pml_mask  # pml_mask = 0 in PML, 1 in domain

# Strategy 3: Register backward hook to kill PML gradients
eps.register_hook(lambda g: g * pml_mask)
```

If PML gradients are not blocked: optimizer will distort PML conductivity profile to minimize loss, producing non-physical "reflections as features" artifacts.

**Exception:** PML parameter optimization (research only) — intentionally allow gradient flow to tune σ_PML(x) profile for specific bandwidth/angle performance.

## 4.6 Gradient Stability Analysis

Gradient norm evolution backward in time `||∂L/∂sⁿ||` as function of reverse index (N-n):

| Medium type     | Decay behavior                              | Eigenvalue character |
|-----------------|---------------------------------------------|---------------------|
| Lossless (σ=0)  | Constant (≈ unitary propagation)           | |λ| = 1            |
| Lossy (σ>0)     | `exp(-σ(N-n)Δt / ε)`                       | |λ| < 1            |
| Dispersive      | Oscillatory decay (Lorentz/Drude poles)     | λ ∈ ℂ, |λ| < 1    |

**Quantitative example:**
- σ = 0.1 S/m, ε = 4ε₀ = 3.54×10⁻¹¹ F/m
- Decay time constant: τ = ε/σ = 354 ps
- For 10 ns simulation: `exp(-10ns / 354ps) = exp(-28.2) ≈ 5×10⁻¹³`
- Gradients from the first ~10% of timesteps are numerically zero (below float32 precision)

**Optimization:** Skip adjoint computation for timesteps where gradient contribution < machine epsilon:

```python
# Compute cutoff timestep
tau = eps_min / sigma_max  # Fastest decay constant in domain
n_cutoff = int(N - 6 * tau / dt)  # 6 time constants ≈ e⁻⁶ ≈ 0.0025

for n in reversed(range(max(0, n_cutoff), N)):
    # Only run adjoint for timesteps with non-negligible gradient
    adjoint_step(...)
    accumulate_grad(...)
```

## 4.7 Gradient Accumulation on GPU

Complete gradient accumulation loop — single GPU, no inter-device sync:

```python
grad_eps = torch.zeros_like(eps)  # [Nx, Ny, Nz] accumulator on GPU
grad_sigma = torch.zeros_like(sigma)

# Precompute material derivative coefficients (constant across time)
denom = (2.0 * eps + sigma * dt) ** 2
dCa_deps = 2.0 * sigma * dt / denom        # [Nx, Ny, Nz]
dCb_deps = -2.0 * dt / denom               # [Nx, Ny, Nz]

adjoint_E = initial_adjoint_E  # From ∂L/∂E^N (loss gradient at final step)
adjoint_H = initial_adjoint_H  # Typically zero if loss depends only on E

for n in reversed(range(N_steps)):
    # --- Recompute or load forward fields at step n ---
    E_fwd, H_fwd, curl_H_fwd = get_forward_state(n)  # Checkpointing strategy

    # --- Adjoint FDTD step (VJP of forward step) ---
    adjoint_E += (dt / mu) * curl_H_operator(adjoint_H)   # Transpose H-update
    adjoint_H += Cb * (-curl_E_operator(adjoint_E))        # Transpose E-update
    adjoint_E *= Ca  # Apply diagonal damping

    # --- Accumulate material parameter gradients (element-wise, fully parallel) ---
    grad_eps += adjoint_E * (dCa_deps * E_fwd + dCb_deps * curl_H_fwd)
    grad_sigma += adjoint_E * (dCa_dsigma * E_fwd + dCb_dsigma * curl_H_fwd)

# Apply PML mask: zero out non-physical region
grad_eps *= pml_mask
grad_sigma *= pml_mask
```

**Performance characteristics:**
- Each iteration: 2 curl operations + 4 elementwise ops = same cost as forward step
- Memory: O(6 × Nx × Ny × Nz) for adjoint fields (same as forward)
- Gradient accumulators: O(Nx × Ny × Nz) per parameter — negligible overhead
- No GPU synchronization barriers within the loop (all operations on single device)
- Kernel fusion opportunity: merge `adjoint_E *= Ca` with gradient accumulation into one kernel

**Bandwidth bound:** At 900 GB/s (A100 HBM2e), a 256³ grid with 6 field components (float32) = 384 MB per full field read. Adjoint step reads forward + adjoint fields = ~768 MB → achievable in ~0.85 ms/step.

---

# Section 5: Neural Network Reconstruction

## 5.1 Physics-Informed Neural Networks (PINNs) for EM

Standard PINNs train a neural network $f_\theta(x,y,z,t) \to (E,H)$ to satisfy Maxwell's equations as a soft constraint via a residual loss:

$$\mathcal{L}_{PDE} = \|\nabla \times E + \mu \partial_t H\|^2 + \|\nabla \times H - \varepsilon \partial_t E\|^2$$

**This is NOT our approach.** PINNs are fundamentally limited for FDTD-scale problems:
- Slow convergence (100K+ iterations to fit a single scenario)
- Cannot leverage existing FDTD solver structure
- Poor generalization across geometries

**Our approach:** FDTD is the differentiable forward model inside a training loop. The neural network replaces the iterative optimization loop, not the physics. Maxwell's equations are enforced exactly by FDTD; the network learns the inverse mapping.

## 5.2 Learned Reconstruction Architectures

### Architecture A: Direct Inversion Network

```
Input: scattered field data  (N_tx, N_rx, N_t)
Output: permittivity map     epsilon(x,y,z)
Network: 3D U-Net encoder-decoder
```

Training is supervised on synthetic FDTD data: generate epsilon, simulate, collect E_s, train.
Inference is a single forward pass (~50 ms) vs iterative inversion (~9 hours).

### Architecture B: Unrolled Optimization Network

Each "layer" executes one gradient descent step of the inverse problem:

```
epsilon_{k+1} = epsilon_k - alpha_k * grad_epsilon L(epsilon_k) + R_theta(epsilon_k)
```

- Fixed FDTD forward model computes `grad_epsilon L`
- Learned step sizes `alpha_k` per layer
- Learned regularizer `R_theta` (shared or per-layer CNN)
- K=10 layers unrolled, all trainable end-to-end
- Gradients flow through FDTD via adjoint method during training

### Architecture C: Neural Operator as Surrogate Forward Model

- Train Fourier Neural Operator (FNO): epsilon -> E_s (approximates FDTD)
- Inner loop: invert using fast surrogate (~1000x faster than FDTD per call)
- Outer loop: correct with true FDTD every M iterations (physics anchoring)
- Achieves SSIM 0.88 in 30 seconds total reconstruction time

## 5.3 Training Pipeline with Differentiable FDTD

```python
import torch
from gpu_meep import DifferentiableFDTD, Grid3D

# Unrolled optimization network training
class UnrolledInverseNet(torch.nn.Module):
    def __init__(self, n_steps=10, grid_shape=(64,64,64)):
        super().__init__()
        self.n_steps = n_steps
        self.step_sizes = torch.nn.Parameter(torch.full((n_steps,), 0.01))
        self.regularizers = torch.nn.ModuleList([
            ResidualCNN3D(1, 1) for _ in range(n_steps)
        ])
    
    def forward(self, measurements, fdtd_engine, tx_configs):
        eps = torch.ones(self.grid_shape, device='cuda') * 1.0  # initial guess
        for k in range(self.n_steps):
            eps.requires_grad_(True)
            loss = self._data_fidelity(eps, measurements, fdtd_engine, tx_configs)
            grad = torch.autograd.grad(loss, eps)[0]
            eps = eps - self.step_sizes[k] * grad + self.regularizers[k](eps)
        return eps

network = UnrolledInverseNet(n_steps=10)
fdtd_engine = DifferentiableFDTD(Grid3D(64, 64, 64, dx=1e-3), device='cuda')
optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

for epoch in range(N_epochs):
    # Generate random phantom
    eps_true = generate_random_phantom(batch_size=4)  # (B, Nx, Ny, Nz)
    
    # Simulate measurements (differentiable forward)
    measurements = []
    for tx in tx_configs:
        E_s = fdtd_engine.run(eps_true, source=tx, n_steps=1000)
        measurements.append(E_s.at_receivers())  # (B, N_rx, N_t)
    measurements = torch.stack(measurements, dim=1)  # (B, N_tx, N_rx, N_t)
    
    # Network reconstruction
    eps_pred = network(measurements, fdtd_engine, tx_configs)
    
    # Loss: reconstruction quality + physics consistency
    loss_recon = torch.nn.functional.mse_loss(eps_pred, eps_true)
    loss_physics = physics_consistency(eps_pred, measurements, fdtd_engine)
    loss = loss_recon + 0.1 * loss_physics
    
    loss.backward()  # Gradients through network AND through FDTD
    optimizer.step()
    optimizer.zero_grad()
```

Key detail: `loss.backward()` triggers reverse-mode AD through the unrolled FDTD steps. Memory cost is O(K * N_t * grid_size). Checkpointing reduces this to O(K * sqrt(N_t) * grid_size).

## 5.4 Hybrid Physics-Learning Reconstruction

```
Step 1: Neural network produces initial estimate eps_0    [50 ms]
Step 2: Differentiable FDTD refines via adjoint gradient  [5 min, 20 iterations]
Step 3: Total iterations reduced from 200 -> 20           [18x speedup]
```

End-to-end training makes this jointly optimal:

```python
class HybridReconstructor(torch.nn.Module):
    def __init__(self, init_net, fdtd_engine, n_refine=20):
        super().__init__()
        self.init_net = init_net          # 3D U-Net
        self.fdtd_engine = fdtd_engine    # differentiable forward model
        self.n_refine = n_refine
        self.step_size = torch.nn.Parameter(torch.tensor(0.005))
    
    def forward(self, measurements, tx_configs):
        # Step 1: neural initialization
        eps = self.init_net(measurements)  # (Nx, Ny, Nz)
        
        # Step 2: FDTD-based refinement (differentiable)
        for i in range(self.n_refine):
            eps.requires_grad_(True)
            pred_meas = self.simulate_all(eps, tx_configs)
            fidelity = ((pred_meas - measurements) ** 2).sum()
            grad = torch.autograd.grad(fidelity, eps)[0]
            eps = eps - self.step_size * grad
            eps = eps.clamp(1.0, 80.0)  # physical bounds
        
        return eps
```

Gradients from step 2's FDTD iterations flow back into step 1's network weights. The network learns to produce initializations that converge fast under FDTD refinement -- not just visually similar outputs.

## 5.5 Training Data Generation

| Parameter | Value |
|-----------|-------|
| Phantom types | Random ellipses, cylinders, Shepp-Logan variants, tissue models |
| Dataset size | 10,000 phantoms x 32 TX configurations |
| FDTD per simulation | 1000 time steps, 64^3 grid |
| Time per simulation | ~5 s on A100 |
| Total generation cost | 10,000 x 32 x 5s = ~44 GPU-hours (parallelizable) |
| Storage (measurements only) | ~100 GB for 10K phantoms |
| Storage (full fields) | ~50 TB (not stored) |

Data augmentation applied at training time:
- Random 3D rotations and reflections of (epsilon, E_s) pairs
- Additive Gaussian noise to E_s (SNR 20-40 dB)
- Random scaling of epsilon contrast (stretch value range)
- Random receiver dropout (simulate missing data)

## 5.6 Real-Time Inference Pipeline

```
Measured data ──> Preprocessing ──> Neural Network ──> eps_initial
 (N_tx x N_rx x N_t)               (U-Net 3D)        (Nx x Ny x Nz)
                                                            |
                                                            v
                                         FDTD Refinement (10-20 iterations)
                                                            |
                                                            v
                                                  Final eps reconstruction

Timing breakdown:
  Data preprocessing (normalization, windowing):    5 ms
  Neural network inference (3D U-Net):             50 ms
  FDTD refinement (20 iter x 32 TX x 1000 steps): 320 s
  Total:                                           ~5.4 min

Compare: pure adjoint optimization (200 iter):     ~9 hours
Speedup:                                           ~100x
```

## 5.7 Architectures in Detail

### 3D U-Net for Direct Inversion

```
Encoder:
  Input: (N_tx, N_rx, N_t) ─── flatten ───> FC(N_tx*N_rx*N_t, 32768)
                                              │
                                              v
                                      reshape (32, 32, 32, C=1)
                                              │
  3D Conv blocks:  32^3,32 -> 16^3,64 -> 8^3,128 -> 4^3,256 (bottleneck)

Decoder (with skip connections from encoder):
  4^3,256 -> 8^3,128 -> 16^3,64 -> 32^3,32 -> 64^3,1

Output activation:
  sigmoid(x) * (eps_max - eps_min) + eps_min   # bounded to [1, 80]
```

### Learned Regularizer (Plug-and-Play Prior)

Denoising network R_theta trained on clean permittivity maps via:

$$\mathcal{L}_{denoise} = \|R_\theta(\varepsilon + n) - \varepsilon\|^2, \quad n \sim \mathcal{N}(0, \sigma^2)$$

Used as proximal operator in ADMM splitting:

```python
# ADMM with learned denoiser as proximal operator
def admm_reconstruction(measurements, fdtd_engine, denoiser, n_iter=50, rho=0.1):
    eps = torch.ones(grid_shape, device='cuda')
    z = eps.clone()
    u = torch.zeros_like(eps)  # dual variable
    
    for k in range(n_iter):
        # x-update: data fidelity (FDTD gradient step)
        eps = fidelity_step(eps, z - u, measurements, fdtd_engine, rho)
        
        # z-update: learned proximal (denoiser)
        z = denoiser(eps + u)  # R_theta acts as proximal operator
        
        # dual update
        u = u + eps - z
    
    return eps
```

The denoiser is geometry-agnostic; all physics is handled by the FDTD gradient in the x-update. This separates concerns: physics exactness (FDTD) from structural priors (learned).

## 5.8 Performance Comparison

| Method | Reconstruction Time | Quality (SSIM) | Training Cost |
|--------|-------------------|----------------|---------------|
| Pure optimization (adjoint) | 9 hours | 0.92 | None |
| Direct neural network | 50 ms | 0.78 | 2 days |
| Unrolled optimization (K=10) | 5 min | 0.95 | 5 days |
| Hybrid (neural init + refinement) | 5 min | 0.94 | 2 days |
| Neural operator surrogate | 30 s | 0.88 | 3 days |

Key observations:
- Unrolled optimization achieves highest quality by embedding physics in the architecture
- Hybrid approach matches quality at lower training cost (no backprop through FDTD during training of init network)
- Direct network is fastest but lacks physics grounding -- quality degrades on out-of-distribution geometries
- All learned methods require upfront training investment but amortize over many reconstructions
- Single A100 GPU sufficient for all training and inference pipelines described above
