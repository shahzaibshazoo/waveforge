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
