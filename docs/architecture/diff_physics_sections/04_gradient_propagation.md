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
