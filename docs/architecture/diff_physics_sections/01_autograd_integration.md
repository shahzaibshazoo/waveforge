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
