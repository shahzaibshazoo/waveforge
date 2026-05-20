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
