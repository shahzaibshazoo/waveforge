# 3. Module Interactions and Data Flow Graph

## 3.1 Dependency Graph

```
                                    ┌─────────────────┐
                                    │   Application   │
                                    │  (User Script)  │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
          ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
          │  Differentiable  │    │     Imaging      │    │   Visualization  │
          │     Module       │    │     Module       │    │     Module       │
          └────────┬─────────┘    └────────┬─────────┘    └──────────────────┘
                   │                       │
                   │       ┌───────────────┼───────────────┐
                   │       │               │               │
                   ▼       ▼               ▼               ▼
          ┌──────────────────────────────────────────────────────┐
          │              FDTD Engine (Core)                      │
          │  (Orchestrates time-stepping simulation)             │
          └──┬───────┬───────┬────────┬──────────┬──────────┬───┘
             │       │       │        │          │          │
     ┌───────┘   ┌───┘   ┌───┘    ┌───┘      ┌───┘      ┌───┘
     │           │       │        │          │          │
     ▼           ▼       ▼        ▼          ▼          ▼
┌────────┐  ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│  Grid  │  │Materials │  │ Sources │  │Boundaries│ │Detectors │  │  Fields  │
│ Module │  │  Module  │  │ Module  │  │  (PML)   │ │  Module  │  │  Module  │
└────────┘  └──────────┘  └─────────┘  └─────────┘  └──────────┘  └──────────┘
     ▲           ▲            ▲             ▲             ▲             ▲
     └───────────┴────────────┴─────────────┴─────────────┴─────────────┘
                            GPU Tensor Storage
                      (Shared VRAM, zero-copy views)
```

### Dependency Hierarchy (Bottom-Up)

**Layer 0: Foundation**
- `grid`: Computational grid, Yee lattice structure, spatial indexing
- `fields`: E and H field tensor storage with proper staggering

**Layer 1: Physics Configuration**
- `materials`: Permittivity/permeability tensors (depends on grid)
- `sources`: Current density injection masks (depends on grid)
- `boundaries`: PML coefficient tensors (depends on grid)
- `detectors`: Field sampling indices and DFT accumulators (depends on grid)

**Layer 2: Simulation Core**
- `fdtd_engine`: Time-stepping orchestrator (depends on all Layer 0-1)

**Layer 3: Analysis**
- `imaging`: Backprojection, SAR, beamforming (depends on fdtd_engine, detectors)
- `differentiable`: Autograd wrappers, adjoint state (depends on fdtd_engine)

**Layer 4: Interface**
- User applications, optimization loops, neural networks

## 3.2 Simulation Lifecycle

### Phase 1: Construction (CPU, Pre-allocation)

```
1. Grid Instantiation
   │
   ├─→ Allocate GridSpec structure
   │   ├─→ Compute Yee grid dimensions (Nx, Ny, Nz)
   │   ├─→ Compute staggered grid offsets
   │   └─→ Store resolution (dx, dy, dz), time step dt
   │
2. Material Assignment
   │
   ├─→ Create MaterialLibrary
   │   ├─→ Define materials (ε_r, μ_r, σ)
   │   └─→ Assign to grid regions via boolean masks
   │
3. Source Placement
   │
   ├─→ Define Source objects (position, waveform, polarization)
   │   ├─→ Gaussian pulse: f(t) = exp(-((t-t₀)/τ)²)
   │   ├─→ CW sinusoid: f(t) = sin(2πft)
   │   └─→ Custom waveforms
   │
4. Boundary Setup
   │
   ├─→ Configure PML absorbers
   │   ├─→ Set PML thickness (typically 8-20 cells)
   │   ├─→ Compute polynomial grading (σ = σ_max(d/L)^m)
   │   └─→ Generate κ and α tensors
   │
5. Detector Placement
   │
   └─→ Define Detector objects (position, frequency list, mode)
       ├─→ Time-domain samplers
       ├─→ Frequency-domain DFT accumulators
       └─→ Far-field transform surfaces
```

### Phase 2: Initialization (GPU Allocation)

```
1. Allocate GPU Tensors
   │
   ├─→ Fields
   │   ├─→ Ex: [Nx+1, Ny, Nz] float32/float64 on device
   │   ├─→ Ey: [Nx, Ny+1, Nz] float32/float64 on device
   │   ├─→ Ez: [Nx, Ny, Nz+1] float32/float64 on device
   │   ├─→ Hx: [Nx, Ny+1, Nz+1] float32/float64 on device
   │   ├─→ Hy: [Nx+1, Ny, Nz+1] float32/float64 on device
   │   └─→ Hz: [Nx+1, Ny+1, Nz] float32/float64 on device
   │
   ├─→ Material Tensors
   │   ├─→ ε_r: [Nx, Ny, Nz, 3] (anisotropic support)
   │   ├─→ μ_r: [Nx, Ny, Nz, 3]
   │   └─→ σ_e: [Nx, Ny, Nz, 3] (conductivity)
   │
   ├─→ PML Tensors
   │   ├─→ σ_pml_x: [Nx, Ny, Nz]
   │   ├─→ κ_pml_x, α_pml_x: [Nx, Ny, Nz]
   │   └─→ (Repeat for y, z directions)
   │
   └─→ Auxiliary PML Fields (split-field formulation)
       ├─→ Ψ_Ex_y, Ψ_Ex_z: [Nx+1, Ny, Nz]
       ├─→ (18 auxiliary tensors total for full 3D PML)
       └─→ Zero-initialized on device
│
2. Compile CUDA Kernels (PyTorch JIT or Triton)
   │
   ├─→ curl_h_kernel: Computes ∇×H for E-field update
   ├─→ curl_e_kernel: Computes ∇×E for H-field update
   ├─→ pml_update_e_kernel: Updates E-fields in PML regions
   ├─→ pml_update_h_kernel: Updates H-fields in PML regions
   └─→ source_inject_kernel: Adds source currents
│
3. Compute PML Coefficients
   │
   ├─→ For each PML region:
   │   ├─→ d = distance from boundary (in cells)
   │   ├─→ σ(d) = σ_max * (d/L)^3
   │   ├─→ κ(d) = 1 + (κ_max - 1) * (d/L)^3
   │   └─→ α(d) = α_max * ((L-d)/L)^2
   │
   └─→ Upload to GPU tensors
│
4. Initialize Detectors
   │
   ├─→ DFT detectors: Create complex accumulators e^(-jωt)
   ├─→ Time-domain detectors: Allocate circular buffers
   └─→ Flux detectors: Compute Poynting vector surfaces
│
5. Validate CFL Condition
   │
   └─→ dt ≤ 1/(c * sqrt(1/dx² + 1/dy² + 1/dz²))
       └─→ Abort if violated (numerical instability)
```

### Phase 3: Time-Stepping Loop (Critical Path)

```
for t in range(0, num_timesteps):
    │
    ├─→ [CUDA Stream 0] Source Injection
    │   │
    │   ├─→ Evaluate waveform at time t
    │   ├─→ Soft source: E_new = E_old + dt*J(t)/ε₀
    │   └─→ Or hard source: E_new = E_source(t)
    │
    ├─→ [CUDA Stream 0] H-Field Update (entire domain)
    │   │
    │   ├─→ Compute curl(E) using finite differences
    │   │   │
    │   │   ├─→ (∇×E)_x = (∂Ez/∂y - ∂Ey/∂z)
    │   │   ├─→ (∇×E)_y = (∂Ex/∂z - ∂Ez/∂x)
    │   │   └─→ (∇×E)_z = (∂Ey/∂x - ∂Ex/∂y)
    │   │
    │   └─→ Update H: H^(n+1/2) = H^(n-1/2) + (dt/μ₀μ_r) * ∇×E^n
    │
    ├─→ [CUDA Stream 0] PML H-Field Update (boundary regions)
    │   │
    │   ├─→ Update auxiliary variables Ψ_Hx, Ψ_Hy, Ψ_Hz
    │   └─→ Apply PML correction to H-fields
    │
    ├─→ [CUDA Stream 1] Detector Recording (H-fields if needed)
    │   │
    │   └─→ Sample H at detector locations (sparse gather)
    │
    ├─→ [CUDA Stream 0] E-Field Update (entire domain)
    │   │
    │   ├─→ Compute curl(H) using finite differences
    │   │
    │   └─→ Update E: E^(n+1) = E^n + (dt/ε₀ε_r) * ∇×H^(n+1/2) - (σ/ε)E^n
    │
    ├─→ [CUDA Stream 0] PML E-Field Update (boundary regions)
    │   │
    │   ├─→ Update auxiliary variables Ψ_Ex, Ψ_Ey, Ψ_Ez
    │   └─→ Apply PML correction to E-fields
    │
    ├─→ [CUDA Stream 1] Detector Recording (E-fields)
    │   │
    │   ├─→ Time-domain: Store E, H samples
    │   ├─→ Frequency-domain: Accumulate DFT
    │   │   └─→ X[ω] += E(t) * exp(-jωt) * dt
    │   └─→ Flux: Accumulate S = E × H
    │
    └─→ [CUDA Stream 2] Visualization Update (if enabled, every N steps)
        │
        └─→ Transfer slice to CPU for rendering
```

**Inner Loop Timing (Typical 512³ Grid, A100 GPU)**
- Curl computation: ~0.5 ms (memory bandwidth bound)
- Field update: ~0.3 ms (arithmetic + memory)
- PML update: ~0.1 ms (only 10-20% of domain)
- Detector sampling: <0.05 ms (sparse operations)
- **Total per timestep: ~1 ms → 1000 timesteps/sec**

### Phase 4: Post-Processing (CPU/GPU Hybrid)

```
1. Finalize DFT Detectors
   │
   ├─→ Normalize by number of timesteps
   ├─→ Extract Fourier magnitudes |E(ω)|
   └─→ Compute Poynting flux S(ω) = (1/2) Re[E × H*]
│
2. Field Extraction
   │
   ├─→ Transfer full 3D fields to CPU (if needed)
   ├─→ Or keep on GPU for imaging pipeline
   └─→ Save to HDF5 or Zarr (chunked storage)
│
3. Imaging Reconstruction (if imaging module used)
   │
   ├─→ SAR (Synthetic Aperture Radar)
   │   ├─→ Backprojection: I(r) = Σ_tx Σ_rx E_rx * exp(jk(R_tx + R_rx))
   │   └─→ Range migration (Stolt interpolation)
   │
   ├─→ MIMO Beamforming
   │   ├─→ Delay-and-sum: I(r) = Σ_i w_i * s_i(t - τ_i(r))
   │   └─→ Minimum variance (Capon)
   │
   └─→ Time Reversal
       ├─→ Record scattered fields at Rx array
       ├─→ Time-reverse and re-inject as sources
       └─→ Focus energy at scatterer location
│
4. Visualization
   │
   ├─→ 2D slices (E_z at z=0)
   ├─→ 3D isosurfaces (|E| > threshold)
   ├─→ Animations (timestep sequence)
   └─→ Far-field radiation patterns
```

## 3.3 Inter-Module Communication Protocol

### Communication Paradigm: GPU-Native Tensor Passing

All modules communicate via **PyTorch CUDA tensors**. No CPU roundtrips in hot path.

```python
# Example: Fields module exposes tensors directly
class Fields:
    def __init__(self, grid, device):
        self.Ex = torch.zeros((grid.Nx+1, grid.Ny, grid.Nz), 
                              dtype=torch.float32, device=device)
        self.Ey = torch.zeros((grid.Nx, grid.Ny+1, grid.Nz), 
                              dtype=torch.float32, device=device)
        # ... (Ez, Hx, Hy, Hz)
    
    def get_field_views(self):
        """Return dict of tensor references (zero-copy)"""
        return {'Ex': self.Ex, 'Ey': self.Ey, ...}

# FDTD engine receives tensor views
class FDTDEngine:
    def __init__(self, fields, materials, sources, boundaries):
        self.fields = fields.get_field_views()  # dict of tensors
        self.eps_r = materials.get_permittivity()  # tensor
        self.sigma_pml = boundaries.get_pml_coefficients()  # tensor
        # ...
    
    def step(self, t):
        """Single timestep update (all on GPU)"""
        # 1. Source injection
        self.sources.inject(self.fields, t)  # modifies fields in-place
        
        # 2. H-field update
        curl_e = self._compute_curl_e(self.fields['Ex'], 
                                      self.fields['Ey'], 
                                      self.fields['Ez'])
        self.fields['Hx'] += self.dt_over_mu * curl_e[..., 0]
        self.fields['Hy'] += self.dt_over_mu * curl_e[..., 1]
        self.fields['Hz'] += self.dt_over_mu * curl_e[..., 2]
        
        # 3. Boundary apply
        self.boundaries.apply_h(self.fields)  # modifies H in-place
        
        # 4. E-field update (similar)
        # ...
        
        # 5. Detector record
        self.detectors.record(self.fields, t)  # reads fields, writes to buffers
```

### Module Interface Contract

Every physics module implements the following lifecycle methods:

```python
class ModuleInterface(ABC):
    @abstractmethod
    def configure(self, **params) -> None:
        """
        CPU-side configuration.
        Define geometry, parameters, metadata.
        No GPU allocation yet.
        """
        pass
    
    @abstractmethod
    def validate(self) -> List[str]:
        """
        Check for configuration errors.
        Return list of error messages (empty if valid).
        Examples:
          - Source outside grid
          - CFL violation
          - Material property unphysical
        """
        pass
    
    @abstractmethod
    def allocate(self, device: torch.device) -> None:
        """
        Allocate GPU tensors.
        Initialize to zero or precomputed values.
        Upload from CPU if needed.
        """
        pass
    
    @abstractmethod
    def step(self, fields: Dict[str, Tensor], t: int) -> None:
        """
        Time-stepping update (called every timestep).
        Modifies fields in-place.
        Must be GPU-kernel efficient (fused operations preferred).
        """
        pass
    
    @abstractmethod
    def extract(self) -> Dict[str, Any]:
        """
        Post-simulation data extraction.
        Transfer results to CPU if needed.
        Return dict of results (numpy arrays, scalars, etc.)
        """
        pass
```

**Example: PML Boundary Module**

```python
class PMLBoundary(ModuleInterface):
    def configure(self, thickness=10, sigma_max=None, kappa_max=15, alpha_max=0.05):
        self.thickness = thickness
        self.sigma_max = sigma_max or 0.8 * (self.m_order + 1) / (Z0 * self.grid.dx)
        # ... store params
    
    def validate(self):
        errors = []
        if self.thickness < 8:
            errors.append("PML thickness < 8 may cause reflections")
        if self.sigma_max < 0:
            errors.append("PML sigma_max must be positive")
        return errors
    
    def allocate(self, device):
        # Compute PML coefficient tensors
        self.sigma_x = self._compute_pml_profile(self.grid.Nx, device)
        self.kappa_x = self._compute_pml_profile(self.grid.Nx, device)
        # ... (y, z directions)
        
        # Allocate auxiliary fields (split-field PML)
        self.Psi_Ex_y = torch.zeros(...)
        self.Psi_Ex_z = torch.zeros(...)
        # ... (18 auxiliary tensors total)
    
    def step(self, fields, t):
        # Called twice per timestep: once for H, once for E
        # Update auxiliary variables in PML regions
        self._update_pml_h(fields)  # CUDA kernel
        self._update_pml_e(fields)  # CUDA kernel
    
    def extract(self):
        # PML doesn't produce output, return diagnostics
        return {'reflection_coefficient': self._measure_reflection()}
```

### Event System for Sparse Operations

Not all operations occur every timestep. Use event scheduling:

```python
class EventScheduler:
    def __init__(self):
        self.events = defaultdict(list)
    
    def register(self, interval: int, callback: Callable):
        """
        Register callback to run every `interval` timesteps.
        Examples:
          - Visualization update: every 50 timesteps
          - Checkpoint save: every 1000 timesteps
          - Adaptive mesh refinement check: every 100 timesteps
        """
        self.events[interval].append(callback)
    
    def trigger(self, t: int, fields: Dict):
        for interval, callbacks in self.events.items():
            if t % interval == 0:
                for cb in callbacks:
                    cb(fields, t)

# Usage in FDTD loop
scheduler = EventScheduler()
scheduler.register(50, lambda f, t: visualizer.update(f))
scheduler.register(1000, lambda f, t: checkpoint.save(f, t))

for t in range(num_timesteps):
    engine.step(t)
    scheduler.trigger(t, engine.fields)
```

## 3.4 Critical Path Analysis

### Computational Bottlenecks (Ranked by Time)

For a typical 512³ grid with single precision on A100 GPU:

| Operation              | Time/Step | % Total | Bandwidth (GB/s) | Overlappable? |
|------------------------|-----------|---------|------------------|---------------|
| curl(E) computation    | 0.5 ms    | 50%     | 2400             | No            |
| E-field update         | 0.3 ms    | 30%     | 2000             | No            |
| curl(H) computation    | 0.5 ms    | 50%     | 2400             | No            |
| H-field update         | 0.3 ms    | 30%     | 2000             | No            |
| PML update             | 0.1 ms    | 10%     | 500              | Yes (stream)  |
| Source injection       | 0.02 ms   | 2%      | -                | Yes (stream)  |
| Detector sampling      | 0.03 ms   | 3%      | -                | Yes (stream)  |
| **Total (sequential)** | **1.0 ms**| **100%**| -                | -             |

**Memory Bandwidth Analysis:**
- A100 theoretical: 1935 GB/s (HBM2e)
- Achieved: ~2400 GB/s effective (80% efficiency with kernel fusion)
- Bottleneck: Memory-bound (not compute-bound)
  - FLOPs: ~5 GFLOP/timestep
  - Memory transfers: ~2.4 GB/timestep (reading 6 fields + writing 6 fields)

### CUDA Stream Assignment for Overlap

```
Stream 0 (Main Physics): Critical path, no overlap
├─→ curl(E)
├─→ H-field update
├─→ curl(H)
└─→ E-field update

Stream 1 (PML/Boundaries): Overlaps with main if using separate regions
├─→ PML H-update (can start after main H-update kernel launches)
└─→ PML E-update (can start after main E-update kernel launches)

Stream 2 (Detectors): Overlaps with field updates (read-only access)
├─→ DFT accumulation
└─→ Time-domain sampling

Stream 3 (Visualization): Asynchronous transfer to CPU
└─→ memcpy_device_to_host (every 50 timesteps)
```

### Timeline Diagram (One Timestep, with Stream Parallelism)

```
Time (μs)  Stream 0           Stream 1        Stream 2         Stream 3
────────── ────────────────── ─────────────── ──────────────── ──────────
0          [Source inject]
           ├─ 20μs
20         [curl(E)]
           ├─ 500μs                           [Record H-det]
           │                                  ├─ 10μs
520        [H-update]                         │
           ├─ 300μs           [PML-H update]  │
           │                  ├─ 50μs         │
820        [curl(H)]          │               └─ done
           ├─ 500μs           └─ done
           │
1320       [E-update]                         [Record E-det]  [Viz copy]
           ├─ 300μs           [PML-E update]  ├─ 10μs         ├─ 100μs
           │                  ├─ 50μs         │               │
1620       └─ done            └─ done         └─ done         └─ done (async)
────────────────────────────────────────────────────────────────────────
Total wall time: 1620 μs (1.62 ms)
Sequential time: 1.0 ms (main path) + 0.1 ms (PML) + 0.02 ms (detectors) = 1.12 ms
Speedup from overlap: 1.12 / 1.62 = negligible (main path dominates)
```

**Key Insight:** FDTD is fundamentally sequential due to data dependencies (E^n → H^(n+1/2) → E^(n+1)). Overlap opportunities are limited to auxiliary operations. Main optimization is **kernel fusion** and **memory bandwidth**.

### Kernel Fusion Opportunities

1. **Fused curl + field update:**
   ```cuda
   // Instead of:
   //   curl_e = curl(E)    // writes to temp buffer
   //   H += dt/mu * curl_e // reads from temp buffer
   // Do:
   __global__ void fused_h_update(E, H, dt_over_mu) {
       // Compute curl(E) on-the-fly, immediately update H
       // Saves one full field read/write (2.4 GB)
   }
   ```
   **Benefit:** 30% speedup by eliminating intermediate buffer

2. **Fused material update:**
   ```cuda
   __global__ void fused_e_update(E, H, eps_r, sigma) {
       curl_h = compute_curl(H);  // on registers
       E_new = E + dt/eps_r * curl_h - sigma/eps_r * E;  // fused
   }
   ```

3. **Fused PML + main update:**
   - Challenging due to divergent control flow (PML only in boundary)
   - Requires careful thread masking or separate kernels

## 3.5 API Contract Pattern Example

### Complete Module Example: Time-Harmonic Source

```python
class HarmonicSource(ModuleInterface):
    """
    Sinusoidal current source: J(t) = J0 * sin(2πft) * δ(r - r0)
    """
    def __init__(self, grid):
        self.grid = grid
        self.configured = False
        self.allocated = False
    
    # ────────────────────────────────────────────────────────
    # 1. CONFIGURE (CPU, before simulation)
    # ────────────────────────────────────────────────────────
    def configure(self, position, frequency, amplitude, polarization='Ez'):
        """
        position: (x, y, z) in meters
        frequency: Hz
        amplitude: A/m² (current density)
        polarization: 'Ex', 'Ey', or 'Ez'
        """
        self.position = position
        self.frequency = frequency
        self.amplitude = amplitude
        self.polarization = polarization
        
        # Convert position to grid indices
        self.idx = self.grid.position_to_index(position)
        
        self.configured = True
    
    # ────────────────────────────────────────────────────────
    # 2. VALIDATE (CPU, before GPU allocation)
    # ────────────────────────────────────────────────────────
    def validate(self) -> List[str]:
        errors = []
        
        if not self.configured:
            errors.append("Source not configured")
            return errors
        
        # Check if source is inside grid
        if not self.grid.is_inside(self.idx):
            errors.append(f"Source at {self.position} is outside grid")
        
        # Check Nyquist criterion
        wavelength = c / self.frequency
        ppw = wavelength / self.grid.dx  # points per wavelength
        if ppw < 10:
            errors.append(f"Resolution too coarse: {ppw:.1f} points/wavelength (need ≥10)")
        
        # Check polarization
        if self.polarization not in ['Ex', 'Ey', 'Ez']:
            errors.append(f"Invalid polarization: {self.polarization}")
        
        return errors
    
    # ────────────────────────────────────────────────────────
    # 3. ALLOCATE (GPU, one-time before time-stepping)
    # ────────────────────────────────────────────────────────
    def allocate(self, device):
        # Precompute angular frequency
        self.omega = 2 * np.pi * self.frequency
        
        # Create injection mask (sparse tensor or index)
        self.inject_idx = torch.tensor(self.idx, device=device)
        self.amplitude_tensor = torch.tensor(self.amplitude, 
                                            dtype=torch.float32, 
                                            device=device)
        
        self.device = device
        self.allocated = True
    
    # ────────────────────────────────────────────────────────
    # 4. STEP (GPU, called every timestep - HOT PATH)
    # ────────────────────────────────────────────────────────
    def step(self, fields, t):
        """
        Soft source: E_new = E_old + dt * J(t) / ε₀
        """
        # Compute waveform value at current time
        time_sec = t * self.grid.dt
        waveform_value = self.amplitude_tensor * torch.sin(self.omega * time_sec)
        
        # Inject into appropriate field component
        # (Soft source: add to existing field)
        ix, iy, iz = self.inject_idx
        
        if self.polarization == 'Ez':
            fields['Ez'][ix, iy, iz] += waveform_value * self.grid.dt / eps0
        elif self.polarization == 'Ex':
            fields['Ex'][ix, iy, iz] += waveform_value * self.grid.dt / eps0
        elif self.polarization == 'Ey':
            fields['Ey'][ix, iy, iz] += waveform_value * self.grid.dt / eps0
    
    # ────────────────────────────────────────────────────────
    # 5. EXTRACT (CPU/GPU, post-simulation)
    # ────────────────────────────────────────────────────────
    def extract(self):
        """
        Sources don't produce output, return metadata for reproducibility.
        """
        return {
            'type': 'harmonic',
            'position': self.position,
            'frequency': self.frequency,
            'amplitude': self.amplitude,
            'polarization': self.polarization
        }
```

### Complete Simulation Example Using Contract Pattern

```python
# ─────────────────────────────────────────────────────────────
# Phase 1: CONFIGURE (all modules)
# ─────────────────────────────────────────────────────────────
grid = Grid()
grid.configure(size=(10e-3, 10e-3, 10e-3), resolution=50e-6)

source = HarmonicSource(grid)
source.configure(position=(5e-3, 5e-3, 5e-3), frequency=10e9, amplitude=1.0)

detector = FrequencyDetector(grid)
detector.configure(position=(8e-3, 5e-3, 5e-3), frequencies=[10e9])

material = Material()
material.configure(region=Box(center=(7e-3, 5e-3, 5e-3), size=(1e-3, 1e-3, 1e-3)),
                   epsilon_r=4.0)

boundary = PMLBoundary(grid)
boundary.configure(thickness=10)

# ─────────────────────────────────────────────────────────────
# Phase 2: VALIDATE (check for errors)
# ─────────────────────────────────────────────────────────────
modules = [grid, source, detector, material, boundary]
for module in modules:
    errors = module.validate()
    if errors:
        raise ValueError(f"{module.__class__.__name__} validation failed:\n" +
                        "\n".join(errors))

# ─────────────────────────────────────────────────────────────
# Phase 3: ALLOCATE (GPU tensors)
# ─────────────────────────────────────────────────────────────
device = torch.device('cuda:0')
for module in modules:
    module.allocate(device)

# ─────────────────────────────────────────────────────────────
# Phase 4: TIME-STEPPING (critical path)
# ─────────────────────────────────────────────────────────────
fields = grid.get_field_views()
num_steps = 1000

for t in range(num_steps):
    source.step(fields, t)
    grid.update_h(fields, t)
    boundary.step(fields, t)
    grid.update_e(fields, t)
    boundary.step(fields, t)
    detector.step(fields, t)

# ─────────────────────────────────────────────────────────────
# Phase 5: EXTRACT (post-processing)
# ─────────────────────────────────────────────────────────────
results = detector.extract()  # Returns {'E_field': complex array, ...}
```

---

**Design Rationale:**

1. **Separation of concerns:** Configuration (CPU) separate from execution (GPU)
2. **Early validation:** Catch errors before expensive GPU allocation
3. **Zero-copy communication:** All modules share tensor references
4. **Testability:** Each phase can be unit-tested independently
5. **Composability:** New modules follow same contract, plug-and-play
6. **Performance:** Hot path (`step()`) is pure GPU, no CPU sync

This pattern scales to multi-GPU (allocate on different devices) and batched simulations (add batch dimension to tensors).
