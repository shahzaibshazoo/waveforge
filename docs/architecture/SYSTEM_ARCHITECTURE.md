# GPU-MEEP: System Architecture

> GPU-Native Differentiable Electromagnetic Simulation Engine
> Architecture Specification v0.1

---

# Section 1: Framework Overview and Design Principles

## 1.1 Executive Summary

**GPU-MEEP** is a GPU-native differentiable electromagnetic simulation engine implementing the Finite-Difference Time-Domain (FDTD) method. Built on PyTorch's CUDA backend, it targets MIMO/SAR microwave imaging, inverse scattering, through-wall radar, and differentiable physics optimization. The architecture eliminates CPU bottlenecks by keeping all field data and computation GPU-resident across the entire simulation lifecycle. Unlike Meep (CPU-first with optional GPU offload), GPU-MEEP is designed ground-up for tensor-parallel execution on modern NVIDIA hardware.

## 1.2 Design Principles

| # | Principle | Rationale |
|---|-----------|-----------|
| 1 | **GPU-First Execution** | All field updates, source injection, boundary application, and detection execute as GPU kernels. CPU exists only for orchestration. |
| 2 | **Zero-Copy Tensor Pipeline** | Field tensors are allocated once on GPU at initialization. No host↔device transfers during time-stepping. |
| 3 | **Differentiable by Default** | Every material parameter and geometry input supports `requires_grad=True`. Forward simulation produces a computational graph for backpropagation. |
| 4 | **Modular Kernel Architecture** | Physics operations (curl, PML, source) are independent composable kernels. Swap PML implementations without touching the time-stepper. |
| 5 | **Multi-GPU Native** | Domain decomposition and halo exchange are first-class primitives, not bolted-on extensions. Single-GPU is a special case of multi-GPU. |
| 6 | **Research-Friendly API** | High-level Python API hides CUDA complexity. `sim.run(until=T)` triggers the full GPU pipeline. Subclassing for custom physics is straightforward. |
| 7 | **Production-Grade Reliability** | Deterministic results across runs. IEEE-754 compliant. Checkpoint/restart for long simulations. Graceful OOM handling. |
| 8 | **AI-Ready Architecture** | Field tensors are native PyTorch tensors. Plug directly into neural networks for learned reconstruction, physics-informed neural operators, or hybrid EM-AI pipelines. |
| 9 | **Memory-Bandwidth Aware** | FDTD is memory-bound (~0.2 FLOP/byte). All optimizations target memory throughput: coalesced access, cache residency, mixed precision. |
| 10 | **Minimal Abstraction Overhead** | No runtime polymorphism in hot paths. Kernel dispatch is resolved at simulation init. Template-style specialization via PyTorch's JIT/Triton. |

## 1.3 Architecture Tier Model

```
┌─────────────────────────────────────────────────────────────┐
│  TIER 4: APPLICATION LAYER                                   │
│  Imaging pipelines, optimization loops, AI integration       │
├─────────────────────────────────────────────────────────────┤
│  TIER 3: PHYSICS SIMULATION LAYER                            │
│  FDTD engine, materials, sources, boundaries, detectors      │
├─────────────────────────────────────────────────────────────┤
│  TIER 2: COMPUTE ENGINE LAYER                                │
│  Kernel dispatch, stream management, memory pools, autograd  │
├─────────────────────────────────────────────────────────────┤
│  TIER 1: HARDWARE ABSTRACTION LAYER                          │
│  PyTorch CUDA backend, NCCL, cuFFT, cuBLAS, Triton JIT      │
└─────────────────────────────────────────────────────────────┘
```

**Tier 1 — Hardware Abstraction:** Wraps CUDA runtime, GPU memory allocators, inter-GPU communication (NCCL), and vendor libraries. Provides device-agnostic tensor operations. Future: ROCm/Vulkan backends plug in here.

**Tier 2 — Compute Engine:** Manages kernel launch configurations, CUDA stream scheduling, memory pool lifecycle, and PyTorch autograd function registration. Owns the execution timeline.

**Tier 3 — Physics Simulation:** Implements Maxwell's equations discretized on the Yee grid. Each module (materials, sources, boundaries) provides a `step()` method that enqueues GPU work. The FDTD engine orchestrates the leapfrog update sequence.

**Tier 4 — Application:** Domain-specific pipelines (MIMO imaging, SAR reconstruction, topology optimization) compose Tier 3 primitives. Runs multiple simulations, aggregates data, feeds into reconstruction algorithms or neural networks.

## 1.4 System Requirements

### Hardware

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| GPU | NVIDIA SM 7.0+ (V100) | A100 40GB | H100 80GB |
| VRAM | 8 GB | 40 GB | 80 GB |
| GPU Count | 1 | 4 (NVLink) | 8 (NVSwitch) |
| Host RAM | 32 GB | 128 GB | 512 GB |
| Storage | SSD 500 GB | NVMe 2 TB | NVMe RAID 8 TB |

### Software

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | ≥ 3.10 | Runtime |
| PyTorch | ≥ 2.0 | Tensor ops, autograd, CUDA backend |
| CUDA Toolkit | ≥ 11.8 | Kernel compilation, cuFFT, cuBLAS |
| CuPy | ≥ 12.0 | Supplementary GPU ops (optional) |
| NCCL | ≥ 2.18 | Multi-GPU communication |
| Triton | ≥ 2.1 | JIT kernel compilation |
| NumPy | ≥ 1.24 | Host-side utilities |
| HDF5/h5py | ≥ 3.8 | Checkpoint I/O |

## 1.5 Performance Targets

Targets on NVIDIA A100 80GB (FP32, single GPU, 3D FDTD with PML):

| Grid Size | Cells | Memory | Throughput (Mcells/s) | Timestep Rate |
|-----------|-------|--------|----------------------|---------------|
| 128³ | 2.1M | 0.3 GB | 4,200 | 2,000 steps/s |
| 256³ | 16.8M | 2.1 GB | 3,800 | 226 steps/s |
| 512³ | 134M | 16.5 GB | 3,200 | 24 steps/s |
| 768³ | 453M | 55 GB | 2,800 | 6.2 steps/s |
| 1024³ | 1.07B | OOM* | — | — |

*1024³ requires multi-GPU (2× A100) or mixed precision (FP16 fits in 55 GB).

**Mixed Precision (BF16) Targets:**

| Grid Size | Memory | Throughput (Mcells/s) | Speedup vs FP32 |
|-----------|--------|----------------------|-----------------|
| 512³ | 8.3 GB | 5,400 | 1.69× |
| 768³ | 28 GB | 4,700 | 1.68× |
| 1024³ | 65 GB | 4,100 | — (fits single GPU) |

**Comparison Target:** Meep on 32-core CPU achieves ~50-100 Mcells/s for equivalent problems. GPU-MEEP targets **30-80× speedup** over Meep for production grid sizes.

---

# Section 2: Core Module Architecture

## 2.1 Module Overview

GPU-MEEP consists of 10 core modules organized in a strict dependency DAG. Each module manages its own GPU tensors and exposes a uniform lifecycle interface: `configure() → validate() → allocate() → step() → extract()`.

---

## 2.2 Module: `grid`

**Purpose:** Constructs and manages the Yee lattice discretization. Owns spatial metadata and coordinate transformations.

**Key Classes:**
- `YeeGrid` — Primary grid object. Stores resolution (dx, dy, dz), dimensions (Nx, Ny, Nz), cell coordinates, and staggering offsets.
- `GridRegion` — Named subregion for PML zones, source planes, detector surfaces.
- `SubpixelAverager` — Computes effective permittivity at material interfaces.

**Tensor Shapes:**
- Coordinate arrays: `(Nx,)`, `(Ny,)`, `(Nz,)` — 1D position vectors
- Cell volume tensor: `(Nx, Ny, Nz)` — for non-uniform grids
- Staggering offset table: `(6, 3)` — half-cell shifts for each field component

**Dependencies:** None (foundational module)

---

## 2.3 Module: `fdtd_engine`

**Purpose:** Orchestrates the leapfrog time-stepping loop. Dispatches E-field and H-field update kernels, enforces CFL stability, manages simulation clock.

**Key Classes:**
- `FDTDEngine` — Main simulation driver. Owns the time loop and kernel dispatch sequence.
- `LeapfrogStepper` — Implements the staggered-time update: H at t+½Δt, E at t+Δt.
- `CurlOperator` — Finite-difference curl via shifted tensor indexing (roll operations or custom kernels).

**Tensor Shapes:**
- Field components: `(Nx, Ny, Nz)` × 6 tensors (Ex, Ey, Ez, Hx, Hy, Hz)
- Update coefficients: `(Nx, Ny, Nz)` × 6 (pre-computed Ca, Cb per component)

**Dependencies:** `grid`, `materials`, `sources`, `boundaries`, `detectors`

---

## 2.4 Module: `materials`

**Purpose:** Defines electromagnetic material properties as GPU-resident tensors. Supports isotropic, anisotropic, and dispersive media with differentiable parameters.

**Key Classes:**
- `MaterialMap` — Maps grid cells to material properties. Stores epsilon/mu/sigma as 3D tensors.
- `DispersiveModel` — Base class for Debye, Drude, Lorentz models with auxiliary differential equation (ADE) state.
- `MaterialLibrary` — Predefined materials (air, FR4, skin tissue, concrete, water).

**Tensor Shapes:**
- Permittivity: `(Nx, Ny, Nz)` or `(Nx, Ny, Nz, 3, 3)` for full anisotropic tensor
- Conductivity: `(Nx, Ny, Nz)` per axis
- Dispersive ADE state: `(N_poles, Nx, Ny, Nz)` per dispersive region
- Material index: `(Nx, Ny, Nz)` int16 — lookup into property table

**Dependencies:** `grid`

---

## 2.5 Module: `sources`

**Purpose:** Generates and injects electromagnetic excitation into the simulation domain. All waveforms pre-computed or generated on GPU.

**Key Classes:**
- `PointSource` — Injects at single cell. Additive or hard source.
- `PlaneWaveSource` — TFSF (Total-Field/Scattered-Field) implementation via surface current injection.
- `GaussianPulse` — Waveform: `exp(-((t-t0)/τ)²) * sin(2πf₀t)`
- `WaveformBuffer` — Pre-computed time series stored as GPU tensor.

**Tensor Shapes:**
- Waveform: `(N_timesteps,)` — temporal profile
- Injection mask: `(Nx, Ny, Nz)` sparse or `(surface_cells,)` indexed
- TFSF correction fields: `(perimeter_cells, 2)` — incident E and H at boundary

**Dependencies:** `grid`

---

## 2.6 Module: `boundaries`

**Purpose:** Implements absorbing (PML), reflective (PEC/PMC), and periodic boundary conditions as tensor operations applied after field updates.

**Key Classes:**
- `CPML` — Convolutional PML with stretched-coordinate formulation. Maintains auxiliary psi tensors.
- `PeriodicBC` — Wraps field edges via circular indexing.
- `BlochBC` — Periodic with phase shift for oblique incidence.
- `PEC` / `PMC` — Zero tangential E / zero tangential H at boundary faces.

**Tensor Shapes:**
- PML psi fields: `(6, PML_depth, surface_Ny, surface_Nz)` per face (6 faces × 6 components)
- PML coefficients (b, c): `(PML_depth,)` — 1D grading profiles
- PML kappa/sigma/alpha: `(PML_depth,)` — stretched coordinate parameters
- Boundary masks: `(Nx, Ny, Nz)` bool — identifies boundary cells

**Dependencies:** `grid`

---

## 2.7 Module: `detectors`

**Purpose:** Records electromagnetic field data during simulation for post-processing. Supports time-domain probes and frequency-domain (DFT) monitors.

**Key Classes:**
- `FieldProbe` — Records field at point/line/plane at every timestep or subsampled.
- `FluxMonitor` — Computes Poynting flux through a surface (DFT-based).
- `DFTMonitor` — Running discrete Fourier transform at specified frequencies. Accumulates on GPU.
- `NearFieldSurface` — Records tangential fields on closed surface for near-to-far transform.

**Tensor Shapes:**
- Time-domain buffer: `(N_probes, N_timesteps)` or `(surface_cells, N_timesteps)`
- DFT accumulator: `(N_freqs, surface_cells)` complex64
- Flux result: `(N_freqs,)` — integrated Poynting vector

**Dependencies:** `grid`, `fdtd_engine` (subscribes to step events)

---

## 2.8 Module: `differentiable`

**Purpose:** Wraps the FDTD forward pass in PyTorch's autograd framework. Implements adjoint-state method for memory-efficient gradient computation through long time sequences.

**Key Classes:**
- `DifferentiableFDTD` — `torch.autograd.Function` subclass. Forward: run N steps, save checkpoints. Backward: adjoint time-reversal.
- `AdjointSolver` — Runs time-reversed simulation with adjoint sources from loss gradient.
- `CheckpointSchedule` — Binomial checkpointing (Griewank) to trade compute for memory in backward pass.
- `GradientAccumulator` — Aggregates parameter gradients across timesteps.

**Tensor Shapes:**
- Checkpoint storage: `(N_checkpoints, 6, Nx, Ny, Nz)` — field snapshots at selected times
- Adjoint fields: same as forward fields `(Nx, Ny, Nz)` × 6
- Parameter gradients: same shape as material tensors

**Dependencies:** `fdtd_engine`, `materials`, `boundaries`

---

## 2.9 Module: `imaging`

**Purpose:** Implements MIMO radar imaging pipelines: multi-TX/RX orchestration, signal processing, and reconstruction algorithms on GPU.

**Key Classes:**
- `MIMOArray` — Defines transmitter/receiver positions, waveforms, and sequencing.
- `BackprojectionReconstructor` — Delay-and-sum imaging on GPU. Vectorized over all TX-RX pairs.
- `SARProcessor` — Synthetic aperture focusing via matched filter + migration.
- `TimeReversalImager` — Time-reversal focusing for unknown medium.

**Tensor Shapes:**
- Raw signals: `(N_tx, N_rx, N_timesteps)` — collected multi-static data
- Image volume: `(Nx_img, Ny_img, Nz_img)` — reconstructed permittivity/reflectivity
- Steering vectors: `(N_tx, N_rx, Nx_img, Ny_img, Nz_img)` — precomputed delays
- Green's functions: `(N_freq, N_spatial)` — for frequency-domain methods

**Dependencies:** `fdtd_engine`, `detectors`, `sources`

---

## 2.10 Module: `io`

**Purpose:** Serialization, checkpointing, and data export. All I/O is async to avoid blocking GPU execution.

**Key Classes:**
- `CheckpointManager` — Periodic full-state saves to HDF5. Async GPU→pinned→disk pipeline.
- `FieldExporter` — Exports field snapshots in HDF5/Zarr/VTK formats.
- `SimulationConfig` — YAML/JSON configuration serialization and validation.

**Tensor Shapes:** N/A (operates on all field/material tensors)

**Dependencies:** `grid`, `fdtd_engine`, `materials`

---

## 2.11 Module: `viz`

**Purpose:** Real-time and post-hoc visualization of fields, materials, and reconstruction results.

**Key Classes:**
- `SlicePlotter` — 2D field slices via matplotlib. Updates in-place for animation.
- `VolumeRenderer` — 3D field visualization via PyVista (isosurfaces, volume rendering).
- `LiveMonitor` — WebSocket-based streaming for remote visualization during long runs.

**Tensor Shapes:** Consumes field tensors directly (GPU→CPU transfer only for display frames).

**Dependencies:** `grid`, `detectors`, `imaging`

---

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

---

# Section 4: GPU Execution Model

## 4.1 Why FDTD Maps to GPU

The FDTD algorithm exhibits **massive data parallelism**: each cell's field update depends only on its immediate neighbors (6-point stencil). For an N³ grid, N³ independent updates execute per half-timestep.

### Arithmetic Intensity Analysis

Per cell E-field update:
- **Reads:** 4 H-field neighbors + 2 material coefficients + 1 current E value = 7 floats = 28 bytes
- **Writes:** 1 updated E value = 4 bytes
- **FLOPs:** 2 subtractions (curl) + 2 multiplications (coefficient scaling) + 1 addition = ~5 FLOPs
- **Per component:** 5 FLOPs / 32 bytes = **0.16 FLOP/byte**

Full cell (6 components): ~30 FLOPs, ~192 bytes transferred.

**Conclusion:** FDTD is **memory-bandwidth bound**. The A100's 2 TB/s HBM bandwidth is the ceiling, not its 19.5 TFLOPS FP32 compute. All optimization must target memory throughput.

### Theoretical Peak Performance

```
A100 80GB: 2,039 GB/s HBM bandwidth
Bytes per cell per step: 192 B (6 components, read+write, FP32)
Max cells/s = 2,039e9 / 192 = 10.6 Gcells/s (theoretical)
Achievable (70% efficiency): ~7.4 Gcells/s = 7,400 Mcells/s
```

With mixed precision (FP16): 96 bytes/cell → **14.8 Gcells/s theoretical**.

---

## 4.2 Kernel Architecture

### E-Field Update Kernel

```
__global__ void update_E(
    float* Ex, float* Ey, float* Ez,           // Output (read-modify-write)
    const float* Hx, const float* Hy, const float* Hz,  // Input (read-only)
    const float* Ca, const float* Cb,          // Material coefficients
    int Nx, int Ny, int Nz, float dt_dx, float dt_dy, float dt_dz
)
```

**Grid mapping:**
- 1 CUDA thread = 1 grid cell
- Thread block: `(8, 8, 8)` = 512 threads (good occupancy, fits 3D locality)
- Grid dims: `(ceil(Nx/8), ceil(Ny/8), ceil(Nz/8))`

**Update equation (Ex component):**
```
Ex[i,j,k] = Ca[i,j,k] * Ex[i,j,k]
           + Cb[i,j,k] * ( (Hz[i,j,k] - Hz[i,j-1,k]) * dt_dy
                          - (Hy[i,j,k] - Hy[i,j,k-1]) * dt_dz )
```

### H-Field Update Kernel

Identical structure to E-update but reads E neighbors and writes H. Operates at half-timestep offset.

### PML Kernel (CPML Formulation)

```
psi_Exy[i,j,k] = b_y[j] * psi_Exy[i,j,k]
                + c_y[j] * (Hz[i,j,k] - Hz[i,j-1,k])
Ex[i,j,k] += Cb[i,j,k] * psi_Exy[i,j,k]
```

**PML-specific considerations:**
- Only executes on boundary cells (PML_depth × surface_area)
- Can be fused with E-field kernel using predicated execution (branch on cell position)
- Separate kernel preferred when PML region << interior (avoid branch divergence in bulk)

### Source Injection Kernel

```
__global__ void inject_source(
    float* field_component,
    const int* indices,     // Sparse cell indices
    const float* amplitudes, // Per-cell amplitude
    float waveform_value,   // Current time sample
    int N_source_cells
)
```

Launched with N_source_cells threads. Scatter pattern — low occupancy but overlaps with bulk updates.

### Detector Kernel (DFT Accumulation)

```
// For each monitor frequency f_m:
dft_real[m, cell] += field[cell] * cos(2π * f_m * t * dt)
dft_imag[m, cell] += field[cell] * sin(2π * f_m * t * dt)
```

Gather + FMA pattern. Launched on detector cells only.

---

## 4.3 CUDA Stream Strategy

### Stream Assignment

| Stream | Purpose | Sync Requirements |
|--------|---------|-------------------|
| 0 (default) | Field update kernels (H→E→H→...) | Sequential within stream |
| 1 | Source injection | Event sync before field update reads source cells |
| 2 | Detector recording | Event after field update completes |
| 3 | Halo exchange (multi-GPU) | Event after boundary region update |
| 4 | Async I/O (checkpoint) | No sync needed (double-buffered) |

### Timeline for One Timestep

```
         ┌────────────────────── Timestep n ──────────────────────┐
Stream 0: [═══ H_update_interior ═══][wait_halo][H_boundary][═══ E_update ═══]
Stream 1: [src_H]                              [src_E]
Stream 2:                                      [detect_H]         [detect_E]
Stream 3: [══ halo_send ══][halo_recv]
Stream 4:                                                   [checkpoint_async]
```

### CUDA Graph Capture

For steady-state time-stepping (no dynamic branches):
```python
with torch.cuda.graph(graph):
    step_body()  # Entire timestep captured as static graph

for t in range(N_steps):
    graph.replay()  # Near-zero CPU overhead per step
```

CUDA Graphs eliminate kernel launch overhead (~5-10 μs per launch × ~10 kernels = 50-100 μs saved/step).

---

## 4.4 Kernel Fusion Opportunities

### Beneficial Fusions

| Fusion | Benefit | Condition |
|--------|---------|-----------|
| E_update + PML_E | 1 kernel launch, shared H reads | Always (PML cells are subset) |
| H_update + PML_H | Same as above | Always |
| Field_update + material_lookup | Avoid extra global load | When materials fit in shared memory |
| DFT_accumulate across frequencies | Single field read, multiple accumulates | N_freqs ≤ register budget |

### When NOT to Fuse

- Source injection + field update: Source is sparse (1% of cells), field update is dense. Fusing adds branch divergence to 99% of threads.
- Checkpoint copy + field update: Copy is on different stream for overlap. Fusing serializes them.

---

## 4.5 Occupancy Analysis

### Thread Block Sizing

| Block Shape | Threads | Registers/Thread (est.) | Shared Mem | Occupancy (SM 8.0) |
|-------------|---------|------------------------|------------|---------------------|
| (8,8,8) | 512 | 32 | 0 | 100% (2 blocks/SM) |
| (16,16,4) | 1024 | 32 | 0 | 50% (1 block/SM) |
| (8,8,4) | 256 | 40 | 2 KB | 100% (4 blocks/SM) |
| (32,8,4) | 1024 | 28 | 4 KB | 50% (1 block/SM) |

**Selected default: `(8, 8, 8)` = 512 threads.**

Rationale:
- 3D locality matches 3D stencil access pattern
- 2 blocks/SM = 1024 threads → good latency hiding
- 32 registers per thread × 512 threads = 16K registers (SM has 64K)
- Leaves headroom for compiler register spilling

### Register Pressure

Per-thread state for E-field update:
- 4 H neighbor values: 4 registers
- 2 material coefficients: 2 registers
- 1 current E value: 1 register
- Index computation: 3-4 registers
- Temporaries: 2-3 registers
- **Total: ~12-13 registers** (well within budget)

### Shared Memory Optimization

For stencil operations, neighboring threads read overlapping H values. Shared memory tiling:

```
Tile size: (8+2) × (8+2) × (8+2) = 1000 floats = 4 KB per H component
3 H components needed per E-component update: 12 KB
```

**Trade-off:** Shared memory reduces global memory reads by ~30% (halo reuse) but limits blocks/SM. Beneficial for large grids where cache misses dominate; less impactful on small grids fitting in L2.

---

## 4.6 PyTorch Integration

### Custom Autograd Function

```python
class FDTDStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, E, H, materials, dt, dx):
        # Launch CUDA kernels (custom or Triton)
        H_new = update_H_kernel(E, H, materials, dt, dx)
        E_new = update_E_kernel(H_new, E, materials, dt, dx)
        ctx.save_for_backward(E, H, materials)
        return E_new, H_new

    @staticmethod
    def backward(ctx, grad_E, grad_H):
        # Adjoint update (time-reversed)
        E, H, materials = ctx.saved_tensors
        grad_materials = adjoint_kernel(grad_E, grad_H, E, H)
        return grad_E_prev, grad_H_prev, grad_materials, None, None
```

### Triton Kernel Path (Rapid Prototyping)

```python
@triton.jit
def update_Ex_kernel(
    Ex_ptr, Hz_ptr, Hy_ptr, Ca_ptr, Cb_ptr,
    Nx, Ny, Nz, dt_dy, dt_dz,
    BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr, BLOCK_Z: tl.constexpr
):
    # Triton handles block/grid mapping, bounds checking, memory coalescing
    pid = tl.program_id(0)
    # ... stencil computation in Triton DSL
```

Triton advantages: auto-tuning block sizes, no CUDA boilerplate, integrates with torch.compile.

### torch.compile Compatibility

All operations use standard PyTorch tensor ops where possible:
```python
def update_H_pytorch(E, H, coeff, dt_dx):
    curl_E_x = (E.Ez.roll(-1, 1) - E.Ez) * dt_dx - (E.Ey.roll(-1, 2) - E.Ey) * dt_dx
    H.Hx -= coeff.Db * curl_E_x
```

`torch.compile(mode='max-autotune')` fuses these into efficient kernels automatically. Custom CUDA kernels used only where torch.compile underperforms (measured >10% gap).

---

# Section 5: Memory Management and Tensor Layout

## 5.1 VRAM Budget Analysis

### Per-Component Memory Formula

```
mem_per_component = Nx × Ny × Nz × sizeof(dtype)
```

### Full Simulation Memory Model

| Category | Formula | Notes |
|----------|---------|-------|
| E-fields (3 components) | 3 × N³ × 4B | Ex, Ey, Ez (FP32) |
| H-fields (3 components) | 3 × N³ × 4B | Hx, Hy, Hz (FP32) |
| Material coefficients | 6 × N³ × 4B | Ca, Cb per component (or 2 if uniform) |
| PML psi fields | 12 × D × S × 4B | 6 faces × 2 psi per face, D=depth, S=surface |
| PML coefficients | 6 × D × 4B | Negligible |
| Source buffers | N_src × N_t × 4B | Usually small |
| Detector DFT | N_freq × N_det × 8B | Complex64 |
| **Total (dominant)** | **(12 + N_mat) × N³ × 4B** | |

### VRAM Requirements Table (FP32)

| Grid Size | Cells | Fields (6) | +Materials (6) | +PML (D=10) | Total Est. |
|-----------|-------|-----------|----------------|-------------|------------|
| 128³ | 2.1M | 50 MB | 100 MB | +12 MB | **~120 MB** |
| 256³ | 16.8M | 403 MB | 806 MB | +48 MB | **~900 MB** |
| 512³ | 134M | 3.2 GB | 6.4 GB | +190 MB | **~7.0 GB** |
| 768³ | 453M | 10.8 GB | 21.6 GB | +430 MB | **~23 GB** |
| 1024³ | 1.07B | 25.6 GB | 51.2 GB | +770 MB | **~54 GB** |

### Mixed Precision (BF16 fields, FP32 materials)

| Grid Size | Fields (BF16) | Materials (FP32) | Total Est. |
|-----------|---------------|------------------|------------|
| 512³ | 1.6 GB | 3.2 GB | **~5.2 GB** |
| 768³ | 5.4 GB | 10.8 GB | **~17 GB** |
| 1024³ | 12.8 GB | 25.6 GB | **~40 GB** |

---

## 5.2 Tensor Memory Layout

### Structure of Arrays (SoA) Design

Each field component is a separate contiguous 3D tensor:

```python
fields = {
    'Ex': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
    'Ey': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
    'Ez': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
    'Hx': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
    'Hy': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
    'Hz': torch.zeros(Nx, Ny, Nz, dtype=torch.float32, device='cuda'),
}
```

**Why SoA over AoS:**
- E-field update reads only H components → SoA loads exactly what's needed
- Coalesced access: adjacent threads read adjacent memory (contiguous in Z)
- AoS (interleaved Ex,Ey,Ez,Hx,Hy,Hz per cell) wastes 50% bandwidth loading unused components

### Memory Ordering

PyTorch default: **row-major (C-contiguous)**, last dimension varies fastest.

For tensor shape `(Nx, Ny, Nz)`:
- Stride: `(Ny*Nz, Nz, 1)`
- Adjacent threads (threadIdx.x mapped to Z) access contiguous memory ✓
- **Thread block (8,8,8):** innermost 8 threads read consecutive Z addresses = 32-byte aligned (perfect coalescing)

### Padding Strategy

```python
def pad_to_warp(N, warp_size=32):
    return ((N + warp_size - 1) // warp_size) * warp_size

Nz_padded = pad_to_warp(Nz)  # Ensures last dimension is multiple of 32
```

Padding the fastest-varying dimension to multiples of 32 ensures:
- Full warp coalescing (no partial transactions)
- 128-byte cache line alignment
- Negligible memory overhead (~3% worst case for Nz=33→64)

---

## 5.3 Memory Pool Architecture

### Pre-Allocation Policy

```python
class TensorPool:
    def __init__(self, grid_shape, device, dtype=torch.float32):
        self.fields = self._allocate_fields(grid_shape, device, dtype)
        self.pml = self._allocate_pml(grid_shape, pml_depth, device, dtype)
        self.materials = self._allocate_materials(grid_shape, device, dtype)
        self.scratch = self._allocate_scratch(grid_shape, device, dtype)
        # All allocations happen here. ZERO allocations during time-stepping.
```

### Design Rules

1. **No runtime allocation in hot loop.** All tensors pre-allocated at `initialize()`. Violation → assertion failure in debug mode.
2. **Scratch buffers** for temporary computations reuse same memory across steps.
3. **Double buffering** for checkpoint: buffer A writes to disk while buffer B captures next checkpoint.
4. **Pinned host memory** (`torch.cuda.HostAllocator`) for async GPU→CPU transfers during checkpointing.

### CUDA Memory Allocator Integration

```python
torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use 95% of VRAM
torch.cuda.memory.empty_cache()  # Defragment before simulation
```

PyTorch's caching allocator handles sub-allocation. We configure:
- Large initial pool (avoid fragmentation)
- `max_split_size_mb=512` to prevent excessive fragmentation
- Explicit `torch.cuda.synchronize()` before measuring peak memory

---

## 5.4 Mixed Precision Strategy

### Precision Assignment by Data Type

| Data | Precision | Rationale |
|------|-----------|-----------|
| E, H fields | BF16 or FP32 | BF16 for speed; FP32 for accuracy-critical runs |
| Material coefficients (Ca, Cb) | FP32 | Computed once, precision matters for stability |
| PML psi fields | FP32 | Accumulation over many steps; BF16 drifts |
| PML grading (b, c, kappa) | FP32 | Small tensors, precision-sensitive |
| DFT accumulators | FP32 or FP64 | Long accumulation (N_steps terms); FP32 OK with Kahan summation |
| Source waveforms | FP32 | Small, precision-sensitive for phase accuracy |
| Gradient tensors (adjoint) | FP32 | Gradient underflow risk with FP16 |

### BF16 Field Update Precision Impact

Numerical dispersion error in FDTD: `δ(kΔx)` depends on floating-point rounding.

- FP32 (24-bit mantissa): relative error ~10⁻⁷ per step, accumulates to ~10⁻⁴ over 10³ steps
- BF16 (8-bit mantissa): relative error ~10⁻², accumulates to ~1.0 over 10³ steps → **UNSTABLE for long runs**
- **Mitigation:** Kahan compensated summation in BF16, or periodic FP32 correction steps

**Recommendation:** Use BF16 for field components only when:
- Simulation is short (<1000 steps) OR
- Periodic FP32 renormalization applied (every 100 steps) OR
- Application is gradient computation (short forward pass + adjoint)

### AMP (Automatic Mixed Precision) Integration

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    H_new = update_H(E, H, coeffs)  # Runs in BF16
    E_new = update_E(H_new, E, coeffs)  # Runs in BF16

# PML always in FP32:
with torch.autocast(enabled=False):
    apply_pml(E_new, psi, pml_coeffs.float())
```

---

## 5.5 Memory Bandwidth Optimization

### Achievable Bandwidth

| GPU | Peak BW | Achievable (80%) | Cells/s (FP32, 192B/cell) |
|-----|---------|-------------------|---------------------------|
| V100 | 900 GB/s | 720 GB/s | 3.75 Gcells/s |
| A100 | 2,039 GB/s | 1,631 GB/s | 8.5 Gcells/s |
| H100 | 3,350 GB/s | 2,680 GB/s | 14.0 Gcells/s |

### Access Pattern Optimization

**Coalesced Access:** Threads in a warp access consecutive 4-byte addresses. The stencil `Hz[i,j,k] - Hz[i,j-1,k]` requires:
- `Hz[i,j,k]`: coalesced (threads differ in k)
- `Hz[i,j-1,k]`: coalesced (same stride pattern, shifted by Nz)

Both are coalesced. The problematic pattern is `Hz[i,j,k-1]` (stride-1 neighbor in fastest dimension) — also coalesced since threads at k and k-1 are both within the same cache line for block size 8.

### L2 Cache Residency Control (SM 8.0+)

```cpp
cudaAccessPolicyWindow policy;
policy.base_ptr = (void*)Hz_ptr;
policy.num_bytes = Nx * Ny * Nz * sizeof(float);
policy.hitRatio = 0.6;  // 60% of L2 reserved for H-fields
policy.hitProp = cudaAccessPropertyPersisting;
policy.missProp = cudaAccessPropertyStreaming;
cudaCtxSetAccessPolicyWindow(&policy);
```

For 512³: 6 field tensors = 3.2 GB. A100 L2 = 40 MB → caches ~1.2% of fields. Residency control pins most-reused data (e.g., the z-1 plane being read by many warps).

### Register Blocking

Keep stencil values in registers across E-field components:
```
// Hz[i,j,k] needed by both Ex and Ey updates
register float hz_ijk = Hz[idx];
Ex_new = Ca_x * Ex + Cb_x * (hz_ijk - hz_ijm1k) * dt_dy - ...
Ey_new = Ca_y * Ey - Cb_y * (hz_ijk - hz_im1jk) * dt_dx + ...
```

Avoids redundant global memory loads when updating multiple components per thread.

---

# Section 6: CUDA Core Parallelism and GPU Execution Strategy

## 6.1 Single-GPU Architecture Target

GPU-MEEP targets **maximum utilization of CUDA cores within a single GPU**. The entire FDTD grid lives in one GPU's VRAM. Parallelism is achieved through thousands of concurrent threads mapped to grid cells — not through distributing work across multiple devices.

**Primary targets:**
- NVIDIA A100: 6,912 CUDA cores, 108 SMs, 80 GB HBM2e
- NVIDIA RTX 4090: 16,384 CUDA cores, 128 SMs, 24 GB GDDR6X
- NVIDIA H100: 14,592 CUDA cores, 132 SMs, 80 GB HBM3

**Design principle:** One CUDA thread = one grid cell update. A 512³ grid = 134M cells = 134M threads dispatched per half-timestep. The GPU's warp scheduler handles the mapping.

---

## 6.2 Thread Hierarchy and Grid Mapping

### CUDA Execution Model Applied to FDTD

```
FDTD Grid (Nx × Ny × Nz)
    ↓ maps to
CUDA Grid (gridDim.x × gridDim.y × gridDim.z)
    ↓ composed of
Thread Blocks (blockDim.x × blockDim.y × blockDim.z)
    ↓ executed as
Warps (32 threads, SIMT execution)
```

### Thread-to-Cell Mapping

```
cell(i, j, k) → thread:
    blockIdx  = (i / BLOCK_X, j / BLOCK_Y, k / BLOCK_Z)
    threadIdx = (i % BLOCK_X, j % BLOCK_Y, k % BLOCK_Z)
```

**3D block shape:** `(8, 8, 8)` = 512 threads per block

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Block size | 8×8×8 = 512 | 2 blocks/SM → 1024 resident threads, good latency hiding |
| Threads per SM | 1024 | Below max (2048) to leave register headroom |
| Blocks per SM | 2 | Balance between occupancy and register availability |
| Total blocks (512³) | 64×64×64 = 262,144 | Far exceeds SM count → full saturation |

### Why 3D Blocks Match FDTD

The Yee stencil accesses `(i±1, j±1, k±1)` neighbors. A 3D thread block ensures:
- Threads needing the same neighbor data are co-located in the same block
- Shared memory tiling captures the 3D neighborhood efficiently
- L1 cache locality is maximized (threads in a block access nearby addresses)

---

## 6.3 Warp-Level Execution

### Warp Formation

A 8×8×8 block contains 512 threads = **16 warps**. Warp linearization:
```
warp_id = (threadIdx.z * 64 + threadIdx.y * 8 + threadIdx.x) / 32
```

First warp: threads (0,0,0)→(3,3,1) — spans a 4×4×2 sub-block in the grid. These threads access contiguous Z-addresses → **coalesced memory access by construction**.

### Warp Divergence Analysis

FDTD field updates are **uniform** — every cell executes identical arithmetic. No branching in the hot path.

**Exception: PML boundary cells.** Cells in the PML region execute additional auxiliary field updates. Two strategies:

| Strategy | Approach | Divergence Cost |
|----------|----------|-----------------|
| Predicated | All threads execute PML code, non-PML threads masked out | ~15% wasted cycles in PML blocks |
| Separate kernel | PML cells handled by dedicated kernel launch | Zero divergence, extra launch overhead |
| **Hybrid (chosen)** | Fused kernel with early-exit for interior blocks | Zero cost for interior; ~15% for boundary blocks |

```
if (block_is_entirely_interior):
    // Fast path: no PML check per thread
    update_field_standard()
else:
    // Boundary block: per-thread PML predicate
    if (cell_in_pml):
        update_field_with_pml()
    else:
        update_field_standard()
```

Block-level divergence check avoids per-thread branching for 90%+ of blocks.

---

## 6.4 SM Occupancy Optimization

### Occupancy Calculation

```
Registers per thread: ~32 (measured for E-field kernel)
Shared memory per block: 0 KB (baseline) or 4 KB (with tiling)
Block size: 512 threads

SM resources (A100, SM 8.0):
- 65,536 registers per SM
- 164 KB shared memory per SM
- Max 2048 threads per SM
- Max 32 blocks per SM

Register-limited: 65,536 / 32 = 2048 threads → 4 blocks of 512 ✓
Thread-limited: 2048 / 512 = 4 blocks ✓
Block-limited: 32 ≥ 4 ✓
Shared mem: 164 KB / 4 KB = 41 blocks ✓ (not limiting)

Achieved occupancy: 2048/2048 = 100% (4 blocks/SM)
```

### Occupancy vs Performance Tradeoff

| Occupancy | Blocks/SM | Registers/Thread | Performance Impact |
|-----------|-----------|------------------|-------------------|
| 100% | 4 | 32 | Maximum latency hiding, register spills possible |
| 75% | 3 | 43 | More registers, fewer spills, slightly less hiding |
| 50% | 2 | 64 | Maximum registers, minimal hiding (bad for memory-bound) |

**For memory-bound FDTD: maximize occupancy.** Latency hiding (more warps in flight) compensates for memory stalls. Register pressure is low (stencil needs ~12-15 values), so 100% occupancy is achievable.

### Occupancy Tuning Knobs

```python
# PyTorch/Triton kernel launch configuration
BLOCK_SIZE = (8, 8, 8)  # 512 threads

# For Triton kernels:
@triton.jit
def update_E_kernel(..., BLOCK_X: tl.constexpr = 8, BLOCK_Y: tl.constexpr = 8, BLOCK_Z: tl.constexpr = 8):
    ...

# Auto-tune across block sizes:
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 8, 'BLOCK_Y': 8, 'BLOCK_Z': 8}),
        triton.Config({'BLOCK_X': 16, 'BLOCK_Y': 16, 'BLOCK_Z': 4}),
        triton.Config({'BLOCK_X': 32, 'BLOCK_Y': 8, 'BLOCK_Z': 4}),
    ],
    key=['Nx', 'Ny', 'Nz']
)
```

---

## 6.5 Memory Access Patterns and Coalescing

### Coalesced Global Memory Access

A warp issues a single memory transaction when 32 threads access 32 consecutive 4-byte addresses (128-byte cache line).

**FDTD stencil access pattern for Ex update:**

```
Ex[i,j,k] reads: Hz[i,j,k], Hz[i,j-1,k], Hy[i,j,k], Hy[i,j,k-1]
```

Memory layout `(Nx, Ny, Nz)` row-major (Z fastest):
- `Hz[i,j,k]` — threads in warp differ in k → consecutive addresses ✓ **COALESCED**
- `Hz[i,j-1,k]` — same stride, shifted by Nz → consecutive ✓ **COALESCED**
- `Hy[i,j,k-1]` — stride-1 shift in fastest dim → still consecutive ✓ **COALESCED**
- `Hy[i,j,k]` — same as Hz pattern ✓ **COALESCED**

**All accesses coalesced.** This is why SoA layout + Z-fastest-varying is critical.

### Cache Line Utilization

| Access Pattern | Bytes Loaded | Bytes Used | Efficiency |
|----------------|-------------|-----------|-----------|
| Hz[i,j,k] (aligned) | 128 B (1 line) | 128 B (32 floats) | 100% |
| Hz[i,j-1,k] (shifted) | 128 B | 128 B | 100% |
| Hz[i,j,k-1] (stride-1) | 128 B | 128 B | 100% |
| Scattered access (AoS) | 128 B | 16 B (4 floats) | 12.5% |

SoA achieves 100% cache line utilization. AoS would waste 87.5% of loaded bytes.

---

## 6.6 Shared Memory Tiling (Optional Optimization)

### Stencil Reuse Opportunity

Adjacent threads read overlapping H-field values. Thread (i,j,k) and (i,j+1,k) both read Hz[i,j,k].

**Tiling strategy:**
```
1. Load (BLOCK_X+1) × (BLOCK_Y+1) × (BLOCK_Z+1) H-values into shared memory
2. Synchronize block (__syncthreads)
3. Compute E-field updates from shared memory (fast, no global stall)
```

### Shared Memory Budget

```
Tile for Hz: (8+1)×(8+1)×(8+1) = 729 floats = 2,916 bytes
Three H-components needed: 3 × 2,916 = 8,748 bytes ≈ 9 KB per block
With 4 blocks/SM: 36 KB (within A100's 164 KB budget)
```

### When Shared Memory Helps

| Grid Size | L2 Hit Rate (no SM) | With Shared Memory | Speedup |
|-----------|---------------------|-------------------|---------|
| 128³ | ~85% | ~95% | 1.05× |
| 256³ | ~60% | ~92% | 1.15× |
| 512³ | ~30% | ~90% | 1.25× |
| 768³ | ~15% | ~88% | 1.35× |

**Conclusion:** Shared memory tiling becomes worthwhile for grids ≥256³ where L2 cache can't hold the working set. For smaller grids, the overhead of loading/syncing outweighs the benefit.

---

## 6.7 Kernel Launch Configuration

### Full Kernel Launch Spec

```python
def launch_E_update(Ex, Ey, Ez, Hx, Hy, Hz, Ca, Cb, Nx, Ny, Nz, dt_dx, dt_dy, dt_dz):
    block = (8, 8, 8)
    grid = (
        (Nx + block[0] - 1) // block[0],
        (Ny + block[1] - 1) // block[1],
        (Nz + block[2] - 1) // block[2],
    )
    # For 512³: grid = (64, 64, 64) = 262,144 blocks
    # Total threads: 262,144 × 512 = 134,217,728
    # A100 can have 108 SMs × 4 blocks = 432 blocks active simultaneously
    # All 262,144 blocks complete in ~607 waves
```

### Wave Execution Model

```
Total blocks: 262,144
Active blocks (A100): 108 SM × 4 blocks/SM = 432 concurrent blocks
Waves needed: 262,144 / 432 = 607 waves
Time per wave: ~0.3 μs (memory-bound, HBM latency hidden by occupancy)
Total kernel time: ~180 μs for 512³ E-field update
```

### Thread Block Cluster (SM 9.0+ / H100)

On Hopper architecture, thread block clusters allow cooperative groups across SMs:
```
// Cluster of 2×2×2 = 8 blocks share distributed shared memory
// Adjacent blocks can read each other's shared memory directly
// Eliminates halo redundancy at block boundaries
```

Future optimization path for H100+ hardware.

---

## 6.8 Instruction-Level Parallelism (ILP)

### Loop Unrolling for Throughput

Each thread updates one cell, but can compute multiple output components:

```cuda
// Single thread computes Ex, Ey, Ez at (i,j,k)
// Shares loaded H-values across component updates
float hz_ijk = Hz[idx];          // Loaded once
float hz_jm1 = Hz[idx - Nz];    // Loaded once
float hy_ijk = Hy[idx];          // Loaded once
float hy_km1 = Hy[idx - 1];     // Loaded once
float hx_ijk = Hx[idx];         // Loaded once
float hx_km1 = Hx[idx - 1];    // Loaded once

Ex[idx] = Ca_x * Ex[idx] + Cb_x * ((hz_ijk - hz_jm1) * dt_dy - (hy_ijk - hy_km1) * dt_dz);
Ey[idx] = Ca_y * Ey[idx] + Cb_y * ((hx_ijk - hx_km1) * dt_dz - (hz_ijk - hz_im1) * dt_dx);
Ez[idx] = Ca_z * Ez[idx] + Cb_z * ((hy_ijk - hy_im1) * dt_dx - (hx_ijk - hx_jm1) * dt_dy);
```

**Benefit:** 6 global loads serve 3 output computations. Arithmetic ops (subtract, multiply, FMA) pipeline while subsequent loads are in-flight. ILP = 3 independent FMA chains per thread.

---

## 6.9 Performance Model Summary

### Single GPU Roofline

```
             Compute Roof (A100 FP32: 19.5 TFLOPS)
            ╱
           ╱
          ╱
         ╱    ┌────── FDTD operating point (0.16 FLOP/byte)
        ╱     │       → Memory-bound
       ╱      ▼
      ╱───────●──────────────── Memory Roof (2 TB/s)
     ╱
    ╱
   ╱
  Arithmetic Intensity (FLOP/byte) →
```

**FDTD lives firmly in the memory-bound regime.** All optimization effort targets:
1. Reducing bytes transferred (mixed precision, fusion)
2. Maximizing effective bandwidth (coalescing, occupancy, cache)
3. NOT increasing FLOPS (already compute-underutilized)

### Expected Throughput by GPU

| GPU | BW (GB/s) | Cores | FDTD Throughput (Mcells/s) | Max Grid (80% VRAM) |
|-----|-----------|-------|---------------------------|---------------------|
| RTX 3090 (24GB) | 936 | 10,496 | 3,400 | 384³ FP32 |
| RTX 4090 (24GB) | 1,008 | 16,384 | 3,700 | 384³ FP32 |
| A100 (80GB) | 2,039 | 6,912 | 7,400 | 768³ FP32 |
| H100 (80GB) | 3,350 | 14,592 | 12,200 | 768³ FP32 |

---

## 6.10 Future: Multi-GPU as Extension

Multi-GPU support is a future extension (v2.0+) for problems exceeding single-GPU VRAM. When implemented:
- Domain decomposition with NCCL halo exchange
- NVLink-aware subdomain placement
- Overlap interior compute with boundary communication

**Current scope:** Single GPU, maximize CUDA core utilization, saturate memory bandwidth.

---

# Section 7: Data Pipeline and I/O Architecture

## 7.1 Simulation Data Flow

### End-to-End Pipeline

```
INPUT                          RUNTIME                         OUTPUT
─────                          ───────                         ──────
Geometry (CSG/STL/SDF)         Source waveform gen (GPU)       Detector time series
       │                              │                              │
       ▼                              ▼                              ▼
Material mapping (CPU)  ──►    Field updates (GPU)    ──►    DFT results (GPU→CPU)
       │                              │                              │
       ▼                              ▼                              ▼
Grid voxelization (CPU) ──►    Detector accumulation  ──►    Imaging recon (GPU)
       │                         (GPU)                               │
       ▼                              │                              ▼
GPU tensor upload       ──►    [checkpoint async]     ──►    HDF5/Zarr export
(one-time)                     (GPU→pinned→disk)             (async)
```

**Key Invariant:** Zero CPU↔GPU transfers during the time-stepping loop. All runtime tensors are GPU-resident. Checkpoints use async DMA and do not stall the compute pipeline.

---

## 7.2 Geometry Pipeline

### Input Formats

| Format | Use Case | Processing |
|--------|----------|------------|
| CSG primitives | Analytical shapes (sphere, box, cylinder) | Direct voxelization |
| STL mesh | CAD imports | Ray-casting voxelization |
| SDF field | Smooth boundaries | Subpixel averaging at interfaces |
| Image stack (DICOM/PNG) | Medical CT, MRI data | Direct mapping to epsilon |
| NumPy array | Programmatic geometry | Direct assignment |

### Voxelization Pipeline

```
1. CSG tree → evaluate SDF at each grid cell center
2. SDF → material index: material_idx[i,j,k] = argmin(SDF_m(x_i, y_j, z_k)) for m in materials
3. Subpixel averaging at interfaces:
   eps_eff[i,j,k] = Σ_m (volume_fraction_m × eps_m)   [for cells crossing boundaries]
4. Upload material_idx (int16) and eps_eff (float32) to GPU
```

**Tensor shapes after voxelization:**
- `material_index`: `(Nx, Ny, Nz)` int16
- `epsilon_xx, epsilon_yy, epsilon_zz`: `(Nx, Ny, Nz)` float32
- `sigma_xx, sigma_yy, sigma_zz`: `(Nx, Ny, Nz)` float32
- `mu_xx, mu_yy, mu_zz`: `(Nx, Ny, Nz)` float32 (usually uniform → scalar)

### Subpixel Averaging

At material interfaces, staircasing artifacts degrade accuracy. Subpixel smoothing:

```
For cell (i,j,k) containing interface:
  eps_eff = (1/V_cell) ∫∫∫_cell eps(x,y,z) dV
  ≈ Σ_{sub} eps(x_sub) / N_sub    (Monte Carlo or tensor-product quadrature)
```

GPU-accelerated: launch kernel over interface cells (identified by SDF sign change between neighbors), compute weighted average using 8-27 sub-samples.

---

## 7.3 Checkpoint and Restart

### Async Checkpoint Pipeline

```
GPU field tensors ──► Pinned host buffer ──► Background thread ──► HDF5 on disk
     (no stall)         (async DMA)            (Python thread)       (chunked write)
```

### Implementation

```python
class CheckpointManager:
    def __init__(self, path, interval_steps=1000):
        self.buf_A = torch.empty(field_shape, pin_memory=True)  # Double buffer
        self.buf_B = torch.empty(field_shape, pin_memory=True)
        self.stream = torch.cuda.Stream()
        self.writer_thread = threading.Thread(target=self._disk_writer, daemon=True)

    def save_async(self, fields, step):
        buf = self.buf_A if step % 2 == 0 else self.buf_B
        with torch.cuda.stream(self.stream):
            buf.copy_(fields, non_blocking=True)  # GPU→pinned (DMA)
        self.stream.synchronize()  # Wait for DMA, not compute stream
        self.write_queue.put((buf, step))  # Background thread writes to disk
```

### Checkpoint Format (HDF5)

```
checkpoint_step_005000.h5
├── fields/
│   ├── Ex  [dataset: (Nx,Ny,Nz) float32, chunked (64,64,64), lz4 compressed]
│   ├── Ey  [...]
│   ├── Ez  [...]
│   ├── Hx  [...]
│   ├── Hy  [...]
│   └── Hz  [...]
├── pml/
│   ├── psi_Exy [...]
│   └── ... (12 psi tensors)
├── metadata/
│   ├── step (int)
│   ├── time (float64)
│   ├── grid_shape (3,)
│   ├── dx, dy, dz (float64)
│   └── config (JSON string)
```

### Restart Protocol

```python
def restart_from_checkpoint(path):
    with h5py.File(path, 'r') as f:
        fields = {name: torch.from_numpy(f[f'fields/{name}'][:]).cuda()
                  for name in ['Ex','Ey','Ez','Hx','Hy','Hz']}
        step = f['metadata/step'][()]
    engine.load_state(fields, step)
    engine.run(until=T_final)  # Resumes from checkpoint step
```

---

## 7.4 Detector Data Pipeline

### Time-Domain Detectors

```
Recording:    field[probe_idx] → ring_buffer[probe_idx, t % buf_size]
Flush:        ring_buffer (GPU) → pinned_host (DMA) → output_array (CPU)
Flush trigger: buffer full OR simulation end
```

**Tensor shapes:**
- Point probes: `(N_probes, buf_size)` float32 — ring buffer on GPU
- Surface probes: `(N_surface_cells, buf_size)` float32 — larger, flush more frequently

### Frequency-Domain (DFT) Detectors

Running DFT avoids storing full time series:

```python
# On GPU, every timestep:
for m in range(N_freqs):
    phase = 2 * pi * freqs[m] * t * dt
    dft_real[m, :] += field[monitor_cells] * cos(phase)
    dft_imag[m, :] += field[monitor_cells] * sin(phase)
```

**Tensor shape:** `(N_freqs, N_monitor_cells)` complex64 (stored as 2× float32)

At simulation end: `S_param[freq] = dft_complex / N_steps` (normalized)

### Near-to-Far Field Transform

```
Surface currents: J_s[surface_cells, N_freqs], M_s[surface_cells, N_freqs]
Far-field: E_ff(theta, phi, f) = ∫∫ (J_s × r̂ + M_s) × Green's × e^{jkr} dS
```

Computed as GPU matrix-vector product: `E_ff = G @ J_s` where G is the Green's function matrix `(N_angles, N_surface_cells)` complex64.

---

## 7.5 Imaging Reconstruction Pipeline

### Multi-Simulation Orchestration

```
for tx_idx in range(N_tx):           # Or batched across GPUs
    configure_source(tx_positions[tx_idx], waveform)
    sim.run(until=T_max)
    raw_data[tx_idx, :, :] = detectors.extract()  # (N_rx, N_t)
    sim.reset_fields()               # Zero fields, keep geometry
```

**Batching strategy:** If VRAM allows, run M simulations in parallel (M independent grids on same GPU) or distribute across GPUs (1 TX per GPU).

### Reconstruction Kernels

**Delay-and-Sum (DAS) Backprojection:**

```
image[x,y,z] = Σ_{tx} Σ_{rx} signal[tx, rx, τ(tx,rx,x,y,z)]
where τ = (|pos_tx - r| + |pos_rx - r|) / c
```

GPU implementation: one thread per image voxel, loops over TX/RX pairs, interpolates signal at computed delay.

**Tensor shapes:**
- `raw_signals`: `(N_tx, N_rx, N_t)` float32 — input
- `delays`: `(N_tx, N_rx, Nx_img, Ny_img, Nz_img)` float32 — precomputed or computed on-the-fly
- `image`: `(Nx_img, Ny_img, Nz_img)` float32 — output

### Pipeline Timing (32 TX, 32 RX, 256³ image, 4096 time samples)

| Stage | GPU Time | Notes |
|-------|----------|-------|
| Forward simulations (32×) | 32 × 5s = 160s | Parallelizable across GPUs |
| Delay computation | 0.5s | Precomputed geometry |
| Backprojection | 2.1s | Memory-bound, all on GPU |
| Post-processing (filter) | 0.1s | |
| **Total (4 GPUs)** | **~42s** | 4× speedup on forward sims |

---

## 7.6 Streaming and Real-Time Monitoring

### Field Snapshot Streaming

```python
class LiveStreamer:
    def __init__(self, decimation=100, slice_axis='z', slice_idx=None):
        self.zmq_pub = zmq.Context().socket(zmq.PUB)
        self.zmq_pub.bind("tcp://*:5555")

    def on_step(self, fields, step):
        if step % self.decimation == 0:
            slice_data = fields.Ex[:, :, self.slice_idx].cpu().numpy()
            self.zmq_pub.send_pyobj({'step': step, 'field': slice_data})
```

### Decimation Strategy

- **Spatial:** Send every Nth cell (2× decimation = 8× data reduction in 3D)
- **Temporal:** Send every Mth step (M=100 typical for 10k+ step sims)
- **Component:** Send only requested field component (6× reduction)
- **Combined:** 100× temporal × 8× spatial = 800× reduction → 8 KB/frame for 512³ grid

### Output Format Summary

| Consumer | Format | Transport |
|----------|--------|-----------|
| Post-processing scripts | HDF5/Zarr | Filesystem |
| Jupyter notebooks | NumPy arrays (in-memory) | Direct return |
| Live visualization | Decimated slices | ZMQ PUB/SUB |
| Web dashboard | PNG/JPEG frames | WebSocket |
| ML pipelines | PyTorch tensors | Direct (same GPU) |

---

# Section 8: Directory Structure and Build System

## 8.1 Complete Project Tree

```
gpu-meep/
├── pyproject.toml                          # Build config, dependencies, metadata
├── README.md                               # Project overview
├── LICENSE                                 # Apache 2.0
├── CHANGELOG.md                            # Version history
├── Makefile                                # Dev shortcuts (test, bench, lint)
│
├── gpumeep/                                # Main Python package
│   ├── __init__.py                         # Public API exports, version
│   │
│   ├── core/                               # Grid and field fundamentals
│   │   ├── __init__.py
│   │   ├── grid.py                         # YeeGrid: lattice construction, cell coordinates
│   │   ├── fields.py                       # FieldSet: tensor container for Ex,Ey,Ez,Hx,Hy,Hz
│   │   ├── timestepper.py                  # LeapfrogStepper: dt management, CFL check, step loop
│   │   ├── constants.py                    # c0, eps0, mu0, eta0
│   │   ├── dtypes.py                       # Precision policies (FP32, BF16, mixed)
│   │   └── regions.py                      # Box, Sphere, Cylinder, CSG ops for geometry
│   │
│   ├── engine/                             # GPU compute dispatch
│   │   ├── __init__.py
│   │   ├── dispatch.py                     # KernelDispatcher: selects kernel variant by grid/device
│   │   ├── streams.py                      # StreamManager: CUDA stream pool, event sync
│   │   ├── graph.py                        # CUDAGraphCapture: record/replay for steady-state
│   │   ├── profiler.py                     # Integrated nsight/torch.profiler hooks
│   │   └── kernels/                        # Kernel implementations
│   │       ├── __init__.py
│   │       ├── update_e.py                 # E-field update (PyTorch ops / Triton)
│   │       ├── update_h.py                 # H-field update (PyTorch ops / Triton)
│   │       ├── pml.py                      # CPML auxiliary field updates
│   │       ├── source_inject.py            # Sparse source injection kernel
│   │       ├── dft_accumulate.py           # Running DFT for freq-domain monitors
│   │       ├── curl_ops.py                 # Discrete curl via roll/shift operations
│   │       └── triton/                     # Triton JIT kernels (optional fast path)
│   │           ├── update_e_triton.py
│   │           ├── update_h_triton.py
│   │           └── fused_pml_triton.py
│   │
│   ├── materials/                          # Electromagnetic material models
│   │   ├── __init__.py
│   │   ├── material.py                     # Material base class, property containers
│   │   ├── material_map.py                 # MaterialMap: grid→properties tensor assignment
│   │   ├── dispersive.py                   # Debye, Drude, Lorentz ADE models
│   │   ├── anisotropic.py                  # Full 3×3 tensor permittivity
│   │   ├── library.py                      # Predefined materials (tissue, concrete, FR4...)
│   │   └── subpixel.py                     # Subpixel averaging at material interfaces
│   │
│   ├── sources/                            # EM excitation sources
│   │   ├── __init__.py
│   │   ├── source.py                       # Source base class, injection protocol
│   │   ├── point_source.py                 # Point dipole source
│   │   ├── plane_wave.py                   # TFSF plane wave injection
│   │   ├── gaussian_beam.py                # Focused Gaussian beam
│   │   ├── waveforms.py                    # GaussianPulse, CW, Chirp, Custom waveform
│   │   └── antenna.py                      # Realistic antenna patterns (patch, dipole, horn)
│   │
│   ├── boundaries/                         # Boundary conditions
│   │   ├── __init__.py
│   │   ├── pml.py                          # CPML: grading, coefficients, psi field management
│   │   ├── periodic.py                     # Periodic BC (wrap-around indexing)
│   │   ├── bloch.py                        # Bloch periodic (phase shift for oblique)
│   │   ├── pec.py                          # Perfect Electric Conductor
│   │   ├── pmc.py                          # Perfect Magnetic Conductor
│   │   └── absorbing.py                    # First-order absorbing (Mur) for comparison
│   │
│   ├── detectors/                          # Field monitors and probes
│   │   ├── __init__.py
│   │   ├── probe.py                        # FieldProbe: point/line/surface recording
│   │   ├── flux.py                         # FluxMonitor: Poynting flux via DFT
│   │   ├── dft_monitor.py                  # DFTMonitor: freq-domain field snapshots
│   │   ├── near2far.py                     # Near-to-far field transform
│   │   └── energy.py                       # EnergyMonitor: EM energy in volume
│   │
│   ├── differentiable/                     # Autograd and adjoint methods
│   │   ├── __init__.py
│   │   ├── autograd_fdtd.py               # torch.autograd.Function wrapping FDTD steps
│   │   ├── adjoint.py                      # AdjointSolver: time-reversed gradient computation
│   │   ├── checkpointing.py               # Binomial checkpointing (Griewank algorithm)
│   │   ├── loss_functions.py              # EM-specific losses (mode overlap, transmission)
│   │   └── optimizer.py                    # Physics-constrained optimization loop
│   │
│   ├── imaging/                            # MIMO/SAR imaging pipelines
│   │   ├── __init__.py
│   │   ├── mimo_array.py                   # MIMOArray: TX/RX geometry, sequencing
│   │   ├── backprojection.py              # Delay-and-sum GPU kernel
│   │   ├── sar.py                          # SAR focusing (range migration, omega-k)
│   │   ├── beamforming.py                  # Capon, MUSIC, MVDR beamformers
│   │   ├── time_reversal.py               # TR-MUSIC, DORT decomposition
│   │   ├── inverse_scattering.py          # Born/Rytov iterative solvers
│   │   └── matched_filter.py             # Pulse compression, range processing
│   │
│   ├── multigpu/                           # Multi-GPU domain decomposition
│   │   ├── __init__.py
│   │   ├── decomposer.py                  # DomainDecomposer: partition grid across GPUs
│   │   ├── halo.py                         # HaloExchanger: pack/send/recv/unpack protocol
│   │   ├── topology.py                     # Auto-detect NVLink/PCIe, plan placement
│   │   └── distributed_engine.py          # DistributedFDTD: multi-GPU time-stepping
│   │
│   ├── io/                                 # Input/Output and serialization
│   │   ├── __init__.py
│   │   ├── checkpoint.py                   # CheckpointManager: async save/restore
│   │   ├── hdf5_io.py                      # HDF5 read/write for fields and geometry
│   │   ├── zarr_io.py                      # Zarr format (cloud-friendly chunked arrays)
│   │   ├── vtk_export.py                   # VTK export for ParaView visualization
│   │   ├── config.py                       # YAML/JSON simulation config loader/saver
│   │   └── stl_import.py                   # STL mesh → voxelized geometry
│   │
│   └── viz/                                # Visualization
│       ├── __init__.py
│       ├── slice_plot.py                   # 2D field slices (matplotlib)
│       ├── volume_render.py               # 3D volume rendering (PyVista)
│       ├── animation.py                    # Time-lapse field animation (mp4/gif)
│       ├── live_stream.py                  # ZMQ-based real-time field streaming
│       └── dashboard.py                    # Web dashboard (optional, plotly/dash)
│
├── kernels/                                # Raw CUDA source (optional advanced path)
│   ├── update_e.cu                         # Custom CUDA E-field kernel
│   ├── update_h.cu                         # Custom CUDA H-field kernel
│   ├── pml_fused.cu                        # Fused field+PML kernel
│   ├── backprojection.cu                   # Imaging backprojection kernel
│   └── build.py                            # torch.utils.cpp_extension build script
│
├── tests/                                  # Test suite
│   ├── conftest.py                         # Fixtures: small grids, reference solutions
│   ├── test_grid.py                        # Grid construction, coordinate correctness
│   ├── test_fdtd_1d.py                     # 1D propagation, analytical comparison
│   ├── test_fdtd_2d.py                     # 2D point source, Green's function check
│   ├── test_fdtd_3d.py                     # 3D cavity modes, resonance frequencies
│   ├── test_pml.py                         # PML reflection coefficient < -40 dB
│   ├── test_materials.py                   # Dispersive model vs analytical
│   ├── test_sources.py                     # TFSF correctness, waveform fidelity
│   ├── test_detectors.py                   # DFT accuracy vs offline FFT
│   ├── test_differentiable.py             # Gradient check (finite difference vs adjoint)
│   ├── test_multigpu.py                    # Halo exchange correctness, multi-GPU match
│   ├── test_imaging.py                     # Reconstruction of known target
│   ├── test_checkpoint.py                  # Save/restore field continuity
│   └── test_numerical_stability.py        # Long-run energy conservation
│
├── benchmarks/                             # Performance benchmarks
│   ├── bench_fdtd_scaling.py              # Grid size vs throughput
│   ├── bench_multigpu.py                   # Strong/weak scaling
│   ├── bench_precision.py                  # FP32 vs BF16 throughput
│   ├── bench_pml_overhead.py              # PML cost relative to free-space
│   ├── bench_vs_meep.py                    # Direct comparison with Meep
│   └── bench_imaging.py                    # Reconstruction pipeline throughput
│
├── examples/                               # Usage examples
│   ├── 01_basic_propagation.py            # Simplest: pulse in free space
│   ├── 02_dielectric_slab.py             # Reflection/transmission
│   ├── 03_waveguide_mode.py              # Rectangular waveguide
│   ├── 04_antenna_pattern.py             # Dipole antenna far-field
│   ├── 05_mimo_imaging.py                 # MIMO breast imaging example
│   ├── 06_inverse_design.py              # Topology optimization of a splitter
│   ├── 07_through_wall.py                # Through-wall radar imaging
│   ├── 08_multigpu_large.py              # Multi-GPU large domain
│   └── 09_ai_reconstruction.py           # Neural network + FDTD hybrid
│
└── docs/                                   # Documentation
    ├── architecture/                       # System architecture (this document)
    │   ├── SYSTEM_ARCHITECTURE.md
    │   └── sections/
    ├── api/                                # Auto-generated API reference
    ├── theory/                             # Physics and numerical methods
    │   ├── fdtd_formulation.md
    │   ├── pml_theory.md
    │   └── adjoint_method.md
    └── tutorials/                          # Step-by-step guides
```

---

## 8.2 Package Dependencies

```toml
[project]
name = "gpumeep"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "h5py>=3.8",
    "pyyaml>=6.0",
    "tqdm>=4.65",
]

[project.optional-dependencies]
triton = ["triton>=2.1"]
distributed = ["torch>=2.0"]  # NCCL included with torch
viz = ["matplotlib>=3.7", "pyvista>=0.40", "pyzmq>=25.0"]
imaging = ["scipy>=1.10", "scikit-image>=0.21"]
dev = [
    "pytest>=7.4",
    "pytest-benchmark>=4.0",
    "ruff>=0.1",
    "mypy>=1.5",
    "pre-commit>=3.0",
]
all = ["gpumeep[triton,viz,imaging,dev]"]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
include = ["gpumeep*"]
```

---

## 8.3 Build System

### Installation Tiers

| Tier | Command | What's Built | Use Case |
|------|---------|-------------|----------|
| Basic | `pip install gpumeep` | Pure Python + PyTorch ops | Development, prototyping |
| Triton | `pip install gpumeep[triton]` | + JIT Triton kernels | Production (auto-tuned) |
| CUDA ext | `pip install gpumeep[cuda]` + build | + Custom .cu kernels | Maximum performance |
| Full | `pip install gpumeep[all]` | Everything | Research environments |

### Custom CUDA Kernel Build (Optional)

```python
# kernels/build.py
from torch.utils.cpp_extension import load

fdtd_cuda = load(
    name='fdtd_cuda',
    sources=['kernels/update_e.cu', 'kernels/update_h.cu', 'kernels/pml_fused.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_80'],
    verbose=True
)
```

Only needed when benchmarks show >10% gap vs Triton/PyTorch paths. Most users never build custom kernels.

### CI Pipeline

```yaml
# .github/workflows/test.yml
jobs:
  test-cpu:        # Correctness on CPU (no GPU runner needed)
  test-gpu:        # Full GPU tests on self-hosted runner (A100)
  benchmark:       # Performance regression detection
  lint:            # ruff + mypy
```

---

## 8.4 Namespace and Import Design

### Top-Level Convenience API

```python
import gpumeep as gm

sim = gm.Simulation(
    grid=gm.Grid(resolution=0.5e-3, size=(0.1, 0.1, 0.05)),
    materials=[gm.Material(eps=4.0, region=gm.Box(center, size))],
    sources=[gm.GaussianSource(freq=2.4e9, center=src_pos)],
    boundaries=[gm.PML(thickness=10)],
    detectors=[gm.FluxMonitor(surface, freqs=[2.4e9])],
)
sim.run(until=10e-9)
results = sim.results()
```

### Submodule Access (Advanced Users)

```python
from gpumeep.core import YeeGrid, FieldSet
from gpumeep.engine import KernelDispatcher, StreamManager
from gpumeep.materials import DebyeModel, MaterialMap
from gpumeep.imaging import MIMOArray, BackprojectionReconstructor
from gpumeep.differentiable import AdjointSolver, DifferentiableFDTD
from gpumeep.multigpu import DomainDecomposer, HaloExchanger
```

### Import DAG (Enforced)

```
core (no internal imports)
  └── engine (imports core)
       └── materials (imports core)
       └── sources (imports core)
       └── boundaries (imports core)
            └── detectors (imports core, engine)
                 └── differentiable (imports core, engine, materials, boundaries)
                      └── imaging (imports core, engine, sources, detectors, differentiable)
                           └── io (imports all above)
                           └── viz (imports all above)
```

Circular imports are caught by `importlib` tests in CI. Violation = build failure.
