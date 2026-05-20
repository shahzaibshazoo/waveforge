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
