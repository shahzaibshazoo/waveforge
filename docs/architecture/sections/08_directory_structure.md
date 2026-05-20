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
