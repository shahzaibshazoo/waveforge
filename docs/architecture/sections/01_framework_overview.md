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
