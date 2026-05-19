# CUDA-MEEP: GPU-Native FDTD Electromagnetic Simulation Engine

A GPU-accelerated 2D TM FDTD electromagnetic simulator built with PyTorch/CUDA.  
Designed as a high-performance alternative to Meep for microwave imaging, MIMO radar, and inverse scattering.

---

## Benchmark Results (Tesla T4 vs Laptop CPU)

### Throughput Comparison

![Benchmark Comparison](assets/benchmark_comparison.png)

### GPU Speedup Scaling

![Speedup Scaling](assets/speedup_scaling.png)

**Measured results (Tesla T4 GPU, Colab free tier):**

| Grid | CUDA-MEEP GPU | CUDA-MEEP CPU | Meep CPU | **GPU / Meep** |
|------|:-------------:|:-------------:|:--------:|:--------------:|
| 64²  | 24 Mcells/s   | 3 Mcells/s    | 18 Mcells/s | 0.2× |
| 128² | 25 Mcells/s   | 12 Mcells/s   | 20 Mcells/s | 1.0× |
| 256² | 94 Mcells/s   | 28 Mcells/s   | 15 Mcells/s | **6.0×** |
| 512² | 350 Mcells/s  | 24 Mcells/s   | 16 Mcells/s | **21.8×** |

> **At 512² grid: 21.8× faster than Meep on a free Colab T4 GPU.**  
> An A100 (2 TB/s HBM vs T4's 300 GB/s) projects to ~140× at 512², and more at 1024²+.

**Why GPU loses at small grids:** CUDA kernel launch overhead (~5μs) dominates when the grid is tiny (4K cells).  
GPU wins decisively at 256²+ where the 2,560 CUDA cores are fully saturated.

---

## Brain Tumor Detection (MIMO Microwave Imaging)

16-antenna circular array, 1 GHz, delay-and-sum backprojection:

![Brain MIMO Imaging](assets/brain_mimo_imaging.png)

- **Top-left:** εᵣ material map — free space (1), skull (8), brain (40), tumor (55)
- **Top-right:** DAS backprojection — bright spot correctly localizes the 12mm tumor
- **Bottom-left:** Scattered signal energy — tumor response arrives at ~3 ns
- **Bottom-right:** TX0→RX8 signal comparison — healthy vs tumor (difference ×10 shown)

---

## Features

- GPU-accelerated FDTD via PyTorch tensor ops — no Python loops over spatial indices
- 2D TM mode: {Ex, Ey, Hz} with correct Yee-grid staggering and leapfrog
- Lossy dispersive materials — per-cell εᵣ, σ coefficients (skull, brain, tumor at 1 GHz)
- First-order Mur absorbing boundary conditions
- CPML coefficient scaffold (ready for Phase 2)
- MIMO circular array imaging — delay-and-sum backprojection
- Batch simulation support (multiple simultaneous TX runs)
- Real-time visualization and MP4/GIF animation export
- 21-test physics verification suite (CFL, Faraday/Ampere signs, propagation, absorption)

---

## Quickstart (Google Colab — GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shahzaibshazoo/cuda-meep/blob/main/notebooks/colab_benchmark.ipynb)

1. Click the badge above
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells — benchmarks + brain tumor demo run automatically

---

## Local Installation

```bash
git clone https://github.com/shahzaibshazoo/cuda-meep.git
cd cuda-meep
pip install torch numpy matplotlib pytest
```

---

## Basic Usage

```python
import sys
sys.path.insert(0, 'src')
from core import YeeGrid, FieldSet, MurABC, GaussianPulse, PointSource, SourceCollection, FDTD2D

# Grid: 512×512, 1mm cells, CUDA
grid     = YeeGrid(512, 512, dx=1e-3, dy=1e-3, device='cuda')
fields   = FieldSet(grid)
boundary = MurABC(grid, fields.Hz)

# Gaussian pulse source at center
pulse   = GaussianPulse(amplitude=1.0, sigma=30*grid.dt)
src     = PointSource(pulse, i=256, j=256, component='Hz', grid=grid, N_steps=1000)
sources = SourceCollection([src])

sim = FDTD2D(grid, fields, boundary, sources)
sim.run(1000, verbose=True)

print(f"Throughput: {sim.mcells_per_second:.0f} Mcells/s")
# → ~350 Mcells/s on Tesla T4
```

---

## Lossy Materials (Brain Imaging)

```python
from core import MaterialMap, TISSUE_LIBRARY

mat_map = MaterialMap(grid, default=TISSUE_LIBRARY['free_space'])
mat_map.add_circle(center=(75,75), radius=55, material=TISSUE_LIBRARY['skull'])
mat_map.add_circle(center=(75,75), radius=51, material=TISSUE_LIBRARY['brain'])
mat_map.add_circle(center=(95,75), radius=6,  material=TISSUE_LIBRARY['tumor'])
Ca, Cb = mat_map.build()

sim = FDTD2D(grid, fields, boundary, sources, Ca=Ca, Cb=Cb)
```

---

## Brain Tumor Detection Example

```bash
python examples/brain_mimo_imaging.py
# Runs 32 FDTD simulations (16 TX healthy + 16 TX with tumor)
# Saves: examples/output/brain_mimo_imaging.png
```

---

## Run Benchmarks

**Local CPU (requires pymeep):**
```bash
conda run -n pymeep python benchmarks/cpu_benchmark.py
# Saves: benchmarks/cpu_results.json
```

**GPU vs CPU vs Meep comparison:**
```bash
# Open notebooks/colab_benchmark.ipynb on Colab with T4 GPU
```

---

## Project Structure

```
cuda-meep/
├── src/core/
│   ├── grid.py          # YeeGrid, CFL enforcement
│   ├── fields.py        # SoA field tensor container (batch-ready)
│   ├── materials.py     # Per-cell εᵣ, σ → Ca/Cb FDTD coefficients
│   ├── sources.py       # Gaussian, Ricker, Sinusoidal sources
│   ├── boundaries.py    # Mur ABC + dual-staggered CPML scaffold
│   └── fdtd2d.py        # 2D TM FDTD engine (Maxwell, Yee leapfrog)
├── src/visualization/
│   └── plot2d.py        # Field snapshots + FuncAnimation
├── examples/
│   ├── basic_2d_wave.py         # Free-space propagation
│   └── brain_mimo_imaging.py    # Brain tumor detection
├── tests/
│   └── test_fdtd2d.py           # 21 physics + stability tests
├── benchmarks/
│   ├── benchmark_gpu_vs_cpu.py  # Throughput scaling
│   ├── cpu_benchmark.py         # Local Meep comparison
│   └── cpu_results.json         # Pre-measured CPU results
├── notebooks/
│   └── colab_benchmark.ipynb    # GPU vs Meep notebook
└── assets/                      # Plots for README
```

---

## Tests

```bash
pytest tests/test_fdtd2d.py -v
# 21/21 passed
```

Tests cover: CFL stability, Yee staggering, Faraday/Ampere sign correctness,
energy propagation, Mur absorption, batch dimension safety, telemetry accuracy.

---

## Roadmap

- [ ] CPML full implementation (psi field updates)
- [ ] TFSF plane wave source (far-field radar scenarios)
- [ ] Near-to-far field transform
- [ ] 3D extension (Ez, Hx, Hy → full 3D TM/TE)
- [ ] Differentiable physics (adjoint method for inverse design)
- [ ] Multi-GPU domain decomposition
- [ ] Triton kernel fusion for additional 2× speedup

---

## License

MIT
