# CUDA-MEEP: GPU-Native FDTD Electromagnetic Simulation Engine

A high-performance 2D TM FDTD electromagnetic simulator built with PyTorch/CUDA.  
Designed as a GPU-native alternative to Meep for microwave imaging, MIMO radar, and inverse scattering applications.

## Features

- GPU-accelerated FDTD via PyTorch tensor operations (no Python loops)
- 2D TM mode: {Ex, Ey, Hz} with correct Yee-grid staggering
- Lossy dispersive materials (per-cell ε, σ coefficients)
- Mur absorbing boundary conditions
- MIMO circular array imaging
- Brain tumor detection example (microwave imaging)
- Batch simulation support (multiple simultaneous runs)
- Real-time visualization and animation

## Quickstart (Google Colab)

Open the benchmark notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shahzaibshazoo/cuda-meep/blob/main/notebooks/colab_benchmark.ipynb)

## Installation

```bash
git clone https://github.com/shahzaibshazoo/cuda-meep.git
cd cuda-meep
pip install torch numpy matplotlib pytest
```

## Basic Usage

```python
import sys
sys.path.insert(0, 'src')

from core import YeeGrid, FieldSet, MurABC, GaussianPulse, PointSource, SourceCollection, FDTD2D

# Build grid (128x128, 1mm spacing)
grid = YeeGrid(128, 128, dx=1e-3, dy=1e-3, device='cuda')

# Initialize fields and boundary
fields = FieldSet(grid)
boundary = MurABC(grid, fields.Hz)

# Add Gaussian pulse source at center
pulse = GaussianPulse(amplitude=1.0, sigma=30*grid.dt)
src = PointSource(pulse, i=64, j=64, component='Hz', grid=grid, N_steps=1000)
sources = SourceCollection([src])

# Run simulation
sim = FDTD2D(grid, fields, boundary, sources)
sim.run(1000, verbose=True)

print(f"Throughput: {sim.mcells_per_second:.1f} Mcells/s")
```

## Brain Tumor Detection (MIMO)

```python
python examples/brain_mimo_imaging.py
```

Runs a 16-antenna MIMO circular array microwave imaging simulation.  
Detects a 12mm tumor in a brain phantom using delay-and-sum backprojection.  
Saves results to `examples/output/brain_mimo_imaging.png`.

## Project Structure

```
cuda-meep/
├── src/
│   ├── core/
│   │   ├── grid.py          # Yee grid, CFL enforcement
│   │   ├── fields.py        # Field tensor container
│   │   ├── materials.py     # Per-cell ε, σ → Ca/Cb coefficients
│   │   ├── sources.py       # Gaussian, Ricker, Sinusoidal sources
│   │   ├── boundaries.py    # Mur ABC + CPML scaffold
│   │   └── fdtd2d.py        # 2D TM FDTD engine
│   └── visualization/
│       └── plot2d.py        # Field visualization + animation
├── examples/
│   ├── basic_2d_wave.py     # Simple propagation demo
│   └── brain_mimo_imaging.py # Brain tumor detection
├── tests/
│   └── test_fdtd2d.py       # 21 physics + stability tests
├── benchmarks/
│   └── benchmark_gpu_vs_cpu.py
└── notebooks/
    └── colab_benchmark.ipynb # GPU vs CPU vs Meep comparison
```

## Performance

On NVIDIA A100 (FP32):

| Grid    | Throughput     |
|---------|----------------|
| 128²    | ~1,400 steps/s |
| 256²    | ~340 steps/s   |
| 512²    | ~85 steps/s    |

**Expected 30-80× speedup over CPU-based Meep** for equivalent problems.

## Tests

```bash
pytest tests/test_fdtd2d.py -v   # 21/21 tests
```

## License

MIT
