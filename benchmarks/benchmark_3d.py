"""
benchmark_3d.py — Comprehensive GPU benchmark suite for WaveForge 3D FDTD.

Benchmarks:
  1. Grid scaling — free-space throughput across 32³ to 128³ on the best device.
  2. torch.compile speedup — baseline vs compiled step on 64³.
  3. Material path overhead — free-space vs dielectric sphere on 48³.

Results are printed to stdout and saved to benchmark_3d_results.json.
"""

import sys
import time
import json
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.fdtd3d import FDTD3D
from core.materials import Material, MaterialMap3D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZES: list[int] = [32, 48, 64, 96, 128]
N_WARMUP: int = 10
N_STEPS: int = 200

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device_label() -> str:
    """Return a human-readable device string including GPU model if available."""
    if DEVICE.startswith("cuda"):
        name = torch.cuda.get_device_name(0)
        return f"cuda ({name})"
    return "cpu"


def _memory_mb(N: int) -> float:
    """Field memory in MB: 6 float32 tensors of N³ cells."""
    return 6 * N * N * N * 4 / 1e6


def _sync() -> None:
    """Synchronize CUDA device if running on GPU."""
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def _build_sim(N: int, n_total: int) -> FDTD3D:
    """Construct a free-space FDTD3D simulation on a uniform N³ grid."""
    grid = YeeGrid(N, N, dx=1e-3, dy=1e-3, Nz=N, dz=1e-3, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)
    sigma = 20 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    src = PointSource(pulse, N // 2, N // 2, "Ez", k=N // 2, grid=grid, N_steps=n_total)
    sources = SourceCollection([src])
    return FDTD3D(grid, fields, boundary, sources, n_check=1000)


def _time_run(sim: FDTD3D, n_steps: int) -> float:
    """Run *n_steps* with proper CUDA synchronization and return wall time in s."""
    _sync()
    t0 = time.perf_counter()
    sim.run(n_steps)
    _sync()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Benchmark 1: Grid scaling
# ---------------------------------------------------------------------------


def bench_grid_scaling() -> list[dict]:
    """Measure free-space throughput across GRID_SIZES on DEVICE."""
    print("[Benchmark 1: Grid Scaling]")
    header = f"{'Grid':<10} {'Mcells/s':<12} {'Memory(MB)':<12} {'Steps':<8} {'Time(s)'}"
    print(header)

    results: list[dict] = []
    for N in GRID_SIZES:
        n_total = N_WARMUP + N_STEPS
        sim = _build_sim(N, n_total)

        # Warmup — not timed
        sim.run(N_WARMUP)

        # Timed run
        elapsed = _time_run(sim, N_STEPS)
        mcells = (N_STEPS * N * N * N) / elapsed / 1e6
        mem_mb = _memory_mb(N)

        label = f"{N}³"   # e.g. "32³"
        print(
            f"{label:<10} {mcells:<12.1f} {mem_mb:<12.1f} {N_STEPS:<8} {elapsed:.2f}"
        )
        results.append({"grid": N, "mcells_per_s": mcells, "memory_mb": mem_mb})

    print()
    return results


# ---------------------------------------------------------------------------
# Benchmark 2: torch.compile speedup
# ---------------------------------------------------------------------------


def bench_compile_speedup() -> dict:
    """Compare baseline vs torch.compile step on 64³."""
    N = 64
    print("[Benchmark 2: torch.compile speedup on 64³]")

    # -- Baseline --
    n_total = N_WARMUP + N_STEPS
    sim_base = _build_sim(N, n_total)
    sim_base.run(N_WARMUP)
    elapsed_base = _time_run(sim_base, N_STEPS)
    mcells_base = (N_STEPS * N * N * N) / elapsed_base / 1e6

    # -- Compiled --
    mcells_compiled: float | None = None
    speedup: float | None = None
    compiled_note: str = ""

    try:
        sim_comp = _build_sim(N, n_total + 5)
        sim_comp.compile_step()
        # Warmup to let compilation settle
        sim_comp.run(5)
        elapsed_comp = _time_run(sim_comp, N_STEPS)
        mcells_compiled = (N_STEPS * N * N * N) / elapsed_comp / 1e6
        speedup = mcells_compiled / mcells_base
        print(f"Baseline:  {mcells_base:.1f} Mcells/s")
        print(f"Compiled:  {mcells_compiled:.1f} Mcells/s  (speedup: {speedup:.2f}x)")
    except AttributeError:
        compiled_note = "compile_step() not yet implemented — skipped"
        print(f"Baseline:  {mcells_base:.1f} Mcells/s")
        print(f"Compiled:  {compiled_note}")
    except Exception as exc:
        compiled_note = f"compile_step() raised {type(exc).__name__}: {exc} — skipped"
        print(f"Baseline:  {mcells_base:.1f} Mcells/s")
        print(f"Compiled:  {compiled_note}")

    print()
    return {
        "baseline": mcells_base,
        "compiled": mcells_compiled,
        "speedup": speedup,
        "note": compiled_note,
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Material path overhead
# ---------------------------------------------------------------------------


def bench_material_overhead() -> dict:
    """Compare free-space vs dielectric sphere on 48³."""
    N = 48
    print("[Benchmark 3: Material path overhead on 48³]")

    n_total = N_WARMUP + N_STEPS
    radius = N // 6   # sphere radius in cells

    # -- Free-space --
    sim_free = _build_sim(N, n_total)
    sim_free.run(N_WARMUP)
    elapsed_free = _time_run(sim_free, N_STEPS)
    mcells_free = (N_STEPS * N * N * N) / elapsed_free / 1e6

    # -- With dielectric sphere (eps_r=4) --
    grid = YeeGrid(N, N, dx=1e-3, dy=1e-3, Nz=N, dz=1e-3, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)
    sigma_pulse = 20 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma_pulse)
    src = PointSource(
        pulse, N // 2, N // 2, "Ez", k=N // 2, grid=grid, N_steps=n_total
    )
    sources = SourceCollection([src])

    mat_eps4 = Material("dielectric", eps_r=4.0, sigma=0.0)
    mm = MaterialMap3D(grid)
    mm.add_sphere(center=(N // 2, N // 2, N // 2), radius=radius, material=mat_eps4)
    Ca, Cb = mm.build3d()

    sim_mat = FDTD3D(grid, fields, boundary, sources, Ca=Ca, Cb=Cb, n_check=1000)
    sim_mat.run(N_WARMUP)
    elapsed_mat = _time_run(sim_mat, N_STEPS)
    mcells_mat = (N_STEPS * N * N * N) / elapsed_mat / 1e6

    overhead_pct = (mcells_free - mcells_mat) / mcells_free * 100.0

    print(f"Free-space: {mcells_free:.1f} Mcells/s")
    print(f"Materials:  {mcells_mat:.1f} Mcells/s  (overhead: {overhead_pct:.1f}%)")
    print()

    return {
        "free_space": mcells_free,
        "materials": mcells_mat,
        "overhead_pct": overhead_pct,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all three benchmarks, print results, and save to JSON."""
    sep = "=" * 60
    print(sep)
    print("WaveForge 3D — GPU Benchmark Suite")
    print(f"Device: {_device_label()} | torch {torch.__version__}")
    print(sep)
    print()

    scaling = bench_grid_scaling()
    compile_result = bench_compile_speedup()
    material_result = bench_material_overhead()

    best_mcells = max(r["mcells_per_s"] for r in scaling)
    print(f"WAVEFORGE_BENCH_3D: {best_mcells:.1f} Mcells/s (best grid)")
    print(sep)

    # Save JSON results
    output = {
        "device": DEVICE,
        "device_label": _device_label(),
        "torch_version": torch.__version__,
        "grid_scaling": scaling,
        "compile_speedup": compile_result,
        "material_overhead": material_result,
    }
    results_path = Path(__file__).parent / "benchmark_3d_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
