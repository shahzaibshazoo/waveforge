"""
benchmark_gpu_vs_cpu.py — FDTD2D throughput benchmark for GPU-MEEP.

Measures Mcells/s across grid sizes on CPU and GPU (when available).
Produces an aligned results table and saves it to benchmarks/results.txt.
"""

import sys
import time
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.fdtd2d import FDTD2D

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

GRID_SIZES = [64, 128, 256, 512]   # Nx == Ny for each run
N_WARMUP   = 20                     # warmup steps (not timed)
N_STEPS    = 200                    # timed steps
DX = DY    = 1e-3                   # 1 mm cell spacing


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def run_benchmark(
    Nx: int,
    Ny: int,
    device: str,
    n_warmup: int,
    n_steps: int,
) -> dict:
    """Build a full simulation and measure FDTD2D throughput.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions (cells).
    device : str
        ``"cpu"`` or ``"cuda"``.
    n_warmup : int
        Steps run before timing begins (allows JIT / kernel compilation to settle).
    n_steps : int
        Steps that are timed.

    Returns
    -------
    dict
        Keys: device, Nx, Ny, n_steps, elapsed_s, mcells_per_second,
        ms_per_step, vram_mb.
    """
    total_steps = n_warmup + n_steps

    grid = YeeGrid(Nx, Ny, DX, DY, device=device)
    fields = FieldSet(grid)
    boundary = MurABC(grid, fields.Hz)

    cx, cy = Nx // 2, Ny // 2
    pulse = GaussianPulse(amplitude=1.0, freq=30e9)
    src = PointSource(pulse, cx, cy, "Hz", grid=grid, N_steps=total_steps)
    sources = SourceCollection([src])

    sim = FDTD2D(grid, fields, boundary, sources)

    # Warmup — lets CUDA JIT compilation finish before we start the clock.
    sim.run(n_warmup, verbose=False)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    sim.run(n_steps, verbose=False)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    elapsed = t1 - t0

    mcells_per_second = (n_steps * Nx * Ny) / elapsed / 1e6
    ms_per_step = elapsed / n_steps * 1e3
    # 3 active float32 tensors (Ex, Ey, Hz), each Nx*Ny*1 cells * 4 bytes
    vram_mb = Nx * Ny * 3 * 4 / 1e6

    return {
        "device": device,
        "Nx": Nx,
        "Ny": Ny,
        "n_steps": n_steps,
        "elapsed_s": elapsed,
        "mcells_per_second": mcells_per_second,
        "ms_per_step": ms_per_step,
        "vram_mb": vram_mb,
    }


# ---------------------------------------------------------------------------
# Results table printer
# ---------------------------------------------------------------------------

def print_results_table(results: list) -> str:
    """Print a box-drawing aligned benchmark results table.

    When both CPU and GPU results exist for the same grid size, a speedup
    column is appended.

    Parameters
    ----------
    results : list[dict]
        List of dicts returned by :func:`run_benchmark`.
    """
    # Build speedup lookup: Nx -> gpu_mcells / cpu_mcells
    by_size: dict[int, dict] = {}
    for r in results:
        by_size.setdefault(r["Nx"], {})[r["device"]] = r

    has_speedup = any(
        "cpu" in devs and "cuda" in devs for devs in by_size.values()
    )

    if has_speedup:
        header = (
            "╔══════╦"
            "════════╦"
            "══════════╦"
            "══════════╦"
            "══════════╦"
            "══════════╗"
        )
        title = (
            "║ Grid ║ Device  ║"
            " Mcells/s  ║  ms/step  ║  VRAM MB  ║  Speedup  ║"
        )
        sep = (
            "╠══════╬"
            "════════╬"
            "══════════╬"
            "══════════╬"
            "══════════╬"
            "══════════╣"
        )
        footer = (
            "╚══════╩"
            "════════╩"
            "══════════╩"
            "══════════╩"
            "══════════╩"
            "══════════╝"
        )
    else:
        header = (
            "╔══════╦"
            "════════╦"
            "══════════╦"
            "══════════╦"
            "══════════╗"
        )
        title = (
            "║ Grid ║ Device  ║"
            " Mcells/s  ║  ms/step  ║  VRAM MB  ║"
        )
        sep = (
            "╠══════╬"
            "════════╬"
            "══════════╬"
            "══════════╬"
            "══════════╣"
        )
        footer = (
            "╚══════╩"
            "════════╩"
            "══════════╩"
            "══════════╩"
            "══════════╝"
        )

    lines = [header, title, sep]

    for r in results:
        Nx = r["Nx"]
        grid_label = f"{Nx}²"
        speedup_col = ""
        if has_speedup:
            devs = by_size.get(Nx, {})
            if r["device"] == "cuda" and "cpu" in devs:
                ratio = r["mcells_per_second"] / devs["cpu"]["mcells_per_second"]
                speedup_col = f"║ {ratio:>7.1f}x  "
            else:
                speedup_col = "║    ---    "

        if has_speedup:
            row = (
                f"║ {grid_label:<4} ║ {r['device']:<6}  ║"
                f" {r['mcells_per_second']:>8.1f}  ║"
                f" {r['ms_per_step']:>8.2f}  ║"
                f" {r['vram_mb']:>8.3f}  "
                f"{speedup_col}║"
            )
        else:
            row = (
                f"║ {grid_label:<4} ║ {r['device']:<6}  ║"
                f" {r['mcells_per_second']:>8.1f}  ║"
                f" {r['ms_per_step']:>8.2f}  ║"
                f" {r['vram_mb']:>8.3f}  ║"
            )
        lines.append(row)

    lines.append(footer)

    output = "\n".join(lines)
    print(output)
    return output


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full benchmark suite and print/save results."""
    print("GPU-MEEP Benchmark -- FDTD2D Throughput")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    results = []
    for device in devices:
        for N in GRID_SIZES:
            print(f"  Benchmarking {N}×{N} on {device}...")
            result = run_benchmark(N, N, device, N_WARMUP, N_STEPS)
            results.append(result)

    print()
    table_str = print_results_table(results)

    # Speedup summary when both CPU and GPU data are present
    if len(devices) == 2:
        print()
        print("GPU speedup summary:")
        by_size: dict[int, dict] = {}
        for r in results:
            by_size.setdefault(r["Nx"], {})[r["device"]] = r
        for N in GRID_SIZES:
            devs = by_size.get(N, {})
            if "cpu" in devs and "cuda" in devs:
                ratio = devs["cuda"]["mcells_per_second"] / devs["cpu"]["mcells_per_second"]
                print(f"  {N}²:  {ratio:.1f}× GPU speedup")

    # Save results to benchmarks/results.txt
    results_path = Path(__file__).parent / "results.txt"
    with open(results_path, "w", encoding="utf-8") as fh:
        fh.write("GPU-MEEP Benchmark -- FDTD2D Throughput\n\n")
        fh.write(table_str + "\n")
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
