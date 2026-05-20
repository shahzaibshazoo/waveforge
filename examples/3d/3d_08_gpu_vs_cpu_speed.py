"""
3d_08_gpu_vs_cpu_speed.py — WaveForge 3D GPU vs CPU speed benchmark.

Measures computational throughput (Mcells/s) for the WaveForge 3D FDTD engine
across multiple grid sizes on CPU, GPU (when CUDA is available), and a pure
NumPy reference implementation that approximates PyMEEP-level CPU performance.

Run:  python examples/3d/3d_08_gpu_vs_cpu_speed.py
Out:  examples/output/3d_08_gpu_vs_cpu_speed.png
      examples/output/3d_08_gpu_vs_cpu_results.json
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.fdtd3d import FDTD3D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent / "output"
GRID_SIZES = [24, 32, 48, 64]
N_WARMUP = 10
N_TIMED = 100
DX = 1.5e-3          # 1.5 mm — matches other 3D examples
NUMPY_GRID_SIZE = 32
NUMPY_N_STEPS = 50

# Performance target lines (Mcells/s) for reference annotation
TARGET_T4_128 = 200.0   # T4 GPU target at 128^3
TARGET_T4_64 = 500.0    # T4 GPU target at 64^3


# ---------------------------------------------------------------------------
# Helper: build and warm up a WaveForge FDTD3D simulation
# ---------------------------------------------------------------------------

def _build_sim(N: int, device: str) -> FDTD3D:
    """Construct a minimal free-space FDTD3D simulation of size N^3.

    Parameters
    ----------
    N : int
        Cubic grid size (N x N x N cells).
    device : str
        PyTorch device string: "cpu" or "cuda".

    Returns
    -------
    FDTD3D
        Freshly constructed simulation, fields zeroed, ready to step.
    """
    grid = YeeGrid(N, N, dx=DX, dy=DX, Nz=N, dz=DX, device=device)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    sigma = 20 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    cx = cy = cz = N // 2
    total_steps = N_WARMUP + N_TIMED + 1
    src = PointSource(pulse, cx, cy, "Ez", k=cz, grid=grid, N_steps=total_steps)
    sources = SourceCollection([src])

    return FDTD3D(grid, fields, boundary, sources, n_check=N_WARMUP + N_TIMED + 1)


def _bench_waveforge(N: int, device: str) -> float:
    """Benchmark WaveForge FDTD3D at grid size N^3 on *device*.

    Runs N_WARMUP steps (discarded) then N_TIMED steps with precise wall-clock
    timing. CUDA synchronisation is applied before and after the timed block to
    account for asynchronous kernel dispatch.

    Parameters
    ----------
    N : int
        Cubic grid size.
    device : str
        "cpu" or "cuda".

    Returns
    -------
    float
        Throughput in Mcells/s over the timed block.
    """
    sim = _build_sim(N, device)
    is_cuda = device == "cuda"

    # Warmup — not timed
    with torch.no_grad():
        for _ in range(N_WARMUP):
            sim.step()

    # Flush any pending GPU work before starting the clock
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(N_TIMED):
            sim.step()

    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    cells_per_step = N ** 3
    return N_TIMED * cells_per_step / elapsed / 1e6


# ---------------------------------------------------------------------------
# Part 3: Pure NumPy reference FDTD
# ---------------------------------------------------------------------------

def numpy_fdtd_bench(N: int = NUMPY_GRID_SIZE, n_steps: int = NUMPY_N_STEPS) -> float:
    """Pure NumPy FDTD — TM mode (Ez, Hx, Hy, Hz). Reference baseline.

    Implements a minimal 3D FDTD using only NumPy array slicing, with four
    field components.  This approximates PyMEEP-level (C-backed) throughput
    but runs in pure Python/NumPy, so it represents the slower end of
    CPU-native FDTD implementations.

    Parameters
    ----------
    N : int
        Cubic grid size (default 32).
    n_steps : int
        Number of FDTD steps to time (default 50).

    Returns
    -------
    float
        Throughput in Mcells/s.
    """
    dx = DX
    dt = 0.99 * dx / (3e8 * np.sqrt(3))
    Dh = dt / (4 * np.pi * 1e-7)
    De = dt / 8.854e-12

    Ez = np.zeros((N, N, N), dtype=np.float32)
    Hx = np.zeros((N, N, N), dtype=np.float32)
    Hy = np.zeros((N, N, N), dtype=np.float32)
    Hz = np.zeros((N, N, N), dtype=np.float32)

    t0 = time.perf_counter()
    for step in range(n_steps):
        # H-field updates (Faraday)
        Hx[:, :-1, :-1] += Dh * (
            (Ez[:, 1:, :-1] - Ez[:, :-1, :-1]) / dx
        )
        Hy[:-1, :, :-1] += Dh * (
            (Ez[1:, :, :-1] - Ez[:-1, :, :-1]) / dx
        )
        Hz[:-1, :-1, :] += Dh * (
            -(Hy[1:, :-1, :] - Hy[:-1, :-1, :]) / dx
        )
        # E-field update (Ampere — Ez only in TM mode)
        Ez[1:, 1:, :] += De * (
            (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) / dx
            - (Hx[1:, 1:, :] - Hx[1:, :-1, :]) / dx
        )
        # Soft point source injection
        Ez[N // 2, N // 2, N // 2] += float(
            np.exp(-((step - 30) ** 2) / (2 * 10 ** 2))
        )

    elapsed = time.perf_counter() - t0
    return n_steps * N ** 3 / elapsed / 1e6


# ---------------------------------------------------------------------------
# Part 1 & 2: Run WaveForge benchmarks
# ---------------------------------------------------------------------------

def run_waveforge_benchmarks(device: str) -> dict[int, float]:
    """Run WaveForge throughput benchmarks for all GRID_SIZES on *device*.

    Parameters
    ----------
    device : str
        "cpu" or "cuda".

    Returns
    -------
    dict[int, float]
        Mapping grid size -> Mcells/s.
    """
    results: dict[int, float] = {}
    for N in GRID_SIZES:
        try:
            mcells = _bench_waveforge(N, device)
            results[N] = mcells
            print(f"  {N}^3:  {mcells:.1f} Mcells/s")
        except Exception as exc:
            print(f"  {N}^3:  FAILED ({exc})")
            results[N] = 0.0
    return results


# ---------------------------------------------------------------------------
# Part 4: Print report
# ---------------------------------------------------------------------------

def _print_report(
    numpy_mcells: float,
    cpu_results: dict[int, float],
    gpu_results: Optional[dict[int, float]],
) -> None:
    """Print the formatted benchmark report to stdout.

    Parameters
    ----------
    numpy_mcells : float
        NumPy reference throughput (Mcells/s).
    cpu_results : dict[int, float]
        WaveForge CPU results per grid size.
    gpu_results : dict[int, float] or None
        WaveForge GPU results per grid size, or None if CUDA unavailable.
    """
    print()
    print("=" * 40)
    print("WaveForge 3D — GPU vs CPU Speed Benchmark")
    print("=" * 40)
    print(f"NumPy reference ({NUMPY_GRID_SIZE}^3, {NUMPY_N_STEPS} steps):  "
          f"{numpy_mcells:.1f} Mcells/s")
    print()
    print("WaveForge CPU:")
    for N, mc in cpu_results.items():
        ratio = mc / numpy_mcells if numpy_mcells > 0 else float("nan")
        print(f"  {N}^3:  {mc:.1f} Mcells/s  (speedup vs NumPy: {ratio:.1f}x)")

    print()
    if gpu_results is not None:
        print("WaveForge GPU:")
        for N, mc in gpu_results.items():
            ratio = mc / numpy_mcells if numpy_mcells > 0 else float("nan")
            cpu_mc = cpu_results.get(N, 0.0)
            gpu_cpu = mc / cpu_mc if cpu_mc > 0 else float("nan")
            print(f"  {N}^3:  {mc:.1f} Mcells/s  "
                  f"(speedup vs NumPy: {ratio:.1f}x, "
                  f"vs CPU: {gpu_cpu:.1f}x)")
        best_gpu = max(gpu_results.values())
        best_cpu = max(cpu_results.values())
        best = max(best_gpu, best_cpu)
    else:
        print("WaveForge GPU: CUDA not available")
        best = max(cpu_results.values())

    print()
    print(f"WAVEFORGE_BENCH_GPU_VS_CPU: {best:.1f} Mcells/s peak")
    print("=" * 40)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _make_figure(
    numpy_mcells: float,
    cpu_results: dict[int, float],
    gpu_results: Optional[dict[int, float]],
    out_path: Path,
) -> None:
    """Render the 1x2 benchmark comparison figure and save to *out_path*.

    Left panel: grouped throughput bars (NumPy-ref, WaveForge-CPU, GPU).
    Right panel: speedup vs NumPy reference across grid sizes.

    Parameters
    ----------
    numpy_mcells : float
        NumPy reference throughput (Mcells/s), shown only at 32^3 bar group.
    cpu_results : dict[int, float]
        WaveForge CPU Mcells/s per grid size.
    gpu_results : dict[int, float] or None
        WaveForge GPU Mcells/s per grid size, or None.
    out_path : Path
        Destination PNG file path.
    """
    sizes = GRID_SIZES
    n = len(sizes)
    x = np.arange(n)

    cpu_vals = np.array([cpu_results.get(N, 0.0) for N in sizes])
    gpu_vals = (
        np.array([gpu_results.get(N, 0.0) for N in sizes])
        if gpu_results is not None
        else None
    )

    # numpy reference only meaningful at 32^3 — shown as single bar
    numpy_bar = np.zeros(n, dtype=np.float64)
    numpy_idx = sizes.index(NUMPY_GRID_SIZE) if NUMPY_GRID_SIZE in sizes else None
    if numpy_idx is not None:
        numpy_bar[numpy_idx] = numpy_mcells

    # --- colour palette (colour-blind safe) ---
    C_NUMPY = "#6baed6"   # muted blue
    C_CPU   = "#fd8d3c"   # orange
    C_GPU   = "#74c476"   # green

    n_groups = 2 + (1 if gpu_vals is not None else 0)
    width = 0.7 / n_groups

    fig, (ax_bar, ax_speed) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "WaveForge 3D — GPU vs CPU vs NumPy Reference",
        fontsize=13, fontweight="bold",
    )

    # ---- Left: throughput bars ----
    offsets = np.linspace(-(n_groups - 1) * width / 2,
                          (n_groups - 1) * width / 2,
                          n_groups)

    ax_bar.bar(x + offsets[0], numpy_bar, width,
               label=f"NumPy ref ({NUMPY_GRID_SIZE}^3 only)",
               color=C_NUMPY, alpha=0.85, zorder=3)
    ax_bar.bar(x + offsets[1], cpu_vals, width,
               label="WaveForge CPU", color=C_CPU, alpha=0.85, zorder=3)
    if gpu_vals is not None:
        ax_bar.bar(x + offsets[2], gpu_vals, width,
                   label="WaveForge GPU", color=C_GPU, alpha=0.85, zorder=3)

    # Target lines
    ax_bar.axhline(TARGET_T4_128, color="#9e9ac8", linestyle="--", linewidth=1.2,
                   label=f"T4 target 128^3 ({TARGET_T4_128:.0f} Mcells/s)")
    ax_bar.axhline(TARGET_T4_64, color="#de2d26", linestyle="--", linewidth=1.2,
                   label=f"T4 target 64^3 ({TARGET_T4_64:.0f} Mcells/s)")

    ax_bar.set_yscale("log")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{N}^3" for N in sizes])
    ax_bar.set_xlabel("Grid size")
    ax_bar.set_ylabel("Throughput (Mcells/s, log scale)")
    ax_bar.set_title("Throughput by Grid Size")
    ax_bar.legend(fontsize=8, loc="upper left")
    ax_bar.grid(axis="y", which="both", linestyle=":", alpha=0.5, zorder=0)
    ax_bar.set_axisbelow(True)

    # Annotate bar tops with numeric values
    for i, (nb, cb) in enumerate(zip(numpy_bar, cpu_vals)):
        if nb > 0:
            ax_bar.text(i + offsets[0], nb * 1.08, f"{nb:.0f}",
                        ha="center", va="bottom", fontsize=7, color=C_NUMPY)
        if cb > 0:
            ax_bar.text(i + offsets[1], cb * 1.08, f"{cb:.0f}",
                        ha="center", va="bottom", fontsize=7, color=C_CPU)
    if gpu_vals is not None:
        for i, gb in enumerate(gpu_vals):
            if gb > 0:
                ax_bar.text(i + offsets[2], gb * 1.08, f"{gb:.0f}",
                            ha="center", va="bottom", fontsize=7, color=C_GPU)

    # ---- Right: speedup vs NumPy ----
    if numpy_mcells > 0:
        cpu_speedup = cpu_vals / numpy_mcells
    else:
        cpu_speedup = np.zeros(n)

    ax_speed.plot(x, cpu_speedup, "o-", color=C_CPU, linewidth=2,
                  markersize=6, label="WaveForge CPU / NumPy")

    if gpu_vals is not None and numpy_mcells > 0:
        gpu_speedup = gpu_vals / numpy_mcells
        ax_speed.plot(x, gpu_speedup, "s-", color=C_GPU, linewidth=2,
                      markersize=6, label="WaveForge GPU / NumPy")

    ax_speed.axhline(1.0, color=C_NUMPY, linestyle="--", linewidth=1.0,
                     label="NumPy baseline (1x)")

    ax_speed.set_xticks(x)
    ax_speed.set_xticklabels([f"{N}^3" for N in sizes])
    ax_speed.set_xlabel("Grid size")
    ax_speed.set_ylabel("Speedup vs NumPy reference")
    ax_speed.set_title("Speedup vs NumPy Reference (CPU baseline)")
    ax_speed.legend(fontsize=9)
    ax_speed.grid(linestyle=":", alpha=0.5)

    # Annotate speedup values on the lines
    for i, sv in enumerate(cpu_speedup):
        ax_speed.annotate(f"{sv:.1f}x", (i, sv),
                          textcoords="offset points", xytext=(0, 8),
                          ha="center", fontsize=8, color=C_CPU)
    if gpu_vals is not None and numpy_mcells > 0:
        for i, sv in enumerate(gpu_speedup):
            ax_speed.annotate(f"{sv:.1f}x", (i, sv),
                              textcoords="offset points", xytext=(0, -14),
                              ha="center", fontsize=8, color=C_GPU)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out_path}")


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------

def _save_json(
    numpy_mcells: float,
    cpu_results: dict[int, float],
    gpu_results: Optional[dict[int, float]],
    cuda_available: bool,
    out_path: Path,
) -> None:
    """Persist all benchmark results to a JSON file.

    Parameters
    ----------
    numpy_mcells : float
        NumPy reference throughput (Mcells/s).
    cpu_results : dict[int, float]
        WaveForge CPU results.
    gpu_results : dict[int, float] or None
        WaveForge GPU results, or None.
    cuda_available : bool
        Whether CUDA was detected at runtime.
    out_path : Path
        Destination JSON file path.
    """
    payload = {
        "benchmark": "WaveForge 3D GPU vs CPU Speed",
        "grid_sizes": GRID_SIZES,
        "n_warmup": N_WARMUP,
        "n_timed": N_TIMED,
        "dx_m": DX,
        "cuda_available": cuda_available,
        "numpy_ref": {
            "grid_size": NUMPY_GRID_SIZE,
            "n_steps": NUMPY_N_STEPS,
            "mcells_per_sec": round(numpy_mcells, 2),
        },
        "waveforge_cpu": {str(k): round(v, 2) for k, v in cpu_results.items()},
        "waveforge_gpu": (
            {str(k): round(v, 2) for k, v in gpu_results.items()}
            if gpu_results is not None
            else None
        ),
        "peak_mcells_per_sec": round(
            max(
                list(cpu_results.values())
                + (list(gpu_results.values()) if gpu_results else [])
            ),
            2,
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved JSON:   {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: run all benchmark parts, print report, save outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    cuda_available = torch.cuda.is_available()

    print("=" * 40)
    print("WaveForge 3D — GPU vs CPU Speed Benchmark")
    print("=" * 40)
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Grid sizes: {GRID_SIZES}")
    print(f"Warmup steps: {N_WARMUP} | Timed steps: {N_TIMED}")
    print(f"dx = {DX * 1e3:.1f} mm")
    print()

    # --- Part 1: WaveForge CPU ---
    print("Part 1 — WaveForge CPU:")
    cpu_results = run_waveforge_benchmarks("cpu")

    # --- Part 2: WaveForge GPU ---
    gpu_results: Optional[dict[int, float]] = None
    if cuda_available:
        print("\nPart 2 — WaveForge GPU:")
        gpu_results = run_waveforge_benchmarks("cuda")
    else:
        print("\nPart 2 — WaveForge GPU: CUDA not available, skipping.")

    # --- Part 3: NumPy reference ---
    print(f"\nPart 3 — NumPy reference FDTD ({NUMPY_GRID_SIZE}^3, {NUMPY_N_STEPS} steps):")
    numpy_mcells = numpy_fdtd_bench(NUMPY_GRID_SIZE, NUMPY_N_STEPS)
    print(f"  NumPy:  {numpy_mcells:.1f} Mcells/s")

    # --- Part 4: Report ---
    _print_report(numpy_mcells, cpu_results, gpu_results)

    # --- Outputs ---
    fig_path = OUTPUT_DIR / "3d_08_gpu_vs_cpu_speed.png"
    json_path = OUTPUT_DIR / "3d_08_gpu_vs_cpu_results.json"

    _make_figure(numpy_mcells, cpu_results, gpu_results, fig_path)
    _save_json(numpy_mcells, cpu_results, gpu_results, cuda_available, json_path)


if __name__ == "__main__":
    main()
