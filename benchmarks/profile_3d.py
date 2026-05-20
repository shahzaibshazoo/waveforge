"""
profile_3d.py — torch.profiler analysis of WaveForge 3D FDTD.

Runs 20 profiled time steps on a 48^3 grid and prints operator-level
breakdowns sorted by time and by memory.  On CUDA builds a Chrome trace
JSON is also exported to the same directory.

Usage:
    python benchmarks/profile_3d.py
"""

import sys
import time
import math
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.fdtd3d import FDTD3D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
N: int = 48          # grid cells per side — 48^3 = 110 592 cells
N_PROFILE: int = 20  # steps captured by the profiler
N_WARMUP: int = 5    # steps run before attaching the profiler
DX: float = 1e-3     # 1 mm isotropic cell spacing
FREQ: float = 1e9    # 1 GHz source


# ---------------------------------------------------------------------------
# Simulation factory
# ---------------------------------------------------------------------------

def build_sim(n_steps_hint: int) -> FDTD3D:
    """Construct a full 3D FDTD simulation on DEVICE.

    Parameters
    ----------
    n_steps_hint : int
        Total step count used to pre-compute the source waveform tensor.

    Returns
    -------
    FDTD3D
        Ready-to-step simulation object.
    """
    grid = YeeGrid(N, N, dx=DX, dy=DX, Nz=N, dz=DX, device=DEVICE)
    fields = FieldSet(grid)

    boundary = MurABC3D(
        grid,
        fields.Hx,
        fields.Hy,
        fields.Hz,
    )

    pulse = GaussianPulse(amplitude=1.0, freq=FREQ)

    source = PointSource(
        waveform=pulse,
        i=N // 2,
        j=N // 2,
        component="Ez",
        k=N // 2,
        grid=grid,
        N_steps=n_steps_hint,
    )

    sources = SourceCollection([source])

    sim = FDTD3D(
        grid=grid,
        fields=fields,
        boundary=boundary,
        sources=sources,
    )
    return sim


# ---------------------------------------------------------------------------
# Main profiling routine
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"WaveForge 3D FDTD — torch.profiler analysis")
    print(f"  grid      : {N}^3 = {N**3:,} cells")
    print(f"  device    : {DEVICE}")
    print(f"  warmup    : {N_WARMUP} steps (unprofiled)")
    print(f"  profiled  : {N_PROFILE} steps")

    total_steps = N_WARMUP + N_PROFILE
    sim = build_sim(n_steps_hint=total_steps)

    # --- warmup (not profiled) -------------------------------------------
    print("\nRunning warmup steps...")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            sim.step()

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # --- profiler setup --------------------------------------------------
    activities = [torch.profiler.ProfilerActivity.CPU]
    if DEVICE == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # --- profiled run ----------------------------------------------------
    print(f"Profiling {N_PROFILE} steps...")
    t_wall_start = time.perf_counter()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(N_PROFILE):
                sim.step()
                prof.step()

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    t_wall = time.perf_counter() - t_wall_start
    cells_per_sec = (N_PROFILE * N**3) / t_wall / 1e6
    print(f"Wall time for {N_PROFILE} steps: {t_wall*1000:.1f} ms  "
          f"({cells_per_sec:.2f} Mcells/s)")

    # --- results tables --------------------------------------------------
    sort_key = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"

    print("\n[Top 15 by time]")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))

    print("\n[Top 10 by memory usage]")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    # --- top-3 bottleneck summary ----------------------------------------
    top3 = sorted(
        prof.key_averages(),
        key=lambda x: getattr(x, sort_key),
        reverse=True,
    )[:3]

    print("\nTop-3 bottlenecks:")
    for i, evt in enumerate(top3):
        val_ms = getattr(evt, sort_key) / 1000.0  # µs -> ms
        print(f"  {i + 1}. {evt.key:40s}  {val_ms:.3f} ms total")

    print(f"\nProfile complete. Grid: {N}^3, device={DEVICE}, "
          f"{N_PROFILE} steps profiled.")

    # --- optional Chrome trace export ------------------------------------
    if DEVICE == "cuda":
        trace_path = str(Path(__file__).parent / "profile_trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"Chrome trace saved: {trace_path}")


if __name__ == "__main__":
    main()
