"""
meep_comparison.py — WaveForge vs Meep: per-scenario throughput comparison.

Run with:
    /home/zuu/miniconda3/bin/conda run -n pymeep python benchmarks/meep_comparison.py

Saves:
    benchmarks/meep_comparison_results.json
    benchmarks/meep_comparison.png
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — must happen before any local imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import (
    GaussianPulse,
    RickerWavelet,
    SinusoidalSource,
    PointSource,
    LineSource,
    SourceCollection,
)
from core.materials import Material, MaterialMap
from core.fdtd2d import FDTD2D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WARMUP = 20
N_STEPS = 100
BENCHMARKS_DIR = Path(__file__).parent
C0 = 299_792_458.0  # m/s


# ---------------------------------------------------------------------------
# Typed result container
# ---------------------------------------------------------------------------

class BenchResult:
    """Holds throughput and per-step latency for a single run."""

    def __init__(self, mcells_s: float, ms_step: float) -> None:
        self.mcells_s = mcells_s
        self.ms_step = ms_step

    def to_dict(self) -> dict:
        return {"mcells_s": round(self.mcells_s, 4), "ms_step": round(self.ms_step, 6)}


# ---------------------------------------------------------------------------
# WaveForge benchmark driver
# ---------------------------------------------------------------------------

def bench_cuda_meep(
    Nx: int,
    Ny: int,
    dx: float,
    setup_fn: Callable,
    n_warmup: int = N_WARMUP,
    n_steps: int = N_STEPS,
) -> BenchResult:
    """Run a WaveForge (CPU) scenario and return throughput.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    dx : float
        Cell spacing in metres.
    setup_fn : callable
        ``(grid: YeeGrid) -> (Ca, Cb, SourceCollection)``
        where Ca / Cb may be None for free-space scenarios.
    """
    grid = YeeGrid(Nx, Ny, dx=dx, dy=dx, device="cpu")
    fields = FieldSet(grid)
    boundary = MurABC(grid, fields.Hz)
    ca, cb, sources = setup_fn(grid)

    kwargs: dict = {}
    if ca is not None:
        kwargs["Ca"] = ca
        kwargs["Cb"] = cb

    sim = FDTD2D(grid, fields, boundary, sources, n_check=10_000, **kwargs)
    sim.run(n_warmup)

    t0 = time.perf_counter()
    sim.run(n_steps)
    elapsed = time.perf_counter() - t0

    mcells_s = n_steps * Nx * Ny / elapsed / 1e6
    ms_step = elapsed / n_steps * 1000.0
    return BenchResult(mcells_s, ms_step)


# ---------------------------------------------------------------------------
# Meep benchmark driver
# ---------------------------------------------------------------------------

def bench_meep(
    Nx: int,
    Ny: int,
    dx: float,
    sources_fn: Callable,
    geometry_fn: Optional[Callable] = None,
    n_warmup: int = N_WARMUP,
    n_steps: int = N_STEPS,
) -> BenchResult:
    """Run an equivalent Meep scenario and return throughput.

    Parameters
    ----------
    sources_fn : callable
        ``() -> list[mp.Source]``
    geometry_fn : callable or None
        ``() -> list[mp.GeometricObject]``
    """
    import meep as mp

    os.environ["MEEP_VERBOSITY"] = "0"

    # Meep: 1 unit = 1 m, resolution = cells per unit length
    resolution = int(round(1.0 / dx))
    cell = mp.Vector3(Nx * dx, Ny * dx)

    sources = sources_fn(Nx, Ny, dx)
    geometry = geometry_fn(Nx, Ny, dx) if geometry_fn is not None else []

    absorber_thick = min(Nx, Ny) * dx * 0.1
    pml_layers = [mp.Absorber(absorber_thick)]

    # Meep time step: dt_meep = courant / resolution (in Meep time units)
    # One Meep time unit = 1/c0 seconds when length unit = 1 m → not quite;
    # in Meep unit system with length in metres, 1 time unit = 1/c = 1/1 = 1 s
    # but c=1 in Meep natural units.  The Courant default is 0.5, so:
    #   dt_meep = 0.5 / resolution  [Meep time units]
    # which maps to dt_SI = 0.5 / resolution / c0 seconds.
    courant = 0.5  # Meep default
    t_per_step = courant / resolution  # Meep time units per FDTD step

    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        sources=sources,
        geometry=geometry,
        boundary_layers=pml_layers,
    )

    # Warmup
    sim.run(until=n_warmup * t_per_step)
    sim.reset_meep()

    # Timed run: re-create to get a clean state
    sim2 = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        sources=sources,
        geometry=geometry,
        boundary_layers=pml_layers,
    )
    sim2.run(until=n_warmup * t_per_step)

    t0 = time.perf_counter()
    sim2.run(until=(n_warmup + n_steps) * t_per_step)
    elapsed = time.perf_counter() - t0

    sim2.reset_meep()

    mcells_s = n_steps * Nx * Ny / elapsed / 1e6
    ms_step = elapsed / n_steps * 1000.0
    return BenchResult(mcells_s, ms_step)


# ---------------------------------------------------------------------------
# Scenario 1: free_space — 128x128, Gaussian pulse
# ---------------------------------------------------------------------------

def setup_free_space_cuda(grid: YeeGrid) -> Tuple:
    pulse = GaussianPulse(amplitude=1.0, sigma=40 * grid.dt)
    src = PointSource(pulse, grid.Nx // 2, grid.Ny // 2, "Hz",
                      grid=grid, N_steps=N_WARMUP + N_STEPS)
    return None, None, SourceCollection([src])


def sources_free_space_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    resolution = int(round(1.0 / dx))
    courant = 0.5
    sigma_steps = 40
    sigma_t = sigma_steps * courant / resolution  # Meep time units
    freq = 1.0 / (2.0 * 3.14159 * sigma_t)
    cx = (Nx // 2) * dx - Nx * dx / 2.0
    cy = (Ny // 2) * dx - Ny * dx / 2.0
    return [
        mp.Source(
            mp.GaussianSource(frequency=freq, fwidth=freq),
            component=mp.Hz,
            center=mp.Vector3(cx, cy),
        )
    ]


# ---------------------------------------------------------------------------
# Scenario 2: dielectric_slab — 200x64, eps_r=4
# ---------------------------------------------------------------------------

def setup_dielectric_slab_cuda(grid: YeeGrid) -> Tuple:
    pulse = GaussianPulse(amplitude=1.0, sigma=40 * grid.dt)
    src = LineSource(pulse, axis="y", position=20, start=0, stop=grid.Ny,
                     component="Hz", grid=grid, N_steps=N_WARMUP + N_STEPS)
    glass = Material("glass", eps_r=4.0, sigma=0.0)
    mm = MaterialMap(grid)
    slab_x0 = int(grid.Nx * 0.5)
    slab_x1 = int(grid.Nx * 0.695)
    mm.add_rectangle((slab_x0, slab_x1), (0, grid.Ny - 1), glass)
    ca, cb = mm.build()
    return ca, cb, SourceCollection([src])


def sources_dielectric_slab_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    resolution = int(round(1.0 / dx))
    courant = 0.5
    sigma_t = 40 * courant / resolution
    freq = 1.0 / (2.0 * 3.14159 * sigma_t)
    cx = 20 * dx - Nx * dx / 2.0
    return [
        mp.Source(
            mp.GaussianSource(frequency=freq, fwidth=freq),
            component=mp.Hz,
            center=mp.Vector3(cx, 0),
            size=mp.Vector3(0, Ny * dx),
        )
    ]


def geometry_dielectric_slab_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    slab_x0 = int(Nx * 0.5)
    slab_x1 = int(Nx * 0.695)
    slab_w = (slab_x1 - slab_x0) * dx
    slab_cx = (slab_x0 + (slab_x1 - slab_x0) / 2.0) * dx - Nx * dx / 2.0
    return [
        mp.Block(
            size=mp.Vector3(slab_w, Ny * dx),
            center=mp.Vector3(slab_cx, 0),
            material=mp.Medium(epsilon=4.0),
        )
    ]


# ---------------------------------------------------------------------------
# Scenario 3: waveguide — 200x60, 5 GHz
# ---------------------------------------------------------------------------

def setup_waveguide_cuda(grid: YeeGrid) -> Tuple:
    cw = SinusoidalSource(amplitude=1.0, frequency=5e9)
    src = PointSource(cw, 10, grid.Ny // 2, "Hz",
                      grid=grid, N_steps=N_WARMUP + N_STEPS)
    return None, None, SourceCollection([src])


def sources_waveguide_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    resolution = int(round(1.0 / dx))
    # 5 GHz in Meep natural units: f_meep = f_SI * dx / c0
    freq_meep = 5e9 * dx / C0
    cx = 10 * dx - Nx * dx / 2.0
    cy = (Ny // 2) * dx - Ny * dx / 2.0
    return [
        mp.Source(
            mp.ContinuousSource(frequency=freq_meep),
            component=mp.Hz,
            center=mp.Vector3(cx, cy),
        )
    ]


# ---------------------------------------------------------------------------
# Scenario 4: cylinder_scatter — 150x150, eps_r=9
# ---------------------------------------------------------------------------

def setup_cylinder_scatter_cuda(grid: YeeGrid) -> Tuple:
    pulse = GaussianPulse(amplitude=1.0, sigma=40 * grid.dt)
    src = LineSource(pulse, axis="y", position=10, start=0, stop=grid.Ny,
                     component="Hz", grid=grid, N_steps=N_WARMUP + N_STEPS)
    cyl = Material("cylinder", eps_r=9.0, sigma=0.0)
    mm = MaterialMap(grid)
    mm.add_circle((grid.Nx // 2, grid.Ny // 2), 15, cyl)
    ca, cb = mm.build()
    return ca, cb, SourceCollection([src])


def sources_cylinder_scatter_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    resolution = int(round(1.0 / dx))
    courant = 0.5
    sigma_t = 40 * courant / resolution
    freq = 1.0 / (2.0 * 3.14159 * sigma_t)
    cx = 10 * dx - Nx * dx / 2.0
    return [
        mp.Source(
            mp.GaussianSource(frequency=freq, fwidth=freq),
            component=mp.Hz,
            center=mp.Vector3(cx, 0),
            size=mp.Vector3(0, Ny * dx),
        )
    ]


def geometry_cylinder_scatter_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    r = 15 * dx
    return [
        mp.Cylinder(
            radius=r,
            center=mp.Vector3(0, 0),
            material=mp.Medium(epsilon=9.0),
        )
    ]


# ---------------------------------------------------------------------------
# Scenario 5: through_wall — 200x100, concrete wall
# ---------------------------------------------------------------------------

def setup_through_wall_cuda(grid: YeeGrid) -> Tuple:
    wv = RickerWavelet(amplitude=1.0, peak_freq=1e9)
    src = LineSource(wv, axis="y", position=5, start=0, stop=grid.Ny,
                     component="Hz", grid=grid, N_steps=N_WARMUP + N_STEPS)
    concrete = Material("concrete", eps_r=6.0, sigma=0.05)
    metal = Material("metal", eps_r=8.0, sigma=3.0)
    mm = MaterialMap(grid)
    mm.add_rectangle((40, 55), (0, grid.Ny - 1), concrete)
    mm.add_rectangle((100, 108), (35, 65), metal)
    ca, cb = mm.build()
    return ca, cb, SourceCollection([src])


def sources_through_wall_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    # Ricker at 1 GHz: freq_meep = f_SI * dx / c0
    freq_meep = 1e9 * dx / C0
    cx = 5 * dx - Nx * dx / 2.0
    return [
        mp.Source(
            mp.GaussianSource(frequency=freq_meep, fwidth=freq_meep),
            component=mp.Hz,
            center=mp.Vector3(cx, 0),
            size=mp.Vector3(0, Ny * dx),
        )
    ]


def geometry_through_wall_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    domain_x = Nx * dx
    domain_y = Ny * dx

    # Concrete slab: cells 40-55 in x
    wall_x0 = 40 * dx
    wall_x1 = 55 * dx
    wall_cx = (wall_x0 + wall_x1) / 2.0 - domain_x / 2.0
    wall_w = wall_x1 - wall_x0

    # Metal target: cells 100-108 in x, 35-65 in y
    met_x0 = 100 * dx
    met_x1 = 108 * dx
    met_y0 = 35 * dx
    met_y1 = 65 * dx
    met_cx = (met_x0 + met_x1) / 2.0 - domain_x / 2.0
    met_cy = (met_y0 + met_y1) / 2.0 - domain_y / 2.0
    met_w = met_x1 - met_x0
    met_h = met_y1 - met_y0

    return [
        mp.Block(
            size=mp.Vector3(wall_w, domain_y),
            center=mp.Vector3(wall_cx, 0),
            material=mp.Medium(epsilon=6.0, D_conductivity=0.05),
        ),
        mp.Block(
            size=mp.Vector3(met_w, met_h),
            center=mp.Vector3(met_cx, met_cy),
            material=mp.Medium(epsilon=8.0, D_conductivity=3.0),
        ),
    ]


# ---------------------------------------------------------------------------
# Scenario 6: interference — 128x128, two sources 2.4 GHz
# ---------------------------------------------------------------------------

def setup_interference_cuda(grid: YeeGrid) -> Tuple:
    src1 = PointSource(
        SinusoidalSource(1.0, 2.4e9), 32, 64, "Hz",
        grid=grid, N_steps=N_WARMUP + N_STEPS,
    )
    src2 = PointSource(
        SinusoidalSource(1.0, 2.4e9), 96, 64, "Hz",
        grid=grid, N_steps=N_WARMUP + N_STEPS,
    )
    return None, None, SourceCollection([src1, src2])


def sources_interference_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    freq_meep = 2.4e9 * dx / C0
    cx1 = 32 * dx - Nx * dx / 2.0
    cx2 = 96 * dx - Nx * dx / 2.0
    cy = 64 * dx - Ny * dx / 2.0
    return [
        mp.Source(
            mp.ContinuousSource(frequency=freq_meep),
            component=mp.Hz,
            center=mp.Vector3(cx1, cy),
        ),
        mp.Source(
            mp.ContinuousSource(frequency=freq_meep),
            component=mp.Hz,
            center=mp.Vector3(cx2, cy),
        ),
    ]


# ---------------------------------------------------------------------------
# Scenario 7: tissue_layers — 250x80, bio tissue
# ---------------------------------------------------------------------------

def setup_tissue_layers_cuda(grid: YeeGrid) -> Tuple:
    pulse = GaussianPulse(amplitude=1.0, freq=1e9)
    src = LineSource(pulse, axis="y", position=5, start=0, stop=grid.Ny,
                     component="Hz", grid=grid, N_steps=N_WARMUP + N_STEPS)
    mm = MaterialMap(grid)
    mm.add_rectangle((5, 14),  (0, grid.Ny - 1), Material("skin",   eps_r=40.0, sigma=1.5))
    mm.add_rectangle((15, 29), (0, grid.Ny - 1), Material("fat",    eps_r=5.0,  sigma=0.05))
    mm.add_rectangle((30, 79), (0, grid.Ny - 1), Material("muscle", eps_r=50.0, sigma=1.7))
    ca, cb = mm.build()
    return ca, cb, SourceCollection([src])


def sources_tissue_layers_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    freq_meep = 1e9 * dx / C0
    cx = 5 * dx - Nx * dx / 2.0
    return [
        mp.Source(
            mp.GaussianSource(frequency=freq_meep, fwidth=freq_meep),
            component=mp.Hz,
            center=mp.Vector3(cx, 0),
            size=mp.Vector3(0, Ny * dx),
        )
    ]


def geometry_tissue_layers_meep(Nx: int, Ny: int, dx: float):
    import meep as mp
    domain_x = Nx * dx
    domain_y = Ny * dx

    def block(x0: int, x1: int, eps: float, sigma: float):
        cx = ((x0 + x1) / 2.0) * dx - domain_x / 2.0
        w = (x1 - x0) * dx
        return mp.Block(
            size=mp.Vector3(w, domain_y),
            center=mp.Vector3(cx, 0),
            material=mp.Medium(epsilon=eps, D_conductivity=sigma),
        )

    return [
        block(5,  14,  40.0, 1.5),
        block(15, 29,   5.0, 0.05),
        block(30, 79,  50.0, 1.7),
    ]


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "free_space",
        "Nx": 128, "Ny": 128, "dx": 2e-3,
        "cuda_setup": setup_free_space_cuda,
        "meep_sources": sources_free_space_meep,
        "meep_geometry": None,
        "notes": "128x128, pulse propagation",
    },
    {
        "name": "dielectric_slab",
        "Nx": 200, "Ny": 64, "dx": 1e-3,
        "cuda_setup": setup_dielectric_slab_cuda,
        "meep_sources": sources_dielectric_slab_meep,
        "meep_geometry": geometry_dielectric_slab_meep,
        "notes": "200x64, eps_r=4 slab",
    },
    {
        "name": "waveguide",
        "Nx": 200, "Ny": 60, "dx": 1e-3,
        "cuda_setup": setup_waveguide_cuda,
        "meep_sources": sources_waveguide_meep,
        "meep_geometry": None,
        "notes": "200x60, guided mode 5GHz",
    },
    {
        "name": "cylinder_scatter",
        "Nx": 150, "Ny": 150, "dx": 1e-3,
        "cuda_setup": setup_cylinder_scatter_cuda,
        "meep_sources": sources_cylinder_scatter_meep,
        "meep_geometry": geometry_cylinder_scatter_meep,
        "notes": "150x150, cylinder eps_r=9",
    },
    {
        "name": "through_wall",
        "Nx": 200, "Ny": 100, "dx": 5e-3,
        "cuda_setup": setup_through_wall_cuda,
        "meep_sources": sources_through_wall_meep,
        "meep_geometry": geometry_through_wall_meep,
        "notes": "200x100, concrete wall",
    },
    {
        "name": "interference",
        "Nx": 128, "Ny": 128, "dx": 2e-3,
        "cuda_setup": setup_interference_cuda,
        "meep_sources": sources_interference_meep,
        "meep_geometry": None,
        "notes": "128x128, two sources 2.4GHz",
    },
    {
        "name": "tissue_layers",
        "Nx": 250, "Ny": 80, "dx": 1e-3,
        "cuda_setup": setup_tissue_layers_cuda,
        "meep_sources": sources_tissue_layers_meep,
        "meep_geometry": geometry_tissue_layers_meep,
        "notes": "250x80, bio tissue",
    },
]


# ---------------------------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------------------------

def run_all_scenarios() -> dict:
    """Run every scenario with both engines; return nested results dict."""
    results: dict = {}

    for sc in SCENARIOS:
        name = sc["name"]
        Nx, Ny, dx = sc["Nx"], sc["Ny"], sc["dx"]
        print(f"\n[{name}]  {Nx}x{Ny}  dx={dx:.1e} m")

        # --- WaveForge (CPU) ---
        print(f"  WaveForge (CPU) ... ", end="", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_cm = bench_cuda_meep(Nx, Ny, dx, sc["cuda_setup"])
            print(f"{res_cm.mcells_s:.2f} Mcells/s  ({res_cm.ms_step:.3f} ms/step)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            res_cm = None

        # --- Meep ---
        print(f"  Meep            ... ", end="", flush=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_mp = bench_meep(
                    Nx, Ny, dx,
                    sc["meep_sources"],
                    sc["meep_geometry"],
                )
            print(f"{res_mp.mcells_s:.2f} Mcells/s  ({res_mp.ms_step:.3f} ms/step)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            res_mp = None

        results[name] = {
            "waveforge": res_cm.to_dict() if res_cm is not None else None,
            "meep": res_mp.to_dict() if res_mp is not None else None,
            "grid": f"{Nx}x{Ny}",
            "notes": sc["notes"],
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(results: dict) -> None:
    """Produce a 3-row multi-panel comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    scenario_names = [sc["name"] for sc in SCENARIOS]
    labels = [s.replace("_", "\n") for s in scenario_names]

    cuda_vals = []
    meep_vals = []
    speedups = []

    for name in scenario_names:
        r = results.get(name, {})
        cm = r.get("waveforge") or {}
        mp_r = r.get("meep") or {}
        cv = cm.get("mcells_s", 0.0) or 0.0
        mv = mp_r.get("mcells_s", 0.0) or 0.0
        cuda_vals.append(cv)
        meep_vals.append(mv)
        if mv > 0:
            speedups.append(cv / mv)
        else:
            speedups.append(0.0)

    cuda_arr = np.array(cuda_vals)
    meep_arr = np.array(meep_vals)
    sp_arr = np.array(speedups)

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.55, top=0.94, bottom=0.06)

    # -----------------------------------------------------------------------
    # Row 1: Horizontal bar chart — Mcells/s grouped by scenario
    # -----------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    n = len(scenario_names)
    y = np.arange(n)
    bar_h = 0.35

    bars_cm = ax1.barh(y + bar_h / 2, cuda_arr, height=bar_h,
                       color="#2ecc71", label="WaveForge (CPU)")
    bars_mp = ax1.barh(y - bar_h / 2, meep_arr, height=bar_h,
                       color="#e74c3c", label="Meep (pymeep)")

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Throughput (Mcells/s)", fontsize=10)
    ax1.set_title("Throughput Comparison: WaveForge vs Meep", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    # Speedup labels on right margin
    x_max = max(cuda_arr.max(), meep_arr.max(), 1e-6)
    for i, sp in enumerate(sp_arr):
        if sp > 0:
            colour = "#2ecc71" if sp >= 1.0 else "#e74c3c"
            label = f"{sp:.2f}x"
            ax1.text(x_max * 1.02, i, label, va="center", fontsize=8,
                     color=colour, fontweight="bold")

    ax1.set_xlim(0, x_max * 1.18)

    # -----------------------------------------------------------------------
    # Row 2: Speedup ratio bar chart
    # -----------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(n)
    colours = ["#2ecc71" if s >= 1.0 else "#e74c3c" for s in sp_arr]
    bars = ax2.bar(x, sp_arr, color=colours, edgecolor="white", linewidth=0.5)
    ax2.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="parity (1×)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Speedup (WaveForge / Meep)", fontsize=10)
    ax2.set_title("Speedup Ratio per Scenario", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    for bar, sp in zip(bars, sp_arr):
        if sp > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{sp:.2f}x",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    green_patch = mpatches.Patch(color="#2ecc71", label="WaveForge wins")
    red_patch = mpatches.Patch(color="#e74c3c", label="Meep wins")
    ax2.legend(handles=[green_patch, red_patch], fontsize=9, loc="upper right")

    # -----------------------------------------------------------------------
    # Row 3: Summary table
    # -----------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    col_labels = ["Scenario", "Grid", "WaveForge\n(Mcells/s)", "Meep\n(Mcells/s)",
                  "Speedup", "Notes"]
    table_data = []
    for sc in SCENARIOS:
        name = sc["name"]
        r = results.get(name, {})
        cm = r.get("waveforge") or {}
        mp_r = r.get("meep") or {}
        cv = cm.get("mcells_s", 0.0) or 0.0
        mv = mp_r.get("mcells_s", 0.0) or 0.0
        sp = cv / mv if mv > 0 else float("nan")
        row = [
            name,
            r.get("grid", "?"),
            f"{cv:.2f}" if cv > 0 else "fail",
            f"{mv:.2f}" if mv > 0 else "fail",
            f"{sp:.2f}x" if mv > 0 else "N/A",
            sc["notes"],
        ]
        table_data.append(row)

    tbl = ax3.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Header row styling
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#2c3e50")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for row_idx in range(1, len(table_data) + 1):
        bg = "#f2f2f2" if row_idx % 2 == 0 else "#ffffff"
        for col in range(len(col_labels)):
            tbl[row_idx, col].set_facecolor(bg)

    ax3.set_title("Full Benchmark Summary", fontsize=12, fontweight="bold", pad=4)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out_path = BENCHMARKS_DIR / "meep_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    """Print a formatted summary table to stdout."""
    hdr = (
        f"{'Scenario':<20} {'Grid':<10} {'WaveForge':>12} {'Meep':>12} "
        f"{'Speedup':>10}  Notes"
    )
    sep = "-" * len(hdr)
    print("\n" + sep)
    print(hdr)
    print(sep)

    for sc in SCENARIOS:
        name = sc["name"]
        r = results.get(name, {})
        cm = r.get("waveforge") or {}
        mp_r = r.get("meep") or {}
        cv = cm.get("mcells_s", 0.0) or 0.0
        mv = mp_r.get("mcells_s", 0.0) or 0.0
        sp_str = f"{cv/mv:.2f}x" if mv > 0 else "N/A"
        cv_str = f"{cv:.2f} Mc/s" if cv > 0 else "fail"
        mv_str = f"{mv:.2f} Mc/s" if mv > 0 else "fail"
        print(
            f"{name:<20} {r.get('grid','?'):<10} {cv_str:>12} {mv_str:>12} "
            f"{sp_str:>10}  {sc['notes']}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    print("=" * 60)
    print("  WaveForge vs Meep — Throughput Benchmark")
    print(f"  N_WARMUP={N_WARMUP}  N_STEPS={N_STEPS}")
    print("=" * 60)

    results = run_all_scenarios()

    # Save JSON
    json_path = BENCHMARKS_DIR / "meep_comparison_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved: {json_path}")

    print_summary(results)
    make_figure(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
