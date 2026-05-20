"""
3d_10_comprehensive_benchmark.py — WaveForge 3D comprehensive benchmark suite.

Runs five independent sub-tests that together validate stability, symmetry,
phase accuracy, material handling, and raw throughput.  Each test builds its
own simulation, runs it, and tears it down.  No shared state between tests.

Sub-tests
---------
1. CFL Stability          — 32³ free-space, 300 steps, checks finiteness,
                            field magnitude, and energy monotonicity.
2. Octant Symmetry        — 32³ centred point source, 100 steps, checks that
                            Ez is symmetric about the source cell.
3. Phase Velocity         — 16×16×64 plane wave, 300 steps, fits wavefront
                            velocity and compares to c₀.
4. Material Conservation  — 32³ dielectric sphere, 200 steps, validates Ca/Cb
                            bounds and energy positivity.
5. Throughput             — 48³ grid, 100 steps (10 warmup), Mcells/s.

Output
------
PNG:  examples/output/3d_10_comprehensive_benchmark.png
JSON: examples/output/3d_10_benchmark_results.json

Run:  python examples/3d/3d_10_comprehensive_benchmark.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.boundaries import MurABC3D
from core.fdtd3d import FDTD3D
from core.fields import FieldSet
from core.grid import YeeGrid
from core.materials import Material, MaterialMap3D
from core.sources import (
    GaussianPulse,
    PlaneSource,
    PointSource,
    SourceCollection,
)

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
C0 = 299_792_458.0  # m/s


# ---------------------------------------------------------------------------
# Test 1: CFL Stability
# ---------------------------------------------------------------------------


def _test_cfl_stability() -> tuple[float, list[float], list[float]]:
    """32³ free-space simulation, 300 steps.  Returns (score, energy_history, steps)."""
    NX = NY = NZ = 32
    DX = 1e-3
    N_STEPS = 300

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    sigma = 20.0 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    cx = cy = cz = NX // 2
    src = PointSource(pulse, cx, cy, "Ez", k=cz, grid=grid, N_steps=N_STEPS)
    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]), n_check=100)

    energy_history: list[float] = []
    step_indices: list[float] = []

    with torch.no_grad():
        for step in range(N_STEPS):
            sim.step()
            if step % 30 == 0:
                energy_history.append(fields.total_energy())
                step_indices.append(float(step))

    Ez_np = fields.Ez.detach().cpu().numpy()
    all_finite = bool(np.all(np.isfinite(Ez_np)))
    max_field = float(np.abs(Ez_np).max())
    field_bounded = max_field < 1e6

    # Energy monotonicity: energy should not grow unboundedly after the source
    # has decayed (source decays by step ~5*sigma/dt).  We check that the peak
    # energy in the second half does not exceed 10x the peak in the first half —
    # a loose but reliable indicator of non-divergence.
    half = len(energy_history) // 2
    if half > 0 and max(energy_history[:half]) > 0.0:
        ratio = max(energy_history[half:]) / max(energy_history[:half])
        energy_stable = ratio < 10.0
    else:
        energy_stable = len(energy_history) > 0

    score = sum([float(all_finite), float(field_bounded), float(energy_stable)])
    return score, energy_history, step_indices


# ---------------------------------------------------------------------------
# Test 2: Octant Symmetry
# ---------------------------------------------------------------------------


def _test_octant_symmetry() -> tuple[float, list[float], list[float]]:
    """32³ centred Ez point source, 100 steps.  Returns (score, errors, steps)."""
    NX = NY = NZ = 32
    DX = 1e-3
    N_STEPS = 100

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    sigma = 15.0 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    cx = cy = cz = NX // 2
    src = PointSource(pulse, cx, cy, "Ez", k=cz, grid=grid, N_steps=N_STEPS)
    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]), n_check=100)

    symmetry_errors: list[float] = []
    step_indices: list[float] = []

    with torch.no_grad():
        for step in range(N_STEPS):
            sim.step()
            if step % 10 == 0:
                Ez_np = fields.Ez.detach().cpu().numpy()
                ez_max = float(np.abs(Ez_np).max())
                if ez_max < 1e-30:
                    symmetry_errors.append(0.0)
                    step_indices.append(float(step))
                    continue
                err_sum = 0.0
                n_offsets = 0
                for d in range(1, 6):
                    if cx + d < NX and cx - d >= 0:
                        v_pos = float(Ez_np[cx + d, cy, cz])
                        v_neg = float(Ez_np[cx - d, cy, cz])
                        err_sum += abs(v_pos - v_neg) / ez_max
                        n_offsets += 1
                mean_err = err_sum / n_offsets if n_offsets > 0 else 0.0
                symmetry_errors.append(mean_err)
                step_indices.append(float(step))

    mean_symmetry_error = float(np.mean(symmetry_errors)) if symmetry_errors else 1.0
    score = max(0.0, min(1.0, 1.0 - mean_symmetry_error))
    return score, symmetry_errors, step_indices


# ---------------------------------------------------------------------------
# Test 3: Phase Velocity Accuracy
# ---------------------------------------------------------------------------


def _test_phase_velocity() -> tuple[float, float, list[float], list[float]]:
    """16×16×64 plane wave, 300 steps.

    Returns (score, error_pct, wavefront_positions, time_steps).
    """
    NX = NY = 16
    NZ = 64
    DX = 1e-3
    N_STEPS = 300
    SOURCE_K = 2

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    sigma = 20.0 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    src = PlaneSource(
        pulse, plane="xy", position=SOURCE_K, component="Ez",
        grid=grid, N_steps=N_STEPS,
    )
    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]), n_check=100)

    wavefront_positions: list[float] = []
    time_steps_ns: list[float] = []

    with torch.no_grad():
        for step in range(N_STEPS):
            sim.step()
            if step % 10 == 0 and step > 10:
                Ez_np = fields.Ez.detach().cpu().numpy()
                profile = np.abs(Ez_np[NX // 2, NY // 2, :])
                if profile.max() > 1e-30:
                    wavefront_k = int(np.argmax(profile))
                    wavefront_positions.append(float(wavefront_k) * DX)
                    time_steps_ns.append(float(step) * grid.dt * 1e9)

    # Fit velocity from linear regression of position vs time
    if len(wavefront_positions) >= 4:
        times_s = [t * 1e-9 for t in time_steps_ns]
        coeffs = np.polyfit(times_s, wavefront_positions, 1)
        v_fitted = float(coeffs[0])
        v_error = abs(v_fitted - C0) / C0
    else:
        v_fitted = 0.0
        v_error = 1.0

    error_pct = v_error * 100.0
    score = max(0.0, min(1.0, 1.0 - v_error))
    return score, error_pct, wavefront_positions, time_steps_ns


# ---------------------------------------------------------------------------
# Test 4: Material Conservation
# ---------------------------------------------------------------------------


def _test_material_conservation() -> tuple[float, np.ndarray, torch.Tensor, torch.Tensor]:
    """32³ dielectric sphere eps_r=4, 200 steps.

    Returns (score, Ez_xy_slice, Ca_tensor, Cb_tensor).
    """
    NX = NY = NZ = 32
    DX = 1e-3
    N_STEPS = 200
    EPS_R = 4.0
    R_CELLS = 6.0

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    mm = MaterialMap3D(grid)
    cx = cy = cz = NX // 2
    sphere_mat = Material("dielectric", eps_r=EPS_R, sigma=0.0)
    mm.add_sphere(center=(cx, cy, cz), radius=R_CELLS, material=sphere_mat)
    Ca, Cb = mm.build3d()

    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    sigma = 20.0 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    src = PointSource(pulse, cx, cy, "Ez", k=cz, grid=grid, N_steps=N_STEPS)
    sim = FDTD3D(
        grid, fields, boundary, SourceCollection([src]),
        Ca=Ca, Cb=Cb, n_check=100,
    )

    with torch.no_grad():
        for _ in range(N_STEPS):
            sim.step()

    Ca_cpu = Ca.detach().cpu()
    Cb_cpu = Cb.detach().cpu()

    ca_min = float(Ca_cpu.min().item())
    ca_max = float(Ca_cpu.max().item())
    cb_min = float(Cb_cpu.min().item())
    ca_no_nan = not bool(torch.isnan(Ca_cpu).any().item())
    cb_no_nan = not bool(torch.isnan(Cb_cpu).any().item())

    ca_in_range = (0.0 < ca_min) and (ca_max <= 1.0 + 1e-6)
    cb_positive = cb_min > 0.0

    energy_final = fields.total_energy()
    energy_positive = energy_final >= 0.0

    score = 1.0 if (ca_in_range and cb_positive and ca_no_nan and cb_no_nan and energy_positive) else 0.0

    Ez_slice = fields.Ez.detach().cpu().numpy()[:, :, cz]
    return score, Ez_slice, Ca_cpu, Cb_cpu


# ---------------------------------------------------------------------------
# Test 5: Throughput
# ---------------------------------------------------------------------------


def _test_throughput() -> tuple[float, float]:
    """48³ grid, 10 warmup + 100 timed steps.  Returns (score, mcells_per_s)."""
    NX = NY = NZ = 48
    DX = 1e-3
    N_WARMUP = 10
    N_TIMED = 100

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    sigma = 20.0 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    cx = cy = cz = NX // 2
    total_steps = N_WARMUP + N_TIMED
    src = PointSource(pulse, cx, cy, "Ez", k=cz, grid=grid, N_steps=total_steps)
    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]), n_check=200)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            sim.step()

    # Timed run
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(N_TIMED):
            sim.step()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    cells_total = N_TIMED * NX * NY * NZ
    mcells = cells_total / elapsed / 1e6

    # Normalised to 10 Mcells/s as baseline (any modern CPU exceeds this)
    score = min(1.0, mcells / 10.0)
    return score, mcells


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _draw_sphere_circle(ax: plt.Axes, cx: int, cy: int, r: int, dx: float) -> None:
    """Overlay a dashed circle representing a sphere boundary in mm coordinates."""
    import matplotlib.patches as mpatches
    circle = mpatches.Circle(
        (cx * dx * 1e3, cy * dx * 1e3), r * dx * 1e3,
        fill=False, color="white", linewidth=1.5, linestyle="--",
    )
    ax.add_patch(circle)


def _build_figure(
    energy_hist: list[float],
    energy_steps: list[float],
    sym_errors: list[float],
    sym_steps: list[float],
    wf_positions: list[float],
    wf_times_ns: list[float],
    ez_material_slice: np.ndarray,
    mcells: float,
    scores: list[float],
    total: float,
) -> plt.Figure:
    """Build the 3×2 summary figure."""
    categories = [
        "CFL\nStability",
        "Octant\nSymmetry",
        "Phase\nVelocity",
        "Material\nConserv.",
        "Throughput",
    ]
    colors = [
        "green" if s > 0.8 else ("orange" if s > 0.5 else "red")
        for s in scores
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle(
        "WaveForge 3D — Comprehensive Benchmark Suite",
        fontsize=14, fontweight="bold",
    )

    # [0, 0] — Energy vs step (CFL stability)
    ax = axes[0, 0]
    ax.plot(energy_steps, energy_hist, color="steelblue", linewidth=1.6)
    ax.set(
        title="Test 1: CFL Stability — Energy vs Step",
        xlabel="Step",
        ylabel="EM Energy (J)",
    )
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # [0, 1] — Symmetry error vs step
    ax = axes[0, 1]
    ax.plot(sym_steps, sym_errors, color="darkorange", linewidth=1.6, marker="o", markersize=3)
    ax.axhline(y=0.01, color="red", linestyle="--", linewidth=1.0, label="Pass threshold (1%)")
    ax.set(
        title="Test 2: Octant Symmetry — Error vs Step",
        xlabel="Step",
        ylabel="Mean symmetry error",
    )
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # [1, 0] — Wavefront position vs time
    ax = axes[1, 0]
    if len(wf_positions) >= 2:
        times_s = [t * 1e-9 for t in wf_times_ns]
        coeffs = np.polyfit(times_s, wf_positions, 1)
        fit_line = np.polyval(coeffs, times_s)
        ax.plot(wf_times_ns, [p * 1e3 for p in wf_positions],
                "o", color="steelblue", markersize=4, label="Wavefront")
        ax.plot(wf_times_ns, [v * 1e3 for v in fit_line],
                "--", color="red", linewidth=1.4,
                label=f"Fit v={coeffs[0]/C0:.3f} c₀")
    ax.set(
        title="Test 3: Phase Velocity — Wavefront vs Time",
        xlabel="Time (ns)",
        ylabel="Wavefront position (mm)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # [1, 1] — Ez XY slice with sphere overlay (material test)
    ax = axes[1, 1]
    NXY = ez_material_slice.shape[0]
    DX_MAT = 1e-3
    ext_mm = [0.0, NXY * DX_MAT * 1e3, 0.0, NXY * DX_MAT * 1e3]
    vmax = float(np.abs(ez_material_slice).max()) or 1e-12
    im = ax.imshow(
        ez_material_slice.T, origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, extent=ext_mm, aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="Ez (V/m)")
    cx_mat = cy_mat = NXY // 2
    r_cells = 6
    _draw_sphere_circle(ax, cx_mat, cy_mat, r_cells, DX_MAT)
    ax.set(
        title="Test 4: Material Conservation — Ez XY slice",
        xlabel="x (mm)",
        ylabel="y (mm)",
    )

    # [2, 0] — Throughput bar
    ax = axes[2, 0]
    bar_color = "green" if mcells >= 10.0 else "orange"
    ax.bar(["Mcells/s"], [mcells], color=bar_color, width=0.4, edgecolor="black")
    ax.axhline(y=200.0, color="purple", linestyle="--", linewidth=1.2,
               label="GPU target (200 Mcells/s)")
    ax.axhline(y=10.0, color="gray", linestyle=":", linewidth=1.0,
               label="CPU baseline (10 Mcells/s)")
    ax.set(
        title=f"Test 5: Throughput — {mcells:.1f} Mcells/s",
        ylabel="Mcells/s",
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    # [2, 1] — Summary horizontal bar chart (radar substitute)
    ax = axes[2, 1]
    y_pos = list(range(len(categories)))
    ax.barh(y_pos, scores, color=colors, edgecolor="black", height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9)
    ax.axvline(x=0.8, color="gray", linestyle="--", linewidth=1.2,
               label="Pass threshold (0.8)")
    ax.set_xlim(0.0, 1.15)
    ax.set(
        title=f"Overall Score: {total:.1f}/5.0",
        xlabel="Score (0–1)",
    )
    ax.legend(fontsize=8)
    for i, (s, c) in enumerate(zip(scores, categories)):
        ax.text(s + 0.02, i, f"{s:.2f}", va="center", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 40)
    print("WaveForge 3D — Comprehensive Benchmark")
    print(f"Device: {DEVICE}")
    print("=" * 40)

    # Test 1
    print("\nRunning Test 1: CFL Stability ...")
    s1, energy_hist, energy_steps = _test_cfl_stability()
    pass1 = "PASS" if s1 >= 3.0 else "FAIL"
    print(f"  Score: {s1:.1f}/3.0  [{pass1}]")

    # Test 2
    print("Running Test 2: Octant Symmetry ...")
    s2, sym_errors, sym_steps = _test_octant_symmetry()
    pass2 = "PASS" if s2 > 0.99 else "FAIL"
    print(f"  Score: {s2:.4f}     [{pass2}]")

    # Test 3
    print("Running Test 3: Phase Velocity ...")
    s3, error_pct, wf_positions, wf_times = _test_phase_velocity()
    pass3 = "PASS" if error_pct < 5.0 else "FAIL"
    print(f"  Error: {error_pct:.2f}%  [{pass3}]")

    # Test 4
    print("Running Test 4: Material Conservation ...")
    s4, ez_mat_slice, _Ca, _Cb = _test_material_conservation()
    pass4 = "PASS" if s4 >= 1.0 else "FAIL"
    print(f"  [{pass4}]")

    # Test 5
    print("Running Test 5: Throughput ...")
    s5, mcells = _test_throughput()
    print(f"  {mcells:.1f} Mcells/s")

    # Overall
    # Tests 1-4 each contribute 1 normalised point; test 5 contributes 1.
    s1_norm = min(1.0, s1 / 3.0)
    total = s1_norm + s2 + s3 + s4 + s5

    # Print report
    print()
    print("=" * 40)
    print("WaveForge 3D — Comprehensive Benchmark")
    print(f"Device: {DEVICE}")
    print("=" * 40)
    print(f"Test 1: CFL Stability .......... {s1:.1f}/3.0  [{pass1}]")
    print(f"Test 2: Octant Symmetry ........ {s2:.3f}     [{pass2}]")
    print(f"Test 3: Phase Velocity ......... {error_pct:.2f}% error [{pass3}]")
    print(f"Test 4: Material Conservation .. [{pass4}]")
    print(f"Test 5: Throughput ............. {mcells:.1f} Mcells/s")
    print()
    print(f"Overall Score: {total:.1f}/5.0")
    print(f"WAVEFORGE_BENCH_COMPREHENSIVE: {mcells:.1f} Mcells/s")
    print("=" * 40)

    # Save JSON report
    results = {
        "device": DEVICE,
        "tests": {
            "cfl_stability": {
                "score_raw": s1,
                "score_norm": s1_norm,
                "pass": pass1 == "PASS",
            },
            "octant_symmetry": {
                "score": s2,
                "pass": pass2 == "PASS",
            },
            "phase_velocity": {
                "score": s3,
                "error_pct": error_pct,
                "pass": pass3 == "PASS",
            },
            "material_conservation": {
                "score": s4,
                "pass": pass4 == "PASS",
            },
            "throughput": {
                "mcells_per_s": mcells,
                "score": s5,
            },
        },
        "overall_score": total,
        "mcells_per_s": mcells,
    }
    json_path = OUTPUT_DIR / "3d_10_benchmark_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Build and save figure
    scores_list = [s1_norm, s2, s3, s4, s5]
    fig = _build_figure(
        energy_hist=energy_hist,
        energy_steps=energy_steps,
        sym_errors=sym_errors,
        sym_steps=sym_steps,
        wf_positions=wf_positions,
        wf_times_ns=wf_times,
        ez_material_slice=ez_mat_slice,
        mcells=mcells,
        scores=scores_list,
        total=total,
    )
    fig.text(
        0.5, 0.005,
        f"Device: {DEVICE} | Overall: {total:.1f}/5.0 | {mcells:.1f} Mcells/s",
        ha="center", fontsize=10,
    )
    png_path = OUTPUT_DIR / "3d_10_comprehensive_benchmark.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG:  {png_path}")


if __name__ == "__main__":
    main()
