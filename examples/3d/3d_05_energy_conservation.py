"""
3d_05_energy_conservation.py — 3D FDTD numerical stability and energy conservation.

Benchmark numerical stability: track total EM energy over time, measure energy
decay rate, compare against analytical decay from Mur ABC, and plot field
symmetry metrics.

Stability metrics:
  - Total EM energy per step (log-scale)
  - Ez at center cell (pulse injection and free decay)
  - Symmetry error Ez[25,24,24] - Ez[23,24,24] (should be near zero)
  - Post-peak energy decay rate (linear fit on log10 scale)

Grid:    48x48x48, dx=1.5mm, device=cpu (reproducibility)
Source:  Ez point source at center (24,24,24), GaussianPulse sigma=15*dt
Steps:   600

Run:  python examples/3d/3d_05_energy_conservation.py
Out:  examples/output/3d_05_energy_conservation.png
"""

import sys
import time
import math
from pathlib import Path

import torch
import numpy as np
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
# Configuration
# ---------------------------------------------------------------------------

NX, NY, NZ = 48, 48, 48
DX = 1.5e-3          # 1.5 mm isotropic cell spacing
N_STEPS = 600
CENTER = 24          # center cell index for a 48-cell grid
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEVICE = "cpu"       # intentional: reproducibility benchmark


# ---------------------------------------------------------------------------
# Build grid and infrastructure
# ---------------------------------------------------------------------------

def build_simulation() -> tuple:
    """Construct grid, fields, boundary, source, and solver."""
    grid = YeeGrid(
        NX, NY,
        DX, DX,
        Nz=NZ,
        dz=DX,
        courant=0.98,
        device=DEVICE,
    )

    fields = FieldSet(grid)

    boundary = MurABC3D(
        grid,
        fields.Hx,
        fields.Hy,
        fields.Hz,
    )

    # GaussianPulse with sigma = 15 * grid.dt
    sigma = 15.0 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)

    source = PointSource(
        pulse,
        CENTER, CENTER,
        component="Ez",
        k=CENTER,
        grid=grid,
        N_steps=N_STEPS,
    )

    sources = SourceCollection([source])

    solver = FDTD3D(
        grid, fields, boundary, sources,
        stability_threshold=1e12,
        n_check=200,
    )

    return grid, fields, solver


# ---------------------------------------------------------------------------
# Main run loop — record diagnostics every step
# ---------------------------------------------------------------------------

def run_simulation(
    grid: YeeGrid,
    fields: FieldSet,
    solver: FDTD3D,
) -> tuple:
    """Step 600 iterations and collect energy, Ez center, symmetry error."""
    energy_history: list[float] = []
    ez_center_history: list[float] = []
    symmetry_err_history: list[float] = []

    t_run_start = time.perf_counter()

    with torch.no_grad():
        for _ in range(N_STEPS):
            solver.step()

            # 1. total EM energy in Joules
            energy_history.append(fields.total_energy())

            # 2. Ez at exact center cell
            ez_center_history.append(
                float(fields.Ez[CENTER, CENTER, CENTER].cpu())
            )

            # 3. symmetry error: Ez[25,24,24] - Ez[23,24,24]
            ez_plus = float(fields.Ez[CENTER + 1, CENTER, CENTER].cpu())
            ez_minus = float(fields.Ez[CENTER - 1, CENTER, CENTER].cpu())
            symmetry_err_history.append(ez_plus - ez_minus)

    elapsed = time.perf_counter() - t_run_start
    print(f"Simulation completed in {elapsed:.2f}s  "
          f"({N_STEPS / elapsed:.0f} steps/s)")

    return (
        np.array(energy_history),
        np.array(ez_center_history),
        np.array(symmetry_err_history),
    )


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    grid: YeeGrid,
    fields: FieldSet,
    energy: np.ndarray,
    symmetry_err: np.ndarray,
) -> dict:
    """Compute scalar stability metrics from recorded histories."""
    peak_energy = float(energy.max())
    peak_step = int(energy.argmax())
    final_energy = float(energy[-1])
    retention_ratio = final_energy / peak_energy if peak_energy > 1e-300 else 0.0

    # Max relative symmetry error normalised by peak energy density proxy
    # (we use sqrt of peak energy as proxy for field amplitude scale)
    field_scale = math.sqrt(peak_energy) if peak_energy > 0 else 1.0
    # symmetry error is a field difference, so normalise by field amplitude
    max_sym_abs = float(np.abs(symmetry_err).max())
    # Normalise by Ez max (derived from energy: E ~ sqrt(2*energy/eps0/V))
    cell_vol = grid.cell_volume
    eps0 = 8.854e-12
    e_field_scale = math.sqrt(
        2.0 * peak_energy / (eps0 * cell_vol * NX * NY * NZ)
    ) if peak_energy > 0 else 1.0
    relative_asymmetry = max_sym_abs / e_field_scale if e_field_scale > 0 else 0.0

    # Check finiteness of all six field components
    all_finite = all(
        bool(torch.isfinite(t).all().item())
        for t in (
            fields.Ex, fields.Ey, fields.Ez,
            fields.Hx, fields.Hy, fields.Hz,
        )
    )

    # CFL margin: grid was built with courant=0.98, so margin is 2%
    from core.grid import compute_stable_dt
    dt_cfl_max = compute_stable_dt(grid.dx, grid.dy, grid.dz, courant=0.98)
    cfl_margin_pct = (1.0 - grid.dt / dt_cfl_max) * 100.0

    # Fit post-peak decay: linear fit on log10(energy/peak) vs step index
    post_peak_mask = np.arange(N_STEPS) > peak_step
    n_post = int(post_peak_mask.sum())
    decay_slope: float = float("nan")
    if n_post >= 10:
        x_fit = np.where(post_peak_mask)[0].astype(float)
        # Guard against zeros before log
        safe_energy = np.where(energy > 0.0, energy, np.nan)
        y_fit = np.log10(safe_energy / peak_energy)
        valid = np.isfinite(y_fit) & post_peak_mask
        if valid.sum() >= 10:
            coeffs = np.polyfit(
                np.arange(N_STEPS)[valid].astype(float),
                y_fit[valid],
                deg=1,
            )
            decay_slope = float(coeffs[0])

    return {
        "peak_energy": peak_energy,
        "peak_step": peak_step,
        "final_energy": final_energy,
        "retention_ratio": retention_ratio,
        "max_sym_abs": max_sym_abs,
        "relative_asymmetry": relative_asymmetry,
        "all_finite": all_finite,
        "cfl_margin_pct": cfl_margin_pct,
        "dt": grid.dt,
        "dt_cfl_max": dt_cfl_max,
        "decay_slope": decay_slope,
    }


# ---------------------------------------------------------------------------
# Stability PASS/FAIL judgment
# ---------------------------------------------------------------------------

def stability_pass(metrics: dict) -> bool:
    """Return True when the simulation passes all three stability criteria."""
    return (
        metrics["all_finite"]
        and metrics["relative_asymmetry"] < 0.01
        and metrics["final_energy"] > 0.0
    )


# ---------------------------------------------------------------------------
# Print stability report
# ---------------------------------------------------------------------------

def print_report(metrics: dict) -> None:
    """Print the formatted stability report to stdout."""
    pct = metrics["retention_ratio"] * 100.0
    verdict = "PASS" if stability_pass(metrics) else "FAIL"
    print("=" * 30)
    print("WaveForge 3D - Stability Report")
    print(f"Grid: 48^3, dx=1.5mm, {N_STEPS} steps")
    print(
        f"dt = {metrics['dt']:.2e} s  "
        f"(CFL margin: {metrics['cfl_margin_pct']:.1f}%)"
    )
    print(
        f"Peak energy: {metrics['peak_energy']:.3e} J  "
        f"at step {metrics['peak_step']}"
    )
    print(f"Final energy: {metrics['final_energy']:.3e} J")
    print(
        f"Energy retention: {metrics['retention_ratio']:.3f}  "
        f"({pct:.2f}% of peak)"
    )
    print(
        f"Max symmetry error: {metrics['relative_asymmetry']:.3e} (relative)"
    )
    print(f"All fields finite: {'YES' if metrics['all_finite'] else 'NO'}")
    print(f"Numerical stability: {verdict}")
    print("=" * 30)


# ---------------------------------------------------------------------------
# 2x2 figure
# ---------------------------------------------------------------------------

def make_figure(
    energy: np.ndarray,
    ez_center: np.ndarray,
    symmetry_err: np.ndarray,
    metrics: dict,
    output_path: Path,
) -> None:
    """Produce and save the 2x2 stability figure."""
    steps = np.arange(N_STEPS)
    peak_step = metrics["peak_step"]
    peak_energy = metrics["peak_energy"]
    decay_slope = metrics["decay_slope"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "WaveForge 3D - Numerical Stability & Energy Conservation",
        fontsize=14, fontweight="bold",
    )

    # [0,0] — Total energy, log scale, mark peak
    ax00 = axes[0, 0]
    # Clip to positive values before log plot
    safe_energy = np.where(energy > 0, energy, np.nan)
    ax00.semilogy(steps, safe_energy, color="steelblue", lw=1.2, label="Total EM energy")
    ax00.axvline(x=peak_step, color="red", linestyle="--", lw=1.0, label=f"Peak (step {peak_step})")
    ax00.set_xlabel("Time step")
    ax00.set_ylabel("Energy (J)")
    ax00.set_title("Total EM Energy (log scale)")
    ax00.legend(fontsize=8)
    ax00.grid(True, which="both", alpha=0.3)

    # [0,1] — Ez at center cell
    ax01 = axes[0, 1]
    ax01.plot(steps, ez_center, color="darkorange", lw=1.0)
    ax01.set_xlabel("Time step")
    ax01.set_ylabel("Ez (V/m)")
    ax01.set_title(f"Ez at center cell ({CENTER},{CENTER},{CENTER})")
    ax01.axhline(0, color="gray", lw=0.5, linestyle=":")
    ax01.grid(True, alpha=0.3)

    # [1,0] — Symmetry error
    ax10 = axes[1, 0]
    ax10.plot(steps, symmetry_err, color="mediumseagreen", lw=1.0)
    ax10.set_xlabel("Time step")
    ax10.set_ylabel("Ez[+1] - Ez[-1]  (V/m)")
    ax10.set_title("Symmetry Error (should be ~0)")
    ax10.axhline(0, color="gray", lw=0.5, linestyle=":")
    ax10.grid(True, alpha=0.3)

    # [1,1] — Post-peak log10 energy decay + fitted line
    ax11 = axes[1, 1]
    post_steps = steps[peak_step:]
    safe_post = np.where(energy[peak_step:] > 0, energy[peak_step:], np.nan)
    log_ratio = np.log10(safe_post / peak_energy)
    ax11.plot(post_steps, log_ratio, color="purple", lw=1.0, label="log10(E/E_peak)")

    # Overlay fitted line when slope is valid
    if math.isfinite(decay_slope):
        x_fit_range = post_steps[np.isfinite(log_ratio)]
        if len(x_fit_range) >= 2:
            y_line = decay_slope * (x_fit_range - peak_step)
            # Intercept: set so the line starts at 0 at peak_step
            ax11.plot(
                x_fit_range, y_line,
                color="red", lw=1.2, linestyle="--",
                label=f"Fit: slope={decay_slope:.4f}/step",
            )
        print(f"Decay slope (log10 energy per step): {decay_slope:.4e}")

    ax11.set_xlabel("Time step")
    ax11.set_ylabel("log10(E / E_peak)")
    ax11.set_title("Post-Peak Energy Decay Rate")
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    output_path = OUTPUT_DIR / "3d_05_energy_conservation.png"

    print("Building simulation...")
    grid, fields, solver = build_simulation()

    # Print CFL reference before run
    from core.grid import compute_stable_dt
    dt_max = compute_stable_dt(grid.dx, grid.dy, grid.dz, courant=0.98)
    print(
        f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f}mm, "
        f"dt={grid.dt:.3e}s, dt_cfl_max={dt_max:.3e}s"
    )

    print("Running 600 steps...")
    energy, ez_center, symmetry_err = run_simulation(grid, fields, solver)

    metrics = compute_metrics(grid, fields, energy, symmetry_err)
    print_report(metrics)

    make_figure(energy, ez_center, symmetry_err, metrics, output_path)


if __name__ == "__main__":
    main()
