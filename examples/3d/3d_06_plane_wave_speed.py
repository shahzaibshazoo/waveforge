"""
3d_06_plane_wave_speed.py — Phase velocity vs analytical c benchmark.

Injects a plane wave via PlaneSource, tracks the wavefront position over time,
measures the numerical phase velocity by linear regression, and compares it to
the analytical speed of light c = 2.998e8 m/s.

Run:  python examples/3d/3d_06_plane_wave_speed.py
Out:  examples/output/3d_06_plane_wave_speed.png
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PlaneSource, SourceCollection
from core.fdtd3d import FDTD3D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NX: int = 16
NY: int = 16
NZ: int = 80
DX: float = 1.5e-3          # 1.5 mm cell spacing
N_STEPS: int = 400
SOURCE_K: int = 4           # inject at k=4 (near z=0 face)
TRACK_INTERVAL: int = 20    # snapshot Ez line every 20 steps
SNAP_STEPS: tuple[int, ...] = (100, 200, 300, 400)
C_ANALYTICAL: float = 2.998e8  # m/s
DISPERSION_THRESHOLD: float = 5.0  # percent
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Center indices for the z-axis line probe
IX_PROBE: int = NX // 2   # = 8
IY_PROBE: int = NY // 2   # = 8


# ---------------------------------------------------------------------------
# Wavefront tracking helpers
# ---------------------------------------------------------------------------

def extract_ez_line(fields: FieldSet) -> np.ndarray:
    """Return Ez along the z-axis centre as a 1-D numpy array."""
    return fields.Ez[IX_PROBE, IY_PROBE, :].cpu().numpy()


def wavefront_position(ez_line: np.ndarray, dz: float) -> float:
    """Return the wavefront z-position (m) as argmax(|ez_line|) * dz."""
    return float(np.argmax(np.abs(ez_line))) * dz


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Grid and fields ---------------------------------------------------
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    # --- Source: PlaneSource Ez on xy-plane at k=SOURCE_K ------------------
    sigma = 20.0 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    src = PlaneSource(
        pulse,
        plane="xy",
        position=SOURCE_K,
        component="Ez",
        grid=grid,
        N_steps=N_STEPS,
    )
    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]), n_check=100)

    # --- Wavefront tracking storage ----------------------------------------
    track_steps: list[int] = []
    track_positions: list[float] = []
    snap_ez: dict[int, np.ndarray] = {}

    # --- Time loop ----------------------------------------------------------
    for n in range(N_STEPS):
        sim.step()

        step = sim.steps_completed  # n+1 after step()

        if step % TRACK_INTERVAL == 0:
            ez_line = extract_ez_line(fields)
            pos = wavefront_position(ez_line, grid.dz)
            track_steps.append(step)
            track_positions.append(pos)

        if step in SNAP_STEPS:
            snap_ez[step] = extract_ez_line(fields)

    # --- Fit velocity -------------------------------------------------------
    times_s = np.array(track_steps, dtype=np.float64) * grid.dt
    positions_m = np.array(track_positions, dtype=np.float64)

    # Linear fit: position = v * time + offset
    coeffs = np.polyfit(times_s, positions_m, deg=1)
    v_numerical: float = float(coeffs[0])

    dispersion_error_pct: float = (
        abs(v_numerical - C_ANALYTICAL) / C_ANALYTICAL * 100.0
    )
    result_str: str = "PASS" if dispersion_error_pct < DISPERSION_THRESHOLD else "FAIL"

    # --- Print report -------------------------------------------------------
    print("====================================")
    print("WaveForge 3D — Phase Velocity Test")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f}mm, {N_STEPS} steps")
    print(f"PlaneSource Ez on xy-plane at k={SOURCE_K}")
    print("------------------------------------")
    print(f"Numerical v = {v_numerical:.3e} m/s")
    print(f"Analytical c = {C_ANALYTICAL:.3e} m/s")
    print(f"Dispersion error: {dispersion_error_pct:.2f}%")
    print(f"Result: {result_str} (threshold: {DISPERSION_THRESHOLD:.0f}%)")
    print("====================================")

    # --- Figure: 1x2 layout ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("WaveForge 3D — Phase Velocity vs Analytical c", fontsize=13)

    _plot_wavefront(axes[0], times_s, positions_m, coeffs, v_numerical, C_ANALYTICAL)
    _plot_snapshots(axes[1], snap_ez, grid.dz, NZ)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "3d_06_plane_wave_speed.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_wavefront(
    ax,
    times_s: np.ndarray,
    positions_m: np.ndarray,
    coeffs: np.ndarray,
    v_numerical: float,
    c_analytical: float,
) -> None:
    """Plot wavefront z-position vs time with fit and reference line."""
    times_ns = times_s * 1e9
    positions_mm = positions_m * 1e3

    fit_mm = (np.polyval(coeffs, times_s)) * 1e3

    ax.scatter(times_ns, positions_mm, s=18, color="royalblue",
               zorder=5, label="Measured wavefront")
    ax.plot(times_ns, fit_mm, color="crimson", lw=1.8,
            label=f"v = {v_numerical:.2e} m/s")

    # Analytical reference line
    ref_mm = c_analytical * times_s * 1e3
    ax.plot(times_ns, ref_mm, color="forestgreen", lw=1.2,
            linestyle="--", label=f"c = {c_analytical:.3e} m/s")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Wavefront z-position (mm)")
    ax.set_title("Wavefront Position vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_snapshots(
    ax,
    snap_ez: dict[int, np.ndarray],
    dz: float,
    nz: int,
) -> None:
    """Plot Ez along the z-axis at four time snapshots."""
    z_mm = np.arange(nz) * dz * 1e3
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for color, (step, ez) in zip(colors, sorted(snap_ez.items())):
        ax.plot(z_mm, ez, color=color, lw=1.4, label=f"step {step}")

    ax.set_xlabel("z-position (mm)")
    ax.set_ylabel("Ez (V/m)")
    ax.set_title("Ez Along z-Axis at 4 Snapshots")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
