"""
3d_07_mie_scattering_accuracy.py — Quantitative Mie scattering accuracy benchmark.

Physics benchmark: dielectric sphere (eps_r=4) in free space illuminated by a
point Ez source.  Compare the field amplitude inside the sphere to the
analytical Clausius-Mossotti (quasi-static Rayleigh limit) prediction:

    E_inside / E_0 = 3 / (eps_r + 2) = 3 / 6 = 0.500

Two runs are performed:
  1. Free-space baseline — no sphere; reference field at the probe point.
  2. With sphere — dielectric sphere at domain centre; field at same probe point.

The scattered field (difference run 2 - run 1) and the transmission ratio are
compared against the Clausius-Mossotti prediction to quantify numerical accuracy.

Run:  python examples/3d/3d_07_mie_scattering_accuracy.py
Out:  examples/output/3d_07_mie_scattering_accuracy.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.fdtd3d import FDTD3D
from core.materials import Material, MaterialMap3D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NX: int = NY = NZ = 48
DX: float = 2e-3          # 2 mm → 96 mm cube domain
EPS_R: float = 4.0        # relative permittivity of sphere
R_CELLS: int = 8          # sphere radius: 8 cells = 16 mm
N_STEPS: int = 500
SNAP_STEP: int = 300      # step at which to capture comparison fields
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Domain centre (sphere centre)
CX: int = NX // 2   # 24
CY: int = NY // 2   # 24
CZ: int = NZ // 2   # 24

# Point source position — offset in z so it illuminates the sphere
SRC_K: int = 4

# Probe point — 4 cells past sphere centre in z (inside sphere region)
PROBE_I: int = CX
PROBE_J: int = CY
PROBE_K: int = CZ         # sphere centre (most sensitive to loading)

# Clausius-Mossotti (quasi-static) transmission ratio
CM_RATIO: float = 3.0 / (EPS_R + 2.0)   # = 0.5 for eps_r=4
PASS_THRESHOLD: float = 0.20              # allow 20% error

OUTPUT_DIR: Path = Path(__file__).parent.parent / "output"


# ---------------------------------------------------------------------------
# Helper: build sphere material map
# ---------------------------------------------------------------------------

def _build_sphere_material(grid: YeeGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (Ca, Cb) tensors with a dielectric sphere at domain centre."""
    sphere_mat = Material("dielectric_sphere", eps_r=EPS_R, sigma=0.0)
    mm = MaterialMap3D(grid)
    mm.add_sphere(center=(CX, CY, CZ), radius=float(R_CELLS), material=sphere_mat)
    return mm.build3d()


# ---------------------------------------------------------------------------
# Helper: single simulation run
# ---------------------------------------------------------------------------

def run_sim(
    with_sphere: bool,
    n_steps: int,
    snap_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one FDTD3D simulation and return field snapshots.

    Parameters
    ----------
    with_sphere : bool
        When True, build sphere Ca/Cb material tensors.
    n_steps : int
        Total number of time steps.
    snap_step : int
        Step index at which to capture the 3-D Ez snapshot for comparison.

    Returns
    -------
    ez_snap : np.ndarray, shape (NX, NY, NZ)
        Ez field at snap_step.
    ez_final : np.ndarray, shape (NX, NY, NZ)
        Ez field at the final step.
    """
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    sigma_pulse = 12.0 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma_pulse)
    src = PointSource(
        pulse, CX, CY, "Ez", k=SRC_K, grid=grid, N_steps=n_steps
    )
    sources = SourceCollection([src])

    if with_sphere:
        Ca, Cb = _build_sphere_material(grid)
        sim = FDTD3D(grid, fields, boundary, sources, Ca=Ca, Cb=Cb, n_check=100)
    else:
        sim = FDTD3D(grid, fields, boundary, sources, n_check=100)

    ez_snap: np.ndarray | None = None

    with torch.no_grad():
        for step in range(n_steps):
            sim.step()
            if step + 1 == snap_step:
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                ez_snap = fields.Ez.detach().cpu().numpy().copy()

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    ez_final = fields.Ez.detach().cpu().numpy().copy()
    return ez_snap, ez_final


# ---------------------------------------------------------------------------
# Helper: draw sphere boundary circle on a matplotlib Axes
# ---------------------------------------------------------------------------

def _draw_sphere_circle(ax, cx_mm: float, cy_mm: float, r_mm: float) -> None:
    """Overlay a dashed white circle representing the sphere boundary."""
    import matplotlib.patches as mpatches
    circle = mpatches.Circle(
        (cx_mm, cy_mm), r_mm,
        fill=False, color="white", linewidth=1.5, linestyle="--"
    )
    ax.add_patch(circle)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_total = time.perf_counter()

    # --- Run 1: free space ------------------------------------------------
    print(f"Run 1/2: free space  ({N_STEPS} steps)...")
    t0 = time.perf_counter()
    ez_free_snap, ez_free_final = run_sim(
        with_sphere=False, n_steps=N_STEPS, snap_step=SNAP_STEP
    )
    print(f"  Done in {time.perf_counter() - t0:.2f}s")

    # --- Run 2: with sphere -----------------------------------------------
    print(f"Run 2/2: with sphere ({N_STEPS} steps)...")
    t0 = time.perf_counter()
    ez_sphere_snap, ez_sphere_final = run_sim(
        with_sphere=True, n_steps=N_STEPS, snap_step=SNAP_STEP
    )
    print(f"  Done in {time.perf_counter() - t0:.2f}s")

    # --- Metrics ----------------------------------------------------------

    # Field value at sphere centre at snap_step
    ez_free_center = float(ez_free_snap[PROBE_I, PROBE_J, PROBE_K])
    ez_sphere_center = float(ez_sphere_snap[PROBE_I, PROBE_J, PROBE_K])

    # Transmission ratio (signed, then compare magnitudes)
    if abs(ez_free_center) > 1e-30:
        ratio_measured = ez_sphere_center / ez_free_center
    else:
        ratio_measured = float("nan")

    error_pct = abs(abs(ratio_measured) - CM_RATIO) / CM_RATIO * 100.0

    # Scattered field: difference at final step
    ez_scattered_final = ez_sphere_final - ez_free_final
    free_amp = float(np.abs(ez_free_final).max())
    scattered_max = float(np.abs(ez_scattered_final).max())
    scattered_pct = (scattered_max / free_amp * 100.0) if free_amp > 1e-30 else float("nan")

    passed = error_pct < PASS_THRESHOLD * 100.0

    # --- Report -----------------------------------------------------------
    print()
    print("==========================================")
    print("WaveForge 3D — Mie Scattering Accuracy")
    print(f"Grid: {NX}³, dx={DX*1e3:.0f}mm, eps_r={EPS_R} sphere r={R_CELLS*DX*1e3:.0f}mm")
    print("------------------------------------------")
    print(f"Ez at sphere centre (step {SNAP_STEP}):")
    print(f"  Free space:  {ez_free_center:.3e}")
    print(f"  With sphere: {ez_sphere_center:.3e}")
    print(f"  Ratio (measured):           {ratio_measured:.3f}")
    print(f"  Ratio (Clausius-Mossotti):  {CM_RATIO:.3f}")
    print(f"  Error: {error_pct:.1f}%")
    print(f"Scattered field max: {scattered_max:.3e} ({scattered_pct:.1f}% of incident)")
    result_str = "PASS" if passed else "FAIL"
    print(f"Result: {result_str} (threshold: {int(PASS_THRESHOLD*100)}%)")
    print("==========================================")

    # --- Plotting ---------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)

    ext_mm = [0.0, NX * DX * 1e3, 0.0, NY * DX * 1e3]
    cx_mm = CX * DX * 1e3
    cy_mm = CY * DX * 1e3
    r_mm = R_CELLS * DX * 1e3
    z_mm = np.arange(NZ) * DX * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"WaveForge 3D — Mie Scattering Accuracy (eps_r={EPS_R})",
        fontsize=13, fontweight="bold"
    )

    # [0, 0]: Free-space Ez XY slice at final step
    ax = axes[0, 0]
    ez_fs_xy = ez_free_final[:, :, CZ]
    vmax_fs = float(np.abs(ez_fs_xy).max()) or 1e-12
    im = ax.imshow(
        ez_fs_xy.T, origin="lower", cmap="RdBu_r",
        vmin=-vmax_fs, vmax=vmax_fs, extent=ext_mm, aspect="auto"
    )
    ax.set_title(f"Free-space Ez XY (step {N_STEPS})", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, label="Ez (V/m)")

    # [0, 1]: With-sphere Ez XY slice at final step (draw sphere boundary)
    ax = axes[0, 1]
    ez_sp_xy = ez_sphere_final[:, :, CZ]
    vmax_sp = float(np.abs(ez_sp_xy).max()) or 1e-12
    im = ax.imshow(
        ez_sp_xy.T, origin="lower", cmap="RdBu_r",
        vmin=-vmax_sp, vmax=vmax_sp, extent=ext_mm, aspect="auto"
    )
    _draw_sphere_circle(ax, cx_mm, cy_mm, r_mm)
    ax.set_title(f"With-sphere Ez XY (step {N_STEPS})", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, label="Ez (V/m)")

    # [1, 0]: Ez along z-axis at snap_step — both runs overlaid
    ax = axes[1, 0]
    ez_free_z = ez_free_snap[CX, CY, :]   # shape (NZ,)
    ez_sphere_z = ez_sphere_snap[CX, CY, :]
    ax.plot(z_mm, ez_free_z, color="steelblue", linewidth=1.5,
            label="Free space")
    ax.plot(z_mm, ez_sphere_z, color="tomato", linewidth=1.5,
            label="With sphere")
    sphere_lo_mm = (CZ - R_CELLS) * DX * 1e3
    sphere_hi_mm = (CZ + R_CELLS) * DX * 1e3
    ax.axvline(sphere_lo_mm, color="orange", linewidth=1.2, linestyle="--",
               label="Sphere boundary")
    ax.axvline(sphere_hi_mm, color="orange", linewidth=1.2, linestyle="--")
    ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_title(f"Ez along z-axis (x=CX, y=CY) at step {SNAP_STEP}", fontsize=10)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Ez (V/m)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # [1, 1]: Scattered field (difference) XY slice at final step
    ax = axes[1, 1]
    diff_xy = ez_scattered_final[:, :, CZ]
    vmax_diff = float(np.abs(diff_xy).max()) or 1e-12
    im = ax.imshow(
        diff_xy.T, origin="lower", cmap="RdBu_r",
        vmin=-vmax_diff, vmax=vmax_diff, extent=ext_mm, aspect="auto"
    )
    _draw_sphere_circle(ax, cx_mm, cy_mm, r_mm)
    result_label = "PASS" if passed else "FAIL"
    ax.set_title(
        f"Scattered field (sphere − free) XY — {result_label} ({error_pct:.1f}% err)",
        fontsize=10
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, label="Ez diff (V/m)")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "3d_07_mie_scattering_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    print(f"Total wall time: {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
