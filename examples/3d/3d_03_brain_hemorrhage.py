"""
3d_03_brain_hemorrhage.py — 3D FDTD brain hemorrhage imaging simulation.

Models a simplified human head as concentric spheres (scalp, skull bone, CSF,
brain) with an embedded blood clot offset from centre.  A UWB Gaussian pulse
is transmitted from the left, propagates through the head model, and is
received on the right.  The time-domain signal, orthogonal Ez field slices,
and tissue boundary overlays are saved to a single figure.

Run:  python examples/3d/3d_03_brain_hemorrhage.py
Out:  examples/output/3d_03_brain_hemorrhage.png
"""

import sys
import time
import math
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.fdtd3d import FDTD3D
from core.materials import Material, MaterialMap3D, TISSUE_LIBRARY

# ── Configuration ─────────────────────────────────────────────────────────────
NX = NY = NZ = 48
DX = 2e-3           # 2 mm cell → 96 mm cube domain
N_STEPS = 300
SNAP_STEPS = [100, 200, 300]
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Domain centre in cell indices
CX = CY = CZ = NX // 2  # = 24

# Transmitter and receiver locations
TX_X, TX_Y, TX_Z = 8, 24, 24
RX_X, RX_Y, RX_Z = 40, 24, 24

# ── Material definitions ───────────────────────────────────────────────────────
# TISSUE_LIBRARY values differ from the scenario requirements (1 GHz library
# uses eps_r=40 for generic brain, but we need the exact layer values below).
# Custom Material objects are constructed for each concentric-sphere layer.

MAT_SCALP = Material("scalp",      eps_r=40.0, sigma=1.2)
MAT_BONE  = Material("bone",       eps_r=15.0, sigma=0.8)
MAT_CSF   = Material("csf",        eps_r=80.0, sigma=2.5)
MAT_BRAIN = Material("brain",      eps_r=50.0, sigma=2.0)
MAT_CLOT  = Material("blood_clot", eps_r=65.0, sigma=2.5)


def build_head_model(grid: YeeGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct concentric-sphere head model with painter's algorithm.

    Layers are added outermost-first so inner layers overwrite outer ones.

    Returns
    -------
    (Ca, Cb) : per-cell coefficient tensors, shape (Nx, Ny, Nz).
    """
    mm = MaterialMap3D(grid)

    # Layer 1 — Scalp (outermost)
    mm.add_sphere(center=(CX, CY, CZ), radius=20.0, material=MAT_SCALP)
    # Layer 2 — Skull bone
    mm.add_sphere(center=(CX, CY, CZ), radius=17.0, material=MAT_BONE)
    # Layer 3 — CSF
    mm.add_sphere(center=(CX, CY, CZ), radius=14.0, material=MAT_CSF)
    # Layer 4 — Brain (bulk)
    mm.add_sphere(center=(CX, CY, CZ), radius=12.0, material=MAT_BRAIN)
    # Layer 5 — Blood clot (offset from centre)
    mm.add_sphere(center=(24, 22, 22), radius=3.0,  material=MAT_CLOT)

    Ca, Cb = mm.build3d()
    return Ca, Cb


def _draw_sphere_circles(ax: object, cx_mm: float, cy_mm: float,
                         radii_mm: list, colors: list, labels: list) -> None:
    """Overlay dashed circles on a slice plot showing sphere cross-sections.

    Parameters
    ----------
    ax     : matplotlib Axes
    cx_mm, cy_mm : circle centre in mm (plot coordinates)
    radii_mm     : list of radii in mm
    colors, labels: per-circle colour string and legend label
    """
    import matplotlib.patches as mpatches

    for r, col, lbl in zip(radii_mm, colors, labels):
        circle = mpatches.Circle(
            (cx_mm, cy_mm), radius=r,
            fill=False, linestyle="--", linewidth=0.9, color=col, label=lbl
        )
        ax.add_patch(circle)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_start = time.perf_counter()

    print("=" * 62)
    print("WaveForge 3D — Brain Hemorrhage Imaging")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm, device={DEVICE}")
    print(f"Domain: {NX*DX*1e3:.0f}x{NY*DX*1e3:.0f}x{NZ*DX*1e3:.0f} mm")
    print(f"Tx at ({TX_X},{TX_Y},{TX_Z}), Rx at ({RX_X},{RX_Y},{RX_Z})")
    print("=" * 62)

    # ── Grid, fields, boundary ─────────────────────────────────────────
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    print(f"dt = {grid.dt:.4e} s")

    fields   = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    # ── Material map ───────────────────────────────────────────────────
    print("Building head model (concentric spheres + blood clot)...")
    Ca, Cb = build_head_model(grid)
    print(f"Ca range: [{float(Ca.min()):.4f}, {float(Ca.max()):.4f}]")
    print(f"Cb range: [{float(Cb.min()):.3e}, {float(Cb.max()):.3e}]")

    # ── Source ─────────────────────────────────────────────────────────
    sigma_t = 10.0 * grid.dt
    pulse   = GaussianPulse(amplitude=1.0, sigma=sigma_t)
    src     = PointSource(pulse, TX_X, TX_Y, "Ez", k=TX_Z, grid=grid, N_steps=N_STEPS)
    sources = SourceCollection([src])

    # ── Simulator ──────────────────────────────────────────────────────
    sim = FDTD3D(grid, fields, boundary, sources, Ca=Ca, Cb=Cb, n_check=100)

    # ── Time loop ──────────────────────────────────────────────────────
    receiver_signal: list[float] = []
    snaps: dict[int, np.ndarray] = {}

    print(f"\nRunning {N_STEPS} steps...")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(N_STEPS):
        sim.step()
        # Record Ez at receiver (every step)
        receiver_signal.append(float(fields.Ez[RX_X, RX_Y, RX_Z].cpu()))
        if step + 1 in SNAP_STEPS:
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            snaps[step + 1] = fields.Ez.detach().cpu().numpy()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mcells  = N_STEPS * NX * NY * NZ / elapsed / 1e6

    # ── Metrics ────────────────────────────────────────────────────────
    Ez_snap200 = snaps[200]

    # Peak receiver signal
    peak_rx = float(np.abs(receiver_signal).max())

    # Peak inside brain sphere (radius 12 cells from centre)
    # Use snapshot at step 200 to compare interior vs exterior
    ix = np.arange(NX)
    iy = np.arange(NY)
    iz = np.arange(NZ)
    I3, J3, K3 = np.meshgrid(ix, iy, iz, indexing="ij")
    dist2 = (I3 - CX)**2 + (J3 - CY)**2 + (K3 - CZ)**2
    brain_mask = dist2 <= 12.0**2
    outside_mask = dist2 >= 20.0**2
    peak_inside  = float(np.abs(Ez_snap200[brain_mask]).max()) if brain_mask.any() else 0.0
    peak_outside = float(np.abs(Ez_snap200[outside_mask]).max()) if outside_mask.any() else 1e-12

    # Approximate wave arrival time: straight-line path / c0
    tx_rx_dist = (RX_X - TX_X) * DX  # metres
    c0 = 3e8
    t_arrival = tx_rx_dist / c0        # seconds in free space
    step_arrival = int(t_arrival / grid.dt)

    print(f"\nDone: {elapsed:.2f}s | {mcells:.1f} Mcells/s")
    print(f"Peak |Ez| at receiver            : {peak_rx:.3e}")
    print(f"Peak |Ez| inside brain (step 200): {peak_inside:.3e}")
    print(f"Peak |Ez| outside head (step 200): {peak_outside:.3e}")
    if peak_outside > 0:
        attenuation_dB = -20.0 * math.log10(max(peak_inside / peak_outside, 1e-12))
        print(f"Head attenuation (approx)        : {attenuation_dB:.1f} dB")
    print(f"Blood clot eps_r={MAT_CLOT.eps_r} vs brain eps_r={MAT_BRAIN.eps_r} "
          f"=> {(MAT_CLOT.eps_r - MAT_BRAIN.eps_r) / MAT_BRAIN.eps_r * 100:.0f}% "
          f"dielectric contrast (clot detectable via scattered field)")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # ── Figure ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        "WaveForge 3D — Brain Hemorrhage Model (48³, dx=2mm)",
        fontsize=13, fontweight="bold"
    )

    Ez200 = snaps[200]
    vmax  = float(np.abs(Ez200).max()) or 1e-12

    # Convert cell-centre positions to mm for plot extents
    ext_mm = [0, NX * DX * 1e3, 0, NY * DX * 1e3]
    cx_mm  = CX * DX * 1e3
    cy_mm  = CY * DX * 1e3

    # Circle radii in mm for sphere boundary overlays
    radii_mm = [r * DX * 1e3 for r in (20.0, 17.0, 14.0, 12.0)]
    layer_colors  = ["#ff9900", "#66ccff", "#99ff99", "#ff6666"]
    layer_labels  = ["Scalp r=20", "Bone r=17", "CSF r=14", "Brain r=12"]

    # ── Panel [0,0]: Ez XY slice at z=centre (step 200) ────────────────
    ax = axes[0, 0]
    im = ax.imshow(
        Ez200[:, :, CZ].T, origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, extent=ext_mm, aspect="auto"
    )
    _draw_sphere_circles(ax, cx_mm, cy_mm, radii_mm, layer_colors, layer_labels)
    ax.legend(fontsize=6, loc="upper right")
    ax.set(title="Ez XY-slice (z=centre), step 200",
           xlabel="x (mm)", ylabel="y (mm)")
    plt.colorbar(im, ax=ax, label="Ez (V/m)")

    # ── Panel [0,1]: Ez XZ slice at y=centre (step 200) ────────────────
    ax = axes[0, 1]
    im = ax.imshow(
        Ez200[:, CY, :].T, origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, extent=ext_mm, aspect="auto"
    )
    _draw_sphere_circles(ax, cx_mm, cy_mm, radii_mm, layer_colors, layer_labels)
    ax.set(title="Ez XZ-slice (y=centre), step 200",
           xlabel="x (mm)", ylabel="z (mm)")
    plt.colorbar(im, ax=ax, label="Ez (V/m)")

    # ── Panel [1,0]: Ez YZ slice at x=centre (step 200) ────────────────
    ax = axes[1, 0]
    im = ax.imshow(
        Ez200[CX, :, :].T, origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, extent=ext_mm, aspect="auto"
    )
    _draw_sphere_circles(ax, cx_mm, cy_mm, radii_mm, layer_colors, layer_labels)
    ax.set(title="Ez YZ-slice (x=centre), step 200",
           xlabel="y (mm)", ylabel="z (mm)")
    plt.colorbar(im, ax=ax, label="Ez (V/m)")

    # ── Panel [1,1]: Receiver time-domain signal ────────────────────────
    ax = axes[1, 1]
    steps_arr = np.arange(N_STEPS)
    ax.plot(steps_arr, receiver_signal, color="#1a6faf", linewidth=0.9,
            label="Ez at Rx")
    ax.axvline(step_arrival, color="red", linestyle="--", linewidth=1.0,
               label=f"Approx arrival (step {step_arrival})")
    ax.set(title="Tx→Head→Rx signal",
           xlabel="Time step", ylabel="Ez (V/m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotate peak receiver value
    ax.annotate(
        f"peak={peak_rx:.2e}",
        xy=(np.argmax(np.abs(receiver_signal)), peak_rx),
        xytext=(10, 10), textcoords="offset points",
        fontsize=7, color="darkgreen",
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8)
    )

    fig.tight_layout()
    out_path = OUTPUT_DIR / "3d_03_brain_hemorrhage.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed_total = time.perf_counter() - t_start
    print(f"\nSaved: {out_path}")
    print(f"Total time: {elapsed_total:.1f}s")
    print("=" * 62)


if __name__ == "__main__":
    main()
