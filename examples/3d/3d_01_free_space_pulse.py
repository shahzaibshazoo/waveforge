"""
3d_01_free_space_pulse.py — 3D free-space Gaussian pulse propagation.

Validates the 3D FDTD engine with a point Ez source at the centre of a
64×64×64 grid. Saves three orthogonal Ez field snapshots (xy, xz, yz planes)
and reports throughput.

Run:  python examples/3d/3d_01_free_space_pulse.py
Out:  examples/output/3d_01_free_space_pulse.png
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

# ── Config ────────────────────────────────────────────────────────────────────
NX = NY = NZ = 64
DX = 1.5e-3          # 1.5 mm cell → 96 mm cube domain
N_STEPS = 300
SNAP_STEPS = [100, 200, 300]
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_start = time.perf_counter()

    print("=" * 60)
    print("WaveForge 3D — Free-Space Gaussian Pulse")
    print(f"Grid: {NX}×{NY}×{NZ}, dx={DX*1e3:.1f} mm, device={DEVICE}")
    print(f"Domain: {NX*DX*1e3:.0f}×{NY*DX*1e3:.0f}×{NZ*DX*1e3:.0f} mm")
    print("=" * 60)

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    print(f"dt = {grid.dt:.4e} s")

    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    sigma = 20 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    cx = cy = cz = NX // 2
    src = PointSource(pulse, cx, cy, "Ez", k=cz, grid=grid, N_steps=N_STEPS)
    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]), n_check=100)

    snaps = {}

    print(f"\nRunning {N_STEPS} steps...")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(N_STEPS):
        sim.step()
        if step + 1 in SNAP_STEPS:
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            Ez_np = fields.Ez.detach().cpu().numpy()
            snaps[step + 1] = Ez_np

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mcells = N_STEPS * NX * NY * NZ / elapsed / 1e6

    print(f"Done: {elapsed:.2f}s | {mcells:.1f} Mcells/s | field_max={sim.last_field_max:.3e}")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # ── Plot: 3 snapshots × 3 planes = 9 panels ───────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(f"WaveForge 3D — Free-Space Ez Pulse ({NX}³ grid, dx={DX*1e3:.1f}mm)",
                 fontsize=13, fontweight="bold")

    ext_mm = [0, NX * DX * 1e3, 0, NY * DX * 1e3]

    for col, step in enumerate(SNAP_STEPS):
        Ez = snaps[step]
        vmax = float(np.abs(Ez).max()) or 1e-12

        # Row 0: XY plane (z = centre)
        ax = axes[0, col]
        im = ax.imshow(Ez[:, :, cz].T, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=ext_mm, aspect="auto")
        ax.set(title=f"step {step} — XY (z=centre)",
               xlabel="x (mm)", ylabel="y (mm)")
        plt.colorbar(im, ax=ax, label="Ez (V/m)" if col == 2 else "")

        # Row 1: XZ plane (y = centre)
        ax = axes[1, col]
        im = ax.imshow(Ez[:, cy, :].T, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=ext_mm, aspect="auto")
        ax.set(title=f"step {step} — XZ (y=centre)",
               xlabel="x (mm)", ylabel="z (mm)")
        plt.colorbar(im, ax=ax, label="Ez (V/m)" if col == 2 else "")

        # Row 2: YZ plane (x = centre)
        ax = axes[2, col]
        im = ax.imshow(Ez[cx, :, :].T, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=ext_mm, aspect="auto")
        ax.set(title=f"step {step} — YZ (x=centre)",
               xlabel="y (mm)", ylabel="z (mm)")
        plt.colorbar(im, ax=ax, label="Ez (V/m)" if col == 2 else "")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "3d_01_free_space_pulse.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed_total = time.perf_counter() - t_start
    print(f"\nSaved: {out_path}")
    print(f"Total time: {elapsed_total:.1f}s")
    print(f"Peak |Ez|: {sim.last_field_max:.3e}")
    print(f"Total EM energy (final): {fields.total_energy():.3e} J")
    print("=" * 60)


if __name__ == "__main__":
    main()
