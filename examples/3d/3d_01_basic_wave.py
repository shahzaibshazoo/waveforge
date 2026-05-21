"""
3d_01_basic_wave.py — WaveForge 3D smoke test and usage tutorial.

Demonstrates 3D FDTD simulation of a Gaussian pulse propagating in free space.
Grid: 64x64x64, dx=1.5mm, Mur ABC, point Ez source at centre.
Saves 9-panel orthogonal slice figure and reports throughput.

Run: python examples/3d/3d_01_basic_wave.py
Out: examples/output/3d_01_basic_wave.png
"""
import sys
import math
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.fdtd3d import FDTD3D

NX = NY = NZ = 64
DX = 1.5e-3
N_STEPS = 300
SNAP_STEPS = [100, 200, 300]
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("WaveForge 3D — Basic Wave Propagation")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm, device={DEVICE}")
    print(f"Domain: {NX*DX*1e3:.0f}x{NY*DX*1e3:.0f}x{NZ*DX*1e3:.0f} mm")
    print("=" * 60)

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    sigma = 20 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    cx = cy = cz = NX // 2
    src = PointSource(pulse, cx, cy, 'Ez', k=cz, grid=grid, N_steps=N_STEPS)
    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]), n_check=100)

    print(f"dt = {grid.dt:.4e} s")
    print(f"\nRunning {N_STEPS} steps...")

    snaps = {}
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(N_STEPS):
        sim.step()
        if step + 1 in SNAP_STEPS:
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            snaps[step + 1] = fields.Ez.detach().cpu().numpy().copy()

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mcells = N_STEPS * NX * NY * NZ / elapsed / 1e6

    print(f"Done: {elapsed:.2f}s | {mcells:.1f} Mcells/s")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")
    print(f"Peak |Ez|: {sim.last_field_max:.3e}")
    print(f"Total EM energy: {fields.total_energy():.3e} J")

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle(f"WaveForge 3D — Free-Space Ez Pulse ({NX}³, dx={DX*1e3:.1f}mm, {mcells:.1f} Mcells/s)",
                 fontsize=13, fontweight='bold')

    ext_mm = [0, NX * DX * 1e3, 0, NY * DX * 1e3]
    plane_labels = ['XY (z=centre)', 'XZ (y=centre)', 'YZ (x=centre)']

    for col, step in enumerate(SNAP_STEPS):
        Ez = snaps[step]
        # Use 99th percentile for vmax to avoid near-field source dominance
        vmax = float(np.percentile(np.abs(Ez), 99.5)) or 1e-12

        # Row 0: XY plane
        im = axes[0, col].imshow(Ez[:, :, cz].T, origin='lower', cmap='RdBu_r',
                                  vmin=-vmax, vmax=vmax, extent=ext_mm, aspect='auto')
        axes[0, col].set(title=f'step {step} — {plane_labels[0]}',
                         xlabel='x (mm)', ylabel='y (mm)')
        plt.colorbar(im, ax=axes[0, col])

        # Row 1: XZ plane
        im = axes[1, col].imshow(Ez[:, cy, :].T, origin='lower', cmap='RdBu_r',
                                  vmin=-vmax, vmax=vmax, extent=ext_mm, aspect='auto')
        axes[1, col].set(title=f'step {step} — {plane_labels[1]}',
                         xlabel='x (mm)', ylabel='z (mm)')
        plt.colorbar(im, ax=axes[1, col])

        # Row 2: YZ plane
        im = axes[2, col].imshow(Ez[cx, :, :].T, origin='lower', cmap='RdBu_r',
                                  vmin=-vmax, vmax=vmax, extent=ext_mm, aspect='auto')
        axes[2, col].set(title=f'step {step} — {plane_labels[2]}',
                         xlabel='y (mm)', ylabel='z (mm)')
        plt.colorbar(im, ax=axes[2, col])

    fig.tight_layout()
    path = OUTPUT_DIR / '3d_01_basic_wave.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
