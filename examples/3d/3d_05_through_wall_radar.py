"""
3d_05_through_wall_radar.py — Through-wall radar: detect metal target behind concrete.

1 GHz Ricker pulse penetrates a concrete wall (eps_r=6, sigma=0.05) and
scatters off a metallic target (sigma=3.0). Shows pulse progression.

Run: python examples/3d/3d_05_through_wall_radar.py
Out: examples/output/3d_05_through_wall_radar.png
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
from core.sources import RickerWavelet, PlaneSource, SourceCollection
from core.materials import Material, MaterialMap3D
from core.fdtd3d import FDTD3D

NX, NY, NZ = 80, 48, 48
DX = 5e-3
N_STEPS = 400
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("WaveForge 3D — Through-Wall Radar")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.0f} mm, device={DEVICE}")
    print(f"Domain: {NX*DX*1e3:.0f}x{NY*DX*1e3:.0f}x{NZ*DX*1e3:.0f} mm")
    print("=" * 60)

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    wv = RickerWavelet(amplitude=1.0, peak_freq=1e9)
    src = PlaneSource(wv, plane='yz', position=3, component='Ez',
                      grid=grid, N_steps=N_STEPS)

    concrete = Material('concrete', eps_r=6.0, sigma=0.05)
    metal = Material('metal', eps_r=8.0, sigma=3.0)
    mm = MaterialMap3D(grid)
    mm.add_box((16, 0, 0), (23, NY - 1, NZ - 1), concrete)
    mm.add_box((40, 18, 18), (44, 30, 30), metal)
    Ca, Cb = mm.build3d()

    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]),
                 Ca=Ca, Cb=Cb, n_check=200)

    cy, cz = NY // 2, NZ // 2

    print(f"dt = {grid.dt:.4e} s")
    print(f"Concrete wall: x=[16,23], eps_r=6, sigma=0.05")
    print(f"Metal target: x=[40,44], y=[18,30], z=[18,30], sigma=3.0")
    print(f"Running {N_STEPS} steps...")

    snaps_xz = {}
    snaps_xy = {}
    snap_steps = [150, 250, 350]

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for n in range(N_STEPS):
        sim.step()
        if n + 1 in snap_steps:
            Ez = fields.Ez.detach().cpu().numpy()
            snaps_xz[n + 1] = Ez[:, cy, :].T.copy()
            snaps_xy[n + 1] = Ez[:, :, cz].T.copy()

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mcells = N_STEPS * NX * NY * NZ / elapsed / 1e6
    print(f"Done: {elapsed:.2f}s | {mcells:.1f} Mcells/s")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # Physics analysis
    MU0 = 1.2566370614e-6
    omega = 2 * math.pi * 1e9
    sd = math.sqrt(2 / (omega * MU0 * 0.05)) * 1e3
    wall_mm = (23 - 16 + 1) * DX * 1e3
    loss_dB = 20 * math.log10(math.exp(-wall_mm * 1e-3 / (sd * 1e-3)))
    print(f"\nPhysics: wall={wall_mm:.0f}mm, skin depth={sd:.1f}mm")
    print(f"  One-way wall loss ~ {loss_dB:.1f} dB")

    # Build material map for visualization
    eps_map_xz = np.ones((NZ, NX), dtype=np.float32)
    eps_map_xz[:, 16:24] = 6.0
    eps_map_xz[18:31, 40:45] = 8.0

    eps_map_xy = np.ones((NY, NX), dtype=np.float32)
    eps_map_xy[:, 16:24] = 6.0
    eps_map_xy[18:31, 40:45] = 8.0

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('3D Through-Wall Radar — 1 GHz Ricker, Concrete + Metal Target',
                 fontsize=13, fontweight='bold')

    ext_xz = [0, NX * DX * 1e3, 0, NZ * DX * 1e3]
    ext_xy = [0, NX * DX * 1e3, 0, NY * DX * 1e3]

    # Row 0: XZ plane
    im0 = axes[0, 0].imshow(eps_map_xz, origin='lower', cmap='viridis',
                             extent=ext_xz, aspect='auto')
    axes[0, 0].set(xlabel='x (mm)', ylabel='z (mm)', title='Material map (XZ)')
    plt.colorbar(im0, ax=axes[0, 0], label='eps_r')

    for col, step in enumerate(snap_steps[1:], 1):
        hz = snaps_xz[step]
        vmax = max(np.abs(hz).max(), 1e-12)
        lbl = 'hitting wall' if step == 250 else 'echo returning'
        im = axes[0, col].imshow(hz, origin='lower', cmap='RdBu_r',
                                  vmin=-vmax, vmax=vmax, extent=ext_xz, aspect='auto')
        axes[0, col].axvline(16 * DX * 1e3, color='yellow', lw=1, ls='--')
        axes[0, col].axvline(24 * DX * 1e3, color='yellow', lw=1, ls='--')
        axes[0, col].set(xlabel='x (mm)', ylabel='z (mm)',
                         title=f'Ez step {step} — {lbl}')
        plt.colorbar(im, ax=axes[0, col])

    # Row 1: XY plane
    im0 = axes[1, 0].imshow(eps_map_xy, origin='lower', cmap='viridis',
                             extent=ext_xy, aspect='auto')
    axes[1, 0].set(xlabel='x (mm)', ylabel='y (mm)', title='Material map (XY)')
    plt.colorbar(im0, ax=axes[1, 0], label='eps_r')

    for col, step in enumerate(snap_steps[1:], 1):
        hz = snaps_xy[step]
        vmax = max(np.abs(hz).max(), 1e-12)
        lbl = 'hitting wall' if step == 250 else 'echo returning'
        im = axes[1, col].imshow(hz, origin='lower', cmap='RdBu_r',
                                  vmin=-vmax, vmax=vmax, extent=ext_xy, aspect='auto')
        axes[1, col].axvline(16 * DX * 1e3, color='yellow', lw=1, ls='--')
        axes[1, col].axvline(24 * DX * 1e3, color='yellow', lw=1, ls='--')
        axes[1, col].set(xlabel='x (mm)', ylabel='y (mm)',
                         title=f'Ez step {step} — {lbl}')
        plt.colorbar(im, ax=axes[1, col])

    fig.tight_layout()
    path = OUTPUT_DIR / '3d_05_through_wall_radar.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
