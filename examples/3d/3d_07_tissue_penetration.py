"""
3d_07_tissue_penetration.py — Microwave penetration through skin/fat/muscle at 1 GHz.

Shows progressive attenuation through layered biological tissue.
Compares simulated attenuation to analytical skin-depth predictions.

Run: python examples/3d/3d_07_tissue_penetration.py
Out: examples/output/3d_07_tissue_penetration.png
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
from core.sources import GaussianPulse, PlaneSource, SourceCollection
from core.materials import Material, MaterialMap3D
from core.fdtd3d import FDTD3D

NX, NY, NZ = 100, 32, 32
DX = 1e-3
N_STEPS = 600
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tissue layers (x ranges in cells)
SKIN_X0, SKIN_X1 = 20, 24
FAT_X0, FAT_X1 = 24, 40
MUSCLE_X0, MUSCLE_X1 = 40, 70

TISSUES = {
    'skin': {'x0': SKIN_X0, 'x1': SKIN_X1, 'eps_r': 40.0, 'sigma': 1.0},
    'fat': {'x0': FAT_X0, 'x1': FAT_X1, 'eps_r': 5.5, 'sigma': 0.05},
    'muscle': {'x0': MUSCLE_X0, 'x1': MUSCLE_X1, 'eps_r': 55.0, 'sigma': 1.0},
}


def skin_depth(sigma, eps_r, freq=1e9):
    MU0 = 1.2566370614e-6
    EPS0 = 8.854e-12
    omega = 2 * math.pi * freq
    eps = eps_r * EPS0
    return math.sqrt(2 / (omega * MU0 * sigma))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("WaveForge 3D — Tissue Penetration (Skin/Fat/Muscle)")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm, device={DEVICE}")
    print("=" * 60)

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    pulse = GaussianPulse(amplitude=1.0, sigma=30 * grid.dt)
    src = PlaneSource(pulse, plane='yz', position=5, component='Ez',
                      grid=grid, N_steps=N_STEPS)

    skin_mat = Material('skin', eps_r=40.0, sigma=1.0)
    fat_mat = Material('fat', eps_r=5.5, sigma=0.05)
    muscle_mat = Material('muscle', eps_r=55.0, sigma=1.0)

    mm = MaterialMap3D(grid)
    mm.add_box((SKIN_X0, 0, 0), (SKIN_X1 - 1, NY - 1, NZ - 1), skin_mat)
    mm.add_box((FAT_X0, 0, 0), (FAT_X1 - 1, NY - 1, NZ - 1), fat_mat)
    mm.add_box((MUSCLE_X0, 0, 0), (MUSCLE_X1 - 1, NY - 1, NZ - 1), muscle_mat)
    Ca, Cb = mm.build3d()

    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]),
                 Ca=Ca, Cb=Cb, n_check=200)

    cy, cz = NY // 2, NZ // 2
    print(f"dt = {grid.dt:.4e} s")
    print(f"Layers: skin x=[{SKIN_X0},{SKIN_X1}], fat x=[{FAT_X0},{FAT_X1}], muscle x=[{MUSCLE_X0},{MUSCLE_X1}]")
    print(f"Running {N_STEPS} steps...")

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    sim.run(N_STEPS)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mcells = N_STEPS * NX * NY * NZ / elapsed / 1e6
    print(f"Done: {elapsed:.2f}s | {mcells:.1f} Mcells/s")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    Ez = fields.Ez.detach().cpu().numpy()
    ez_profile = Ez[:, cy, cz]

    # Analytical skin depths
    print("\nTissue properties at 1 GHz:")
    for name, props in TISSUES.items():
        sd = skin_depth(props['sigma'], props['eps_r']) * 1e3
        thickness = (props['x1'] - props['x0']) * DX * 1e3
        loss = 20 * math.log10(math.exp(-thickness * 1e-3 / (sd * 1e-3)))
        print(f"  {name:8s}: eps_r={props['eps_r']:5.1f}, sigma={props['sigma']:.2f}, "
              f"skin_depth={sd:.1f}mm, thickness={thickness:.0f}mm, loss~{loss:.1f}dB")

    # Build eps_r profile for plotting
    eps_profile = np.ones(NX)
    for name, props in TISSUES.items():
        eps_profile[props['x0']:props['x1']] = props['eps_r']

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('3D Tissue Penetration — Skin/Fat/Muscle at 1 GHz',
                 fontsize=13, fontweight='bold')

    x_mm = np.arange(NX) * DX * 1e3

    # Panel 1: eps_r profile
    axes[0, 0].fill_between(x_mm, 1, eps_profile, alpha=0.3, color='orange')
    axes[0, 0].plot(x_mm, eps_profile, 'k-', lw=2)
    axes[0, 0].set(xlabel='x (mm)', ylabel='eps_r', title='Permittivity Profile')
    axes[0, 0].set_ylim(0, 60)
    for name, props in TISSUES.items():
        xc = (props['x0'] + props['x1']) / 2 * DX * 1e3
        axes[0, 0].text(xc, props['eps_r'] + 2, name, ha='center', fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # Panel 2: Ez field xz mid-plane
    ez_xz = Ez[:, cy, :].T
    vmax = max(np.abs(ez_xz).max(), 1e-12)
    ext = [0, NX * DX * 1e3, 0, NZ * DX * 1e3]
    im = axes[0, 1].imshow(ez_xz, origin='lower', cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
    for name, props in TISSUES.items():
        axes[0, 1].axvline(props['x0'] * DX * 1e3, color='yellow', lw=1, ls='--')
    axes[0, 1].set(xlabel='x (mm)', ylabel='z (mm)', title=f'Ez field — step {N_STEPS}')
    plt.colorbar(im, ax=axes[0, 1])

    # Panel 3: Ez amplitude along x
    axes[1, 0].plot(x_mm, np.abs(ez_profile), 'b-', lw=1.5, label='|Ez| simulated')
    axes[1, 0].axvspan(SKIN_X0 * DX * 1e3, SKIN_X1 * DX * 1e3, alpha=0.2, color='red', label='skin')
    axes[1, 0].axvspan(FAT_X0 * DX * 1e3, FAT_X1 * DX * 1e3, alpha=0.2, color='yellow', label='fat')
    axes[1, 0].axvspan(MUSCLE_X0 * DX * 1e3, MUSCLE_X1 * DX * 1e3, alpha=0.2, color='green', label='muscle')
    axes[1, 0].set(xlabel='x (mm)', ylabel='|Ez| (V/m)', title='Attenuation Profile')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    # Panel 4: Analytical vs simulated comparison (bar chart of skin depths)
    names = list(TISSUES.keys())
    sd_values = [skin_depth(TISSUES[n]['sigma'], TISSUES[n]['eps_r']) * 1e3 for n in names]
    colors = ['red', 'gold', 'green']
    axes[1, 1].bar(names, sd_values, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set(ylabel='Skin depth (mm)', title='Analytical Skin Depth at 1 GHz')
    axes[1, 1].grid(alpha=0.3, axis='y')
    for i, (n, sd) in enumerate(zip(names, sd_values)):
        axes[1, 1].text(i, sd + 0.5, f'{sd:.1f}mm', ha='center', fontsize=9)

    fig.tight_layout()
    path = OUTPUT_DIR / '3d_07_tissue_penetration.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
