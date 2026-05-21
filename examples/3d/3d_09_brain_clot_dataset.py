"""
3d_09_brain_clot_dataset.py — Mini 3D brain clot dataset generator.

Creates 2 brain phantom simulations (healthy + clot). For each: runs 4 TX
FDTD simulations, collects signals, shows differential detection.

Run: python examples/3d/3d_09_brain_clot_dataset.py
Out: examples/output/3d_09_brain_clot_dataset.png
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
from core.materials import Material, MaterialMap3D
from core.fdtd3d import FDTD3D

NX = NY = NZ = 48
DX = 2e-3
N_STEPS = 200
N_TX = 4
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CX = CY = CZ = NX // 2

# TX positions (ring in z=centre plane)
TX_POS = [(3, CY, CZ), (NX - 4, CY, CZ), (CX, 3, CZ), (CX, NY - 4, CZ)]

# Brain tissue properties
SCALP = Material('scalp', eps_r=40.0, sigma=0.8)
SKULL = Material('skull', eps_r=12.0, sigma=0.1)
BRAIN = Material('brain', eps_r=50.0, sigma=0.7)
CLOT = Material('clot', eps_r=60.0, sigma=1.5)

CLOT_POS = (30, 24, 24)
CLOT_R = 3


def build_materials(include_clot=False):
    """Build Ca/Cb for brain phantom (with or without clot)."""
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    mm = MaterialMap3D(grid)
    # Painter's algorithm: add in order, later overwrites earlier
    mm.add_sphere(center=(CX, CY, CZ), radius=22, material=SCALP)
    mm.add_sphere(center=(CX, CY, CZ), radius=20, material=SKULL)
    mm.add_sphere(center=(CX, CY, CZ), radius=17, material=BRAIN)
    if include_clot:
        mm.add_sphere(center=CLOT_POS, radius=CLOT_R, material=CLOT)
    Ca, Cb = mm.build3d()
    return grid, Ca, Cb


def run_sample(Ca, Cb, label=""):
    """Run all TX positions for one sample, return signals array [N_TX, N_TX, N_STEPS]."""
    signals = np.zeros((N_TX, N_TX, N_STEPS), dtype=np.float32)
    snap = None

    for tx_idx in range(N_TX):
        grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
        fields = FieldSet(grid)
        boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)
        ti, tj, tk = TX_POS[tx_idx]
        src = PointSource(pulse, ti, tj, 'Ez', k=tk, grid=grid, N_steps=N_STEPS)
        sim = FDTD3D(grid, fields, boundary, SourceCollection([src]),
                     Ca=Ca, Cb=Cb, n_check=200)

        for n in range(N_STEPS):
            sim.step()
            for rx in range(N_TX):
                ri, rj, rk = TX_POS[rx]
                signals[tx_idx, rx, n] = fields.Ez[ri, rj, rk].item()

        if tx_idx == 0 and snap is None:
            snap = fields.Ez.detach().cpu().numpy().copy()

    return signals, snap


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("WaveForge 3D — Brain Clot Detection Dataset")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.0f} mm, device={DEVICE}")
    print(f"Samples: healthy + clot at {CLOT_POS}, r={CLOT_R}")
    print("=" * 60)

    # Run healthy
    print("\n--- Sample 0: Healthy brain ---")
    grid_h, Ca_h, Cb_h = build_materials(include_clot=False)

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    sigs_healthy, snap_healthy = run_sample(Ca_h, Cb_h, "healthy")
    print(f"  Healthy done ({N_TX} TX x {N_STEPS} steps)")

    # Run with clot
    print("--- Sample 1: Brain + clot ---")
    grid_c, Ca_c, Cb_c = build_materials(include_clot=True)
    sigs_clot, snap_clot = run_sample(Ca_c, Cb_c, "clot")
    print(f"  Clot done ({N_TX} TX x {N_STEPS} steps)")

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_cells = 2 * N_TX * N_STEPS * NX * NY * NZ
    mcells = total_cells / elapsed / 1e6
    print(f"\nTotal: {elapsed:.1f}s | {mcells:.1f} Mcells/s")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # Differential signal
    diff = sigs_clot - sigs_healthy
    diff_energy = float(np.sum(diff ** 2))
    print(f"\nDifferential energy (clot - healthy): {diff_energy:.4e}")
    print(f"Detection metric: {'DETECTED' if diff_energy > 1e-6 else 'NOT detected'}")

    # Build eps_r map for visualization
    eps_map = np.ones((NY, NX), dtype=np.float32)
    I, J = np.meshgrid(np.arange(NX), np.arange(NY))
    mask_scalp = (I - CX) ** 2 + (J - CY) ** 2 <= 22 ** 2
    mask_skull = (I - CX) ** 2 + (J - CY) ** 2 <= 20 ** 2
    mask_brain = (I - CX) ** 2 + (J - CY) ** 2 <= 17 ** 2
    mask_clot = (I - CLOT_POS[0]) ** 2 + (J - CLOT_POS[1]) ** 2 <= CLOT_R ** 2
    eps_map[mask_scalp] = 40
    eps_map[mask_skull] = 12
    eps_map[mask_brain] = 50

    eps_map_clot = eps_map.copy()
    eps_map_clot[mask_clot] = 60

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('3D Brain Clot Detection — Healthy vs Clot',
                 fontsize=13, fontweight='bold')

    ext = [0, NX * DX * 1e3, 0, NY * DX * 1e3]

    # Row 0: Healthy
    im = axes[0, 0].imshow(eps_map, origin='lower', cmap='viridis', extent=ext, aspect='auto')
    axes[0, 0].set(xlabel='x (mm)', ylabel='y (mm)', title='Healthy — tissue map')
    plt.colorbar(im, ax=axes[0, 0], label='eps_r')

    if snap_healthy is not None:
        ez = snap_healthy[:, :, CZ].T
        vmax = float(np.percentile(np.abs(ez), 99)) or 1e-12
        im = axes[0, 1].imshow(ez, origin='lower', cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
        axes[0, 1].set(xlabel='x (mm)', ylabel='y (mm)', title='Healthy — Ez TX[0] final')
        plt.colorbar(im, ax=axes[0, 1])

    t_ns = np.arange(N_STEPS) * grid_h.dt * 1e9
    for rx in range(N_TX):
        axes[0, 2].plot(t_ns, sigs_healthy[0, rx], label=f'RX[{rx}]', alpha=0.7)
    axes[0, 2].set(xlabel='Time (ns)', ylabel='Ez', title='Healthy — TX[0] signals')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(alpha=0.3)

    # Row 1: Clot
    im = axes[1, 0].imshow(eps_map_clot, origin='lower', cmap='viridis', extent=ext, aspect='auto')
    axes[1, 0].set(xlabel='x (mm)', ylabel='y (mm)', title='Clot — tissue map')
    plt.colorbar(im, ax=axes[1, 0], label='eps_r')

    if snap_clot is not None:
        ez = snap_clot[:, :, CZ].T
        vmax = float(np.percentile(np.abs(ez), 99)) or 1e-12
        im = axes[1, 1].imshow(ez, origin='lower', cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
        axes[1, 1].set(xlabel='x (mm)', ylabel='y (mm)', title='Clot — Ez TX[0] final')
        plt.colorbar(im, ax=axes[1, 1])

    # Differential signals
    for rx in range(N_TX):
        axes[1, 2].plot(t_ns, diff[0, rx], label=f'RX[{rx}]', alpha=0.7)
    axes[1, 2].set(xlabel='Time (ns)', ylabel='ΔEz', title='Differential (clot − healthy)')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / '3d_09_brain_clot_dataset.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
