"""
3d_02_dielectric_slab.py — Plane-wave reflection/transmission at a 3D dielectric slab.

A Gaussian pulse PlaneSource hits a glass slab (eps_r=4). Detectors record
reflected and transmitted signals. Compares to analytical Fresnel coefficients.

Run: python examples/3d/3d_02_dielectric_slab.py
Out: examples/output/3d_02_dielectric_slab.png
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
N_STEPS = 500
SLAB_X0, SLAB_X1 = 45, 65
EPS_R = 4.0
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("WaveForge 3D — Dielectric Slab Reflection/Transmission")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm, eps_r={EPS_R}")
    print(f"Slab: x=[{SLAB_X0},{SLAB_X1}] cells, device={DEVICE}")
    print("=" * 60)

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    pulse = GaussianPulse(amplitude=1.0, sigma=40 * grid.dt)
    src = PlaneSource(pulse, plane='yz', position=10, component='Ez',
                      grid=grid, N_steps=N_STEPS)

    glass = Material('glass', eps_r=EPS_R, sigma=0.0)
    mm = MaterialMap3D(grid)
    mm.add_box((SLAB_X0, 0, 0), (SLAB_X1, NY - 1, NZ - 1), glass)
    Ca, Cb = mm.build3d()

    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]),
                 Ca=Ca, Cb=Cb, n_check=200)

    det_refl = np.zeros(N_STEPS, dtype=np.float32)
    det_trans = np.zeros(N_STEPS, dtype=np.float32)
    cy, cz = NY // 2, NZ // 2

    print(f"dt = {grid.dt:.4e} s")
    print(f"Running {N_STEPS} steps...")

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for n in range(N_STEPS):
        sim.step()
        det_refl[n] = fields.Ez[30, cy, cz].item()
        det_trans[n] = fields.Ez[80, cy, cz].item()

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mcells = N_STEPS * NX * NY * NZ / elapsed / 1e6
    print(f"Done: {elapsed:.2f}s | {mcells:.1f} Mcells/s")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # Analytical Fresnel (normal incidence)
    n_idx = math.sqrt(EPS_R)
    R_theory = ((n_idx - 1) / (n_idx + 1)) ** 2
    T_theory = 1.0 - R_theory

    peak_refl = float(np.abs(det_refl).max())
    peak_trans = float(np.abs(det_trans).max())
    peak_inc = max(peak_refl, peak_trans, 1e-30)
    R_meas = (peak_refl / peak_inc) ** 2 if peak_inc > 0 else 0
    T_meas = (peak_trans / peak_inc) ** 2 if peak_inc > 0 else 0

    print(f"\nPhysics: eps_r={EPS_R}, n={n_idx:.3f}")
    print(f"  Analytical: R={R_theory*100:.1f}%, T={T_theory*100:.1f}%")
    print(f"  Measured peaks: reflected={peak_refl:.3e}, transmitted={peak_trans:.3e}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'3D Dielectric Slab (eps_r={EPS_R}) — Reflection & Transmission',
                 fontsize=13, fontweight='bold')

    # Panel 1: Ez field xz mid-plane
    Ez = fields.Ez.detach().cpu().numpy()
    ez_xz = Ez[:, cy, :].T
    vmax = max(np.abs(ez_xz).max(), 1e-12)
    ext_xz = [0, NX * DX * 1e3, 0, NZ * DX * 1e3]
    im = axes[0, 0].imshow(ez_xz, origin='lower', cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, extent=ext_xz, aspect='auto')
    axes[0, 0].axvline(SLAB_X0 * DX * 1e3, color='yellow', lw=1.5, ls='--', label='slab')
    axes[0, 0].axvline((SLAB_X1 + 1) * DX * 1e3, color='yellow', lw=1.5, ls='--')
    axes[0, 0].set(xlabel='x (mm)', ylabel='z (mm)', title='Ez — XZ plane (y=centre)')
    axes[0, 0].legend(fontsize=8)
    plt.colorbar(im, ax=axes[0, 0])

    # Panel 2: Ez amplitude along x-axis (1D profile showing reflection/transmission)
    ez_x = Ez[:, cy, cz]
    axes[0, 1].plot(np.arange(NX) * DX * 1e3, ez_x, 'b-', lw=1.5)
    axes[0, 1].axvspan(SLAB_X0 * DX * 1e3, (SLAB_X1 + 1) * DX * 1e3,
                        alpha=0.2, color='orange', label='slab')
    axes[0, 1].set(xlabel='x (mm)', ylabel='Ez (V/m)',
                   title='Ez profile along x (y=z=centre)')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # Panel 3: Detector time signals
    t_ns = np.arange(N_STEPS) * grid.dt * 1e9
    axes[1, 0].plot(t_ns, det_refl, label='x=30 (incident+reflected)', color='blue')
    axes[1, 0].plot(t_ns, det_trans, label='x=80 (transmitted)', color='red')
    axes[1, 0].set(xlabel='Time (ns)', ylabel='Ez (V/m)', title='Detector Signals')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)

    # Panel 4: R/T comparison bar chart
    x_bar = np.arange(2)
    width = 0.35
    axes[1, 1].bar(x_bar - width / 2, [R_theory * 100, T_theory * 100],
                   width, label='Analytical (Fresnel)', color='steelblue')
    axes[1, 1].bar(x_bar + width / 2, [R_meas * 100, T_meas * 100],
                   width, label='FDTD Measured', color='coral')
    axes[1, 1].set_xticks(x_bar)
    axes[1, 1].set_xticklabels(['R (%)', 'T (%)'])
    axes[1, 1].set(ylabel='Percentage', title='Fresnel vs FDTD')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_ylim(0, 110)

    fig.tight_layout()
    path = OUTPUT_DIR / '3d_02_dielectric_slab.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
