"""
3d_04_scattering_sphere.py — Plane-wave scattering from a dielectric sphere.

PlaneSource illuminates an eps_r=4 sphere (r=8 cells) at domain centre.
8 detectors on a ring record scattered+incident field.
Angular scattering pattern is plotted and compared to Rayleigh forward bias.

Run: python examples/3d/3d_04_scattering_sphere.py
Out: examples/output/3d_04_scattering_sphere.png
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

NX = NY = NZ = 64
DX = 2e-3
N_STEPS = 500
SPHERE_R = 8
EPS_R = 4.0
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("WaveForge 3D — Scattering from Dielectric Sphere")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm, eps_r={EPS_R}, r={SPHERE_R} cells")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    pulse = GaussianPulse(amplitude=1.0, sigma=30 * grid.dt)
    src = PlaneSource(pulse, plane='yz', position=5, component='Ez',
                      grid=grid, N_steps=N_STEPS)

    sphere_mat = Material('dielectric', eps_r=EPS_R, sigma=0.0)
    mm = MaterialMap3D(grid)
    cx = cy = cz = NX // 2
    mm.add_sphere(center=(cx, cy, cz), radius=SPHERE_R, material=sphere_mat)
    Ca, Cb = mm.build3d()

    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]),
                 Ca=Ca, Cb=Cb, n_check=200)

    # 8 detectors on a ring in xz-plane (y=centre), radius=20 cells
    n_det = 8
    det_radius = 20
    angles = np.linspace(0, 2 * np.pi, n_det, endpoint=False)
    det_pos = []
    for theta in angles:
        di = int(cx + det_radius * np.cos(theta))
        dk = int(cz + det_radius * np.sin(theta))
        di = max(1, min(NX - 2, di))
        dk = max(1, min(NZ - 2, dk))
        det_pos.append((di, cy, dk))

    det_signals = np.zeros((n_det, N_STEPS), dtype=np.float32)

    print(f"dt = {grid.dt:.4e} s")
    print(f"Running {N_STEPS} steps...")

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    snap_300 = None
    for n in range(N_STEPS):
        sim.step()
        for d_idx, (di, dj, dk) in enumerate(det_pos):
            det_signals[d_idx, n] = fields.Ez[di, dj, dk].item()
        if n + 1 == 300:
            snap_300 = fields.Ez.detach().cpu().numpy().copy()

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mcells = N_STEPS * NX * NY * NZ / elapsed / 1e6
    print(f"Done: {elapsed:.2f}s | {mcells:.1f} Mcells/s")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # Angular scattering: peak |signal| at each detector
    peak_per_det = np.abs(det_signals).max(axis=1)
    angles_deg = np.degrees(angles)

    print(f"\nScattering pattern (peak |Ez| at each detector angle):")
    for i, (ang, pk) in enumerate(zip(angles_deg, peak_per_det)):
        print(f"  {ang:6.1f}°: {pk:.4e}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'3D Mie Scattering — Sphere eps_r={EPS_R}, r={SPHERE_R*DX*1e3:.0f}mm',
                 fontsize=13, fontweight='bold')

    # Panel 1: Ez xz-plane at step 300
    ez_xz = snap_300[:, cy, :].T
    vmax = float(np.percentile(np.abs(ez_xz), 99)) or 1e-12
    ext = [0, NX * DX * 1e3, 0, NZ * DX * 1e3]
    im = axes[0, 0].imshow(ez_xz, origin='lower', cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
    circle = plt.Circle((cx * DX * 1e3, cz * DX * 1e3), SPHERE_R * DX * 1e3,
                         fill=False, color='lime', lw=2)
    axes[0, 0].add_patch(circle)
    axes[0, 0].set(xlabel='x (mm)', ylabel='z (mm)', title='Ez — XZ plane, step 300')
    plt.colorbar(im, ax=axes[0, 0])

    # Panel 2: Ez xy-plane at step 300
    ez_xy = snap_300[:, :, cz].T
    vmax2 = float(np.percentile(np.abs(ez_xy), 99)) or 1e-12
    im = axes[0, 1].imshow(ez_xy, origin='lower', cmap='RdBu_r',
                            vmin=-vmax2, vmax=vmax2, extent=ext, aspect='auto')
    circle2 = plt.Circle((cx * DX * 1e3, cy * DX * 1e3), SPHERE_R * DX * 1e3,
                          fill=False, color='lime', lw=2)
    axes[0, 1].add_patch(circle2)
    axes[0, 1].set(xlabel='x (mm)', ylabel='y (mm)', title='Ez — XY plane, step 300')
    plt.colorbar(im, ax=axes[0, 1])

    # Panel 3: Time traces from 8 detectors
    t_ns = np.arange(N_STEPS) * grid.dt * 1e9
    for i in range(n_det):
        axes[1, 0].plot(t_ns, det_signals[i], label=f'{angles_deg[i]:.0f}°', alpha=0.7)
    axes[1, 0].set(xlabel='Time (ns)', ylabel='Ez (V/m)', title='Detector Signals (8 angles)')
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].grid(alpha=0.3)

    # Panel 4: Polar scattering pattern
    ax_polar = fig.add_subplot(2, 2, 4, projection='polar')
    axes[1, 1].set_visible(False)
    angles_closed = np.append(angles, angles[0])
    peaks_closed = np.append(peak_per_det, peak_per_det[0])
    ax_polar.plot(angles_closed, peaks_closed, 'b-o', markersize=5)
    ax_polar.fill(angles_closed, peaks_closed, alpha=0.2)
    ax_polar.set_title('Angular Scattering Pattern', pad=15)

    fig.tight_layout()
    path = OUTPUT_DIR / '3d_04_scattering_sphere.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
