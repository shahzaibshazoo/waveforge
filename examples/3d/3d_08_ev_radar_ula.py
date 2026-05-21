"""
3d_08_ev_radar_ula.py — Automotive radar: 8-element ULA detecting target at angle.

Simulates a 10 GHz-class automotive radar with 8-element Uniform Linear Array.
Target sphere at ~45 degrees from broadside. Delay-and-sum beamforming estimates
target angle and compares to ground truth.

Run: python examples/3d/3d_08_ev_radar_ula.py
Out: examples/output/3d_08_ev_radar_ula.png
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

NX = NY = NZ = 64
DX = 1.5e-3
N_STEPS = 250
N_ELEMENTS = 8
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ULA along x-axis at y=5, z=32 (8 elements spaced 4 cells apart)
ULA_Y = 5
ULA_Z = NZ // 2
ULA_X = [12 + i * 6 for i in range(N_ELEMENTS)]  # x=12,18,24,30,36,42,48,54

# Target at ~45 degrees from broadside (+y direction)
# broadside = +y, angle measured from +y axis
# 45 degrees → target at roughly equal dx and dy from array centre
TGT_X, TGT_Y, TGT_Z = 48, 42, 32  # offset in +x and +y → ~45 deg
TGT_R = 4
TGT_EPS = 12.0

# Ground truth angle
ARRAY_CX = np.mean(ULA_X) * DX
ARRAY_CY = ULA_Y * DX
TGT_DX = TGT_X * DX - ARRAY_CX
TGT_DY = TGT_Y * DX - ARRAY_CY
TRUE_ANGLE = math.degrees(math.atan2(TGT_DX, TGT_DY))
TRUE_RANGE = math.sqrt(TGT_DX**2 + TGT_DY**2)


def run_tx(tx_idx, Ca, Cb):
    """Run one TX and record signals at all elements."""
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)
    pulse = GaussianPulse(amplitude=1.0, sigma=15 * grid.dt)
    src = PointSource(pulse, ULA_X[tx_idx], ULA_Y, 'Ez', k=ULA_Z,
                      grid=grid, N_steps=N_STEPS)
    sim = FDTD3D(grid, fields, boundary, SourceCollection([src]),
                 Ca=Ca, Cb=Cb, n_check=300)

    signals = np.zeros((N_ELEMENTS, N_STEPS), dtype=np.float32)
    snap = None

    with torch.no_grad():
        for n in range(N_STEPS):
            sim.step()
            for rx in range(N_ELEMENTS):
                signals[rx, n] = fields.Ez[ULA_X[rx], ULA_Y, ULA_Z].item()
            if tx_idx == 0 and n + 1 == 150:
                snap = fields.Ez.detach().cpu().numpy().copy()

    return signals, snap


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("WaveForge 3D — EV Radar ULA (8 elements)")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm, device={DEVICE}")
    print(f"ULA: {N_ELEMENTS} elements at y={ULA_Y}, x={ULA_X}")
    print(f"Target: sphere eps_r={TGT_EPS} at ({TGT_X},{TGT_Y},{TGT_Z}), r={TGT_R}")
    print(f"True angle: {TRUE_ANGLE:.1f}° from broadside, range={TRUE_RANGE*1e3:.1f}mm")
    print("=" * 60)

    grid_ref = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    target_mat = Material('target', eps_r=TGT_EPS, sigma=0.0)
    mm = MaterialMap3D(grid_ref)
    mm.add_sphere(center=(TGT_X, TGT_Y, TGT_Z), radius=TGT_R, material=target_mat)
    Ca, Cb = mm.build3d()

    # Also need free-space reference for background subtraction
    Ca_free = torch.ones_like(Ca)
    Cb_free = torch.full_like(Cb, grid_ref.dt / 8.8541878128e-12)

    print(f"Running {N_ELEMENTS} TX x 2 (target + reference) = {N_ELEMENTS*2} sims...")
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    all_signals = np.zeros((N_ELEMENTS, N_ELEMENTS, N_STEPS), dtype=np.float32)
    ref_signals = np.zeros((N_ELEMENTS, N_ELEMENTS, N_STEPS), dtype=np.float32)
    snap_tx0 = None

    for tx in range(N_ELEMENTS):
        sigs, snap = run_tx(tx, Ca, Cb)
        all_signals[tx] = sigs
        if snap is not None:
            snap_tx0 = snap
        # Reference (no target)
        ref_sigs, _ = run_tx(tx, Ca_free, Cb_free)
        ref_signals[tx] = ref_sigs

    # Background subtraction: scattered = total - incident
    scattered = all_signals - ref_signals

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_cells = 2 * N_ELEMENTS * N_STEPS * NX * NY * NZ
    mcells = total_cells / elapsed / 1e6
    print(f"Done: {elapsed:.1f}s | {mcells:.1f} Mcells/s")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # Near-field DAS beamforming — scan angle and range
    C0 = 3e8
    dt = grid_ref.dt

    # Scan in angle (from broadside=+y) at the known target range
    angles_scan = np.linspace(-80, 80, 161)
    beam_power = np.zeros(len(angles_scan))
    scan_range = TRUE_RANGE  # use known range for 1D angle scan

    for ai, angle_deg in enumerate(angles_scan):
        angle_rad = math.radians(angle_deg)
        # Candidate pixel position at this angle and range
        px = ARRAY_CX + scan_range * math.sin(angle_rad)
        py = ARRAY_CY + scan_range * math.cos(angle_rad)
        total = 0.0
        for tx in range(N_ELEMENTS):
            tx_pos = np.array([ULA_X[tx] * DX, ULA_Y * DX])
            for rx in range(N_ELEMENTS):
                rx_pos = np.array([ULA_X[rx] * DX, ULA_Y * DX])
                pix = np.array([px, py])
                d_tx = np.linalg.norm(pix - tx_pos)
                d_rx = np.linalg.norm(pix - rx_pos)
                delay_idx = int((d_tx + d_rx) / C0 / dt)
                if 0 <= delay_idx < N_STEPS:
                    total += scattered[tx, rx, delay_idx]
        beam_power[ai] = total ** 2

    # Normalize
    beam_power /= beam_power.max() + 1e-30

    # Find estimated angle from beamformer
    est_idx = np.argmax(beam_power)
    est_angle_bf = angles_scan[est_idx]

    print(f"\nBeamforming results:")
    print(f"  True target angle:      {TRUE_ANGLE:.1f}°")
    print(f"  Beamformer estimate:    {est_angle_bf:.1f}°")

    # DAS image in xy-plane
    ny_img, nx_img = 50, 50
    das_img = np.zeros((ny_img, nx_img), dtype=np.float64)
    for iy in range(ny_img):
        for ix in range(nx_img):
            px = (ix + 7) * DX
            py = (iy + 7) * DX
            for tx in range(0, N_ELEMENTS, 2):  # subsample for speed
                tx_pos = np.array([ULA_X[tx] * DX, ULA_Y * DX])
                for rx in range(0, N_ELEMENTS, 2):
                    rx_pos = np.array([ULA_X[rx] * DX, ULA_Y * DX])
                    pix_pos = np.array([px, py])
                    d_tx = np.linalg.norm(pix_pos - tx_pos)
                    d_rx = np.linalg.norm(pix_pos - rx_pos)
                    delay_idx = int((d_tx + d_rx) / C0 / dt)
                    if 0 <= delay_idx < N_STEPS:
                        das_img[iy, ix] += scattered[tx, rx, delay_idx]
    das_img = das_img ** 2

    # Estimate angle from DAS image peak
    peak_iy, peak_ix = np.unravel_index(np.argmax(das_img), das_img.shape)
    peak_x_m = (peak_ix + 7) * DX
    peak_y_m = (peak_iy + 7) * DX
    est_dx = peak_x_m - ARRAY_CX
    est_dy = peak_y_m - ARRAY_CY
    est_angle_das = math.degrees(math.atan2(est_dx, est_dy))
    das_error = abs(est_angle_das - TRUE_ANGLE)
    print(f"  DAS image estimate:     {est_angle_das:.1f}°")
    print(f"  DAS angular error:      {das_error:.1f}°")

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'3D EV Radar — 8-element ULA, Target at {TRUE_ANGLE:.0f}°',
                 fontsize=13, fontweight='bold')

    ext = [0, NX * DX * 1e3, 0, NY * DX * 1e3]

    # Panel 1: Geometry
    axes[0, 0].set_xlim(0, NX * DX * 1e3)
    axes[0, 0].set_ylim(0, NY * DX * 1e3)
    circle = plt.Circle((TGT_X * DX * 1e3, TGT_Y * DX * 1e3), TGT_R * DX * 1e3,
                         color='red', alpha=0.5, label=f'target ({TRUE_ANGLE:.0f}°)')
    axes[0, 0].add_patch(circle)
    for x in ULA_X:
        axes[0, 0].plot(x * DX * 1e3, ULA_Y * DX * 1e3, 'b^', markersize=8)
    axes[0, 0].plot([ARRAY_CX * 1e3, TGT_X * DX * 1e3],
                    [ARRAY_CY * 1e3, TGT_Y * DX * 1e3], 'r--', lw=1, alpha=0.5)
    axes[0, 0].set(xlabel='x (mm)', ylabel='y (mm)', title='Geometry (XY, z=centre)')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(alpha=0.3)

    # Panel 2: Ez field from TX0
    if snap_tx0 is not None:
        ez_xy = snap_tx0[:, :, ULA_Z].T
        vmax = float(np.percentile(np.abs(ez_xy), 99)) or 1e-12
        im = axes[0, 1].imshow(ez_xy, origin='lower', cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
        axes[0, 1].set(xlabel='x (mm)', ylabel='y (mm)', title='Ez from TX[0], step 150')
        plt.colorbar(im, ax=axes[0, 1])

    # Panel 3: Angular beamforming
    axes[1, 0].plot(angles_scan, 10 * np.log10(beam_power + 1e-30), 'b-', lw=1.5)
    axes[1, 0].axvline(TRUE_ANGLE, color='red', ls='--', lw=1.5, label=f'true={TRUE_ANGLE:.1f}°')
    axes[1, 0].axvline(est_angle_bf, color='green', ls=':', lw=1.5, label=f'BF={est_angle_bf:.1f}°')
    axes[1, 0].set(xlabel='Angle (degrees)', ylabel='Power (dB)',
                   title=f'Angular Beamforming (near-field)')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlim(-80, 80)

    # Panel 4: DAS image
    img_ext = [7 * DX * 1e3, 57 * DX * 1e3, 7 * DX * 1e3, 57 * DX * 1e3]
    im = axes[1, 1].imshow(das_img, origin='lower', cmap='hot', extent=img_ext, aspect='auto')
    axes[1, 1].plot(TGT_X * DX * 1e3, TGT_Y * DX * 1e3, 'c+', markersize=15, mew=2,
                    label='true pos')
    axes[1, 1].set(xlabel='x (mm)', ylabel='y (mm)', title='DAS Image')
    axes[1, 1].legend(fontsize=9)
    plt.colorbar(im, ax=axes[1, 1], label='Power')

    fig.tight_layout()
    path = OUTPUT_DIR / '3d_08_ev_radar_ula.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
