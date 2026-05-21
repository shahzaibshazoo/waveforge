"""
3d_10_breast_tumor_mimo.py — 3D Microwave MIMO Imaging of Breast Tumor.

Simulates a 3D breast phantom with a malignant tumor using a 4-element MIMO
antenna array at 1 GHz.  The breast is modelled as concentric spheres:
  - Skin shell (eps_r=40, sigma=0.8)
  - Adipose fat interior (eps_r=5, sigma=0.05)
  - Fibroglandular core (eps_r=12, sigma=0.2)
  - Tumor (eps_r=55, sigma=1.5) — offset from centre

For each of 4 transmit positions, a separate FDTD simulation is run with
an Ez point source, and the received signals at the other 3 positions are
collected.  A delay-and-sum (DAS) beamforming image is formed by
backprojecting all TX/RX pairs into the xy-plane at z=centre.

Physical setup:
  Domain:    64x64x64 cells, dx=dy=dz=2mm -> 128mm cube
  Breast:    Spherical, skin radius=28, adipose radius=26, fibro radius=12
  Tumor:     r=4 at (38,36,32) — offset in +x, +y from centre
  Antennas:  4 TX at (5,32,32), (59,32,32), (32,5,32), (32,59,32)

Output: 4-panel figure saved to examples/output/3d_10_breast_tumor_mimo.png
  Panel 1: Material cross-section (xy at z=32) showing tissue layers
  Panel 2: Ez field from TX0 at step 200 (xy at z=32)
  Panel 3: Received signals (time traces from all TX/RX pairs)
  Panel 4: Simple DAS image (delay-and-sum backprojection in xy plane)

Run:  python examples/3d/3d_10_breast_tumor_mimo.py
Out:  examples/output/3d_10_breast_tumor_mimo.png
"""

import sys
import math
from pathlib import Path
import numpy as np
import time
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.materials import Material, MaterialMap3D
from core.fdtd3d import FDTD3D

# ── Configuration ─────────────────────────────────────────────────────────────
NX = NY = NZ = 64
DX = 2e-3               # 2 mm cell -> 128 mm cube domain
N_STEPS = 300
FREQ = 1e9              # 1 GHz centre frequency
CENTER = (32, 32, 32)   # domain centre in cell indices

# Breast geometry (spherical model)
SKIN_RADIUS = 28        # outer skin shell
ADIPOSE_RADIUS = 26     # adipose interior (inside skin)
FIBRO_RADIUS = 12       # fibroglandular core
TUMOR_CENTER = (38, 36, 32)  # offset toward +x, +y
TUMOR_RADIUS = 4

# Tissue materials at 1 GHz
MAT_SKIN    = Material('skin',            eps_r=40.0, sigma=0.8)
MAT_ADIPOSE = Material('adipose',         eps_r=5.0,  sigma=0.05)
MAT_FIBRO   = Material('fibroglandular',  eps_r=12.0, sigma=0.2)
MAT_TUMOR   = Material('tumor',           eps_r=55.0, sigma=1.5)

# 4 transmit positions (ring in z=32 plane)
TX_POSITIONS = [
    (5, 32, 32),    # left
    (59, 32, 32),   # right
    (32, 5, 32),    # bottom
    (32, 59, 32),   # top
]

OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_breast_model(grid: YeeGrid):
    """Construct spherical breast phantom with painter's algorithm.

    Layers added outermost-first; inner layers overwrite outer ones.

    Returns
    -------
    (Ca, Cb) : per-cell coefficient tensors, shape (Nx, Ny, Nz).
    eps_map  : numpy array (Nx, Ny) of eps_r values at z=32 for visualization.
    """
    mm = MaterialMap3D(grid)

    # Painter's algorithm: skin first (outermost), then inward
    mm.add_sphere(center=CENTER, radius=SKIN_RADIUS, material=MAT_SKIN)
    mm.add_sphere(center=CENTER, radius=ADIPOSE_RADIUS, material=MAT_ADIPOSE)
    mm.add_sphere(center=CENTER, radius=FIBRO_RADIUS, material=MAT_FIBRO)
    mm.add_sphere(center=TUMOR_CENTER, radius=TUMOR_RADIUS, material=MAT_TUMOR)

    Ca, Cb = mm.build3d()

    # Build eps_r visualization map at z=32 (xy cross-section)
    cx, cy, cz = float(CENTER[0]), float(CENTER[1]), float(CENTER[2])
    tx, ty, tz = float(TUMOR_CENTER[0]), float(TUMOR_CENTER[1]), float(TUMOR_CENTER[2])
    kz = 32  # z-slice index

    I = np.arange(NX).reshape(-1, 1) * np.ones((1, NY))
    J = np.ones((NX, 1)) * np.arange(NY).reshape(1, -1)

    eps_map = np.ones((NX, NY), dtype=np.float32)  # air background

    # Spherical cross-sections at z=kz (circle in xy plane)
    r_skin_xy = np.sqrt(max(SKIN_RADIUS**2 - (kz - cz)**2, 0))
    r_adip_xy = np.sqrt(max(ADIPOSE_RADIUS**2 - (kz - cz)**2, 0))
    r_fibro_xy = np.sqrt(max(FIBRO_RADIUS**2 - (kz - cz)**2, 0))
    r_tumor_xy = np.sqrt(max(TUMOR_RADIUS**2 - (kz - tz)**2, 0))

    dist_center = np.sqrt((I - cx)**2 + (J - cy)**2)
    dist_tumor = np.sqrt((I - tx)**2 + (J - ty)**2)

    in_skin = dist_center <= r_skin_xy
    in_adip = dist_center <= r_adip_xy
    in_fibro = dist_center <= r_fibro_xy
    in_tumor = dist_tumor <= r_tumor_xy

    # Apply layers (painter's algorithm for 2D visualization)
    eps_map[in_skin] = MAT_SKIN.eps_r
    eps_map[in_adip] = MAT_ADIPOSE.eps_r
    eps_map[in_fibro] = MAT_FIBRO.eps_r
    eps_map[in_tumor] = MAT_TUMOR.eps_r

    return Ca, Cb, eps_map


def run_mimo_3d(grid, Ca, Cb, tx_positions, n_steps):
    """Run separate FDTD simulation for each TX, collect RX signals.

    Parameters
    ----------
    grid : YeeGrid
    Ca, Cb : material coefficient tensors
    tx_positions : list of (i, j, k) tuples
    n_steps : int

    Returns
    -------
    all_sigs : ndarray shape (N_TX, N_TX, n_steps) — signal at each RX for each TX
    ez_snap  : ndarray — Ez field snapshot from TX0 at step 200
    """
    N_TX = len(tx_positions)
    all_sigs = np.zeros((N_TX, N_TX, n_steps), dtype=np.float32)
    ez_snap = None

    # Gaussian pulse source (broadband, centred at ~1 GHz)
    sigma_t = 1.0 / (2.0 * math.pi * FREQ / 3.0)
    t0 = 5.0 * sigma_t

    for tx_idx in range(N_TX):
        ti, tj, tk = tx_positions[tx_idx]
        print(f"  TX {tx_idx+1}/{N_TX}: source at ({ti},{tj},{tk})", end='')

        # Fresh fields and boundary for each TX
        fields = FieldSet(grid)
        boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

        # Source: Ez point source with Gaussian pulse
        pulse = GaussianPulse(amplitude=1000.0, sigma=sigma_t, t0=t0)
        src = PointSource(pulse, ti, tj, 'Ez', k=tk, grid=grid, N_steps=n_steps)
        sources = SourceCollection([src])

        # Simulator
        sim = FDTD3D(grid, fields, boundary, sources, Ca=Ca, Cb=Cb, n_check=500)

        # RX indices on device
        rx_i = torch.tensor([p[0] for p in tx_positions], dtype=torch.long, device=grid.device)
        rx_j = torch.tensor([p[1] for p in tx_positions], dtype=torch.long, device=grid.device)
        rx_k = torch.tensor([p[2] for p in tx_positions], dtype=torch.long, device=grid.device)

        # Time-step loop
        for step in range(n_steps):
            sim.step()
            # Record Ez at all antenna positions
            all_sigs[tx_idx, :, step] = fields.Ez[rx_i, rx_j, rx_k].cpu().numpy()

            # Save snapshot from TX0 at step 200
            if tx_idx == 0 and step == 199:
                ez_snap = fields.Ez[:, :, CENTER[2]].detach().cpu().numpy()

        print(f" done")

    return all_sigs, ez_snap


def das_backprojection_3d(signals, tx_positions, grid, n_steps):
    """Delay-and-sum backprojection onto the xy plane at z=centre.

    Parameters
    ----------
    signals : ndarray (N_TX, N_TX, n_steps)
    tx_positions : list of (i, j, k) tuples
    grid : YeeGrid
    n_steps : int

    Returns
    -------
    image : ndarray (NX, NY) — DAS power image
    """
    dt = grid.dt
    dx = grid.dx
    c0 = 3e8

    # Effective permittivity for wave speed in breast tissue
    # Weighted average: mostly adipose with some fibroglandular
    eps_eff = 0.70 * MAT_ADIPOSE.eps_r + 0.20 * MAT_FIBRO.eps_r + 0.10 * MAT_SKIN.eps_r
    v_bg = c0 / math.sqrt(eps_eff)

    N_TX = len(tx_positions)
    ant_x = np.array([p[0] * dx for p in tx_positions])
    ant_y = np.array([p[1] * dx for p in tx_positions])

    # Pixel coordinates in metres
    PX = (np.arange(NX) * dx).reshape(-1, 1)
    PY = (np.arange(NY) * dx).reshape(1, -1)

    image = np.zeros((NX, NY), dtype=np.float64)

    for tx in range(N_TX):
        for rx in range(N_TX):
            if tx == rx:
                continue  # skip self-reception (TX==RX same position)
            # Distance from TX to each pixel, and from pixel to RX
            d_tx = np.sqrt((PX - ant_x[tx])**2 + (PY - ant_y[tx])**2)
            d_rx = np.sqrt((PX - ant_x[rx])**2 + (PY - ant_y[rx])**2)
            # Time delay in samples
            n_tau = np.clip(
                np.round((d_tx + d_rx) / v_bg / dt).astype(np.int32),
                0, n_steps - 1
            )
            # Sum signal at computed delay for each pixel
            image += signals[tx, rx, n_tau]

    return image**2


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_start = time.perf_counter()

    print("=" * 64)
    print("WaveForge 3D — Breast Tumor MIMO Imaging (1 GHz)")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f}mm, device={DEVICE}")
    print(f"Domain: {NX*DX*1e3:.0f}x{NY*DX*1e3:.0f}x{NZ*DX*1e3:.0f} mm")
    print(f"Breast model: skin r={SKIN_RADIUS}, adipose r={ADIPOSE_RADIUS}, "
          f"fibro r={FIBRO_RADIUS}")
    print(f"Tumor: r={TUMOR_RADIUS} at {TUMOR_CENTER} "
          f"(eps_r={MAT_TUMOR.eps_r}, sigma={MAT_TUMOR.sigma})")
    print(f"TX positions: {TX_POSITIONS}")
    print("=" * 64)

    # ── Tissue properties ─────────────────────────────────────────────
    print("\nTissue properties at 1 GHz:")
    for mat in [MAT_SKIN, MAT_ADIPOSE, MAT_FIBRO, MAT_TUMOR]:
        sd = mat.skin_depth(FREQ)
        wl = mat.wavelength(FREQ)
        print(f"  {mat.name:18s}: eps_r={mat.eps_r:5.1f}, "
              f"sigma={mat.sigma:5.2f} S/m, "
              f"skin_depth={sd*1e3:6.1f}mm, "
              f"lambda={wl*1e3:5.1f}mm")

    # ── Grid and material model ───────────────────────────────────────
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    print(f"\ndt = {grid.dt:.4e} s")
    print(f"Cells per wavelength (adipose): "
          f"{grid.wavelength_resolution(FREQ / math.sqrt(MAT_ADIPOSE.eps_r)):.1f}")

    print("Building breast model...")
    Ca, Cb, eps_map = build_breast_model(grid)
    print(f"Ca range: [{float(Ca.min()):.4f}, {float(Ca.max()):.4f}]")
    print(f"Cb range: [{float(Cb.min()):.3e}, {float(Cb.max()):.3e}]")

    # ── MIMO simulation ───────────────────────────────────────────────
    print(f"\nRunning MIMO ({len(TX_POSITIONS)} TX x {N_STEPS} steps each)...")
    t_sim_start = time.perf_counter()

    all_sigs, ez_snap = run_mimo_3d(grid, Ca, Cb, TX_POSITIONS, N_STEPS)

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t_sim_end = time.perf_counter()
    sim_elapsed = t_sim_end - t_sim_start

    total_cells_computed = len(TX_POSITIONS) * N_STEPS * NX * NY * NZ
    mcells = total_cells_computed / max(sim_elapsed, 1e-9) / 1e6

    # ── DAS beamforming ───────────────────────────────────────────────
    print("\nRunning DAS backprojection...")
    das_image = das_backprojection_3d(all_sigs, TX_POSITIONS, grid, N_STEPS)

    # Find DAS peak location
    peak_idx = np.unravel_index(np.argmax(das_image), das_image.shape)
    peak_mm = (peak_idx[0] * DX * 1e3, peak_idx[1] * DX * 1e3)
    true_mm = (TUMOR_CENTER[0] * DX * 1e3, TUMOR_CENTER[1] * DX * 1e3)
    err = math.sqrt((peak_mm[0] - true_mm[0])**2 + (peak_mm[1] - true_mm[1])**2)

    print(f"DAS peak:   ({peak_mm[0]:.0f}, {peak_mm[1]:.0f}) mm")
    print(f"True tumor: ({true_mm[0]:.0f}, {true_mm[1]:.0f}) mm")
    print(f"Localization error: {err:.1f} mm")

    # ── Figure ────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        'WaveForge 3D — Breast Tumor MIMO Imaging (64^3, 1 GHz, 4-TX)',
        fontsize=13, fontweight='bold'
    )

    ext_mm = [0, NX * DX * 1e3, 0, NY * DX * 1e3]

    # ── Panel [0,0]: Material cross-section (xy at z=32) ──────────────
    ax = axes[0, 0]
    im = ax.imshow(
        eps_map.T, origin='lower', cmap='viridis',
        extent=ext_mm, aspect='auto', vmin=1, vmax=60
    )
    # Mark antenna positions
    for pi, pj, pk in TX_POSITIONS:
        ax.plot(pi * DX * 1e3, pj * DX * 1e3, 'w^', ms=8,
                markeredgecolor='k', markeredgewidth=0.5)
    # Mark tumor
    tumor_circle = plt.Circle(
        (TUMOR_CENTER[0] * DX * 1e3, TUMOR_CENTER[1] * DX * 1e3),
        TUMOR_RADIUS * DX * 1e3, fill=False, color='cyan', lw=2, linestyle='--'
    )
    ax.add_patch(tumor_circle)
    ax.set(xlabel='x (mm)', ylabel='y (mm)',
           title='Tissue eps_r (xy-slice at z=32)')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('eps_r')

    # ── Panel [0,1]: Ez field from TX0 at step 200 ────────────────────
    ax = axes[0, 1]
    if ez_snap is not None:
        vmax = float(np.percentile(np.abs(ez_snap), 99)) or 1e-12
        im = ax.imshow(
            ez_snap.T, origin='lower', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax, extent=ext_mm, aspect='auto'
        )
        plt.colorbar(im, ax=ax, label='Ez (V/m)')
    # Mark TX0
    ax.plot(TX_POSITIONS[0][0] * DX * 1e3, TX_POSITIONS[0][1] * DX * 1e3,
            'g*', ms=12, label='TX0')
    # Mark tumor
    tumor_circle2 = plt.Circle(
        (TUMOR_CENTER[0] * DX * 1e3, TUMOR_CENTER[1] * DX * 1e3),
        TUMOR_RADIUS * DX * 1e3, fill=False, color='cyan', lw=1.5, linestyle='--'
    )
    ax.add_patch(tumor_circle2)
    ax.legend(fontsize=8, loc='upper right')
    ax.set(xlabel='x (mm)', ylabel='y (mm)',
           title='Ez field from TX0, step 200 (xy at z=32)')

    # ── Panel [1,0]: Received signals ─────────────────────────────────
    ax = axes[1, 0]
    t_ns = np.arange(N_STEPS) * grid.dt * 1e9
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    pair_idx = 0
    N_TX = len(TX_POSITIONS)
    for tx in range(N_TX):
        for rx in range(N_TX):
            if tx == rx:
                continue
            ax.plot(t_ns, all_sigs[tx, rx, :],
                    color=colors[pair_idx % len(colors)],
                    alpha=0.7, linewidth=0.8,
                    label=f'TX{tx}->RX{rx}')
            pair_idx += 1
    ax.set(xlabel='Time (ns)', ylabel='Ez (V/m)',
           title='Received signals (all TX/RX pairs)')
    ax.legend(fontsize=6, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── Panel [1,1]: DAS image ────────────────────────────────────────
    ax = axes[1, 1]
    vmax_das = das_image.max()
    im = ax.imshow(
        das_image.T, origin='lower', cmap='hot',
        vmin=0, vmax=vmax_das, extent=ext_mm, aspect='auto'
    )
    # Mark true tumor location
    tumor_circle3 = plt.Circle(
        (TUMOR_CENTER[0] * DX * 1e3, TUMOR_CENTER[1] * DX * 1e3),
        TUMOR_RADIUS * DX * 1e3, fill=False, color='cyan', lw=2,
        label=f'True tumor'
    )
    ax.add_patch(tumor_circle3)
    ax.plot(peak_mm[0], peak_mm[1], 'c+', ms=14, mew=2,
            label=f'DAS peak ({err:.0f}mm err)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set(xlabel='x (mm)', ylabel='y (mm)',
           title='DAS Backprojection (xy-plane)')
    plt.colorbar(im, ax=ax, label='Power')

    fig.tight_layout()
    out_path = OUTPUT_DIR / '3d_10_breast_tumor_mimo.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - t_start

    print(f"\nSaved: {out_path}")
    print(f"\n{'='*64}")
    print(f"Results summary:")
    print(f"  Simulation time: {sim_elapsed:.1f}s")
    print(f"  Total time:      {elapsed_total:.1f}s")
    print(f"  Adipose->Tumor contrast: delta_eps_r = {MAT_TUMOR.eps_r - MAT_ADIPOSE.eps_r:.0f}")
    print(f"  Max received signal: {np.abs(all_sigs).max():.3e}")
    print(f"  DAS localization error: {err:.1f} mm (tumor r={TUMOR_RADIUS*DX*1e3:.0f}mm)")
    print(f"  Tumor location: ({TUMOR_CENTER[0]*DX*1e3:.0f}, "
          f"{TUMOR_CENTER[1]*DX*1e3:.0f}, {TUMOR_CENTER[2]*DX*1e3:.0f}) mm")
    print(f"{'='*64}")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")


if __name__ == '__main__':
    main()
