"""
08_ev_radar_ula.py — Automotive EV Radar: ULA detecting a cylinder at 120°.

Simulates a 10 GHz automotive radar with a Uniform Linear Array (ULA)
of 8 antennas along the x-axis. A cylindrical target (eps_r=9, like a car
corner reflector) is placed at 120° from the array boresight (which points
along +y). The backscattered signals at all ULA elements are collected and
delay-and-sum beamforming reconstructs the angular image.

Physical setup:
  Domain: 200×200 cells, dx=dy=1mm → 200mm × 200mm
  ULA: 8 elements at y=10, x=76..99 (spaced by lambda/2 = 15mm at 10 GHz)
  Target: cylinder eps_r=9, r=8mm, at 120° from broadside
    → broadside = +y, 120° = 30° from -x axis
    → target center ≈ (cx - R*sin(30°), cy + R*cos(30°)... adjusted for grid)
    → placed at (50, 120) cells from ULA center
  Frequency: 10 GHz → lambda=30mm in free space, lambda/2=15mm

Run: python examples/08_ev_radar_ula.py
"""
import sys, math
from pathlib import Path
import numpy as np
import time
import torch
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.grid import YeeGrid, C0
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import RickerWavelet, PointSource, SourceCollection
from core.materials import Material, MaterialMap
from core.fdtd2d import FDTD2D

NX, NY     = 200, 200
DX         = 1e-3          # 1 mm
N_STEPS    = 600
FREQ       = 10e9          # 10 GHz
LAM        = C0 / FREQ     # 30 mm
LAM_HALF   = LAM / 2       # 15 mm → element spacing

# ULA: 8 elements along x at y=10, centred at x=100
N_ELEM     = 8
ULA_Y      = 10
ULA_CX     = 100
ULA_SPACING_CELLS = max(1, int(round(LAM_HALF / DX)))  # 15 cells
ULA_XS     = [ULA_CX + (k - N_ELEM//2) * ULA_SPACING_CELLS for k in range(N_ELEM)]

# Target: cylinder at 120° from broadside (+y direction)
# 120° from broadside = 30° from -x axis (left side, far up)
TARGET_R_CELLS = 100          # 100 mm range
ANGLE_DEG      = 120.0        # degrees from broadside
ANGLE_RAD      = math.radians(ANGLE_DEG)
# boresight = +y (90°), 120° from boresight clockwise →
# absolute angle from +x axis = 90° - 120° + 90° = 60° from +y toward -x
# Target is left-forward: (x=100-sin(30°)*100, y=10+cos(30°)*100)
TX_CX = int(round(ULA_CX - TARGET_R_CELLS * math.sin(math.radians(30))))   # 50
TX_CY = int(round(ULA_Y  + TARGET_R_CELLS * math.cos(math.radians(30))))   # ~97
TARGET_RAD = 8   # 8mm radius cylinder
OUTPUT_DIR = Path(__file__).parent / 'output'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

def das_beamform(signals, ula_xs, ula_y, grid, n_angles=181):
    """Delay-and-sum beamforming across angle sweep."""
    dt   = grid.dt
    c0   = C0
    angles = np.linspace(-90, 90, n_angles)  # degrees from broadside
    power  = np.zeros(n_angles)
    ula_x_m = np.array(ula_xs) * grid.dx
    ula_y_m = ula_y * grid.dy
    for ai, ang in enumerate(angles):
        phase_delays = ula_x_m * math.sin(math.radians(ang)) / c0
        coherent_sum = np.zeros(signals.shape[1])
        for k in range(len(ula_xs)):
            delay_steps = int(round(phase_delays[k] / dt))
            sig = np.roll(signals[k], -delay_steps)
            coherent_sum += sig
        power[ai] = float(np.sum(coherent_sum**2))
    return angles, power

def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid     = YeeGrid(NX, NY, dx=DX, dy=DX, device=DEVICE)

    # Material: cylinder target
    cyl = Material('cylinder', eps_r=9.0, sigma=0.0)
    mm  = MaterialMap(grid)
    mm.add_circle((TX_CX, TX_CY), TARGET_RAD, cyl)
    Ca, Cb = mm.build()

    # Signals buffer: (N_ELEM, N_STEPS)
    all_signals = np.zeros((N_ELEM, N_STEPS), dtype=np.float32)

    # Run one simulation (single TX at array center, all elements receive)
    fields   = FieldSet(grid)
    boundary = MurABC(grid, fields.Hz)
    wv       = RickerWavelet(amplitude=1.0, peak_freq=FREQ)
    tx_x     = ULA_XS[N_ELEM//2]
    src      = PointSource(wv, tx_x, ULA_Y, 'Hz', grid=grid, N_steps=N_STEPS)
    sim      = FDTD2D(grid, fields, boundary, SourceCollection([src]), Ca=Ca, Cb=Cb, n_check=300)

    snap = None
    rx_i = torch.tensor(ULA_XS, dtype=torch.long, device=grid.device)
    rx_j = torch.full((N_ELEM,), ULA_Y, dtype=torch.long, device=grid.device)

    print(f"Running {N_STEPS} steps on {DEVICE}...")
    print(f"ULA: {N_ELEM} elements, spacing={ULA_SPACING_CELLS}mm, f={FREQ/1e9:.0f}GHz")
    print(f"Target: cylinder r={TARGET_RAD}mm at ({TX_CX},{TX_CY}) → {ANGLE_DEG}° from broadside")

    _t0_bench = time.perf_counter()
    for n in range(N_STEPS):
        sim.step()
        all_signals[:, n] = fields.Hz[rx_i, rx_j, 0].cpu().numpy()
        if n == 300: snap = fields.Hz[:,:,0].detach().cpu().numpy().T.copy()

    _bench_mc = N_STEPS * NX * NY / max(time.perf_counter() - _t0_bench, 1e-9) / 1e6
    print(f"WAVEFORGE_BENCH: {_bench_mc:.1f} Mcells/s")
    if snap is None: snap = fields.Hz[:,:,0].detach().cpu().numpy().T.copy()

    # Beamforming
    angles, power = das_beamform(all_signals, ULA_XS, ULA_Y, grid)
    peak_angle    = angles[np.argmax(power)]

    # Material map
    eps_map = np.ones((NY, NX), dtype=np.float32)
    I, J = np.meshgrid(range(NX), range(NY))
    eps_map[(I-TX_CX)**2 + (J-TX_CY)**2 <= TARGET_RAD**2] = 9.0

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ext = [0, NX*DX*1e3, 0, NY*DX*1e3]

    # Panel 1: scene
    im0 = axes[0].imshow(eps_map, origin='lower', cmap='viridis', extent=ext, aspect='auto')
    axes[0].scatter([x*DX*1e3 for x in ULA_XS], [ULA_Y*DX*1e3]*N_ELEM,
                    c='red', s=40, zorder=5, label='ULA elements')
    axes[0].scatter([TX_CX*DX*1e3], [TX_CY*DX*1e3], c='white', s=80,
                    marker='*', zorder=5, label=f'Target ({ANGLE_DEG}°)')
    # Draw boresight line
    axes[0].plot([ULA_CX*DX*1e3]*2, [ULA_Y*DX*1e3, NY*DX*1e3],
                 'w--', lw=0.8, alpha=0.5, label='Boresight (90°)')
    axes[0].set(xlabel='x (mm)', ylabel='y (mm)', title='Scene layout')
    axes[0].legend(fontsize=7); fig.colorbar(im0, ax=axes[0]).set_label('eps_r')

    # Panel 2: Hz snapshot
    vmax = max(np.abs(snap).max(), 1e-12)
    im1  = axes[1].imshow(snap, origin='lower', cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
    axes[1].scatter([x*DX*1e3 for x in ULA_XS], [ULA_Y*DX*1e3]*N_ELEM,
                    c='red', s=20, zorder=5)
    axes[1].set(xlabel='x (mm)', ylabel='y (mm)', title='Hz field — step 300')
    fig.colorbar(im1, ax=axes[1]).set_label('Hz (A/m)')

    # Panel 3: beamforming angular spectrum
    axes[2].plot(angles, 10*np.log10(power / max(power.max(), 1e-30)), lw=1.8)
    axes[2].axvline(ANGLE_DEG-90, color='red', ls='--', lw=1.5,
                    label=f'True target: {ANGLE_DEG-90:.0f}° from broadside')
    axes[2].axvline(peak_angle, color='green', ls=':', lw=1.5,
                    label=f'DAS peak: {peak_angle:.1f}°')
    axes[2].set(xlabel='Angle from broadside (deg)', ylabel='Power (dB)',
                title='DAS Beamforming — Angular Spectrum')
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    axes[2].set_xlim(-90, 90); axes[2].set_ylim(-40, 2)

    fig.suptitle(f'EV Radar Simulation: {N_ELEM}-element ULA at {FREQ/1e9:.0f} GHz — target at {ANGLE_DEG}° from boresight')
    fig.tight_layout()
    path = OUTPUT_DIR / '08_ev_radar_ula.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {path}")

    print(f"\nPhysics:")
    print(f"  Frequency: {FREQ/1e9:.0f} GHz, lambda={LAM*1e3:.0f}mm, lambda/2={LAM_HALF*1e3:.0f}mm")
    print(f"  ULA: {N_ELEM} elements, spacing={ULA_SPACING_CELLS}mm ({ULA_SPACING_CELLS*DX/LAM_HALF:.2f}λ/2)")
    print(f"  Angular resolution ≈ lambda/(N*d) = {math.degrees(LAM/(N_ELEM*LAM_HALF)):.1f}°")
    print(f"  True target angle: {ANGLE_DEG-90:.0f}° from broadside")
    print(f"  DAS peak angle:    {peak_angle:.1f}°")
    print(f"  Throughput: {sim.mcells_per_second:.1f} Mcells/s")

if __name__ == '__main__': main()
