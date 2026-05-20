"""
06_multipath_interference.py — Two coherent sources at 2.4 GHz (WiFi).

Constructive/destructive interference pattern. Fringe spacing = lambda/2.
Time-averaged |Hz|² shows the standing-wave structure.

Run: python examples/06_multipath_interference.py
"""
import sys, math
from pathlib import Path
import numpy as np
import time
import torch
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import SinusoidalSource, PointSource, SourceCollection
from core.fdtd2d import FDTD2D

NX, NY   = 128, 128
DX       = 2e-3       # 2mm → 256mm × 256mm
FREQ     = 2.4e9
N_STEPS  = 800
AVG_WIN  = 200
OUTPUT_DIR = Path(__file__).parent / 'output'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid     = YeeGrid(NX, NY, dx=DX, dy=DX, device=DEVICE)
    fields   = FieldSet(grid)
    src1     = PointSource(SinusoidalSource(1.0, FREQ), 32, 64, 'Hz', grid=grid, N_steps=N_STEPS)
    src2     = PointSource(SinusoidalSource(1.0, FREQ), 96, 64, 'Hz', grid=grid, N_steps=N_STEPS)
    boundary = MurABC(grid, fields.Hz)
    sim      = FDTD2D(grid, fields, boundary, SourceCollection([src1, src2]), n_check=400)

    hz_sq_sum = np.zeros((NY, NX), dtype=np.float64)
    snap = None
    print(f"Running {N_STEPS} steps on {DEVICE}...")
    _t0_bench = time.perf_counter()
    for n in range(N_STEPS):
        sim.step()
        if n >= N_STEPS - AVG_WIN:
            hz_sq_sum += fields.Hz[:,:,0].detach().cpu().numpy().T.astype(np.float64)**2
        if n+1 == N_STEPS:
            snap = fields.Hz[:,:,0].detach().cpu().numpy().T.copy()

    _bench_mc = N_STEPS * NX * NY / max(time.perf_counter() - _t0_bench, 1e-9) / 1e6
    print(f"WAVEFORGE_BENCH: {_bench_mc:.1f} Mcells/s")
    hz_sq_avg = (hz_sq_sum / AVG_WIN).astype(np.float32)

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ext = [0, NX*DX*1e3, 0, NY*DX*1e3]

    vmax = max(np.abs(snap).max(), 1e-12)
    im0  = axes[0].imshow(snap, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
    axes[0].scatter([32*DX*1e3,96*DX*1e3],[64*DX*1e3,64*DX*1e3],c='white',s=60,zorder=5,marker='+',linewidths=2)
    axes[0].set(xlabel='x (mm)', ylabel='y (mm)', title='Hz snapshot — interference fringes')
    fig.colorbar(im0, ax=axes[0]).set_label('Hz (A/m)')

    im1 = axes[1].imshow(hz_sq_avg, origin='lower', cmap='hot', extent=ext, aspect='auto')
    axes[1].set(xlabel='x (mm)', ylabel='y (mm)', title='|Hz|² time-averaged (last 200 steps)')
    fig.colorbar(im1, ax=axes[1]).set_label('|Hz|² (A²/m²)')

    x_mm = np.arange(NX)*DX*1e3
    axes[2].plot(x_mm, snap[64,:], lw=1.5, color='steelblue')
    axes[2].axhline(0, color='k', lw=0.5)
    axes[2].set(xlabel='x (mm)', ylabel='Hz (A/m)', title='Hz along centre line y=128mm')
    axes[2].grid(alpha=0.3)

    fig.suptitle(f'Two-source interference at {FREQ/1e9:.1f} GHz (WiFi)')
    fig.tight_layout()
    path = OUTPUT_DIR / '06_multipath_interference.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {path}")

    C0 = 299_792_458.0
    lam = C0/FREQ
    print(f"\nPhysics: f={FREQ/1e9:.1f}GHz, lambda={lam*1e3:.1f}mm")
    print(f"  Fringe spacing = lambda/2 = {lam/2*1e3:.1f}mm")
    print(f"  Source separation = {(96-32)*DX*1e3:.0f}mm = {(96-32)*DX/lam:.1f}λ")
    print(f"  Throughput: {sim.mcells_per_second:.1f} Mcells/s")

if __name__ == '__main__': main()
