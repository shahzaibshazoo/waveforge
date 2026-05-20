"""
05_through_wall_radar.py — Through-wall radar: detect metal target behind concrete.

1 GHz Ricker pulse penetrates a concrete wall (eps_r=6, sigma=0.05)
and scatters off a highly-conductive target (sigma=50 S/m ≈ metal).
Return echo is visible at step 500.

Run: python examples/05_through_wall_radar.py
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
from core.sources import RickerWavelet, LineSource, SourceCollection
from core.materials import Material, MaterialMap
from core.fdtd2d import FDTD2D

NX, NY   = 200, 100
DX       = 5e-3       # 5 mm → 1m × 0.5m domain
N_STEPS  = 600
OUTPUT_DIR = Path(__file__).parent / 'output'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid     = YeeGrid(NX, NY, dx=DX, dy=DX, device=DEVICE)
    fields   = FieldSet(grid)
    wv       = RickerWavelet(amplitude=1.0, peak_freq=1e9)
    src      = LineSource(wv, axis='y', position=5, start=0, stop=NY,
                          component='Hz', grid=grid, N_steps=N_STEPS)
    concrete = Material('concrete', eps_r=6.0, sigma=0.05)
    metal    = Material('metal',    eps_r=8.0, sigma=3.0)   # high-loss conductor (sigma capped for CFL stability)
    mm       = MaterialMap(grid)
    mm.add_rectangle((40, 55),   (0, NY-1), concrete)
    mm.add_rectangle((100, 108), (35, 65),  metal)
    Ca, Cb   = mm.build()
    boundary = MurABC(grid, fields.Hz)
    sim      = FDTD2D(grid, fields, boundary, SourceCollection([src]), Ca=Ca, Cb=Cb, n_check=300)

    snaps = {}
    print(f"Running {N_STEPS} steps on {DEVICE}...")
    _t0_bench = time.perf_counter()
    for n in range(N_STEPS):
        sim.step()
        if n+1 in (300, 500):
            snaps[n+1] = fields.Hz[:,:,0].detach().cpu().numpy().T.copy()

    _bench_mc = N_STEPS * NX * NY / max(time.perf_counter() - _t0_bench, 1e-9) / 1e6
    print(f"WAVEFORGE_BENCH: {_bench_mc:.1f} Mcells/s")
    # eps_r map for visualization
    eps_map = np.ones((NY, NX), dtype=np.float32)
    eps_map[:, 40:56] = 6.0
    eps_map[35:66, 100:109] = 3.0  # mark metal differently

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ext = [0, NX*DX*1e3, 0, NY*DX*1e3]

    im0 = axes[0].imshow(eps_map, origin='lower', cmap='viridis', extent=ext, aspect='auto')
    axes[0].set(xlabel='x (mm)', ylabel='y (mm)', title='Material map')
    fig.colorbar(im0, ax=axes[0]).set_label('eps_r')

    for ax, step in zip(axes[1:], [300, 500]):
        hz = snaps[step]
        vmax = max(np.abs(hz).max(), 1e-12)
        im = ax.imshow(hz, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
        lbl = 'pulse hitting target' if step==300 else 'return echo visible'
        ax.set(xlabel='x (mm)', ylabel='y (mm)', title=f'Hz step {step} — {lbl}')
        ax.axvline(40*DX*1e3, color='yellow', lw=1, ls='--')
        ax.axvline(56*DX*1e3, color='yellow', lw=1, ls='--')
        fig.colorbar(im, ax=ax).set_label('Hz (A/m)')

    fig.suptitle('Through-Wall Radar — 1 GHz, concrete + metal target')
    fig.tight_layout()
    path = OUTPUT_DIR / '05_through_wall_radar.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {path}")

    MU0 = 1.2566370614e-6
    omega = 2*math.pi*1e9
    sd = math.sqrt(2/(omega*MU0*0.05))*1e3
    wall_mm = 15*DX*1e3
    loss_dB = 20*math.log10(math.exp(-wall_mm*1e-3/(sd*1e-3)))
    print(f"\nPhysics: wall thickness={wall_mm:.0f}mm, concrete skin depth={sd:.1f}mm")
    print(f"  One-way wall loss ≈ {loss_dB:.1f} dB")
    print(f"  Throughput: {sim.mcells_per_second:.1f} Mcells/s")

if __name__ == '__main__': main()
