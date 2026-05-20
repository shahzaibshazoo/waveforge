"""
03_waveguide.py — EM wave propagation in a parallel-plate waveguide.

PEC walls at y=0 and y=Ny-1. 5 GHz sinusoidal source at center.
Cutoff frequency = c0/(2*width) ≈ 2.5 GHz — mode propagates since 5 GHz > f_c.

Run: python examples/03_waveguide.py
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.grid import YeeGrid, C0
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import SinusoidalSource, PointSource, SourceCollection
from core.fdtd2d import FDTD2D

NX, NY   = 200, 60
DX       = 1e-3
N_STEPS  = 1000
FREQ     = 5e9
OUTPUT_DIR = Path(__file__).parent / 'output'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid     = YeeGrid(NX, NY, dx=DX, dy=DX, device=DEVICE)
    fields   = FieldSet(grid)
    cw       = SinusoidalSource(amplitude=1.0, frequency=FREQ)
    src      = PointSource(cw, 10, 30, 'Hz', grid=grid, N_steps=N_STEPS)
    boundary = MurABC(grid, fields.Hz)
    sim      = FDTD2D(grid, fields, boundary, SourceCollection([src]), n_check=200)

    det_hz = np.zeros(N_STEPS, dtype=np.float32)
    print(f"Running {N_STEPS} steps on {DEVICE}...")
    for n in range(N_STEPS):
        sim.step()
        fields.Hz[:, 0, :]    = 0.0   # PEC wall bottom
        fields.Hz[:, NY-1, :] = 0.0   # PEC wall top
        det_hz[n] = fields.Hz[100, 30, 0].item()

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    hz  = fields.Hz[:,:,0].detach().cpu().numpy().T
    vmax = max(np.abs(hz).max(), 1e-12)
    ext  = [0, NX*DX*1e3, 0, NY*DX*1e3]
    im = axes[0].imshow(hz, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
    axes[0].axhline(0,            color='black', lw=2.5, label='PEC wall')
    axes[0].axhline(NY*DX*1e3,   color='black', lw=2.5)
    axes[0].scatter([100*DX*1e3], [30*DX*1e3], color='yellow', s=40, zorder=5, label='detector')
    axes[0].set(xlabel='x (mm)', ylabel='y (mm)', title=f'Hz guided mode — step {sim.steps_completed}')
    axes[0].legend(fontsize=8); fig.colorbar(im, ax=axes[0]).set_label('Hz (A/m)')
    t = np.arange(N_STEPS)*grid.dt*1e9
    axes[1].plot(t, det_hz)
    axes[1].set(xlabel='Time (ns)', ylabel='Hz (A/m)', title='Detector (100,30)')
    axes[1].grid(alpha=0.3)
    fig.suptitle(f'Parallel-plate waveguide  f={FREQ/1e9:.1f} GHz')
    fig.tight_layout()
    path = OUTPUT_DIR / '03_waveguide.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {path}")

    w   = NY*DX
    f_c = C0 / (2*w)
    print(f"\nPhysics: width={w*1e3:.0f}mm, f_c={f_c/1e9:.3f}GHz, f={FREQ/1e9:.1f}GHz")
    print(f"  Mode propagates: {FREQ > f_c}")
    print(f"  Throughput: {sim.mcells_per_second:.1f} Mcells/s")

if __name__ == '__main__': main()
