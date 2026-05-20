"""
02_dielectric_slab.py — Plane-wave reflection and transmission at a dielectric slab.

A Gaussian pulse hits a glass-like slab (eps_r=4).
Two detectors record incident+reflected (x=50) and transmitted (x=170) signals.

Run: python examples/02_dielectric_slab.py
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import GaussianPulse, LineSource, SourceCollection
from core.materials import Material, MaterialMap
from core.fdtd2d import FDTD2D

NX, NY   = 200, 64
DX       = 1e-3
N_STEPS  = 800
SLAB_X0, SLAB_X1 = 100, 139
EPS_R    = 4.0
OUTPUT_DIR = Path(__file__).parent / 'output'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid     = YeeGrid(NX, NY, dx=DX, dy=DX, device=DEVICE)
    fields   = FieldSet(grid)
    pulse    = GaussianPulse(amplitude=1.0, sigma=40*grid.dt)
    src      = LineSource(pulse, axis='y', position=20, start=0, stop=NY,
                          component='Hz', grid=grid, N_steps=N_STEPS)
    glass    = Material('glass', eps_r=EPS_R, sigma=0.0)
    mm       = MaterialMap(grid)
    mm.add_rectangle((SLAB_X0, SLAB_X1), (0, NY-1), glass)
    Ca, Cb   = mm.build()
    boundary = MurABC(grid, fields.Hz)
    sim      = FDTD2D(grid, fields, boundary, SourceCollection([src]), Ca=Ca, Cb=Cb, n_check=200)

    det_a = np.zeros(N_STEPS, dtype=np.float32)
    det_b = np.zeros(N_STEPS, dtype=np.float32)

    print(f"Running {N_STEPS} steps on {DEVICE}...")
    for n in range(N_STEPS):
        sim.step()
        det_a[n] = fields.Hz[50,  NY//2, 0].item()
        det_b[n] = fields.Hz[170, NY//2, 0].item()

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    hz = fields.Hz[:,:,0].detach().cpu().numpy().T
    vmax = max(np.abs(hz).max(), 1e-12)
    ext  = [0, NX*DX*1e3, 0, NY*DX*1e3]
    im = axes[0].imshow(hz, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
    axes[0].axvline(SLAB_X0*DX*1e3, color='yellow', lw=1.5, ls='--', label='slab')
    axes[0].axvline((SLAB_X1+1)*DX*1e3, color='yellow', lw=1.5, ls='--')
    axes[0].set(xlabel='x (mm)', ylabel='y (mm)', title=f'Hz field — step {sim.steps_completed}')
    axes[0].legend(fontsize=8); fig.colorbar(im, ax=axes[0]).set_label('Hz (A/m)')
    t = np.arange(N_STEPS)*grid.dt*1e9
    axes[1].plot(t, det_a, label='x=50 mm (incident+reflected)')
    axes[1].plot(t, det_b, label='x=170 mm (transmitted)')
    axes[1].set(xlabel='Time (ns)', ylabel='Hz (A/m)', title='Detector signals')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle(f'Dielectric slab (eps_r={EPS_R}) — reflection & transmission')
    fig.tight_layout()
    path = OUTPUT_DIR / '02_dielectric_slab.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {path}")

    n_idx = math.sqrt(EPS_R)
    R = ((n_idx-1)/(n_idx+1))**2
    print(f"\nPhysics: eps_r={EPS_R}, n={n_idx:.3f}")
    print(f"  Analytical R={(R*100):.1f}%  T={(1-R)*100:.1f}%")
    print(f"  Det A peak: {np.abs(det_a).max():.3e}  Det B peak: {np.abs(det_b).max():.3e}")
    print(f"  Throughput: {sim.mcells_per_second:.1f} Mcells/s")

if __name__ == '__main__': main()
