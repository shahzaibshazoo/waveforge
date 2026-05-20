"""
04_scattering_cylinder.py — Plane-wave scattering from a dielectric cylinder.

eps_r=9 cylinder (r=15mm) at domain center. Line source creates plane wave.
8 detectors on a ring record total field (incident + scattered).

Run: python examples/04_scattering_cylinder.py
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

NX, NY  = 150, 150
DX      = 1e-3
N_STEPS = 600
CX, CY  = 75, 75
RADIUS  = 15
EPS_R   = 9.0
RING_R  = 50
OUTPUT_DIR = Path(__file__).parent / 'output'
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'

def ring_positions(cx, cy, r, n):
    return [(int(round(cx + r*math.cos(2*math.pi*k/n))),
             int(round(cy + r*math.sin(2*math.pi*k/n)))) for k in range(n)]

def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid     = YeeGrid(NX, NY, dx=DX, dy=DX, device=DEVICE)
    fields   = FieldSet(grid)
    pulse    = GaussianPulse(amplitude=1.0, sigma=40*grid.dt)
    src      = LineSource(pulse, axis='y', position=10, start=0, stop=NY,
                          component='Hz', grid=grid, N_steps=N_STEPS)
    cyl      = Material('cylinder', eps_r=EPS_R, sigma=0.0)
    mm       = MaterialMap(grid)
    mm.add_circle((CX,CY), RADIUS, cyl)
    Ca, Cb   = mm.build()
    boundary = MurABC(grid, fields.Hz)
    sim      = FDTD2D(grid, fields, boundary, SourceCollection([src]), Ca=Ca, Cb=Cb, n_check=200)

    dets   = ring_positions(CX, CY, RING_R, 8)
    sigs   = np.zeros((8, N_STEPS), dtype=np.float32)
    snap   = None

    print(f"Running {N_STEPS} steps on {DEVICE}...")
    for n in range(N_STEPS):
        sim.step()
        if n == 299: snap = fields.Hz[:,:,0].detach().cpu().numpy().T.copy()
        for d,(di,dj) in enumerate(dets):
            sigs[d,n] = fields.Hz[di,dj,0].item()
    if snap is None: snap = fields.Hz[:,:,0].detach().cpu().numpy().T.copy()

    # eps_r map
    eps_map = np.ones((NY,NX), dtype=np.float32)
    I,J = np.meshgrid(range(NX), range(NY))
    eps_map[(I-CX)**2 + (J-CY)**2 <= RADIUS**2] = EPS_R

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ext = [0, NX*DX*1e3, 0, NY*DX*1e3]

    im0 = axes[0].imshow(eps_map, origin='lower', cmap='viridis', extent=ext, aspect='auto')
    axes[0].set(xlabel='x (mm)', ylabel='y (mm)', title='Material map (eps_r)')
    for di,dj in dets: axes[0].scatter([di*DX*1e3],[dj*DX*1e3],c='red',s=20,zorder=5)
    fig.colorbar(im0, ax=axes[0]).set_label('eps_r')

    vmax = max(np.abs(snap).max(), 1e-12)
    im1  = axes[1].imshow(snap, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
    axes[1].set(xlabel='x (mm)', ylabel='y (mm)', title='Hz at step 300 — scattering')
    fig.colorbar(im1, ax=axes[1]).set_label('Hz (A/m)')

    t = np.arange(N_STEPS)*grid.dt*1e9
    labels = ['0° (fwd)', '90° (top)', '180° (back)', '270° (bot)']
    for idx, lbl in zip([0,2,4,6], labels):
        axes[2].plot(t, sigs[idx], label=lbl)
    axes[2].set(xlabel='Time (ns)', ylabel='Hz (A/m)', title='4-direction detector signals')
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

    fig.suptitle(f'Dielectric cylinder scattering  eps_r={EPS_R}, r={RADIUS}mm')
    fig.tight_layout()
    path = OUTPUT_DIR / '04_scattering_cylinder.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {path}")

    n_idx = math.sqrt(EPS_R)
    print(f"\nPhysics: eps_r={EPS_R}, n={n_idx:.2f}, r={RADIUS}mm")
    print(f"  Geometric cross-section (2D) = 2r = {2*RADIUS}mm")
    pf = np.abs(sigs[0]).max(); pb = np.abs(sigs[4]).max()
    print(f"  Forward peak: {pf:.3e}  Back peak: {pb:.3e}  ratio={pb/max(pf,1e-30):.3f}")
    print(f"  Throughput: {sim.mcells_per_second:.1f} Mcells/s")

if __name__ == '__main__': main()
