"""
07_tissue_penetration.py — Microwave penetration through skin/fat/muscle at 1 GHz.

Shows progressive attenuation through layered biological tissue.
Relevant to medical imaging, hyperthermia planning, body-scanner design.

Run: python examples/07_tissue_penetration.py
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
from core.sources import GaussianPulse, LineSource, SourceCollection
from core.materials import Material, MaterialMap
from core.fdtd2d import FDTD2D

NX, NY   = 250, 80
DX       = 1e-3
N_STEPS  = 800
OUTPUT_DIR = Path(__file__).parent / 'output'
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid     = YeeGrid(NX, NY, dx=DX, dy=DX, device=DEVICE)
    fields   = FieldSet(grid)
    pulse    = GaussianPulse(amplitude=1.0, freq=1e9)
    src      = LineSource(pulse, axis='y', position=5, start=0, stop=NY,
                          component='Hz', grid=grid, N_steps=N_STEPS)
    mm       = MaterialMap(grid)
    mm.add_rectangle((5,  14), (0, NY-1), Material('skin',   eps_r=40.0, sigma=1.5))
    mm.add_rectangle((15, 29), (0, NY-1), Material('fat',    eps_r=5.0,  sigma=0.05))
    mm.add_rectangle((30, 79), (0, NY-1), Material('muscle', eps_r=50.0, sigma=1.7))
    Ca, Cb   = mm.build()
    boundary = MurABC(grid, fields.Hz)
    sim      = FDTD2D(grid, fields, boundary, SourceCollection([src]), Ca=Ca, Cb=Cb, n_check=400)

    det_xs  = [10, 20, 40, 60, 80]
    peaks   = {x: 0.0 for x in det_xs}
    print(f"Running {N_STEPS} steps on {DEVICE}...")
    _t0_bench = time.perf_counter()
    for n in range(N_STEPS):
        sim.step()
        for xi in det_xs:
            v = abs(float(fields.Hz[xi, NY//2, 0].item()))
            if v > peaks[xi]: peaks[xi] = v

    _bench_mc = N_STEPS * NX * NY / max(time.perf_counter() - _t0_bench, 1e-9) / 1e6
    print(f"WAVEFORGE_BENCH: {_bench_mc:.1f} Mcells/s")
    hz = fields.Hz[:,:,0].detach().cpu().numpy().T
    eps_map = np.ones((NY,NX),dtype=np.float32)
    eps_map[:,5:15]=40; eps_map[:,15:30]=5; eps_map[:,30:80]=50

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ext = [0, NX*DX*1e3, 0, NY*DX*1e3]

    im0 = axes[0].imshow(eps_map, origin='lower', cmap='plasma', extent=ext, aspect='auto')
    for xi,lbl in ((5,'skin'),(15,'fat'),(30,'muscle')):
        axes[0].axvline(xi*DX*1e3, color='white', lw=0.8, ls='--')
        axes[0].text(xi*DX*1e3+1, 5, lbl, color='white', fontsize=8)
    axes[0].set(xlabel='x (mm)', ylabel='y (mm)', title='Material map (eps_r)')
    fig.colorbar(im0, ax=axes[0]).set_label('eps_r')

    vmax = max(np.abs(hz).max(), 1e-12)
    im1  = axes[1].imshow(hz, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=ext, aspect='auto')
    for xi in (5,15,30): axes[1].axvline(xi*DX*1e3, color='white', lw=0.8, ls='--')
    axes[1].set(xlabel='x (mm)', ylabel='y (mm)', title='Hz field — progressive attenuation')
    fig.colorbar(im1, ax=axes[1]).set_label('Hz (A/m)')

    xs_mm = [xi*DX*1e3 for xi in det_xs]
    pk    = [peaks[xi] for xi in det_xs]
    axes[2].semilogy(xs_mm, pk, 'o-', color='darkorange', lw=1.5, ms=7)
    for xi,lbl in ((5,'skin'),(15,'fat'),(30,'muscle')):
        axes[2].axvline(xi*DX*1e3, color='grey', lw=0.8, ls='--', alpha=0.7)
    axes[2].set(xlabel='Depth x (mm)', ylabel='Peak |Hz| (A/m)', title='Attenuation vs depth')
    axes[2].grid(alpha=0.3, which='both')

    fig.suptitle('Microwave penetration through skin/fat/muscle at 1 GHz')
    fig.tight_layout()
    path = OUTPUT_DIR / '07_tissue_penetration.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {path}")

    MU0 = 1.2566370614e-6; omega = 2*math.pi*1e9
    print("\nPhysics — skin depths at 1 GHz:")
    for name,sigma in [('skin',1.5),('fat',0.05),('muscle',1.7)]:
        sd = math.sqrt(2/(omega*MU0*sigma))*1e3
        print(f"  {name:6s}: sigma={sigma} S/m  delta={sd:.1f}mm")
    print("\nDetector peak amplitudes:")
    for xi,p in zip(det_xs,pk): print(f"  x={xi:3d}mm: {p:.3e} A/m")
    print(f"Throughput: {sim.mcells_per_second:.1f} Mcells/s")

if __name__ == '__main__': main()
