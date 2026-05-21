"""
3d_brain_das_accuracy_test.py — All 4 haemorrhage classes with material maps, DAS imaging.

Follows the same pattern as 3d_09_brain_clot_dataset.py:
  - Explicit MaterialMap3D with tissue eps_r/sigma values
  - PointSource at grid edge (near the skull)
  - GaussianPulse waveform
  - eps_r material map shown for each class
  - Ez field snapshot shown
  - DAS backprojection from differential signals

Classes:
  0 = Healthy         (no bleed)
  1 = Epidural        (between skull and dura)
  2 = Subdural        (between dura and brain surface)
  3 = Intracerebral   (inside brain parenchyma)

Run: python examples/3d/3d_brain_das_accuracy_test.py
Out: examples/output/3d_brain_das_accuracy_test.png
"""
import sys, math, time
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

# ── Grid ──────────────────────────────────────────────────────────────────────
NX = NY = NZ = 64
DX = 3e-3           # 3mm/cell — EXACT match to Kaggle dataset config
N_STEPS = 700       # 3.96ns — covers full round trip (needed: 520 steps = 2.94ns)
N_TX = 4
OUTPUT_DIR = Path(__file__).parent.parent / 'output'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CX = CY = CZ = NX // 2   # 32, 32, 32

# ── Antenna positions (PointSources near grid edges, outside head) ────────────
# Ring of 4 around the head in the z=CZ plane
TX_POS = [
    (3,      CY,    CZ),   # left
    (NX-4,   CY,    CZ),   # right
    (CX,     3,     CZ),   # front
    (CX,     NY-4,  CZ),   # back
]

# ── Tissue properties at 1 GHz (Gabriel 1996) ────────────────────────────────
SCALP  = Material('scalp',  eps_r=40.0, sigma=0.87)
SKULL  = Material('skull',  eps_r=13.1, sigma=0.10)
DURA   = Material('dura',   eps_r=44.0, sigma=0.82)
CSF    = Material('csf',    eps_r=68.0, sigma=2.46)
GM     = Material('gm',     eps_r=52.7, sigma=0.94)
WM     = Material('wm',     eps_r=38.1, sigma=0.61)

# Blood aging values
BLOOD_ACUTE    = Material('blood_acute',    eps_r=61.0, sigma=1.58)
BLOOD_SUBACUTE = Material('blood_subacute', eps_r=50.8, sigma=1.19)
BLOOD_CHRONIC  = Material('blood_chronic',  eps_r=43.7, sigma=1.03)

# ── Head geometry (in cells, centre at 32,32,32) ──────────────────────────────
# At 2mm/cell: scalp outer = 26×2mm = 52mm radius (head ⌀ 104mm — adult)
HEAD_RADII = {
    'scalp':  26,   # outer head surface
    'skull':  24,   # inner skull
    'dura':   21,   # epidural space = skull(24)-dura(21) = 3 cells = 6mm
    'csf':    18,   # subdural/CSF = dura(21)-csf(18) = 3 cells = 6mm
    'gm':     18,   # gray matter surface
    'wm':     11,   # white matter core
}

# Bleed configurations per class
# (center_offset from CX,CY,CZ, radius, material, label_name)
# Bleed positions at 3mm/cell:
#   skull_inner=24 cells → epidural zone: 21-24 cells from centre
#   dura_inner=21 cells  → subdural zone: 18-21 cells from centre
#   gray_matter=18 cells → intracerebral: <18 cells from centre
BLEEDS = {
    'healthy':       None,
    'epidural':      {'center': (CX+22, CY,    CZ), 'r': 2, 'mat': BLOOD_ACUTE,    'label': 1},
    'subdural':      {'center': (CX+19, CY,    CZ), 'r': 2, 'mat': BLOOD_SUBACUTE, 'label': 2},
    'intracerebral': {'center': (CX+10, CY+6,  CZ), 'r': 4, 'mat': BLOOD_CHRONIC,  'label': 3},
}


def build_phantom(bleed_name: str):
    """Build Ca/Cb material tensors for a given class."""
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    mm = MaterialMap3D(grid)
    cx = cy = cz = NX // 2

    # Layers outside-in (painter's algorithm — last wins)
    mm.add_sphere((cx, cy, cz), HEAD_RADII['scalp'], SCALP)
    mm.add_sphere((cx, cy, cz), HEAD_RADII['skull'], SKULL)
    mm.add_sphere((cx, cy, cz), HEAD_RADII['dura'],  DURA)
    mm.add_sphere((cx, cy, cz), HEAD_RADII['csf'],   CSF)
    mm.add_sphere((cx, cy, cz), HEAD_RADII['gm'],    GM)
    mm.add_sphere((cx, cy, cz), HEAD_RADII['wm'],    WM)

    # Bleed overlay
    b = BLEEDS[bleed_name]
    if b is not None:
        mm.add_sphere(b['center'], b['r'], b['mat'])

    Ca, Cb = mm.build3d()
    return grid, Ca, Cb


def build_eps_map(bleed_name: str):
    """Build 2D eps_r map (axial slice at z=CZ) for visualisation."""
    eps = np.ones((NY, NX), dtype=np.float32)
    I, J = np.meshgrid(np.arange(NX), np.arange(NY))

    def sphere_mask(cx, cy, r):
        return (I - cx)**2 + (J - cy)**2 <= r**2

    eps[sphere_mask(CX, CY, HEAD_RADII['scalp'])] = SCALP.eps_r
    eps[sphere_mask(CX, CY, HEAD_RADII['skull'])] = SKULL.eps_r
    eps[sphere_mask(CX, CY, HEAD_RADII['dura'])]  = DURA.eps_r
    eps[sphere_mask(CX, CY, HEAD_RADII['csf'])]   = CSF.eps_r
    eps[sphere_mask(CX, CY, HEAD_RADII['gm'])]    = GM.eps_r
    eps[sphere_mask(CX, CY, HEAD_RADII['wm'])]    = WM.eps_r

    b = BLEEDS[bleed_name]
    if b is not None:
        cx_b, cy_b, _ = b['center']
        eps[sphere_mask(cx_b, cy_b, b['r'])] = b['mat'].eps_r

    return eps


def run_mimo(Ca, Cb):
    """Run all 4 TX simulations and return signals (N_TX, N_TX, N_STEPS) + Ez snapshot."""
    signals = np.zeros((N_TX, N_TX, N_STEPS), dtype=np.float32)
    snap = None

    for tx_idx in range(N_TX):
        grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
        fields = FieldSet(grid)
        boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)
        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)
        ti, tj, tk = TX_POS[tx_idx]
        src = PointSource(pulse, ti, tj, 'Ez', k=tk, grid=grid, N_steps=N_STEPS)
        sim = FDTD3D(grid, fields, boundary, SourceCollection([src]),
                     Ca=Ca, Cb=Cb, n_check=99999)

        snap_step = 350  # ~2.0ns: wavefront inside head, good scattering visible
        with torch.no_grad():
            for n in range(N_STEPS):
                sim.step()
                for rx in range(N_TX):
                    ri, rj, rk = TX_POS[rx]
                    signals[tx_idx, rx, n] = fields.Ez[ri, rj, rk].item()
                if tx_idx == 0 and n == snap_step:
                    snap = fields.Ez.detach().cpu().numpy().copy()

    return signals, snap, grid.dt


def das_backprojection(scattered, dt, image_size=64):
    """Delay-and-sum backprojection in the xy-plane at z=CZ."""
    img = np.zeros((image_size, image_size), dtype=np.float64)
    x_px = np.linspace(0, (NX-1)*DX, image_size)
    y_px = np.linspace(0, (NY-1)*DX, image_size)
    ant_xy = np.array([(TX_POS[i][0]*DX, TX_POS[i][1]*DX) for i in range(N_TX)])
    # Effective speed in brain tissue: c/sqrt(eps_r_avg)
    # avg eps_r ≈ 45 (gray matter 52.7, white matter 38.1, CSF 68)
    C0 = 3e8 / math.sqrt(45.0)

    for iy, py in enumerate(y_px):
        for ix, px in enumerate(x_px):
            pix = np.array([px, py])
            acc = 0.0
            for tx in range(N_TX):
                d_tx = np.linalg.norm(pix - ant_xy[tx])
                for rx in range(N_TX):
                    d_rx = np.linalg.norm(pix - ant_xy[rx])
                    idx = int((d_tx + d_rx) / C0 / dt)
                    if 0 <= idx < N_STEPS:
                        acc += float(scattered[tx, rx, idx])
            img[iy, ix] = acc

    return (img ** 2).astype(np.float32)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    print('=' * 65)
    print('WaveForge 3D — Brain Haemorrhage: All 4 Classes + DAS')
    print(f'Grid: {NX}³, dx={DX*1e3:.0f}mm, device={DEVICE}')
    print(f'Antennas: {N_TX} PointSources, {N_STEPS} steps')
    print('=' * 65)

    classes = ['healthy', 'epidural', 'subdural', 'intracerebral']
    class_labels = {0:'Healthy', 1:'Epidural (EDH)', 2:'Subdural (SDH)', 3:'Intracerebral (ICH)'}
    all_signals = {}
    all_snaps = {}
    dt_val = None

    # ── Step 1: Run healthy reference ──────────────────────────────────────
    print('\n[1/5] Running healthy reference...')
    t0 = time.perf_counter()
    grid_ref, Ca_ref, Cb_ref = build_phantom('healthy')
    sigs_ref, snap_ref, dt_val = run_mimo(Ca_ref, Cb_ref)
    all_signals['healthy'] = sigs_ref
    all_snaps['healthy'] = snap_ref
    print(f'  Done in {time.perf_counter()-t0:.0f}s')

    # ── Step 2: Run bleed classes ───────────────────────────────────────────
    for idx, name in enumerate(['epidural', 'subdural', 'intracerebral'], 2):
        print(f'\n[{idx}/5] Running {name}...')
        t0 = time.perf_counter()
        _, Ca, Cb = build_phantom(name)
        sigs, snap, _ = run_mimo(Ca, Cb)
        all_signals[name] = sigs
        all_snaps[name] = snap
        print(f'  Done in {time.perf_counter()-t0:.0f}s')

    print('\n[5/5] Computing DAS images...')

    # ── Step 3: Compute scattered signals and DAS ───────────────────────────
    das_images = {}
    energies = {}
    for name in classes:
        scattered = all_signals[name] - sigs_ref
        energies[name] = float((scattered**2).sum())
        das_images[name] = das_backprojection(scattered, dt_val, image_size=NX)

    # ── Step 4: Detection & localisation ───────────────────────────────────
    # Use differential energy as detection metric (more robust than DAS peak ratio)
    # Threshold: differential energy > 1e-9 (well above numerical noise ~1e-30)
    ENERGY_THRESH = 1e-9
    healthy_peak = float(das_images['healthy'].max()) + 1e-30
    thresh = 5.0

    print('\n' + '=' * 65)
    print('DETECTION RESULTS')
    print('=' * 65)
    print(f'{"Class":22} {"Diff energy":14}  {"Result"}')
    print('-' * 50)

    correct = 0
    for name in classes:
        energy = energies[name]
        detected = energy > ENERGY_THRESH
        true_bleed = name != 'healthy'
        ok = (detected == true_bleed)
        if ok: correct += 1
        status = '✓' if ok else '✗'
        result = 'DETECTED' if detected else 'clean'
        print(f'{name:22} {energy:14.3e}  {result} {status}')

    print('-' * 65)
    print(f'Accuracy: {correct}/{len(classes)} ({correct/len(classes)*100:.0f}%)')

    # ── Step 5: Plot ────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(
        f'WaveForge 3D Brain Haemorrhage — Material Maps, Ez Fields, DAS Imaging\n'
        f'{NX}³ grid, dx={DX*1e3:.0f}mm, {N_TX} antennas, {N_STEPS} steps, {DEVICE}',
        fontsize=13, fontweight='bold'
    )

    ext_mm = [0, NX*DX*1e3, 0, NY*DX*1e3]
    t_ns = np.arange(N_STEPS) * dt_val * 1e9

    # Tissue colormap for eps_r map
    tissue_cmap = plt.cm.get_cmap('tab20', 12)

    for row, name in enumerate(classes):
        b = BLEEDS[name]
        label_id = 0 if b is None else b['label']
        detected = float(das_images[name].max()) / healthy_peak > thresh
        true_bleed = name != 'healthy'
        correct_detect = (detected == true_bleed)

        # ── Col 0: Material map (eps_r axial slice) ────────────────────────
        ax = axes[row, 0]
        eps = build_eps_map(name)
        im = ax.imshow(eps.T, origin='lower', cmap='jet',
                       vmin=1, vmax=70, extent=ext_mm, aspect='auto')
        # Mark antennas
        for ti, tj, tk in TX_POS:
            ax.plot(ti*DX*1e3, tj*DX*1e3, 'w^', markersize=8)
        # Mark bleed
        if b is not None:
            ax.plot(b['center'][0]*DX*1e3, b['center'][1]*DX*1e3, 'r*',
                    markersize=14, markeredgecolor='white', label='bleed')
            ax.legend(fontsize=7)
        ax.set(title=f'{class_labels[label_id]} — material map',
               xlabel='x (mm)', ylabel='y (mm)')
        plt.colorbar(im, ax=ax, label='ε_r')

        # ── Col 1: Ez field snapshot (axial slice, percentile scaled) ─────
        ax = axes[row, 1]
        snap = all_snaps[name]
        if snap is not None:
            ez = snap[:, :, CZ].T
            vmax = float(np.percentile(np.abs(ez), 99)) or 1e-12
            im = ax.imshow(ez, origin='lower', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax, extent=ext_mm, aspect='auto')
            plt.colorbar(im, ax=ax, label='Ez (V/m)')
        if b is not None:
            ax.plot(b['center'][0]*DX*1e3, b['center'][1]*DX*1e3, 'y*',
                    markersize=12, markeredgecolor='black')
        ax.set(title=f'Ez field — TX[0], step {N_STEPS}',
               xlabel='x (mm)', ylabel='y (mm)')

        # ── Col 2: RX signals from TX[0] (total and differential) ────────
        ax = axes[row, 2]
        scattered = all_signals[name] - sigs_ref
        colors = plt.cm.Set1(np.linspace(0, 1, N_TX))
        for rx in range(N_TX):
            ax.plot(t_ns, all_signals[name][0, rx], color=colors[rx],
                    alpha=0.5, lw=0.8, ls='--', label=f'total RX{rx}')
            ax.plot(t_ns, scattered[0, rx], color=colors[rx],
                    alpha=0.9, lw=1.2, label=f'scattered RX{rx}')
        ax.set(title=f'TX[0] signals (dashed=total, solid=scattered)',
               xlabel='Time (ns)', ylabel='Ez (V/m)')
        ax.grid(alpha=0.3)
        if row == 0: ax.legend(fontsize=6, ncol=2)

        # ── Col 3: DAS backprojection image ────────────────────────────────
        ax = axes[row, 3]
        das = das_images[name]
        im = ax.imshow(das.T, origin='lower', cmap='hot',
                       extent=ext_mm, aspect='auto')
        plt.colorbar(im, ax=ax, label='DAS power')
        if b is not None:
            ax.plot(b['center'][0]*DX*1e3, b['center'][1]*DX*1e3, 'c+',
                    markersize=18, mew=2.5, label='true bleed')
            ax.legend(fontsize=8)
        result_str = 'DETECTED ✓' if detected else 'clean ✓' if not true_bleed else 'MISSED ✗'
        border = '#2ecc71' if correct_detect else '#e74c3c'
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(3)
        ax.set(title=f'DAS image  ΔE={energies[name]:.2e}  {result_str}',
               xlabel='x (mm)', ylabel='y (mm)')

    # Column headers
    for col, title in enumerate(['Material map (ε_r)', 'Ez field (TX[0])',
                                  'RX signals (total + scattered)', 'DAS backprojection']):
        axes[0, col].set_title(title + '\n' + axes[0, col].get_title(),
                                fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = OUTPUT_DIR / '3d_brain_das_accuracy_test.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {path}')
    print('=' * 65)


if __name__ == '__main__':
    main()
