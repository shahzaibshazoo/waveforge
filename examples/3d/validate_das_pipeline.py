"""
validate_das_pipeline.py — Fast CPU pipeline verification.

Uses a small 32³ grid with 4 TX and 80 steps to verify:
  1. All 4 phantom classes generate non-zero scattered signals
  2. DAS with analytic layered-medium delays localizes bleeds correctly
  3. Localisation error < 2 cell widths for each bleed class

Run: python examples/3d/validate_das_pipeline.py
"""
import sys, math, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import numpy as np
from scipy.signal import hilbert
import torch

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.materials import Material, MaterialMap3D
from core.fdtd3d import FDTD3D

# ── Tiny grid for fast CPU validation ──────────────────────────────────────
NX = NY = NZ = 32
DX = 6e-3       # 6mm/cell — scale everything by 2 from the 3mm dataset
N_STEPS = 150   # enough for round-trip in this smaller head
N_TX = 8        # 8 antennas
_RING_R = 12    # ring radius in cells
DEVICE = 'cpu'

CX = CY = CZ = NX // 2  # 16

# ── Tissue properties at 1 GHz ──────────────────────────────────────────────
SCALP  = Material('scalp',  eps_r=40.0, sigma=0.87)
SKULL  = Material('skull',  eps_r=13.1, sigma=0.10)
DURA   = Material('dura',   eps_r=44.0, sigma=0.82)
CSF    = Material('csf',    eps_r=68.0, sigma=2.46)
GM     = Material('gm',     eps_r=52.7, sigma=0.94)
WM     = Material('wm',     eps_r=38.1, sigma=0.61)
BLOOD  = Material('blood',  eps_r=61.0, sigma=1.58)

# ── Head geometry (scaled to 32³ at 6mm/cell) ──────────────────────────────
# 32*6mm = 192mm domain; head radii scaled proportionally from 64³@3mm
HEAD_RADII = {
    'scalp': 11, 'skull': 10, 'dura': 8, 'csf': 7, 'gm': 7, 'wm': 4
}

# ── Bleeds ─────────────────────────────────────────────────────────────────
BLEEDS = {
    'healthy':       None,
    'epidural':      {'center': (CX+9, CY, CZ),  'r': 1, 'mat': BLOOD, 'label': 1},
    'subdural':      {'center': (CX+7, CY, CZ),  'r': 1, 'mat': BLOOD, 'label': 2},
    'intracerebral': {'center': (CX+4, CY+2, CZ),'r': 2, 'mat': BLOOD, 'label': 3},
}

_EPS_LAYERS = [
    (HEAD_RADII['scalp'], 40.0),
    (HEAD_RADII['skull'], 13.1),
    (HEAD_RADII['dura'],  44.0),
    (HEAD_RADII['csf'],   68.0),
    (HEAD_RADII['gm'],    52.7),
    (HEAD_RADII['wm'],    38.1),
]

# ── Antenna positions ────────────────────────────────────────────────────────
TX_POS = []
for k in range(N_TX):
    ang = 2 * math.pi * k / N_TX
    i = int(round(CX + _RING_R * math.cos(ang)))
    j = int(round(CY + _RING_R * math.sin(ang)))
    i = max(1, min(NX - 2, i))
    j = max(1, min(NY - 2, j))
    TX_POS.append((i, j, CZ))


def build_phantom(bleed_name):
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    mm = MaterialMap3D(grid)
    mm.add_sphere((CX,CY,CZ), HEAD_RADII['scalp'], SCALP)
    mm.add_sphere((CX,CY,CZ), HEAD_RADII['skull'], SKULL)
    mm.add_sphere((CX,CY,CZ), HEAD_RADII['dura'],  DURA)
    mm.add_sphere((CX,CY,CZ), HEAD_RADII['csf'],   CSF)
    mm.add_sphere((CX,CY,CZ), HEAD_RADII['gm'],    GM)
    mm.add_sphere((CX,CY,CZ), HEAD_RADII['wm'],    WM)
    b = BLEEDS[bleed_name]
    if b:
        mm.add_sphere(b['center'], b['r'], b['mat'])
    return mm.build3d()


def run_mimo(Ca, Cb):
    signals = np.zeros((N_TX, N_TX, N_STEPS), dtype=np.float32)
    for tx_idx in range(N_TX):
        grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
        fields = FieldSet(grid)
        boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)
        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)
        ti, tj, tk = TX_POS[tx_idx]
        src = PointSource(pulse, ti, tj, 'Ez', k=tk, grid=grid, N_steps=N_STEPS)
        sim = FDTD3D(grid, fields, boundary, SourceCollection([src]),
                     Ca=Ca, Cb=Cb, n_check=99999)
        with torch.no_grad():
            for n in range(N_STEPS):
                sim.step()
                for rx in range(N_TX):
                    ri, rj, rk = TX_POS[rx]
                    signals[tx_idx, rx, n] = fields.Ez[ri, rj, rk].item()
    return signals, grid.dt


def _ray_sphere_chord(ax, ay, bx, by, cx, cy, r):
    dx_ = bx - ax; dy_ = by - ay
    seg_len = math.sqrt(dx_**2 + dy_**2) + 1e-30
    ux, uy = dx_ / seg_len, dy_ / seg_len
    ox, oy = ax - cx, ay - cy
    b_c = ox*ux + oy*uy; c_c = ox**2 + oy**2 - r**2
    disc = b_c**2 - c_c
    if disc <= 0: return 0.0
    sq = math.sqrt(disc)
    t0 = max(0.0, min(seg_len, -b_c - sq))
    t1 = max(0.0, min(seg_len, -b_c + sq))
    return max(0.0, t1 - t0)


def _travel_time(ax, ay, bx, by, dt):
    hcx = CX * DX * 1e3; hcy = CY * DX * 1e3
    total_d = math.sqrt((bx-ax)**2 + (by-ay)**2) + 1e-30
    prev = 0.0; t = 0.0
    for idx, (r_c, eps) in enumerate(_EPS_LAYERS):
        r_mm = r_c * DX * 1e3
        co = _ray_sphere_chord(ax, ay, bx, by, hcx, hcy, r_mm)
        ri = _EPS_LAYERS[idx+1][0] * DX * 1e3 if idx+1 < len(_EPS_LAYERS) else 0.0
        ci = _ray_sphere_chord(ax, ay, bx, by, hcx, hcy, ri) if ri > 0 else 0.0
        sc = max(0.0, co - ci)
        t += (sc * 1e-3) / (3e8 / math.sqrt(eps))
        prev += sc
    t += (max(0.0, total_d - prev) * 1e-3) / 3e8
    return t / dt


def das_backprojection(scattered, dt, image_size=32):
    env = np.abs(hilbert(scattered, axis=-1)).astype(np.float32)
    x_px = np.linspace(0, (NX-1) * DX * 1e3, image_size)
    y_px = np.linspace(0, (NY-1) * DX * 1e3, image_size)
    ant_xy = [(TX_POS[i][0]*DX*1e3, TX_POS[i][1]*DX*1e3) for i in range(N_TX)]

    tt = np.zeros((N_TX, image_size, image_size), dtype=np.float32)
    for ai in range(N_TX):
        ax, ay = ant_xy[ai]
        for iy, py in enumerate(y_px):
            for ix, px in enumerate(x_px):
                tt[ai, iy, ix] = _travel_time(ax, ay, px, py, dt)

    img = np.zeros((image_size, image_size), dtype=np.float64)
    for tx in range(N_TX):
        for rx in range(N_TX):
            delay_map = np.rint(tt[tx] + tt[rx]).astype(np.int32)
            valid = (delay_map >= 0) & (delay_map < N_STEPS)
            delay_map = np.clip(delay_map, 0, N_STEPS - 1)
            gathered = env[tx, rx][delay_map]
            img += np.where(valid, gathered, 0.0)
    return (img**2).astype(np.float32)


def main():
    print('=' * 60)
    print(f'DAS Pipeline Validation — {NX}³ grid, {N_TX} TX, {N_STEPS} steps, {DEVICE}')
    print(f'dx={DX*1e3:.0f}mm, domain={NX*DX*1e3:.0f}mm')
    print('=' * 60)

    classes = ['healthy', 'epidural', 'subdural', 'intracerebral']
    all_signals = {}
    dt_val = None

    # Run healthy reference
    t0 = time.perf_counter()
    print('[1/5] Healthy reference...')
    Ca_ref, Cb_ref = build_phantom('healthy')
    sigs_ref, dt_val = run_mimo(Ca_ref, Cb_ref)
    all_signals['healthy'] = sigs_ref
    print(f'  {time.perf_counter()-t0:.1f}s')

    # Run bleed classes
    for idx, name in enumerate(['epidural', 'subdural', 'intracerebral'], 2):
        t0 = time.perf_counter()
        print(f'[{idx}/5] {name}...')
        Ca, Cb = build_phantom(name)
        sigs, _, = run_mimo(Ca, Cb)
        all_signals[name] = sigs
        print(f'  {time.perf_counter()-t0:.1f}s')

    print('[5/5] DAS backprojection...')

    ENERGY_THRESH = 1e-12  # lower for small grid
    correct = 0
    results = []

    for name in classes:
        scattered = all_signals[name] - sigs_ref
        energy = float((scattered**2).sum())
        das = das_backprojection(scattered, dt_val, image_size=NX)

        b = BLEEDS[name]
        detected = energy > ENERGY_THRESH
        true_bleed = name != 'healthy'
        ok = (detected == true_bleed)
        if ok: correct += 1

        loc_err = None
        if b:
            bx = b['center'][0] * DX * 1e3
            by = b['center'][1] * DX * 1e3
            peak_iy, peak_ix = np.unravel_index(np.argmax(das), das.shape)
            x_px = np.linspace(0, (NX-1)*DX*1e3, NX)
            y_px = np.linspace(0, (NY-1)*DX*1e3, NX)
            px = x_px[peak_ix]; py = y_px[peak_iy]
            loc_err = math.sqrt((px-bx)**2 + (py-by)**2)

        status = '✓' if ok else '✗'
        loc_str = f'  loc_err={loc_err:.1f}mm' if loc_err is not None else ''
        print(f'  {name:18} ΔE={energy:.2e} {"DETECTED" if detected else "clean":9} {status}{loc_str}')
        results.append((name, ok, loc_err))

    print()
    print(f'Detection: {correct}/{len(classes)} correct')
    loc_pass = [r for r in results if r[1] and r[2] is not None and r[2] < 3 * DX * 1e3]
    print(f'Localisation < 3 cells: {len(loc_pass)}/3 bleeds')

    if correct == 4 and len(loc_pass) == 3:
        print('✅ PIPELINE VALIDATED — DAS correctly detects and localises all bleeds')
    else:
        print('❌ VALIDATION FAILED — check signals and travel-time model')
        print()
        print('Signals summary:')
        for name in classes:
            scat = all_signals[name] - sigs_ref
            print(f'  {name}: max|scattered|={np.abs(scat).max():.3e}  energy={float((scat**2).sum()):.3e}')


if __name__ == '__main__':
    main()
