"""
10_breast_tumor_mimo.py — Microwave MIMO imaging of breast tumor.

Simulates a 2D cross-section of a breast with a malignant tumor using a
circular MIMO antenna array at 1 GHz.

Physical setup (based on validated dielectric breast models):
  Domain:    200x200 cells, dx=dy=1.5mm → 300mm x 300mm
  Breast:    Elliptical cross-section (semi-axes 70x80 cells)
  Layers:    Skin shell (2 cells), adipose interior (low-eps, main bulk),
             fibroglandular core (higher eps, inner 30% of breast),
             Tumor: circle r=8 cells inside breast at offset position

Tissue properties at 1 GHz:
  Adipose fat:       eps_r=5,  sigma=0.05  (most of breast, low loss)
  Fibroglandular:    eps_r=25, sigma=0.55  (dense tissue, high contrast)
  Tumor (malignant): eps_r=55, sigma=0.70  (~11× eps contrast vs adipose!)
  Skin:              eps_r=36, sigma=0.40

Why breast imaging is easier than brain:
  - Adipose→tumor contrast: delta_eps_r = 50 (vs brain delta = 15)
  - 4× stronger scattered signal
  - Lower overall tissue loss (adipose skin_depth=71mm vs brain 13mm)

Run:  python examples/10_breast_tumor_mimo.py
Out:  examples/output/10_breast_tumor_mimo.png
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.grid import YeeGrid, C0
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import RickerWavelet, PointSource, SourceCollection
from core.materials import Material, MaterialMap
from core.fdtd2d import FDTD2D

# ── Config ────────────────────────────────────────────────────────────────────
NX, NY       = 200, 200
DX           = 1.5e-3       # 1.5 mm → 300mm × 300mm domain
N_STEPS      = 1000
N_TX         = 16           # full MIMO for better image quality
FREQ         = 1e9
ARRAY_R      = 88           # cells — array just outside breast
CENTER       = (100, 100)

# Breast geometry (ellipse)
BREAST_A     = 70           # semi-axis x (cells)
BREAST_B     = 80           # semi-axis y (cells)
SKIN_THICK   = 2            # cells
FIBRO_A      = 35           # fibroglandular core semi-axis x
FIBRO_B      = 40           # fibroglandular core semi-axis y

# Tumor
TUMOR_CENTER = (125, 110)   # offset toward upper-right
TUMOR_R      = 8            # cells = 12mm

# Tissue materials at 1 GHz
MAT_ADIPOSE  = Material('adipose',       eps_r=5.0,  sigma=0.05)
MAT_FIBRO    = Material('fibroglandular',eps_r=25.0, sigma=0.55)
MAT_TUMOR    = Material('tumor',         eps_r=55.0, sigma=0.70)
MAT_SKIN     = Material('skin',          eps_r=36.0, sigma=0.40)

OUTPUT_DIR   = Path(__file__).parent / 'output'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_antenna_positions(n, radius, center):
    cx, cy = center
    return [(int(round(cx + radius*math.cos(2*math.pi*k/n))),
             int(round(cy + radius*math.sin(2*math.pi*k/n)))) for k in range(n)]


def build_breast_map(grid, include_tumor=True):
    """Build elliptical breast phantom with skin, adipose, fibroglandular, tumor."""
    cx, cy = float(CENTER[0]), float(CENTER[1])
    mm = MaterialMap(grid, default=Material('air', eps_r=1.0, sigma=0.0))

    # Outer skin shell (ellipse with extra thickness)
    def skin_mask(I, J):
        outer = ((I-cx)/BREAST_A)**2 + ((J-cy)/BREAST_B)**2 <= 1.0
        inner = ((I-cx)/(BREAST_A-SKIN_THICK))**2 + ((J-cy)/(BREAST_B-SKIN_THICK))**2 <= 1.0
        return outer & ~inner
    mm.add_custom(skin_mask, MAT_SKIN)

    # Adipose interior (full breast interior minus skin)
    def adipose_mask(I, J):
        return ((I-cx)/(BREAST_A-SKIN_THICK))**2 + ((J-cy)/(BREAST_B-SKIN_THICK))**2 <= 1.0
    mm.add_custom(adipose_mask, MAT_ADIPOSE)

    # Fibroglandular core (inner ellipse)
    def fibro_mask(I, J):
        return ((I-cx)/FIBRO_A)**2 + ((J-cy)/FIBRO_B)**2 <= 1.0
    mm.add_custom(fibro_mask, MAT_FIBRO)

    # Tumor (only if requested)
    if include_tumor:
        mm.add_circle(TUMOR_CENTER, TUMOR_R, MAT_TUMOR)

    Ca, Cb = mm.build()

    # Build eps_r map for visualization
    I = np.arange(NX).reshape(-1, 1) * np.ones((1, NY))
    J = np.ones((NX, 1)) * np.arange(NY).reshape(1, -1)
    eps_map = np.ones((NX, NY), dtype=np.float32)
    in_outer  = ((I-cx)/BREAST_A)**2 + ((J-cy)/BREAST_B)**2 <= 1.0
    in_inner  = ((I-cx)/(BREAST_A-SKIN_THICK))**2 + ((J-cy)/(BREAST_B-SKIN_THICK))**2 <= 1.0
    in_fibro  = ((I-cx)/FIBRO_A)**2 + ((J-cy)/FIBRO_B)**2 <= 1.0
    eps_map[in_outer & ~in_inner] = 36.0   # skin
    eps_map[in_inner]              = 5.0    # adipose
    eps_map[in_fibro]              = 25.0   # fibroglandular
    if include_tumor:
        tc_x, tc_y = TUMOR_CENTER
        in_tumor = (I-tc_x)**2 + (J-tc_y)**2 <= TUMOR_R**2
        eps_map[in_tumor] = 55.0

    return Ca, Cb, eps_map


def run_mimo(grid, Ca, Cb, ant_pos, n_steps, label):
    N = len(ant_pos)
    all_sigs = np.zeros((N, N, n_steps), dtype=np.float32)
    rx_i = torch.tensor([p[0] for p in ant_pos], dtype=torch.long, device=grid.device)
    rx_j = torch.tensor([p[1] for p in ant_pos], dtype=torch.long, device=grid.device)

    for tx in range(N):
        print(f"  [{label}] TX {tx+1}/{N}", end='\r')
        fields   = FieldSet(grid)
        boundary = MurABC(grid, fields.Hz)
        ti, tj   = ant_pos[tx]
        t0_wav   = 1.5 / FREQ
        wv       = RickerWavelet(amplitude=1000.0, peak_freq=FREQ, t0=t0_wav)
        src      = PointSource(wv, ti, tj, 'Hz', grid=grid, N_steps=n_steps)
        sim      = FDTD2D(grid, fields, boundary, SourceCollection([src]),
                          Ca=Ca, Cb=Cb, n_check=500)
        buf = torch.zeros(N, n_steps, device=grid.device)
        for step in range(n_steps):
            sim.step()
            buf[:, step] = fields.Hz[rx_i, rx_j, 0]
        all_sigs[tx] = buf.cpu().numpy()
    print(f"  [{label}] TX {N}/{N} — done")
    return all_sigs


def das_backprojection(delta_S, ant_pos, grid, n_steps):
    """Delay-and-sum backprojection. Returns power image (NX, NY)."""
    dt = grid.dt; dx = grid.dx
    # Effective permittivity: volume-weighted breast average
    # ~60% adipose (eps=5) + 30% fibroglandular (eps=25) + 10% skin (eps=36)
    eps_eff = 0.60*5.0 + 0.30*25.0 + 0.10*36.0   # ≈ 13.1
    v_bg = C0 / math.sqrt(eps_eff)
    ant_x = np.array([p[0]*dx for p in ant_pos])
    ant_y = np.array([p[1]*dx for p in ant_pos])
    PX, PY = (np.arange(NX)*dx)[:,None], (np.arange(NY)*dx)[None,:]
    image = np.zeros((NX, NY), dtype=np.float64)
    for tx in range(len(ant_pos)):
        for rx in range(len(ant_pos)):
            d_tx = np.sqrt((PX - ant_x[tx])**2 + (PY - ant_y[tx])**2)
            d_rx = np.sqrt((PX - ant_x[rx])**2 + (PY - ant_y[rx])**2)
            n_tau = np.clip(np.round((d_tx+d_rx)/v_bg/dt).astype(np.int32), 0, n_steps-1)
            image += delta_S[tx, rx, n_tau]
    return image**2


def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    import matplotlib.patches as mpatches
    import time

    t_start = time.perf_counter()
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, device=DEVICE)
    ant_pos = get_antenna_positions(N_TX, ARRAY_R, CENTER)

    print("="*60)
    print("GPU-MEEP: Breast Tumor MIMO Imaging (1 GHz)")
    print(f"Grid: {NX}x{NY}, dx={DX*1e3:.1f}mm, N_TX={N_TX}, device={DEVICE}")
    print(f"Breast: {BREAST_A*2*DX*1e3:.0f}x{BREAST_B*2*DX*1e3:.0f}mm ellipse")
    print(f"Tumor:  r={TUMOR_R*DX*1e3:.0f}mm at ({TUMOR_CENTER[0]*DX*1e3:.0f},{TUMOR_CENTER[1]*DX*1e3:.0f})mm")
    print("="*60)

    # Run healthy
    print("\nRunning healthy breast (no tumor)...")
    Ca_h, Cb_h, eps_h = build_breast_map(grid, include_tumor=False)
    S_healthy = run_mimo(grid, Ca_h, Cb_h, ant_pos, N_STEPS, 'Healthy')

    # Run with tumor
    print("Running breast with tumor...")
    Ca_t, Cb_t, eps_t = build_breast_map(grid, include_tumor=True)
    S_tumor = run_mimo(grid, Ca_t, Cb_t, ant_pos, N_STEPS, 'Tumor')

    # Scattered field and DAS
    delta_S = S_tumor - S_healthy
    print(f"\nMax |delta_S|: {np.abs(delta_S).max():.3e}")
    print("Running DAS backprojection...")
    image = das_backprojection(delta_S, ant_pos, grid, N_STEPS)
    peak_idx = np.unravel_index(np.argmax(image), image.shape)
    peak_mm  = (peak_idx[0]*DX*1e3, peak_idx[1]*DX*1e3)
    true_mm  = (TUMOR_CENTER[0]*DX*1e3, TUMOR_CENTER[1]*DX*1e3)
    print(f"DAS peak:  ({peak_mm[0]:.0f}, {peak_mm[1]:.0f}) mm")
    print(f"True tumor:({true_mm[0]:.0f}, {true_mm[1]:.0f}) mm")
    err = math.sqrt((peak_mm[0]-true_mm[0])**2 + (peak_mm[1]-true_mm[1])**2)
    print(f"Localization error: {err:.1f} mm")

    # ── Plot ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.suptitle('GPU-MEEP: Microwave Breast Tumor Detection (1 GHz MIMO)',
                 fontsize=14, fontweight='bold')

    ext = [0, NX*DX*1e3, 0, NY*DX*1e3]
    cmap4  = plt.colormaps.get_cmap('tab10').resampled(5)
    bounds = [0, 3, 20, 30, 50, 65]
    norm4  = BoundaryNorm(bounds, cmap4.N)
    tissue_labels = ['Air(ε=1)', 'Adipose(ε=5)', 'Fibro(ε=25)', 'Skin(ε=36)', 'Tumor(ε=55)']

    # Row 1, Col 1: Healthy eps_r map
    im0 = axes[0,0].imshow(eps_h.T, origin='lower', cmap=cmap4, norm=norm4,
                            extent=ext, aspect='auto')
    for ai,aj in ant_pos:
        axes[0,0].plot(ai*DX*1e3, aj*DX*1e3, 'w^', ms=4, markeredgecolor='k', markeredgewidth=0.3)
    axes[0,0].set(xlabel='x (mm)', ylabel='y (mm)', title='Healthy Breast — εᵣ map')
    cb0 = plt.colorbar(im0, ax=axes[0,0])
    cb0.set_ticks([1.5, 12.5, 27.5, 43, 57.5])
    cb0.set_ticklabels(['Air\n(ε=1)', 'Adipose\n(ε=5)', 'Fibro\n(ε=25)', 'Skin\n(ε=36)', 'Tumor\n(ε=55)'], fontsize=7)

    # Row 1, Col 2: Tumor eps_r map
    im1 = axes[0,1].imshow(eps_t.T, origin='lower', cmap=cmap4, norm=norm4,
                            extent=ext, aspect='auto')
    for ai,aj in ant_pos:
        axes[0,1].plot(ai*DX*1e3, aj*DX*1e3, 'w^', ms=4, markeredgecolor='k', markeredgewidth=0.3)
    tc = mpatches.Circle((TUMOR_CENTER[0]*DX*1e3, TUMOR_CENTER[1]*DX*1e3),
                          TUMOR_R*DX*1e3, fill=False, color='cyan', lw=2)
    axes[0,1].add_patch(tc)
    axes[0,1].set(xlabel='x (mm)', ylabel='y (mm)', title='Breast with Tumor — εᵣ map (cyan=tumor)')
    plt.colorbar(im1, ax=axes[0,1])

    # Row 1, Col 3: DAS image
    vmax = image.max()
    im2  = axes[0,2].imshow(image.T, origin='lower', cmap='hot',
                             vmin=0, vmax=vmax, extent=ext, aspect='auto')
    tc2  = mpatches.Circle((TUMOR_CENTER[0]*DX*1e3, TUMOR_CENTER[1]*DX*1e3),
                            TUMOR_R*DX*1e3, fill=False, color='cyan', lw=2, label='True tumor')
    axes[0,2].add_patch(tc2)
    axes[0,2].plot(*peak_mm, 'c+', ms=12, mew=2, label=f'DAS peak ({err:.0f}mm err)')
    axes[0,2].set(xlabel='x (mm)', ylabel='y (mm)', title='DAS Backprojection — Tumor Power')
    axes[0,2].legend(fontsize=8)
    plt.colorbar(im2, ax=axes[0,2]).set_label('Power')

    # Row 2, Col 1: Signal comparison TX0 → opposite RX
    rx_opp = N_TX // 2
    t_ns = np.arange(N_STEPS)*grid.dt*1e9
    axes[1,0].plot(t_ns, S_healthy[0, rx_opp, :], label='Healthy', alpha=0.8)
    axes[1,0].plot(t_ns, S_tumor[0, rx_opp, :],   label='With tumor', alpha=0.8)
    axes[1,0].set(xlabel='Time (ns)', ylabel='Hz (A/m)',
                  title=f'TX0 → RX{rx_opp} (opposite) signal comparison')
    axes[1,0].legend(fontsize=9); axes[1,0].grid(alpha=0.3)

    # Row 2, Col 2: Scattered signal (delta_S)
    axes[1,1].plot(t_ns, delta_S[0, rx_opp, :]*100, label='100×ΔS', color='red')
    axes[1,1].axhline(0, color='k', lw=0.5)
    axes[1,1].set(xlabel='Time (ns)', ylabel='Hz (A/m)',
                  title='Scattered signal ΔS = S_tumor − S_healthy')
    axes[1,1].legend(fontsize=9); axes[1,1].grid(alpha=0.3)

    # Row 2, Col 3: Tissue contrast bar chart
    tissue_names = ['Adipose→Fibro', 'Adipose→Skin', 'Adipose→Tumor']
    delta_eps   = [25-5, 36-5, 55-5]
    colors      = ['steelblue', 'orange', 'red']
    bars = axes[1,2].barh(tissue_names, delta_eps, color=colors, alpha=0.85)
    axes[1,2].bar_label(bars, [f'Δε={d}' for d in delta_eps], fontsize=10, padding=3)
    axes[1,2].set(xlabel='Δεᵣ (contrast vs adipose)', title='Dielectric Contrast at 1 GHz')
    axes[1,2].axvline(15, color='grey', ls='--', alpha=0.5, label='Brain clot Δε=15')
    axes[1,2].legend(fontsize=8); axes[1,2].grid(alpha=0.3, axis='x')

    fig.tight_layout()
    path = OUTPUT_DIR / '10_breast_tumor_mimo.png'
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"\nSaved: {path}")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Physics summary:")
    print(f"  Adipose→Tumor contrast: Δεᵣ={55-5}  (brain clot: Δεᵣ={55-40})")
    print(f"  Signal contrast: {np.abs(delta_S).max()/max(np.abs(S_healthy).max(),1e-30)*100:.4f}%")
    print(f"  DAS localization error: {err:.1f} mm (tumor r={TUMOR_R*DX*1e3:.0f}mm)")
    print(f"  Breast window: {BREAST_A*2*DX*1e3:.0f}×{BREAST_B*2*DX*1e3:.0f}mm")
    print(f"{'='*60}")

if __name__ == '__main__': main()
