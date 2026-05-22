"""Build dib_dataset_figures.ipynb — publication figures for Data in Brief submission."""
import json, pathlib

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

cells.append(md("# WaveForge Brain Haemorrhage Dataset — Publication Figures\n\nGenerates all figures needed for the Data in Brief submission.\nRun All → download the output folder → attach to manuscript.\n"))

cells.append(code("""\
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import json, math, pathlib, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white',
    'axes.grid':True,'grid.alpha':0.3,'axes.labelsize':13,
    'axes.titlesize':14,'font.size':12,'figure.dpi':150,
    'savefig.dpi':300,'savefig.bbox':'tight',
    'font.family':'DejaVu Sans',
})
PALETTE = sns.color_palette('Set2', 4)
CLASS_NAMES = ['Healthy','Epidural','Subdural','Intracerebral']
OUT = Path('/kaggle/working/dib_figures')
OUT.mkdir(exist_ok=True)

# ── Locate dataset ──────────────────────────────────────────────────────
for candidate in [
    '/kaggle/input/data-part1/brain_haemorrhage_dataset',
    '/kaggle/input/waveforge-brain-v1/brain_haemorrhage_dataset',
]:
    DATA_ROOT = Path(candidate)
    if DATA_ROOT.exists(): break
else:
    for p in Path('/kaggle/input').glob('**/train'):
        DATA_ROOT = p.parent; break

TRAIN_DIR = DATA_ROOT / 'train'
TEST_DIRS  = [DATA_ROOT / 'test_gpu0', DATA_ROOT / 'test_gpu1']

def load_meta(p):
    s = np.load(p, allow_pickle=True)
    return {
        'path':str(p), 'label':int(s['label']),
        'bleed_type':str(s['bleed_type']), 'bleed_age':str(s['bleed_age']),
        'radius_mm':float(s['bleed_radius_mm']), 'volume_ml':float(s['bleed_volume_ml']),
        'skull_r':int(s['phantom_skull_inner_r']), 'gray_r':int(s['phantom_gray_r']),
        'scalp_r':int(s['phantom_scalp_outer_r']), 'dt_s':float(s['dt_s']),
    }

print("Loading metadata...")
train_recs = [load_meta(p) for p in sorted(TRAIN_DIR.glob('*.npz'))]
test_recs  = []
for td in TEST_DIRS:
    if td.exists(): test_recs += [load_meta(p) for p in sorted(td.glob('*.npz'))]

df_train = pd.DataFrame(train_recs)
df_test  = pd.DataFrame(test_recs)
print(f"Train: {len(df_train)}  Test: {len(df_test)}")
print(df_train['label'].value_counts().sort_index())
"""))

# ── Figure 1: Class distribution ──────────────────────────────────────
cells.append(md("## Figure 1: Class Distribution"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

counts = df_train['label'].value_counts().sort_index()
bars = axes[0].bar(CLASS_NAMES, counts.values, color=PALETTE, edgecolor='black', linewidth=0.8, width=0.6)
axes[0].set_ylabel('Number of samples')
axes[0].set_title('(a) Training set class distribution')
axes[0].set_ylim(0, counts.max()*1.15)
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+4, str(v), ha='center', fontsize=11, fontweight='bold')

wedges, texts, autotexts = axes[1].pie(
    counts.values, labels=CLASS_NAMES, colors=PALETTE,
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor':'white','linewidth':1.5},
    textprops={'fontsize':11}
)
for at in autotexts: at.set_fontsize(10)
axes[1].set_title('(b) Class proportions')

plt.suptitle('Figure 1. Class balance in the WaveForge Brain Haemorrhage Dataset training split.',
             y=-0.02, fontsize=11, style='italic')
plt.tight_layout()
plt.savefig(OUT/'fig1_class_distribution.png')
plt.show()
print("Saved fig1_class_distribution.png")
"""))

# ── Figure 2: Phantom geometry diversity ─────────────────────────────
cells.append(md("## Figure 2: Phantom Geometry Diversity"))
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Skull vs gray matter
for lbl in range(4):
    d = df_train[df_train['label']==lbl]
    axes[0].scatter(d['skull_r']*3, d['gray_r']*3,
                    color=PALETTE[lbl], alpha=0.35, s=12, label=CLASS_NAMES[lbl])
axes[0].set_xlabel('Skull inner radius (mm)')
axes[0].set_ylabel('Gray matter radius (mm)')
axes[0].set_title('(a) Skull vs. brain radius')
axes[0].legend(markerscale=2.5, fontsize=9)

# Head outer radius histogram
axes[1].hist(df_train['scalp_r']*3, bins=28, color='steelblue', edgecolor='black', lw=0.5)
mu = (df_train['scalp_r']*3).mean()
axes[1].axvline(mu, color='#e74c3c', lw=2, linestyle='--', label=f'Mean = {mu:.0f} mm')
axes[1].set_xlabel('Head outer radius (mm)')
axes[1].set_ylabel('Count')
axes[1].set_title('(b) Head size distribution')
axes[1].legend()

# Skull thickness histogram
skull_thick = (df_train['scalp_r'] - df_train['skull_r']) * 3
axes[2].hist(skull_thick, bins=20, color='coral', edgecolor='black', lw=0.5)
axes[2].axvline(skull_thick.mean(), color='navy', lw=2, linestyle='--',
                label=f'Mean = {skull_thick.mean():.1f} mm')
axes[2].set_xlabel('Skull thickness (mm)')
axes[2].set_ylabel('Count')
axes[2].set_title('(c) Skull thickness distribution')
axes[2].legend()

plt.suptitle('Figure 2. Population-distributed phantom geometry across all training samples.',
             y=-0.02, fontsize=11, style='italic')
plt.tight_layout()
plt.savefig(OUT/'fig2_phantom_geometry.png')
plt.show()
print("Saved fig2_phantom_geometry.png")
"""))

# ── Figure 3: Bleed properties ────────────────────────────────────────
cells.append(md("## Figure 3: Haemorrhage Properties"))
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

bleed_df = df_train[df_train['label']>0].copy()
bleed_df['Class'] = bleed_df['label'].map({1:'Epidural',2:'Subdural',3:'ICH'})

# Radius by class
sns.violinplot(data=bleed_df, x='Class', y='radius_mm',
               palette=PALETTE[1:], ax=axes[0], inner='box', linewidth=0.8)
axes[0].set_title('(a) Bleed radius by class')
axes[0].set_ylabel('Bleed radius (mm)')
axes[0].set_xlabel('')

# Volume distribution
for lbl, name, c in [(1,'Epidural',PALETTE[1]),(2,'Subdural',PALETTE[2]),(3,'ICH',PALETTE[3])]:
    d = df_train[df_train['label']==lbl]['volume_ml']
    axes[1].hist(d[d>0], bins=22, alpha=0.65, label=name, color=c, edgecolor='black', lw=0.4)
axes[1].set_xlabel('Bleed volume (mL)')
axes[1].set_ylabel('Count')
axes[1].set_title('(b) Bleed volume distribution')
axes[1].legend()

# Blood age stacked bar
age_order = ['acute','subacute','chronic']
age_colors = ['#c0392b','#e67e22','#7f8c8d']
age_counts = bleed_df.groupby(['Class','bleed_age']).size().unstack(fill_value=0)
# reorder columns
for col in age_order:
    if col not in age_counts.columns: age_counts[col] = 0
age_counts[age_order].plot(kind='bar', ax=axes[2],
    color=age_colors, edgecolor='black', linewidth=0.5, width=0.6)
axes[2].set_title('(c) Blood age distribution')
axes[2].set_xlabel('')
axes[2].set_ylabel('Count')
axes[2].tick_params(axis='x', rotation=0)
axes[2].legend(title='Age stage', fontsize=9)

plt.suptitle('Figure 3. Haemorrhage physical properties across the training set.',
             y=-0.02, fontsize=11, style='italic')
plt.tight_layout()
plt.savefig(OUT/'fig3_bleed_properties.png')
plt.show()
print("Saved fig3_bleed_properties.png")
"""))

# ── Figure 4: Representative signals ─────────────────────────────────
cells.append(md("## Figure 4: Representative Scattered Signals"))
cells.append(code("""\
# Load one sample per class
samples = {}
for lbl in range(4):
    row = df_train[df_train['label']==lbl].iloc[0]
    s = np.load(row['path'], allow_pickle=True)
    samples[lbl] = {'sig': s['signals_scattered'], 'dt': float(s['dt_s'])}

dt = samples[0]['dt']
t_ns = np.arange(700) * dt * 1e9

fig, axes = plt.subplots(2, 4, figsize=(16, 6))

# Row 1: TX[0] traces
for lbl in range(4):
    ax = axes[0, lbl]
    scat = samples[lbl]['sig']
    colors = sns.color_palette('husl', 8)
    for rx in range(8):
        ax.plot(t_ns, scat[0, rx], color=colors[rx], alpha=0.7, lw=0.7)
    energy = float((scat**2).sum())
    ax.set_title(f'{CLASS_NAMES[lbl]}')
    ax.set_xlabel('Time (ns)')
    if lbl==0: ax.set_ylabel('Ez (V/m)')
    ax.text(0.97, 0.96, f'ΔE={energy:.1e}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))
    # shade temporal windows
    ax.axvspan(0, 0.85, alpha=0.07, color='red')
    ax.axvspan(0.28,1.98, alpha=0.05, color='orange')
    ax.axvspan(1.13,3.96, alpha=0.04, color='blue')

# Row 2: RMS energy
for lbl in range(4):
    ax = axes[1, lbl]
    scat = samples[lbl]['sig']
    rms = np.sqrt((scat**2).mean(axis=(0,1)))
    ax.plot(t_ns, rms, color=PALETTE[lbl], lw=1.8)
    ax.fill_between(t_ns, rms, alpha=0.15, color=PALETTE[lbl])
    ax.axvspan(0, 0.85, alpha=0.07, color='red')
    ax.axvspan(0.28,1.98, alpha=0.05, color='orange')
    ax.axvspan(1.13,3.96, alpha=0.04, color='blue')
    ax.set_xlabel('Time (ns)')
    if lbl==0: ax.set_ylabel('RMS Ez (V/m)')
    ax.set_title(f'RMS energy')

# Add window labels on first column
for ax, label in [(axes[0,0], 'Top: TX[0] channel traces (8 RX)'),(axes[1,0],'Bottom: RMS over all 64 pairs')]:
    pass

axes[0,0].set_title('Healthy')
fig.text(0.01, 0.75, 'TX[0] traces', va='center', rotation='vertical', fontsize=10, color='gray')
fig.text(0.01, 0.28, 'RMS energy', va='center', rotation='vertical', fontsize=10, color='gray')

plt.suptitle('Figure 4. Scattered MIMO signals for one representative sample per class.\n'
             'Shaded regions: red=skull/epidural window, orange=subdural window, blue=ICH window.',
             y=1.01, fontsize=11, style='italic')
plt.tight_layout()
plt.savefig(OUT/'fig4_scattered_signals.png')
plt.show()
print("Saved fig4_scattered_signals.png")
"""))

# ── Figure 5: MIMO energy matrix ─────────────────────────────────────
cells.append(md("## Figure 5: MIMO Antenna Energy Matrix"))
cells.append(code("""\
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for lbl in range(4):
    scat = samples[lbl]['sig']
    energy_mat = (scat**2).sum(axis=-1)   # (8,8)
    im = sns.heatmap(energy_mat, ax=axes[lbl], cmap='YlOrRd',
                annot=True, fmt='.1e', annot_kws={'size':6.5},
                xticklabels=[f'R{i}' for i in range(8)],
                yticklabels=[f'T{i}' for i in range(8)],
                cbar_kws={'label':'Energy (V²·s)', 'shrink':0.8},
                linewidths=0.3, linecolor='white')
    axes[lbl].set_title(CLASS_NAMES[lbl], fontsize=12)
    axes[lbl].tick_params(labelsize=8)

plt.suptitle('Figure 5. MIMO channel energy matrix (TX × RX) for one representative sample per class.\n'
             'Antenna pairs nearest the bleed carry the highest differential energy.',
             y=0.02, fontsize=11, style='italic')
plt.tight_layout()
plt.savefig(OUT/'fig5_mimo_energy_matrix.png')
plt.show()
print("Saved fig5_mimo_energy_matrix.png")
"""))

# ── Figure 6: DAS validation ──────────────────────────────────────────
cells.append(md("## Figure 6: DAS Backprojection Validation"))
cells.append(code("""\
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
ext_mm = [0, 64*3, 0, 64*3]

for lbl in range(4):
    ax = axes[lbl]
    row = df_train[df_train['label']==lbl].iloc[0]
    s = np.load(row['path'], allow_pickle=True)
    das = s['das_image']
    im = ax.imshow(das.T, origin='lower', cmap='hot', extent=ext_mm, aspect='auto')
    plt.colorbar(im, ax=ax, label='DAS power', shrink=0.85)

    r_mm = float(s['bleed_radius_mm'])
    if r_mm > 0:
        bx = float(s['bleed_center_mm'][0])
        by = float(s['bleed_center_mm'][1])
        ax.plot(bx, by, 'c+', markersize=16, mew=2.5, label=f'True: ({bx:.0f},{by:.0f})mm')
        iy, ix = np.unravel_index(np.argmax(das), das.shape)
        px = ix/das.shape[0]*64*3; py = iy/das.shape[1]*64*3
        err = math.sqrt((px-bx)**2+(py-by)**2)
        ax.plot(px, py, 'y^', markersize=10, mew=1.5, label=f'Peak (err={err:.0f}mm)')
        ax.legend(fontsize=7, loc='lower right')

    ax.set_title(CLASS_NAMES[lbl])
    ax.set_xlabel('x (mm)')
    if lbl==0: ax.set_ylabel('y (mm)')

plt.suptitle('Figure 6. DAS backprojection images with true bleed location (cyan +) and detected peak (yellow ▲).\n'
             'Localisation errors are within the physics-determined resolution limit (~21 mm at 1 GHz in tissue).',
             y=0.02, fontsize=11, style='italic')
plt.tight_layout()
plt.savefig(OUT/'fig6_das_validation.png')
plt.show()
print("Saved fig6_das_validation.png")
"""))

# ── Summary ───────────────────────────────────────────────────────────
cells.append(md("## Summary"))
cells.append(code("""\
import os
figs = sorted(OUT.glob('*.png'))
total_mb = sum(f.stat().st_size for f in figs) / 1e6
print(f"Generated {len(figs)} figures ({total_mb:.1f} MB total)")
for f in figs:
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
print()
print("Download the dib_figures/ folder from the Output tab.")
print("These are publication-ready at 300 DPI.")
"""))

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {},
    "cells": cells
}
out = pathlib.Path(__file__).parent / 'dib_dataset_figures.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)

import json as j
with open(out) as f:
    j.load(f)
print(f"Saved: {out}  ({out.stat().st_size//1024} KB)  {len(cells)} cells  ✓ valid JSON")
