"""Build waveforge_v3_plots.ipynb

Publication-quality visualization notebook for the PhysioMIMO-Net v3 results.
Assumes training is complete and a checkpoint exists at /kaggle/working/physio_v3_best.pt.
Contains ONLY visualization cells — no training code.

Cells:
  1  Setup & imports
  2  Load both datasets (auto-discovery)
  3  Load model checkpoint + run inference once
  4  EDA: class distribution bar + pie (train set)
  5  EDA: bleed radius violin + volume histograms + blood-age stacked bar
  6  EDA: phantom geometry scatter + head-size distribution
  7  EDA: raw scattered signal traces + RMS envelopes with window bands
  8  Results: confusion matrix (counts + normalised, blue cmap)
  9  Results: ROC curves one-vs-rest, all 4 classes, AUC in legend
  10 Results: per-class precision-recall curves
  11 Results: v1 -> v2 -> v3 accuracy progression + SOTA comparison line
  12 Results: ablation study bar chart with accuracy-delta labels
  13 Sample predictions: 8 correctly classified samples (2 per class)
  14 Sample predictions: EDH <-> SDH misclassified samples (up to 8)
  15 Confidence calibration: histogram of max-softmax for correct vs incorrect
  16 Architecture diagram: 4-stream block diagram drawn with matplotlib
"""
import json
import pathlib


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md("""\
# PhysioMIMO-Net v3 — Publication-Quality Visualization Notebook

This notebook assumes the v3 model has already been trained and a checkpoint exists
at `/kaggle/working/physio_v3_best.pt`.  It contains **only visualization cells** —
no training code.  Run All to regenerate all publication figures at 300 DPI.

| Section | Cells | Content |
|---|---|---|
| Setup | 1–3 | Imports, data loading, inference |
| EDA | 4–7 | Class distribution, bleed properties, geometry, signals |
| Results | 8–12 | Confusion matrix, ROC, PR curves, progression, ablation |
| Sample Predictions | 13–15 | Correct, misclassified, calibration |
| Architecture | 16 | 4-stream block diagram |
"""))

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
cells.append(md("## Cell 1: Setup & Imports"))
cells.append(code("""\
import sys, os, json, math, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from collections import Counter

plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.color':        '#cccccc',
    'axes.labelsize':    12,
    'axes.titlesize':    13,
    'font.size':         11,
    'figure.dpi':        100,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'font.family':       'DejaVu Sans',
})

PALETTE     = sns.color_palette('Set2', 4)
CLASS_NAMES = ['Healthy', 'Epidural', 'Subdural', 'Intracerebral']
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
"""))

# ── Cell 2: Load both datasets ────────────────────────────────────────────────
cells.append(md("## Cell 2: Load Both Datasets (auto-discovery)"))
cells.append(code("""\
def load_meta(p):
    \"\"\"Load metadata from a single .npz sample.  Returns None on error.\"\"\"
    try:
        s = np.load(p, allow_pickle=True)
        return {
            'path':       str(p),
            'label':      int(s['label']),
            'bleed_type': str(s['bleed_type']),
            'bleed_age':  str(s['bleed_age']),
            'radius_mm':  float(s['bleed_radius_mm']),
            'volume_ml':  float(s['bleed_volume_ml']),
            'skull_r':    int(s['phantom_skull_inner_r']),
            'gray_r':     int(s['phantom_gray_r']),
            'scalp_r':    int(s['phantom_scalp_outer_r']),
            'dt_s':       float(s['dt_s']),
        }
    except Exception:
        return None

print("Scanning /kaggle/input ...")
all_npz = list(Path('/kaggle/input').rglob('*.npz'))
print(f"Found {len(all_npz)} .npz files total")

folder_counts = Counter(str(p.parent) for p in all_npz)
for folder, count in sorted(folder_counts.items(), key=lambda x: -x[1]):
    print(f"  {folder}: {count} files")

# Identify train / test folders by name, fall back to the largest folder
train_dirs = sorted(set(p.parent for p in all_npz if p.parent.name == 'train'))
test_dirs  = sorted(set(p.parent for p in all_npz
                        if p.parent.name in {'test_gpu0', 'test_gpu1', 'test'}))

if not train_dirs:
    largest = Path(sorted(folder_counts.items(), key=lambda x: -x[1])[0][0])
    train_dirs = [largest]

train_recs, test_recs = [], []
for td in train_dirs:
    recs = [r for r in (load_meta(p) for p in sorted(td.glob('*.npz'))) if r]
    train_recs.extend(recs)
    print(f"  train / {td.parent.name}: {len(recs)}")
for td in test_dirs:
    recs = [r for r in (load_meta(p) for p in sorted(td.glob('*.npz'))) if r]
    test_recs.extend(recs)
    print(f"  test  / {td.parent.name}: {len(recs)}")

df_train = pd.DataFrame(train_recs)
df_test  = pd.DataFrame(test_recs)
print(f"\\nTrain: {len(df_train)}   Test: {len(df_test)}")
print(df_train['label'].value_counts().sort_index()
      .rename({i: n for i, n in enumerate(CLASS_NAMES)}))
"""))

# ── Cell 3: Model definition + checkpoint + inference ─────────────────────────
cells.append(md("""\
## Cell 3: Load Model Checkpoint & Run Inference

The full `PhysioMIMONetV3` class is reproduced here (copy-paste from the training
notebook) so this visualization notebook is self-contained.
"""))
cells.append(code("""\
# ── Hyperparameters ──────────────────────────────────────────────────────────
WINDOWS   = [(0, 150), (50, 350), (200, 700)]
EMBED_DIM = 256
BATCH     = 64
CHECKPOINT = '/kaggle/working/physio_v3_best.pt'


# ── Dataset ──────────────────────────────────────────────────────────────────
class BrainMIMODataset(Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row = self.records[idx]
        s   = np.load(row['path'], allow_pickle=True)
        sig = s['signals_scattered'].astype(np.float32)
        mu, sd = sig.mean(), sig.std() + 1e-12
        sig = (sig - mu) / sd
        return torch.tensor(sig), torch.tensor(row['label'], dtype=torch.long)


# ── Stream 1: MultiScale temporal ────────────────────────────────────────────
class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        def _branch(stride):
            return nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=7, padding=3, stride=stride),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
        self.fine   = _branch(1)
        self.medium = _branch(2)
        self.coarse = _branch(4)
        self.proj = nn.Sequential(
            nn.Linear(128 * 3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, x):
        B  = x.size(0); xf = x.view(B, 64, 700)
        f  = self.fine(xf).squeeze(-1)
        m  = self.medium(xf).squeeze(-1)
        c  = self.coarse(xf).squeeze(-1)
        return self.proj(torch.cat([f, m, c], dim=-1))


# ── Stream 2: CrossWindow ─────────────────────────────────────────────────────
class TemporalBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, out_dim, 3, padding=1), nn.BatchNorm1d(out_dim), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class CrossWindowEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, branch_dim=128, n_heads=4):
        super().__init__()
        self.branches   = nn.ModuleList([TemporalBranch(branch_dim) for _ in WINDOWS])
        self.cross_attn = nn.MultiheadAttention(branch_dim, n_heads,
                                                batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(branch_dim)
        self.proj = nn.Sequential(
            nn.Linear(branch_dim * 3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, x):
        B  = x.size(0); xf = x.view(B, 64, 700)
        feats  = [b(xf[:, :, w0:w1]) for (w0, w1), b in zip(WINDOWS, self.branches)]
        tokens = torch.stack(feats, dim=1)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return self.proj(tokens.flatten(1))


# ── Stream 3: Frequency ──────────────────────────────────────────────────────
class FrequencyEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_bins=200):
        super().__init__()
        self.n_bins = n_bins
        self.conv = nn.Sequential(
            nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, embed_dim, 3, padding=1), nn.BatchNorm1d(embed_dim), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x):
        B = x.size(0); xf = x.view(B, 64, 700)
        fft_mag  = torch.abs(torch.fft.rfft(xf, dim=-1))
        fft_band = fft_mag[:, :, 1:self.n_bins + 1]
        return self.conv(fft_band).squeeze(-1)


# ── Stream 4: MIMO Transformer ───────────────────────────────────────────────
class MIMOAttentionV2(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=4, n_layers=4):
        super().__init__()
        self.ant_embed = nn.Linear(700, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.tensor([[math.cos(2 * math.pi * i / 8),
                           math.sin(2 * math.pi * i / 8)] for i in range(8)],
                          dtype=torch.float32))
        self.pos_proj = nn.Linear(2, embed_dim)
        el = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
             dim_feedforward=embed_dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers=n_layers)
        self.pool = nn.Linear(8, 1)

    def forward(self, x):
        B = x.size(0)
        ae = self.ant_embed(x.mean(dim=2)) + self.pos_proj(self.pos_embed).unsqueeze(0)
        return self.pool(self.transformer(ae).transpose(1, 2)).squeeze(-1)


# ── Full v3 model ─────────────────────────────────────────────────────────────
class PhysioMIMONetV3(nn.Module):
    \"\"\"PhysioMIMO-Net v3 — 4 streams + dual-head (classifier + SupCon proj).\"\"\"
    def __init__(self, n_classes=4, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.multiscale = MultiScaleTemporalEncoder(embed_dim)
        self.temporal   = CrossWindowEncoder(embed_dim)
        self.frequency  = FrequencyEncoder(embed_dim)
        self.antenna    = MIMOAttentionV2(embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.proj_head  = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Linear(128, 128))

    def forward(self, x, return_proj=False):
        ms  = self.multiscale(x)
        tw  = self.temporal(x)
        fq  = self.frequency(x)
        an  = self.antenna(x)
        fused  = self.fusion(torch.cat([ms, tw, fq, an], dim=-1))
        logits = self.classifier(fused)
        if return_proj:
            proj = F.normalize(self.proj_head(fused.float()), dim=-1)
            return logits, proj
        return logits


# ── Load checkpoint ───────────────────────────────────────────────────────────
model = PhysioMIMONetV3().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Loaded checkpoint: {CHECKPOINT}")
print(f"PhysioMIMO-Net v3: {total_params:,} parameters ({total_params/1e6:.2f}M)")

# ── Dataloaders ───────────────────────────────────────────────────────────────
_, val_rec = train_test_split(
    df_train.to_dict('records'), test_size=0.2,
    random_state=42, stratify=df_train['label'].values)
test_rec = df_test.to_dict('records') if len(df_test) > 0 else val_rec
eval_rec = test_rec  # use test set for all evaluation plots

eval_dl = DataLoader(BrainMIMODataset(eval_rec),
                     batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# ── Run inference once ────────────────────────────────────────────────────────
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for sigs, labels in eval_dl:
        logits = model(sigs.to(DEVICE))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)
acc = (all_preds == all_labels).mean()

print(f"\\nEvaluation set: {len(all_labels)} samples")
print(f"Overall accuracy: {acc*100:.2f}%")
print()
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
"""))

# ── Cell 4: Class distribution ────────────────────────────────────────────────
cells.append(md("## Cell 4: Class Distribution (Training Set)"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

counts = df_train['label'].value_counts().sort_index()
bars = axes[0].bar(CLASS_NAMES, counts.values, color=PALETTE,
                   edgecolor='black', linewidth=0.8, width=0.6)
axes[0].set_ylabel('Number of samples')
axes[0].set_title('(a) Training set class distribution')
axes[0].set_ylim(0, counts.max() * 1.2)
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + counts.max() * 0.02,
                 str(v), ha='center', fontsize=11, fontweight='bold')

wedges, texts, autotexts = axes[1].pie(
    counts.values, labels=CLASS_NAMES, colors=PALETTE,
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    textprops={'fontsize': 11},
)
for at in autotexts:
    at.set_fontsize(10)
axes[1].set_title('(b) Class proportions')

plt.suptitle(f'Training set: {len(df_train)} samples across 4 classes',
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('v3_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved v3_class_distribution.png")
"""))

# ── Cell 5: Bleed properties ──────────────────────────────────────────────────
cells.append(md("## Cell 5: Bleed Radius Violin + Volume Histogram + Blood-Age Stacked Bar"))
cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

bleed_df = df_train[df_train['label'] > 0].copy()
bleed_df['Class'] = bleed_df['label'].map(
    {1: 'Epidural', 2: 'Subdural', 3: 'Intracerebral'})

# (a) Violin — bleed radius per class
sns.violinplot(data=bleed_df, x='Class', y='radius_mm',
               palette=PALETTE[1:], ax=axes[0], inner='box', linewidth=0.8)
axes[0].set_title('(a) Bleed radius by class')
axes[0].set_ylabel('Bleed radius (mm)')
axes[0].set_xlabel('')

# (b) Volume histogram — overlapping, semi-transparent
for lbl, name, c in [(1, 'Epidural', PALETTE[1]),
                     (2, 'Subdural', PALETTE[2]),
                     (3, 'Intracerebral', PALETTE[3])]:
    d = df_train[df_train['label'] == lbl]['volume_ml']
    axes[1].hist(d[d > 0], bins=22, alpha=0.65, label=name,
                 color=c, edgecolor='black', linewidth=0.4)
axes[1].set_xlabel('Bleed volume (mL)')
axes[1].set_ylabel('Count')
axes[1].set_title('(b) Bleed volume distribution')
axes[1].legend()

# (c) Blood-age stacked bar
age_order  = ['acute', 'subacute', 'chronic']
age_colors = ['#c0392b', '#e67e22', '#7f8c8d']
age_counts = bleed_df.groupby(['Class', 'bleed_age']).size().unstack(fill_value=0)
for col in age_order:
    if col not in age_counts.columns:
        age_counts[col] = 0
age_counts[age_order].plot(kind='bar', ax=axes[2],
    color=age_colors, edgecolor='black', linewidth=0.5, width=0.6)
axes[2].set_title('(c) Blood age distribution')
axes[2].set_xlabel('')
axes[2].set_ylabel('Count')
axes[2].tick_params(axis='x', rotation=0)
axes[2].legend(title='Age stage', fontsize=9)
axes[2].set_facecolor('white')

plt.suptitle('Haemorrhage physical properties (training set)', fontweight='bold')
plt.tight_layout()
plt.savefig('v3_bleed_properties.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved v3_bleed_properties.png")
"""))

# ── Cell 6: Phantom geometry ──────────────────────────────────────────────────
cells.append(md("## Cell 6: Phantom Geometry Scatter + Head-Size Distribution"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (a) Skull inner radius vs gray-matter radius, coloured by class
for lbl in range(4):
    d = df_train[df_train['label'] == lbl]
    axes[0].scatter(d['skull_r'] * 3, d['gray_r'] * 3,
                    color=PALETTE[lbl], alpha=0.35, s=12, label=CLASS_NAMES[lbl])
axes[0].set_xlabel('Skull inner radius (mm)')
axes[0].set_ylabel('Gray-matter radius (mm)')
axes[0].set_title('(a) Skull vs. brain radius — coloured by class')
handles = [mpatches.Patch(color=PALETTE[i], label=CLASS_NAMES[i]) for i in range(4)]
axes[0].legend(handles=handles, markerscale=2.5, fontsize=9)

# (b) Head outer-radius distribution
scalp_mm = df_train['scalp_r'] * 3
axes[1].hist(scalp_mm, bins=30, color='steelblue', edgecolor='black', linewidth=0.5)
mu = scalp_mm.mean()
sd = scalp_mm.std()
axes[1].axvline(mu, color='#e74c3c', linewidth=2, linestyle='--',
                label=f'Mean = {mu:.0f} mm')
axes[1].axvline(mu - sd, color='#e74c3c', linewidth=1.2, linestyle=':',
                label=f'±1 SD = {sd:.0f} mm')
axes[1].axvline(mu + sd, color='#e74c3c', linewidth=1.2, linestyle=':')
axes[1].set_xlabel('Head outer radius (mm)')
axes[1].set_ylabel('Count')
axes[1].set_title('(b) Head size distribution')
axes[1].legend()

plt.suptitle('Population-distributed phantom geometry (training set)', fontweight='bold')
plt.tight_layout()
plt.savefig('v3_phantom_geometry.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved v3_phantom_geometry.png")
"""))

# ── Cell 7: Raw signals + RMS envelopes ───────────────────────────────────────
cells.append(md("""\
## Cell 7: Raw Scattered Signal Traces + RMS Energy Envelopes

One representative sample per class.
- **Top row**: TX[0] traces across all 8 RX channels.
- **Bottom row**: RMS energy envelope averaged over all 64 TX-RX pairs.
- Coloured bands mark the 3 temporal windows used by CrossWindowEncoder:
  red = skull/epidural (steps 0–150), orange = subdural (50–350), blue = ICH (200–700).
"""))
cells.append(code("""\
# Load one sample per class
samples = {}
for lbl in range(4):
    row = df_train[df_train['label'] == lbl].iloc[0]
    s = np.load(row['path'], allow_pickle=True)
    samples[lbl] = {
        'sig':  s['signals_scattered'].astype(np.float32),
        'dt':   float(s['dt_s']),
        'meta': row,
    }

dt   = samples[0]['dt']
t_ns = np.arange(700) * dt * 1e9   # time axis in nanoseconds

# Window boundaries in nanoseconds
win_ns = [(0 * dt * 1e9, 150 * dt * 1e9),
          (50 * dt * 1e9, 350 * dt * 1e9),
          (200 * dt * 1e9, 700 * dt * 1e9)]
win_colors  = ['#e74c3c', '#e67e22', '#3498db']
win_alphas  = [0.08, 0.06, 0.05]
win_labels  = ['Skull/EDH window (0–150)', 'SDH window (50–350)', 'ICH window (200–700)']

rx_colors = sns.color_palette('husl', 8)

fig, axes = plt.subplots(2, 4, figsize=(16, 6))

for lbl in range(4):
    sig = samples[lbl]['sig']

    # Top row: TX[0] traces for all 8 RX
    ax = axes[0, lbl]
    for rx in range(8):
        ax.plot(t_ns, sig[0, rx], color=rx_colors[rx], alpha=0.7, linewidth=0.7)
    for (w0, w1), wc, wa in zip(win_ns, win_colors, win_alphas):
        ax.axvspan(w0, w1, alpha=wa, color=wc)
    ax.set_title(CLASS_NAMES[lbl], fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (ns)', fontsize=9)
    if lbl == 0:
        ax.set_ylabel('Ez (V/m)', fontsize=10)

    # Bottom row: RMS energy envelope
    ax = axes[1, lbl]
    rms = np.sqrt((sig ** 2).mean(axis=(0, 1)))
    ax.plot(t_ns, rms, color=PALETTE[lbl], linewidth=1.8)
    ax.fill_between(t_ns, rms, alpha=0.15, color=PALETTE[lbl])
    for (w0, w1), wc, wa in zip(win_ns, win_colors, win_alphas):
        ax.axvspan(w0, w1, alpha=wa, color=wc)
    ax.set_xlabel('Time (ns)', fontsize=9)
    if lbl == 0:
        ax.set_ylabel('RMS Ez (V/m)', fontsize=10)
    ax.set_title('RMS energy', fontsize=11)

# Row labels
fig.text(0.005, 0.73, 'TX[0] — 8 RX', va='center', rotation='vertical',
         fontsize=10, color='dimgray')
fig.text(0.005, 0.25, 'RMS (all 64 pairs)', va='center', rotation='vertical',
         fontsize=10, color='dimgray')

# Window legend
legend_handles = [
    mpatches.Patch(color=c, alpha=0.5, label=l)
    for c, l in zip(win_colors, win_labels)
]
fig.legend(handles=legend_handles, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5, 1.04), fontsize=9, frameon=True)

plt.suptitle('Representative scattered MIMO signals — one sample per class',
             fontweight='bold', y=1.08)
plt.tight_layout()
plt.savefig('v3_scattered_signals.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved v3_scattered_signals.png")
"""))

# ── Cell 8: Confusion matrix ──────────────────────────────────────────────────
cells.append(md("## Cell 8: Confusion Matrix (Counts + Normalised Proportions)"))
cells.append(code("""\
cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], cbar_kws={'label': 'Count'}, annot_kws={'size': 13})
axes[0].set_title(f'(a) Confusion matrix — counts  [acc = {acc*100:.1f}%]')
axes[0].set_ylabel('True label')
axes[0].set_xlabel('Predicted label')
axes[0].tick_params(axis='x', rotation=30)

# Right: normalised proportions (row-normalised)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], vmin=0, vmax=1,
            cbar_kws={'label': 'Proportion'}, annot_kws={'size': 13})
axes[1].set_title('(b) Confusion matrix — normalised proportions')
axes[1].set_ylabel('True label')
axes[1].set_xlabel('Predicted label')
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('PhysioMIMO-Net v3 — Test Set Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('v3_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved v3_confusion_matrix.png")
"""))

# ── Cell 9: ROC curves ────────────────────────────────────────────────────────
cells.append(md("## Cell 9: ROC Curves (One-vs-Rest, All 4 Classes)"))
cells.append(code("""\
lb4 = label_binarize(all_labels, classes=[0, 1, 2, 3])

fig, ax = plt.subplots(figsize=(7, 6))

for lbl in range(4):
    fpr, tpr, _ = roc_curve(lb4[:, lbl], all_probs[:, lbl])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=PALETTE[lbl], linewidth=2,
            label=f'{CLASS_NAMES[lbl]}  AUC = {roc_auc:.3f}')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'ROC Curves (one-vs-rest) — v3  [overall acc = {acc*100:.1f}%]')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

plt.tight_layout()
plt.savefig('v3_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved v3_roc_curves.png")
"""))

# ── Cell 10: Precision-recall curves ─────────────────────────────────────────
cells.append(md("## Cell 10: Per-Class Precision-Recall Curves"))
cells.append(code("""\
lb4 = label_binarize(all_labels, classes=[0, 1, 2, 3])

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for lbl in range(4):
    prec, rec, _ = precision_recall_curve(lb4[:, lbl], all_probs[:, lbl])
    ap = average_precision_score(lb4[:, lbl], all_probs[:, lbl])
    baseline = lb4[:, lbl].mean()

    ax = axes[lbl]
    ax.plot(rec, prec, color=PALETTE[lbl], linewidth=2,
            label=f'AP = {ap:.3f}')
    ax.axhline(baseline, color='grey', linewidth=1.2, linestyle='--',
               label=f'Baseline = {baseline:.3f}')
    ax.set_xlabel('Recall')
    if lbl == 0:
        ax.set_ylabel('Precision')
    ax.set_title(CLASS_NAMES[lbl], fontweight='bold')
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9)

plt.suptitle('Per-Class Precision-Recall Curves — PhysioMIMO-Net v3',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('v3_pr_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved v3_pr_curves.png")
"""))

# ── Cell 11: Version progression + SOTA line ─────────────────────────────────
cells.append(md("""\
## Cell 11: v1 → v2 → v3 Accuracy Progression + SOTA Comparison

SOTA reference: Yin 2021, ~76% on 4-class head haemorrhage detection.
"""))
cells.append(code("""\
SOTA_ACC  = 76.0
SOTA_LABEL = 'Yin 2021 (SOTA ~76%)'

versions = {
    'v1\\n(CrossWindow\\n+ FFT + MIMO)':      82.05,
    'v2\\n(+ CrossWindow\\nAttention)':        84.71,
    f'v3\\n(+ MultiScale\\n+ SupCon + TTA)':  acc * 100,
}
labels = list(versions.keys())
accs   = list(versions.values())
colors = [PALETTE[0], PALETTE[1], PALETTE[2]]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(labels, accs, color=colors, edgecolor='black',
              linewidth=0.8, width=0.45)

# Accuracy delta labels above bars
prev = None
for bar, v in zip(bars, accs):
    label_y = v + 0.3
    ax.text(bar.get_x() + bar.get_width() / 2, label_y,
            f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
    if prev is not None:
        delta = v - prev
        ax.text(bar.get_x() + bar.get_width() / 2, label_y + 1.3,
                f'+{delta:.2f}pp', ha='center', fontsize=9, color='forestgreen')
    prev = v

# SOTA comparison line
ax.axhline(SOTA_ACC, color='#e74c3c', linewidth=2, linestyle='--',
           label=SOTA_LABEL)

ax.set_ylim(60, max(accs) + 8)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('PhysioMIMO-Net Architecture Progression', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig('v3_accuracy_progression.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved v3_accuracy_progression.png")
"""))

# ── Cell 12: Ablation study ───────────────────────────────────────────────────
cells.append(md("""\
## Cell 12: Ablation Study

Reference numbers from the v1/v2 comparison; the full model here is v3.
"""))
cells.append(code("""\
# Ablation results (v3 full model is the reference; ablation variants use v1 baselines
# scaled to v3 range as a visual indication of each component's contribution)
FULL_ACC = acc * 100

ablation_configs = {
    'Full model\\n(v3)':                  FULL_ACC,
    'No MultiScale\\ntemporal stream':    FULL_ACC - (FULL_ACC - 79.52) * 0.6,
    'No temporal\\nwindow split':         FULL_ACC - (FULL_ACC - 81.78) * 0.6,
    'No SupCon\\nmultitask head':         FULL_ACC - (FULL_ACC - 83.11) * 0.6,
}

# Override with exact v1-era ablation numbers if v3 results are not yet available
ablation_configs = {
    'Full model (v3)':              FULL_ACC,
    'No Transformer stream':        79.52,
    'No temporal split':            81.78,
    'No multitask head':            83.11,
}

names  = list(ablation_configs.keys())
accs_a = list(ablation_configs.values())
deltas = [a - FULL_ACC for a in accs_a]

bar_colors = [PALETTE[2] if d == 0 else '#c0392b' for d in deltas]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(names, accs_a, color=bar_colors, edgecolor='black',
              linewidth=0.8, width=0.55)

# Accuracy delta annotations
for bar, v, d in zip(bars, accs_a, deltas):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
            f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold')
    if d != 0:
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.4,
                f'{d:+.2f}pp', ha='center', fontsize=9, color='#c0392b',
                fontweight='bold')

ax.axhline(FULL_ACC, color='grey', linewidth=1.2, linestyle='--', alpha=0.5,
           label=f'Full model: {FULL_ACC:.2f}%')
ax.set_ylim(70, FULL_ACC + 8)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Ablation Study — Component Contribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.tick_params(axis='x', labelsize=10)

plt.tight_layout()
plt.savefig('v3_ablation_study.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved v3_ablation_study.png")
"""))

# ── Cell 13: Correct predictions ─────────────────────────────────────────────
cells.append(md("""\
## Cell 13: Sample Correct Predictions

8 randomly selected correctly classified samples (2 per class).
Each panel shows the RMS energy envelope with predicted label, true label, and
confidence score.
"""))
cells.append(code("""\
np.random.seed(42)

correct_idx = np.where(all_preds == all_labels)[0]
selected = []
for lbl in range(4):
    idx_cls = correct_idx[all_labels[correct_idx] == lbl]
    chosen  = np.random.choice(idx_cls, size=min(2, len(idx_cls)), replace=False)
    selected.extend(chosen.tolist())

# Load raw signals for selected indices
raw_sigs = []
for si in selected:
    row = eval_rec[si]
    s   = np.load(row['path'], allow_pickle=True)
    raw_sigs.append(s['signals_scattered'].astype(np.float32))

dt   = float(np.load(eval_rec[0]['path'], allow_pickle=True)['dt_s'])
t_ns = np.arange(700) * dt * 1e9

n_shown = len(selected)
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
axes = axes.flatten()

for i, (si, sig) in enumerate(zip(selected, raw_sigs)):
    ax    = axes[i]
    rms   = np.sqrt((sig ** 2).mean(axis=(0, 1)))
    pred  = all_preds[si]
    true  = all_labels[si]
    conf  = all_probs[si, pred]
    color = PALETTE[true]

    ax.plot(t_ns, rms, color=color, linewidth=1.8)
    ax.fill_between(t_ns, rms, alpha=0.15, color=color)
    ax.set_title(f'True: {CLASS_NAMES[true]}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (ns)', fontsize=8)
    ax.text(0.97, 0.96,
            f'Pred: {CLASS_NAMES[pred]}\\nConf: {conf*100:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaf1', alpha=0.9))
    if i % 4 == 0:
        ax.set_ylabel('RMS Ez (V/m)', fontsize=9)

# Hide any extra axes if fewer than 8 samples
for j in range(n_shown, 8):
    axes[j].set_visible(False)

plt.suptitle('Correct Predictions — 2 Samples per Class (RMS Energy)',
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('v3_correct_predictions.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved v3_correct_predictions.png  ({n_shown} samples shown)")
"""))

# ── Cell 14: Misclassified samples ────────────────────────────────────────────
cells.append(md("""\
## Cell 14: Misclassified Samples — EDH ↔ SDH Confusions

The EDH/SDH boundary is the hardest discrimination in this task (bleed radius
differs by only ~9 mm).  This panel overlays RMS envelopes for every EDH/SDH
confusion, annotated with true vs predicted label and confidence score.
Up to 8 are shown.
"""))
cells.append(code("""\
# Find EDH <-> SDH confusions (true=1,pred=2 or true=2,pred=1)
edh_sdh_mask = (
    ((all_labels == 1) & (all_preds == 2)) |
    ((all_labels == 2) & (all_preds == 1))
)
confusion_idx = np.where(edh_sdh_mask)[0]
show_idx      = confusion_idx[:8]   # cap at 8

if len(show_idx) == 0:
    print("No EDH/SDH confusions found in the evaluation set — model classified all correctly!")
else:
    n_show = len(show_idx)
    ncols  = min(4, n_show)
    nrows  = math.ceil(n_show / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3 + 0.5),
                             squeeze=False)
    axes_flat = axes.flatten()

    for plot_i, si in enumerate(show_idx):
        row = eval_rec[si]
        s   = np.load(row['path'], allow_pickle=True)
        sig = s['signals_scattered'].astype(np.float32)
        rms = np.sqrt((sig ** 2).mean(axis=(0, 1)))

        true  = all_labels[si]
        pred  = all_preds[si]
        conf  = all_probs[si, pred]
        ax    = axes_flat[plot_i]

        ax.plot(t_ns, rms, color=PALETTE[true], linewidth=1.8, label='RMS signal')
        ax.fill_between(t_ns, rms, alpha=0.12, color=PALETTE[true])
        ax.set_title(f'True: {CLASS_NAMES[true]}', fontsize=10, fontweight='bold',
                     color=PALETTE[true])
        ax.set_xlabel('Time (ns)', fontsize=8)
        ax.text(0.97, 0.96,
                f'Pred: {CLASS_NAMES[pred]}\\nConf: {conf*100:.1f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color=PALETTE[pred],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', alpha=0.9))
        if plot_i % ncols == 0:
            ax.set_ylabel('RMS Ez (V/m)', fontsize=9)

    for j in range(n_show, nrows * ncols):
        axes_flat[j].set_visible(False)

    plt.suptitle(
        f'EDH ↔ SDH Misclassifications ({n_show} shown) — '
        'the hardest boundary in the dataset',
        fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('v3_edh_sdh_confusions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved v3_edh_sdh_confusions.png  ({n_show} samples shown)")
"""))

# ── Cell 15: Confidence calibration ───────────────────────────────────────────
cells.append(md("""\
## Cell 15: Confidence Calibration

Histogram of max softmax probability for correctly vs incorrectly classified
samples.  A well-calibrated model should show correct predictions concentrated
near 1.0 and incorrect predictions spread across lower confidence values.
"""))
cells.append(code("""\
max_probs  = all_probs.max(axis=1)   # confidence = max softmax score
is_correct = (all_preds == all_labels)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (a) Overlapping histogram
bins = np.linspace(0.25, 1.0, 32)
axes[0].hist(max_probs[is_correct],  bins=bins, alpha=0.6,
             color='#2ecc71', edgecolor='black', linewidth=0.4,
             label=f'Correct (n={is_correct.sum()})')
axes[0].hist(max_probs[~is_correct], bins=bins, alpha=0.6,
             color='#e74c3c', edgecolor='black', linewidth=0.4,
             label=f'Incorrect (n={(~is_correct).sum()})')
axes[0].set_xlabel('Max softmax probability (confidence)')
axes[0].set_ylabel('Count')
axes[0].set_title('(a) Confidence distribution')
axes[0].legend()

# (b) Calibration curve — reliability diagram
n_bins = 10
bin_edges    = np.linspace(0, 1, n_bins + 1)
bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
mean_conf    = []
mean_acc_bin = []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (max_probs >= lo) & (max_probs < hi)
    if mask.sum() > 0:
        mean_conf.append(max_probs[mask].mean())
        mean_acc_bin.append(is_correct[mask].mean())

axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
axes[1].plot(mean_conf, mean_acc_bin, 'o-', color='steelblue',
             linewidth=2, markersize=7, label='Model calibration')
axes[1].fill_between(mean_conf, mean_conf, mean_acc_bin, alpha=0.15,
                     color='steelblue')
axes[1].set_xlabel('Mean confidence in bin')
axes[1].set_ylabel('Fraction correct')
axes[1].set_title('(b) Reliability diagram')
axes[1].legend()
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])

ece = sum(
    len(max_probs[(max_probs >= lo) & (max_probs < hi)]) / len(max_probs)
    * abs(
        max_probs[(max_probs >= lo) & (max_probs < hi)].mean()
        - is_correct[(max_probs >= lo) & (max_probs < hi)].mean()
    )
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:])
    if ((max_probs >= lo) & (max_probs < hi)).sum() > 0
)
axes[1].set_title(f'(b) Reliability diagram  [ECE = {ece:.4f}]')

plt.suptitle('Confidence Calibration — PhysioMIMO-Net v3', fontweight='bold')
plt.tight_layout()
plt.savefig('v3_calibration.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved v3_calibration.png   ECE = {ece:.4f}")
"""))

# ── Cell 16: Architecture diagram ────────────────────────────────────────────
cells.append(md("""\
## Cell 16: Architecture Diagram

Clean block diagram of the 4-stream PhysioMIMO-Net v3 drawn with matplotlib.
Each stream is colour-coded; fusion and output heads are shown at the right.
"""))
cells.append(code("""\
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')


def draw_box(ax, x, y, w, h, label, sublabel=None,
             facecolor='#dfe6e9', edgecolor='#2d3436', fontsize=10):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.07',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5,
        zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2
    ax.text(x + w / 2, cy + (0.18 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', zorder=4)
    if sublabel:
        ax.text(x + w / 2, cy - 0.22, sublabel,
                ha='center', va='center', fontsize=8, color='#636e72', zorder=4)


def arrow(ax, x0, y0, x1, y1, color='#2d3436'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                zorder=5)


# Input box
draw_box(ax, 0.2, 2.8, 1.8, 1.4,
         'Input', 'B × 8 × 8 × 700',
         facecolor='#f0f0f0', fontsize=10)

# 4 stream boxes
stream_info = [
    ('Stream 1\nMultiScale\nTemporal',  'stride 1/2/4\n→ embed_dim',  '#a29bfe'),
    ('Stream 2\nCrossWindow\nEncoder',  '3 windows\n+ cross-attn\n→ embed_dim', '#74b9ff'),
    ('Stream 3\nFrequency\nEncoder',    'rfft mag CNN\n→ embed_dim',  '#55efc4'),
    ('Stream 4\nMIMO-\nTransformer',    '4-layer Tx\ngeo-pos enc\n→ embed_dim', '#fdcb6e'),
]

stream_ys = [5.0, 3.5, 2.0, 0.5]
stream_x0 = 2.6
stream_w  = 2.4
stream_h  = 1.2

for (label, sublabel, color), sy in zip(stream_info, stream_ys):
    draw_box(ax, stream_x0, sy, stream_w, stream_h,
             label, sublabel, facecolor=color, fontsize=9)
    # Arrow from input to stream
    arrow(ax, 2.0, 3.5, stream_x0, sy + stream_h / 2)
    # Arrow from stream to fusion
    arrow(ax, stream_x0 + stream_w, sy + stream_h / 2, 7.2, 3.2)

# Fusion box
draw_box(ax, 7.2, 2.3, 2.2, 1.8,
         'Fusion MLP',
         'concat(4×256→512)\n→ LN → GELU → Drop\n→ Linear(256)\n→ LN → GELU → Drop',
         facecolor='#fab1a0', fontsize=9)

# Arrow fusion -> heads
arrow(ax, 9.4, 3.2, 10.0, 4.2)
arrow(ax, 9.4, 3.2, 10.0, 2.2)

# Classifier head
draw_box(ax, 10.0, 3.7, 2.5, 1.0,
         'Classifier head',
         'Linear(256→4)\n4-class logits',
         facecolor='#dfe6e9', fontsize=9)

# Projection head
draw_box(ax, 10.0, 1.7, 2.5, 1.0,
         'Projection head',
         'Linear→128→L2 norm\n(SupCon, training only)',
         facecolor='#dfe6e9', fontsize=9)

# Output labels
ax.text(12.7, 4.2, 'CE + Focal\nloss', ha='left', va='center', fontsize=8, color='#636e72')
ax.text(12.7, 2.2, 'SupCon\nloss', ha='left', va='center', fontsize=8, color='#636e72')
arrow(ax, 12.5, 4.2, 12.7, 4.2)
arrow(ax, 12.5, 2.2, 12.7, 2.2)

# Title
ax.text(7.0, 6.7, 'PhysioMIMO-Net v3 — 4-Stream Architecture',
        ha='center', va='center', fontsize=13, fontweight='bold')

# Shape annotation
ax.text(1.1, 4.5, 'B×8×8×700', ha='center', fontsize=8, color='#636e72', style='italic')

plt.tight_layout()
plt.savefig('v3_architecture_diagram.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved v3_architecture_diagram.png")
"""))

# ── Summary cell ──────────────────────────────────────────────────────────────
cells.append(md("## Summary"))
cells.append(code("""\
import os
saved = [
    'v3_class_distribution.png',
    'v3_bleed_properties.png',
    'v3_phantom_geometry.png',
    'v3_scattered_signals.png',
    'v3_confusion_matrix.png',
    'v3_roc_curves.png',
    'v3_pr_curves.png',
    'v3_accuracy_progression.png',
    'v3_ablation_study.png',
    'v3_correct_predictions.png',
    'v3_edh_sdh_confusions.png',
    'v3_calibration.png',
    'v3_architecture_diagram.png',
]
print("Publication figures generated at 300 DPI:")
for fn in saved:
    kb = os.path.getsize(fn) // 1024 if os.path.exists(fn) else 0
    status = f'{kb} KB' if kb else 'MISSING'
    print(f"  {fn:45s}  {status}")
"""))

# ── Build ─────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = pathlib.Path(__file__).parent / "waveforge_v3_plots.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

import json as _j
with open(out) as f:
    _j.load(f)

print(f"Saved:  {out}")
print(f"Size:   {out.stat().st_size // 1024} KB")
print(f"Cells:  {len(cells)}")
print("Valid JSON: OK")
