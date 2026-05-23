"""Build waveforge_v2_classifier.ipynb
PhysioMIMO-Net v2 — three targeted improvements over v1:
  1. Cross-window attention: temporal branches attend to each other
     (EDH vs SDH = energy RATIO between windows, not just each independently)
  2. Frequency branch: FFT magnitude spectrum for spectral discrimination
  3. Deeper MIMO Transformer: 4 layers instead of 2
  4. No multi-task regression: ablation showed +1.1pp without it
"""
import json, pathlib

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(md("""\
# PhysioMIMO-Net v2 — Improved Architecture

**Key changes over v1 (82.05% → target 85-88%):**

| Component | v1 | v2 | Motivation |
|---|---|---|---|
| Temporal branches | 3 CNN branches, concatenated | 3 CNN branches + **cross-window self-attention** | EDH/SDH confusion is an *energy ratio* between windows, not just each window independently |
| Frequency branch | None | **FFT magnitude CNN** | Dura-layer reflections create spectral interference patterns that distinguish EDH from SDH |
| MIMO Transformer | 2 layers | **4 layers** | Deeper antenna interaction modelling |
| Multi-task head | Classification + regression | **Classification only** | Ablation showed regression hurts accuracy by 1.1pp |

**Datasets:** WaveForge Part 1 + Part 2 (combined ~3,200 train / ~748 test)
"""))

# ── Cell 1: Setup ──────────────────────────────────────────────────────────
cells.append(md("## Cell 1: Setup & Imports"))
cells.append(code("""\
import subprocess, sys, os, json, math, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, f1_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white',
    'axes.grid':True,'grid.alpha':0.3,'grid.color':'#cccccc',
    'axes.labelsize':12,'axes.titlesize':13,'font.size':11,
    'figure.dpi':100,'savefig.dpi':300,'savefig.bbox':'tight',
})
PALETTE     = sns.color_palette('Set2', 4)
CLASS_NAMES = ['Healthy','Epidural','Subdural','Intracerebral']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
"""))

# ── Cell 2: Data loading ───────────────────────────────────────────────────
cells.append(md("## Cell 2: Load Part 1 + Part 2 Datasets"))
cells.append(code("""\
def load_meta(p):
    try:
        s = np.load(p, allow_pickle=True)
        return {
            'path':      str(p),
            'label':     int(s['label']),
            'radius_mm': float(s['bleed_radius_mm']),
            'skull_r':   int(s['phantom_skull_inner_r']),
            'gray_r':    int(s['phantom_gray_r']),
            'scalp_r':   int(s['phantom_scalp_outer_r']),
        }
    except: return None

print("Scanning /kaggle/input ...")
train_dirs = sorted(set(p.parent for p in Path('/kaggle/input').rglob('*.npz')
                         if p.parent.name == 'train'))
test_dirs  = sorted(set(p.parent for p in Path('/kaggle/input').rglob('*.npz')
                         if p.parent.name in {'test_gpu0','test_gpu1','test'}))

train_recs, test_recs = [], []
for td in train_dirs:
    recs = [r for r in (load_meta(p) for p in sorted(td.glob('*.npz'))) if r]
    train_recs.extend(recs)
    print(f"  train/{td.parent.name}: {len(recs)}")
for td in test_dirs:
    recs = [r for r in (load_meta(p) for p in sorted(td.glob('*.npz'))) if r]
    test_recs.extend(recs)
    print(f"  test/{td.parent.name}: {len(recs)}")

df_train = pd.DataFrame(train_recs)
df_test  = pd.DataFrame(test_recs)
print(f"\\nTotal train: {len(df_train)}  test: {len(df_test)}")
print(df_train['label'].value_counts().sort_index()
      .rename({0:'Healthy',1:'Epidural',2:'Subdural',3:'ICH'}))
"""))

# ── Cell 3: Dataset + dataloaders ─────────────────────────────────────────
cells.append(md("## Cell 3: Dataset & Dataloaders"))
cells.append(code("""\
class BrainMIMODataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row = self.records[idx]
        s   = np.load(row['path'], allow_pickle=True)
        sig = s['signals_scattered'].astype(np.float32)
        mu, sd = sig.mean(), sig.std() + 1e-12
        sig = (sig - mu) / sd
        if self.augment and np.random.random() < 0.5:
            perm = np.random.permutation(8)
            sig  = sig[perm][:, perm, :]
        return torch.tensor(sig), torch.tensor(row['label'], dtype=torch.long)

BATCH = 32
tr_rec, val_rec = train_test_split(df_train.to_dict('records'), test_size=0.2,
                                   random_state=42, stratify=df_train['label'].values)
test_rec = df_test.to_dict('records')

train_dl = DataLoader(BrainMIMODataset(tr_rec, augment=True),
                      batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(BrainMIMODataset(val_rec),
                      batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(BrainMIMODataset(test_rec),
                      batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train {len(tr_rec)}  Val {len(val_rec)}  Test {len(test_rec)}")
"""))

# ── Cell 4: Architecture ───────────────────────────────────────────────────
cells.append(md("""\
## Cell 4: PhysioMIMO-Net v2 Architecture

```
Input (8, 8, 700)
       ├─ CrossWindowEncoder ─────────────────── (embed_dim)
       │    ├ W1-CNN [0-150]  ─ 128-dim ─┐
       │    ├ W2-CNN [50-350] ─ 128-dim ─┤→ 3-token self-attention → proj
       │    └ W3-CNN [200-700]─ 128-dim ─┘
       │
       ├─ FrequencyEncoder ──────────────────── (embed_dim)
       │    └ rfft → mag → CNN on UWB+scatter spectrum
       │
       └─ MIMOAttentionV2 (4-layer Transformer) (embed_dim)
              └ 8 antenna embeddings + geometric pos encoding

       → Fusion: concat(3 × embed_dim) → 2-layer MLP → embed_dim
       → Classifier: Linear(embed_dim, 4)
```
"""))
cells.append(code("""\
WINDOWS   = [(0, 150), (50, 350), (200, 700)]
EMBED_DIM = 256

class TemporalBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


class CrossWindowEncoder(nn.Module):
    \"\"\"
    3 parallel temporal CNN branches (one per physics window).
    Cross-window self-attention lets branches communicate:
    the model can learn W1/W2 energy ratios needed for EDH vs SDH.
    \"\"\"
    def __init__(self, embed_dim=EMBED_DIM, branch_dim=128, n_heads=4):
        super().__init__()
        self.branches   = nn.ModuleList([TemporalBranch(branch_dim) for _ in WINDOWS])
        # 3 tokens of dim branch_dim — self-attention across windows
        self.cross_attn = nn.MultiheadAttention(branch_dim, n_heads, batch_first=True,
                                                 dropout=0.1)
        self.norm       = nn.LayerNorm(branch_dim)
        self.proj       = nn.Sequential(
            nn.Linear(branch_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, x):
        B  = x.size(0)
        xf = x.view(B, 64, 700)
        # Each branch: (B, branch_dim)
        feats = [b(xf[:, :, w0:w1]) for (w0, w1), b in zip(WINDOWS, self.branches)]
        # Stack as sequence: (B, 3, branch_dim)
        tokens = torch.stack(feats, dim=1)
        # Cross-window self-attention (residual)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        # Flatten and project: (B, embed_dim)
        return self.proj(tokens.flatten(1))


class FrequencyEncoder(nn.Module):
    \"\"\"
    FFT magnitude branch.
    EDH and SDH create different spectral interference patterns at the
    dura boundary — this branch captures spectral fingerprints that the
    time-domain branches miss.
    Uses bins 1-200 (0 to ~50 GHz) — the model learns which bins matter.
    \"\"\"
    def __init__(self, embed_dim=EMBED_DIM, n_bins=200):
        super().__init__()
        self.n_bins = n_bins
        self.conv   = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        B  = x.size(0)
        xf = x.view(B, 64, 700)
        # FFT magnitude: (B, 64, 351)
        fft_mag = torch.abs(torch.fft.rfft(xf, dim=-1))
        # Discard DC (bin 0); take bins 1:n_bins+1
        fft_band = fft_mag[:, :, 1:self.n_bins + 1]          # (B, 64, n_bins)
        return self.conv(fft_band).squeeze(-1)                 # (B, embed_dim)


class MIMOAttentionV2(nn.Module):
    \"\"\"4-layer Transformer over 8 antenna embeddings (double the depth of v1).\"\"\"
    def __init__(self, embed_dim=EMBED_DIM, n_heads=4, n_layers=4):
        super().__init__()
        self.ant_embed = nn.Linear(700, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.tensor([[math.cos(2*math.pi*i/8),
                           math.sin(2*math.pi*i/8)] for i in range(8)],
                          dtype=torch.float32))
        self.pos_proj  = nn.Linear(2, embed_dim)
        el = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 2, dropout=0.1,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers=n_layers)
        self.pool = nn.Linear(8, 1)

    def forward(self, x):
        B  = x.size(0)
        ae = self.ant_embed(x.mean(dim=2)) + self.pos_proj(self.pos_embed).unsqueeze(0)
        return self.pool(self.transformer(ae).transpose(1, 2)).squeeze(-1)


class PhysioMIMONetV2(nn.Module):
    \"\"\"
    PhysioMIMO-Net v2.

    Three input streams:
      1. CrossWindowEncoder  — physics windows + cross-window attention
      2. FrequencyEncoder    — FFT spectral fingerprints
      3. MIMOAttentionV2     — 4-layer antenna Transformer

    No regression head (ablation: classification-only is +1.1pp over multi-task).
    \"\"\"
    def __init__(self, n_classes=4, embed_dim=EMBED_DIM, dropout=0.3):
        super().__init__()
        self.temporal  = CrossWindowEncoder(embed_dim)
        self.frequency = FrequencyEncoder(embed_dim)
        self.antenna   = MIMOAttentionV2(embed_dim)
        # Two-layer fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        t = self.temporal(x)
        f = self.frequency(x)
        a = self.antenna(x)
        fused = self.fusion(torch.cat([t, f, a], dim=-1))
        return self.classifier(fused)


# ── Verify shapes ──────────────────────────────────────────────────────────
model = PhysioMIMONetV2().to(DEVICE)
total  = sum(p.numel() for p in model.parameters())
print(f"PhysioMIMO-Net v2: {total:,} parameters")
print()
print("  v1 had 2.03M parameters")
print(f"  v2 has {total/1e6:.2f}M parameters")
print()
with torch.no_grad():
    dummy = torch.randn(2, 8, 8, 700).to(DEVICE)
    logits = model(dummy)
    print(f"  Output shape: {logits.shape}  OK")
"""))

# ── Cell 5: Training ──────────────────────────────────────────────────────
cells.append(md("## Cell 5: Training"))
cells.append(code("""\
EPOCHS    = 80   # more epochs — 3 streams need more time to converge
LR        = 8e-4
PATIENCE  = 12
CHECKPOINT = '/kaggle/working/physio_v2_best.pt'

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=40, T_mult=1, eta_min=1e-5)
ce_loss  = nn.CrossEntropyLoss(label_smoothing=0.05)
scaler   = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))

history = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
best_val_acc = 0.0; no_improve = 0

print(f"{'Ep':>4} {'TrLoss':>9} {'TrAcc':>7} {'VaLoss':>9} {'VaAcc':>7}")
print('-'*45)

for epoch in range(1, EPOCHS + 1):
    for phase, dl, train in [('train', train_dl, True), ('val', val_dl, False)]:
        model.train(train)
        tot_loss = tot_n = tot_correct = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for sigs, labels in dl:
                sigs   = sigs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
                    logits = model(sigs)
                    loss   = ce_loss(logits, labels)
                if train:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                tot_loss    += loss.item() * len(labels)
                tot_correct += (logits.argmax(1) == labels).sum().item()
                tot_n       += len(labels)
        l = tot_loss / tot_n; a = tot_correct / tot_n
        history[f'{phase}_loss'].append(l); history[f'{phase}_acc'].append(a)

    scheduler.step()
    va_a = history['val_acc'][-1];   tr_a = history['train_acc'][-1]
    va_l = history['val_loss'][-1];  tr_l = history['train_loss'][-1]

    if va_a > best_val_acc:
        best_val_acc = va_a
        torch.save(model.state_dict(), CHECKPOINT)
        no_improve = 0; mk = ' *'
    else:
        no_improve += 1; mk = ''

    print(f"{epoch:>4} {tr_l:>9.4f} {tr_a*100:>6.2f}% {va_l:>9.4f} {va_a*100:>6.2f}%{mk}")
    if no_improve >= PATIENCE:
        print(f"Early stop at epoch {epoch}"); break

print(f"\\nBest val accuracy: {best_val_acc*100:.2f}%")
"""))

# ── Cell 6: Training curves ───────────────────────────────────────────────
cells.append(md("## Cell 6: Training Curves"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ep = range(1, len(history['train_loss']) + 1)
axes[0].plot(ep, history['train_loss'], label='Train', color='steelblue', lw=2)
axes[0].plot(ep, history['val_loss'],   label='Val',   color='coral',     lw=2)
axes[0].set(title='Loss', xlabel='Epoch', ylabel='Loss'); axes[0].legend()
axes[1].plot(ep, [a*100 for a in history['train_acc']], label='Train', color='steelblue', lw=2)
axes[1].plot(ep, [a*100 for a in history['val_acc']],   label='Val',   color='coral',     lw=2)
axes[1].axhline(82.05, color='grey', linestyle='--', lw=1.5, label='v1 baseline (82.05%)')
axes[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy (%)'); axes[1].legend()
plt.suptitle('PhysioMIMO-Net v2 Training History', fontweight='bold')
plt.tight_layout(); plt.savefig('v2_training.png'); plt.show()
"""))

# ── Cell 7: Evaluation ────────────────────────────────────────────────────
cells.append(md("## Cell 7: Test Evaluation"))
cells.append(code("""\
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for sigs, labels in test_dl:
        logits = model(sigs.to(DEVICE))
        all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)
acc = (all_preds == all_labels).mean()

print(f"Test Accuracy: {acc*100:.2f}%  ({(all_preds==all_labels).sum()}/{len(all_labels)})")
print()
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], cbar_kws={'label':'Proportion'}, annot_kws={'size':13})
axes[0].set_title(f'v2 Confusion Matrix — {acc*100:.1f}%')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

# ROC curves
lb = label_binarize(all_labels, classes=[0,1,2,3])
for lbl in range(4):
    fpr, tpr, _ = roc_curve(lb[:,lbl], all_probs[:,lbl])
    axes[1].plot(fpr, tpr, color=PALETTE[lbl], lw=2,
                 label=f'{CLASS_NAMES[lbl]} AUC={auc(fpr,tpr):.3f}')
axes[1].plot([0,1],[0,1],'k--',lw=1)
axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC Curves — v2'); axes[1].legend(loc='lower right')
plt.tight_layout(); plt.savefig('v2_results.png'); plt.show()
"""))

# ── Cell 8: Component analysis ────────────────────────────────────────────
cells.append(md("## Cell 8: What the New Components Learned"))
cells.append(code("""\
# ── Cross-window attention weights ───────────────────────────────────────
# Hook into the cross-window self-attention to see which window pairs interact most
attn_store = {}
def hook_fn(module, inp, out):
    if isinstance(out, tuple) and out[1] is not None:
        attn_store['weights'] = out[1].detach().cpu()
hook = model.temporal.cross_attn.register_forward_hook(hook_fn)

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
window_labels = ['W1 Skull\\n(0-150)', 'W2 Subdural\\n(50-350)', 'W3 ICH\\n(200-700)']

for lbl in range(4):
    sample = df_test[df_test['label'] == lbl].iloc[0]
    s = np.load(sample['path'], allow_pickle=True)
    sig = torch.tensor(s['signals_scattered'].astype(np.float32)).unsqueeze(0).to(DEVICE)
    mu, sd = sig.mean(), sig.std() + 1e-12
    sig = (sig - mu) / sd
    with torch.no_grad():
        _ = model(sig)
    if 'weights' in attn_store:
        w = attn_store['weights'][0].numpy()  # (3, 3) — already head-averaged, do NOT .mean(0)
        sns.heatmap(w, ax=axes[lbl], cmap='YlOrRd', vmin=0, vmax=w.max(),
                    xticklabels=window_labels, yticklabels=window_labels,
                    annot=True, fmt='.2f', annot_kws={'size':8})
        axes[lbl].set_title(CLASS_NAMES[lbl])
        axes[lbl].tick_params(axis='both', labelsize=7)
    else:
        axes[lbl].text(0.5, 0.5, 'No weights\ncaptured',
                       ha='center', va='center', transform=axes[lbl].transAxes)
        axes[lbl].set_title(CLASS_NAMES[lbl])

hook.remove()
plt.suptitle('Cross-Window Attention Weights — how temporal windows interact per class',
             fontweight='bold', y=1.05)
plt.tight_layout(); plt.savefig('v2_crosswindow_attn.png'); plt.show()
print("Key: EDH attention should weight W1 (skull/epidural) heavily;")
print("     SDH should show stronger W2 (subdural zone) attention.")
"""))

# ── Cell 9: v1 vs v2 comparison ───────────────────────────────────────────
cells.append(md("## Cell 9: v1 vs v2 Comparison"))
cells.append(code("""\
# v1 results (Part1+Part2, from combined classifier notebook)
v1 = {
    'acc':   82.05,
    'f1':    0.8213,
    'per_class_f1': [0.997, 0.684, 0.647, 0.957],
    'per_class_auc':[1.000, 0.917, 0.894, 0.997],
}

f1_v2  = f1_score(all_labels, all_preds, average=None)
lb4    = label_binarize(all_labels, classes=[0,1,2,3])
auc_v2 = [auc(*roc_curve(lb4[:,i], all_probs[:,i])[:2]) for i in range(4)]
acc_v2 = acc * 100
f1m_v2 = f1_score(all_labels, all_preds, average='macro')

print("=" * 62)
print(f"{'':20} {'v1':>10} {'v2':>10} {'Delta':>10}")
print("-" * 62)
print(f"{'Test Accuracy':20} {v1['acc']:>9.2f}% {acc_v2:>9.2f}% {acc_v2-v1['acc']:>+9.2f}pp")
print(f"{'Macro F1':20} {v1['f1']:>10.4f} {f1m_v2:>10.4f} {f1m_v2-v1['f1']:>+10.4f}")
print()
print("Per-class F1:")
for i, name in enumerate(CLASS_NAMES):
    d = f1_v2[i] - v1['per_class_f1'][i]
    print(f"  {name:16} {v1['per_class_f1'][i]:>10.3f} {f1_v2[i]:>10.3f} {d:>+10.3f}")
print()
print("Per-class AUC:")
for i, name in enumerate(CLASS_NAMES):
    d = auc_v2[i] - v1['per_class_auc'][i]
    print(f"  {name:16} {v1['per_class_auc'][i]:>10.3f} {auc_v2[i]:>10.3f} {d:>+10.3f}")
print("=" * 62)

# Bar chart comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(4); w = 0.35
axes[0].bar(x-w/2, v1['per_class_f1'], w, label='v1', color='steelblue',  edgecolor='black', lw=0.7)
axes[0].bar(x+w/2, f1_v2,              w, label='v2', color='darkorange', edgecolor='black', lw=0.7)
axes[0].set_xticks(x); axes[0].set_xticklabels(CLASS_NAMES)
axes[0].set_title('Per-Class F1: v1 vs v2'); axes[0].set_ylabel('F1')
axes[0].set_ylim(0, 1.12); axes[0].legend()
for i in range(4):
    axes[0].text(i-w/2, v1['per_class_f1'][i]+0.01, f"{v1['per_class_f1'][i]:.3f}",
                 ha='center', fontsize=8)
    axes[0].text(i+w/2, f1_v2[i]+0.01, f"{f1_v2[i]:.3f}", ha='center', fontsize=8)

axes[1].bar(['v1 (82.05%)','v2'], [v1['acc'], acc_v2],
            color=['steelblue','darkorange'], edgecolor='black', lw=0.8, width=0.4)
axes[1].set_title('Overall Test Accuracy'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(70, 100)
for i, v in enumerate([v1['acc'], acc_v2]):
    axes[1].text(i, v+0.3, f'{v:.2f}%', ha='center', fontweight='bold')

plt.suptitle('PhysioMIMO-Net v1 vs v2 — Full Dataset', fontweight='bold')
plt.tight_layout(); plt.savefig('v1_vs_v2.png'); plt.show()
"""))

# ── Cell 10: Final summary ────────────────────────────────────────────────
cells.append(md("## Cell 10: Final Summary"))
cells.append(code("""\
from sklearn.metrics import f1_score as _f1

f1_per = _f1(all_labels, all_preds, average=None)
f1m    = _f1(all_labels, all_preds, average='macro')
lb4    = label_binarize(all_labels, classes=[0,1,2,3])
aucs   = [auc(*roc_curve(lb4[:,i], all_probs[:,i])[:2]) for i in range(4)]
total_params = sum(p.numel() for p in model.parameters())

print("=" * 60)
print("PhysioMIMO-Net v2 — Final Results")
print("=" * 60)
print(f"  Test Accuracy:    {acc*100:.2f}%")
print(f"  Macro F1:         {f1m:.4f}")
print()
print("  Per-class F1 / AUC:")
for i, name in enumerate(CLASS_NAMES):
    print(f"    {name:18} F1={f1_per[i]:.3f}  AUC={aucs[i]:.3f}")
print()
print(f"  Model parameters: {total_params:,}")
print(f"  Training epochs:  {len(history['train_acc'])}")
print()
print("Architecture changes from v1:")
print("  + Cross-window attention between temporal branches")
print("  + FFT frequency encoder branch")
print("  + 4-layer MIMO Transformer (was 2 layers)")
print("  - Multi-task regression head removed (ablation: +1.1pp)")
print()
print("Comparison to published systems:")
print("  Yin 2021 (4-class, raw MIMO, unconfirmed):  ~76%")
print(f"  Our v1 (Part1+Part2, raw MIMO):            82.05%")
print(f"  Our v2 (Part1+Part2, raw MIMO):            {acc*100:.2f}%")
print("=" * 60)
"""))

# ── Build notebook ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "cells": cells
}

out = pathlib.Path(__file__).parent / 'waveforge_v2_classifier.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)

import json as _j
with open(out) as f: _j.load(f)
print(f"Saved: {out}  ({out.stat().st_size//1024} KB)  {len(cells)} cells  OK")
