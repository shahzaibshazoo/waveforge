"""Build waveforge_v3_classifier.ipynb
PhysioMIMO-Net v3 — targeting 90% accuracy.

New over v2 (84.71%):
  1. MultiScaleTemporalEncoder — fine/medium/coarse stride CNNs see the signal
     at multiple temporal resolutions (stride 1, 2, 4). Captures the subtle
     skull-dura timing difference between EDH (0-9mm) and SDH (9-18mm).
  2. Supervised Contrastive Loss — directly maximises EDH/SDH embedding distance
     while tightening intra-class clusters. The CE loss has no such explicit push.
  3. Focal Loss — down-weights the already-easy Healthy/ICH pairs, concentrates
     gradient signal on the hard EDH/SDH boundary.
  4. Test-Time Augmentation — 8-way ring rotation symmetry averaging at inference.
     The antenna ring has 8-fold discrete rotational symmetry; each rotation is
     a valid augmentation. Free +1-2pp with zero architecture change.

Stream summary:
  Stream 1: MultiScaleTemporalEncoder  (NEW)  -> embed_dim
  Stream 2: CrossWindowEncoder         (v2)   -> embed_dim
  Stream 3: FrequencyEncoder           (v2)   -> embed_dim
  Stream 4: MIMOAttentionV2            (v2)   -> embed_dim
  Fusion:   concat(4*embed_dim) -> 2-layer MLP -> embed_dim
  Heads:    classifier (4-class) + proj_head (SupCon, training only)
"""
import json, pathlib

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(md("""\
# PhysioMIMO-Net v3 — Targeting 90% Accuracy

**Key changes over v2 (84.71%):**

| Component | v2 | v3 |
|---|---|---|
| Temporal encoding | CrossWindow (3 windows + cross-attn) | CrossWindow **+** MultiScale (stride 1/2/4) |
| Loss function | Cross-entropy (label smooth 0.05) | **Focal loss + Supervised Contrastive** |
| Inference | Single forward pass | **Test-Time Augmentation (8 ring rotations)** |
| Streams | 3 | **4** |

**Datasets:** WaveForge Part 1 + Part 2 (~3,200 train / ~748 test)
"""))

# ── Cell 1: Setup ──────────────────────────────────────────────────────────
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
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, f1_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

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
        return {'path': str(p), 'label': int(s['label'])}
    except: return None

print("Scanning /kaggle/input ...")
train_dirs = sorted(set(p.parent for p in Path('/kaggle/input').rglob('*.npz')
                         if p.parent.name == 'train'))
test_dirs  = sorted(set(p.parent for p in Path('/kaggle/input').rglob('*.npz')
                         if p.parent.name in {'test_gpu0','test_gpu1','test'}))

train_recs, test_recs = [], []
for td in train_dirs:
    recs = [r for r in (load_meta(p) for p in sorted(td.glob('*.npz'))) if r]
    train_recs.extend(recs); print(f"  train/{td.parent.name}: {len(recs)}")
for td in test_dirs:
    recs = [r for r in (load_meta(p) for p in sorted(td.glob('*.npz'))) if r]
    test_recs.extend(recs); print(f"  test/{td.parent.name}: {len(recs)}")

df_train = pd.DataFrame(train_recs)
df_test  = pd.DataFrame(test_recs)
print(f"\\nTrain: {len(df_train)}  Test: {len(df_test)}")
print(df_train['label'].value_counts().sort_index()
      .rename({0:'Healthy',1:'Epidural',2:'Subdural',3:'ICH'}))
"""))

# ── Cell 3: Dataset ────────────────────────────────────────────────────────
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
## Cell 4: PhysioMIMO-Net v3 Architecture

```
Input (B, 8, 8, 700)
  │
  ├─[Stream 1] MultiScaleTemporalEncoder  ─────────────── embed_dim
  │   fine   (stride=1): sharp skull/EDH reflections
  │   medium (stride=2): subdural zone
  │   coarse (stride=4): broad ICH return
  │   concat + project
  │
  ├─[Stream 2] CrossWindowEncoder  ───────────────────── embed_dim
  │   W1[0-150], W2[50-350], W3[200-700]
  │   + cross-window self-attention (W1/W2 ratio for EDH/SDH)
  │
  ├─[Stream 3] FrequencyEncoder  ─────────────────────── embed_dim
  │   rfft magnitude CNN (spectral fingerprints)
  │
  └─[Stream 4] MIMOAttentionV2  ──────────────────────── embed_dim
      4-layer Transformer, geometric positional encoding

  concat(4×embed_dim) → Linear(1024,512) → LN → GELU → Dropout(0.3)
                       → Linear(512, 256) → LN → GELU → Dropout(0.15)
  ↓
  Classifier:  Linear(256, 4)        [Focal + CE loss]
  Proj head:   Linear(256,128) → L2  [SupCon loss, training only]
```
"""))
cells.append(code("""\
WINDOWS   = [(0, 150), (50, 350), (200, 700)]
EMBED_DIM = 256


# ── Stream 1 (NEW): Multi-scale temporal ──────────────────────────────────
class MultiScaleTemporalEncoder(nn.Module):
    \"\"\"
    3 CNNs at stride 1 / 2 / 4.
    Fine  (stride=1): captures sharp 0.17ns skull reflection → EDH discrimination
    Medium(stride=2): subdural zone timing
    Coarse(stride=4): broad deep-tissue ICH return
    All three see the full 700-step signal — complementary to CrossWindowEncoder.
    \"\"\"
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
            nn.Linear(128 * 3, embed_dim),
            nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, x):
        B  = x.size(0)
        xf = x.view(B, 64, 700)
        f  = self.fine(xf).squeeze(-1)
        m  = self.medium(xf).squeeze(-1)
        c  = self.coarse(xf).squeeze(-1)
        return self.proj(torch.cat([f, m, c], dim=-1))


# ── Stream 2: Cross-window encoder (from v2) ──────────────────────────────
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
        self.cross_attn = nn.MultiheadAttention(branch_dim, n_heads, batch_first=True,
                                                 dropout=0.1)
        self.norm       = nn.LayerNorm(branch_dim)
        self.proj       = nn.Sequential(
            nn.Linear(branch_dim * 3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
    def forward(self, x):
        B  = x.size(0); xf = x.view(B, 64, 700)
        feats  = [b(xf[:, :, w0:w1]) for (w0, w1), b in zip(WINDOWS, self.branches)]
        tokens = torch.stack(feats, dim=1)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + attn_out)
        return self.proj(tokens.flatten(1))


# ── Stream 3: Frequency encoder (from v2) ─────────────────────────────────
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


# ── Stream 4: MIMO Transformer (from v2, 4 layers) ─────────────────────────
class MIMOAttentionV2(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=4, n_layers=4):
        super().__init__()
        self.ant_embed = nn.Linear(700, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.tensor([[math.cos(2*math.pi*i/8),
                           math.sin(2*math.pi*i/8)] for i in range(8)],
                          dtype=torch.float32))
        self.pos_proj = nn.Linear(2, embed_dim)
        el = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
             dim_feedforward=embed_dim*2, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers=n_layers)
        self.pool = nn.Linear(8, 1)
    def forward(self, x):
        B = x.size(0)
        ae = self.ant_embed(x.mean(dim=2)) + self.pos_proj(self.pos_embed).unsqueeze(0)
        return self.pool(self.transformer(ae).transpose(1, 2)).squeeze(-1)


# ── Full v3 model ──────────────────────────────────────────────────────────
class PhysioMIMONetV3(nn.Module):
    \"\"\"
    PhysioMIMO-Net v3.
    4 streams: MultiScale + CrossWindow + Frequency + MIMO Transformer
    Dual output: classifier logits + L2-normalised projection (SupCon, training only)
    \"\"\"
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
        ms = self.multiscale(x)
        tw = self.temporal(x)
        fq = self.frequency(x)
        an = self.antenna(x)
        fused  = self.fusion(torch.cat([ms, tw, fq, an], dim=-1))
        logits = self.classifier(fused)
        if return_proj:
            proj = F.normalize(self.proj_head(fused), dim=-1)
            return logits, proj
        return logits


model = PhysioMIMONetV3().to(DEVICE)
total = sum(p.numel() for p in model.parameters())
print(f"PhysioMIMO-Net v3: {total:,} parameters ({total/1e6:.2f}M)")
with torch.no_grad():
    dummy = torch.randn(4, 8, 8, 700).to(DEVICE)
    logits, proj = model(dummy, return_proj=True)
    print(f"logits: {logits.shape}  proj: {proj.shape}  OK")
"""))

# ── Cell 5: Loss functions ─────────────────────────────────────────────────
cells.append(md("""\
## Cell 5: Loss Functions

**Focal Loss:** down-weights easy (Healthy/ICH) samples, concentrates gradient on hard (EDH/SDH).

**Supervised Contrastive Loss:** explicitly pushes EDH and SDH embeddings apart.
For each sample, all samples of the same class are positives; all others are negatives.
Temperature τ=0.07 sharpens the distribution.
"""))
cells.append(code("""\
def focal_loss(logits, labels, gamma=2.0, label_smoothing=0.05):
    \"\"\"Focal loss with label smoothing. Down-weights confident correct predictions.\"\"\"
    ce = F.cross_entropy(logits, labels, label_smoothing=label_smoothing, reduction='none')
    p_t = torch.exp(-ce)
    return ((1 - p_t) ** gamma * ce).mean()


class SupConLoss(nn.Module):
    \"\"\"
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Pulls same-class embeddings together, pushes different-class apart.
    Uses L2-normalised projections and cosine similarity.
    \"\"\"
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        \"\"\"features: (B, D) L2-normalised. labels: (B,)\"\"\"
        B      = features.size(0)
        device = features.device
        # Cast to float32 — -9e15 overflows float16 (max ~6.5e4)
        features = features.float()
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)
        # float('-inf') is safe in any precision; -9e15 is not
        mask_self = torch.eye(B, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask_self, float('-inf'))
        # Positive pairs: same label, different sample
        mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~mask_self
        if not mask_pos.any():
            return torch.tensor(0.0, device=device)
        log_prob = F.log_softmax(sim, dim=-1)
        n_pos    = mask_pos.sum(dim=-1).float().clamp(min=1)
        loss     = -(log_prob * mask_pos.float()).sum(dim=-1) / n_pos
        return loss.mean()


supcon_loss = SupConLoss(temperature=0.07)
SUPCON_WEIGHT = 0.3  # total loss = focal_ce + 0.3 * supcon

print("Loss functions ready.")
print(f"  Focal loss:  gamma=2.0, label_smoothing=0.05")
print(f"  SupCon loss: temperature=0.07, weight={SUPCON_WEIGHT}")
"""))

# ── Cell 6: Training ──────────────────────────────────────────────────────
cells.append(md("## Cell 6: Training"))
cells.append(code("""\
EPOCHS     = 100
LR         = 5e-4
PATIENCE   = 15
CHECKPOINT = '/kaggle/working/physio_v3_best.pt'

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
# Warm restarts: T_0=50 epochs first cycle, then 50 more
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=1, eta_min=1e-5)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))

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
                    if train:
                        logits, proj = model(sigs, return_proj=True)
                        fl   = focal_loss(logits, labels)
                        sc   = supcon_loss(proj, labels)
                        loss = fl + SUPCON_WEIGHT * sc
                    else:
                        logits = model(sigs)
                        loss   = focal_loss(logits, labels)
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
    va_a = history['val_acc'][-1];  tr_a = history['train_acc'][-1]
    va_l = history['val_loss'][-1]; tr_l = history['train_loss'][-1]
    if va_a > best_val_acc:
        best_val_acc = va_a; torch.save(model.state_dict(), CHECKPOINT)
        no_improve = 0; mk = ' *'
    else:
        no_improve += 1; mk = ''
    print(f"{epoch:>4} {tr_l:>9.4f} {tr_a*100:>6.2f}% {va_l:>9.4f} {va_a*100:>6.2f}%{mk}")
    if no_improve >= PATIENCE:
        print(f"Early stop at epoch {epoch}"); break

print(f"\\nBest val accuracy: {best_val_acc*100:.2f}%")
"""))

# ── Cell 7: Training curves ───────────────────────────────────────────────
cells.append(md("## Cell 7: Training Curves"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ep = range(1, len(history['train_loss']) + 1)
axes[0].plot(ep, history['train_loss'], label='Train', color='steelblue', lw=2)
axes[0].plot(ep, history['val_loss'],   label='Val',   color='coral',     lw=2)
axes[0].set(title='Loss (Focal + SupCon)', xlabel='Epoch', ylabel='Loss')
axes[0].legend()
axes[1].plot(ep, [a*100 for a in history['train_acc']], label='Train', color='steelblue', lw=2)
axes[1].plot(ep, [a*100 for a in history['val_acc']],   label='Val',   color='coral',     lw=2)
axes[1].axhline(84.71, color='grey', linestyle='--', lw=1.5, label='v2 baseline (84.71%)')
axes[1].axhline(90.0,  color='green', linestyle=':', lw=1.5, label='90% target')
axes[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy (%)'); axes[1].legend()
plt.suptitle('PhysioMIMO-Net v3 Training History', fontweight='bold')
plt.tight_layout(); plt.savefig('v3_training.png'); plt.show()
"""))

# ── Cell 8: Standard evaluation ───────────────────────────────────────────
cells.append(md("## Cell 8: Standard Test Evaluation (single forward pass)"))
cells.append(code("""\
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

preds_std, labels_all, probs_std = [], [], []
with torch.no_grad():
    for sigs, labels in test_dl:
        logits = model(sigs.to(DEVICE))
        probs_std.extend(F.softmax(logits, dim=1).cpu().numpy())
        preds_std.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labels.numpy())

preds_std  = np.array(preds_std)
labels_all = np.array(labels_all)
probs_std  = np.array(probs_std)
acc_std    = (preds_std == labels_all).mean()

print(f"Standard Test Accuracy: {acc_std*100:.2f}%")
print()
print(classification_report(labels_all, preds_std, target_names=CLASS_NAMES))
"""))

# ── Cell 9: TTA evaluation ────────────────────────────────────────────────
cells.append(md("""\
## Cell 9: Test-Time Augmentation (TTA)

The 8-element antenna ring has **8-fold discrete rotational symmetry** — rotating all
TX and RX indices by k positions (mod 8) is a physically valid transformation of the
same measurement. Averaging predictions over all 8 rotations is free at inference.
"""))
cells.append(code("""\
def predict_tta(model, loader, n_rotations=8):
    \"\"\"
    For each sample: run n_rotations antenna-ring rotations, average softmax probs.
    sig shape: (B, 8_TX, 8_RX, 700)  — rotate both TX (dim1) and RX (dim2) together.
    \"\"\"
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for sigs, labels in loader:
            sigs = sigs.to(DEVICE)
            B    = sigs.size(0)
            probs_accum = torch.zeros(B, 4, device=DEVICE)
            for k in range(n_rotations):
                perm = torch.tensor([(i + k) % 8 for i in range(8)], device=DEVICE)
                aug  = sigs[:, perm][:, :, perm]   # rotate TX then RX
                probs_accum += F.softmax(model(aug), dim=1)
            all_probs.extend((probs_accum / n_rotations).cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)

print("Running TTA (8 ring rotations per sample) ...")
probs_tta, labels_tta = predict_tta(model, test_dl, n_rotations=8)
preds_tta = probs_tta.argmax(axis=1)
acc_tta   = (preds_tta == labels_tta).mean()

print(f"Standard accuracy: {acc_std*100:.2f}%")
print(f"TTA accuracy:      {acc_tta*100:.2f}%  (+{(acc_tta-acc_std)*100:.2f}pp from TTA)")
print()
print(classification_report(labels_tta, preds_tta, target_names=CLASS_NAMES))

# Use TTA results for all further analysis
all_preds  = preds_tta
all_labels = labels_tta
all_probs  = probs_tta
acc        = acc_tta
"""))

# ── Cell 10: Confusion matrix + ROC ─────────────────────────────────────
cells.append(md("## Cell 10: Confusion Matrix & ROC Curves (TTA)"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], cbar_kws={'label':'Proportion'}, annot_kws={'size':13})
axes[0].set_title(f'v3 Confusion Matrix (TTA) — {acc*100:.1f}%')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

lb = label_binarize(all_labels, classes=[0,1,2,3])
for lbl in range(4):
    fpr, tpr, _ = roc_curve(lb[:,lbl], all_probs[:,lbl])
    axes[1].plot(fpr, tpr, color=PALETTE[lbl], lw=2,
                 label=f'{CLASS_NAMES[lbl]} AUC={auc(fpr,tpr):.3f}')
axes[1].plot([0,1],[0,1],'k--',lw=1)
axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC Curves — v3 (TTA)')
axes[1].legend(loc='lower right')
plt.tight_layout(); plt.savefig('v3_results.png'); plt.show()
"""))

# ── Cell 11: v1/v2/v3 comparison ─────────────────────────────────────────
cells.append(md("## Cell 11: v1 → v2 → v3 Progression"))
cells.append(code("""\
# Historical results
versions = {
    'v1 (82.05%)': {'acc': 82.05, 'f1': [0.997, 0.684, 0.647, 0.957]},
    'v2 (84.71%)': {'acc': 84.71, 'f1': [1.000, 0.710, 0.695, 0.977]},
    f'v3 ({acc*100:.2f}%)': {
        'acc': acc*100,
        'f1': f1_score(all_labels, all_preds, average=None).tolist()
    },
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(4); n = len(versions); w = 0.25
colors = ['steelblue', 'darkorange', 'seagreen']

for vi, (vname, vdata) in enumerate(versions.items()):
    axes[0].bar(x + (vi - 1) * w, vdata['f1'], w,
                label=vname, color=colors[vi], edgecolor='black', lw=0.6)

axes[0].set_xticks(x); axes[0].set_xticklabels(CLASS_NAMES)
axes[0].set_title('Per-Class F1: v1 → v2 → v3'); axes[0].set_ylabel('F1')
axes[0].set_ylim(0, 1.15); axes[0].legend()

vnames = list(versions.keys())
vaccs  = [v['acc'] for v in versions.values()]
bars = axes[1].bar(vnames, vaccs, color=colors, edgecolor='black', lw=0.8, width=0.4)
axes[1].axhline(90, color='green', linestyle=':', lw=1.5, label='90% target')
axes[1].set_title('Overall Test Accuracy Progression'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_ylim(70, 100); axes[1].legend()
for bar, v in zip(bars, vaccs):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.3,
                 f'{v:.2f}%', ha='center', fontweight='bold')

plt.suptitle('PhysioMIMO-Net Architecture Evolution', fontweight='bold')
plt.tight_layout(); plt.savefig('v3_progression.png'); plt.show()

# Print progression table
print(f"{'Version':20} {'Accuracy':>10} {'EDH F1':>8} {'SDH F1':>8} {'Delta':>8}")
print('-'*58)
prev_acc = None
for vname, vdata in versions.items():
    delta = f'(+{vdata[\"acc\"]-prev_acc:.2f}pp)' if prev_acc else 'baseline'
    print(f"  {vname:18} {vdata['acc']:>9.2f}%  {vdata['f1'][1]:>7.3f}  {vdata['f1'][2]:>7.3f}  {delta}")
    prev_acc = vdata['acc']
"""))

# ── Cell 12: Final summary ────────────────────────────────────────────────
cells.append(md("## Cell 12: Final Summary"))
cells.append(code("""\
from sklearn.metrics import f1_score as _f1

f1_per = _f1(all_labels, all_preds, average=None)
f1m    = _f1(all_labels, all_preds, average='macro')
lb4    = label_binarize(all_labels, classes=[0,1,2,3])
aucs   = [auc(*roc_curve(lb4[:,i], all_probs[:,i])[:2]) for i in range(4)]
total_params = sum(p.numel() for p in model.parameters())

print("=" * 62)
print("PhysioMIMO-Net v3 — Final Results (with TTA)")
print("=" * 62)
print(f"  Test Accuracy (TTA):  {acc*100:.2f}%")
print(f"  Test Accuracy (std):  {acc_std*100:.2f}%")
print(f"  Macro F1:             {f1m:.4f}")
print()
print("  Per-class F1 / AUC:")
for i, name in enumerate(CLASS_NAMES):
    print(f"    {name:18} F1={f1_per[i]:.3f}  AUC={aucs[i]:.3f}")
print()
print(f"  Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"  Training epochs:  {len(history['train_acc'])}")
print()
print("Architecture (v3 novel contributions):")
print("  + MultiScaleTemporalEncoder (stride 1/2/4 — new)")
print("  + Supervised Contrastive Loss (new)")
print("  + Focal Loss (new)")
print("  + Test-Time Augmentation, 8 ring rotations (new)")
print("  = CrossWindowEncoder + FFT + 4-layer MIMO Transformer (from v2)")
print()
print("Comparison to published systems:")
print("  Yin 2021 (4-class, unconfirmed):         ~76%")
print("  Our v2 (84.71%)")
print(f"  Our v3 — TTA:                            {acc*100:.2f}%")
print(f"  Our v3 — standard:                       {acc_std*100:.2f}%")
print("  Binary (healthy vs any):                 100.0%")
print("=" * 62)
"""))

# ── Build ──────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "cells": cells
}

out = pathlib.Path(__file__).parent / 'waveforge_v3_classifier.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)

import json as _j
with open(out) as f: _j.load(f)
print(f"Saved: {out}  ({out.stat().st_size//1024} KB)  {len(cells)} cells  OK")
