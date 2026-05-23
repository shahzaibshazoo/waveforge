"""Build waveforge_combined_classifier.ipynb
Combined Part1+Part2 dataset — 4-class + ablation study + binary classification
No Kaggle metadata — import cleanly.
"""
import json, pathlib

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────
cells.append(md("""\
# WaveForge Combined Classifier: Part 1 + Part 2 — Final Results

**Architecture:** PhysioMIMO-Net v1 (Physics-Informed MIMO Radar)
**Input:** Raw scattered MIMO signals (8×8×700) — no DAS pre-processing
**Tasks:**
- 4-class classification (Healthy / Epidural / Subdural / Intracerebral)
- Bleed radius regression (multi-task head)
- Ablation study (3 experiments)
- Binary classification (Healthy vs Any Haemorrhage)

**Datasets:** WaveForge Part 1 + Part 2
**Frequency:** UWB 0.5–1.5 GHz | Grid: 64³ at 3mm/cell | 8-element ring at 90mm radius
"""))

# ── Cell 1: Setup ─────────────────────────────────────────────────────────
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
                              roc_curve, auc, f1_score, accuracy_score)
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
BIN_NAMES   = ['Healthy','Haemorrhage']

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
            'label_bin': 0 if int(s['label']) == 0 else 1,
            'bleed_type':str(s['bleed_type']),
            'bleed_age': str(s['bleed_age']),
            'radius_mm': float(s['bleed_radius_mm']),
            'volume_ml': float(s['bleed_volume_ml']),
            'skull_r':   int(s['phantom_skull_inner_r']),
            'gray_r':    int(s['phantom_gray_r']),
            'scalp_r':   int(s['phantom_scalp_outer_r']),
        }
    except Exception as e:
        return None

print("Scanning /kaggle/input ...")
train_dirs = sorted(set(p.parent for p in Path('/kaggle/input').rglob('*.npz')
                         if p.parent.name == 'train'))
test_dirs  = sorted(set(p.parent for p in Path('/kaggle/input').rglob('*.npz')
                         if p.parent.name in {'test_gpu0','test_gpu1','test'}))

print(f"Train folders found: {len(train_dirs)}")
for d in train_dirs: print(f"  {d}")
print(f"Test folders found:  {len(test_dirs)}")
for d in test_dirs:  print(f"  {d}")

train_recs, test_recs = [], []
for td in train_dirs:
    recs = [r for r in (load_meta(p) for p in sorted(td.glob('*.npz'))) if r]
    train_recs.extend(recs)
    print(f"  train/{td.parent.name}: {len(recs)} samples loaded")

for td in test_dirs:
    recs = [r for r in (load_meta(p) for p in sorted(td.glob('*.npz'))) if r]
    test_recs.extend(recs)
    print(f"  {td.name}/{td.parent.name}: {len(recs)} samples loaded")

df_train = pd.DataFrame(train_recs)
df_test  = pd.DataFrame(test_recs)

print(f"\\nTotal train: {len(df_train)}  |  Total test: {len(df_test)}")
print("\\nClass distribution (train):")
print(df_train['label'].value_counts().sort_index()
      .rename({0:'Healthy',1:'Epidural',2:'Subdural',3:'ICH'}))
"""))

# ── Cell 3: Dataset class ──────────────────────────────────────────────────
cells.append(md("## Cell 3: PyTorch Dataset"))
cells.append(code("""\
class BrainMIMODataset(Dataset):
    def __init__(self, records, augment=False, binary=False):
        self.records = records
        self.augment = augment
        self.binary  = binary

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        s   = np.load(row['path'], allow_pickle=True)
        sig = s['signals_scattered'].astype(np.float32)
        label  = row['label_bin'] if self.binary else row['label']
        radius = row['radius_mm']

        mu, sd = sig.mean(), sig.std() + 1e-12
        sig = (sig - mu) / sd

        if self.augment and np.random.random() < 0.5:
            perm = np.random.permutation(8)
            sig  = sig[perm][:, perm, :]

        return (torch.tensor(sig),
                torch.tensor(label,  dtype=torch.long),
                torch.tensor(radius, dtype=torch.float32))

print("Dataset class ready.")
"""))

# ── Cell 4: Architecture ────────────────────────────────────────────────────
cells.append(md("""\
## Cell 4: PhysioMIMO-Net v1 Architecture

Physics-informed temporal windows:
- **W1 [0–150]:** skull / epidural reflections
- **W2 [50–350]:** subdural zone
- **W3 [200–700]:** deep ICH
"""))
cells.append(code("""\
WINDOWS = [(0, 150), (50, 350), (200, 700)]

class TemporalBranch(nn.Module):
    def __init__(self, win_len, out_dim=128):
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

class PhysicsTemporalEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.branches = nn.ModuleList([TemporalBranch(w1-w0) for w0,w1 in WINDOWS])
        self.proj = nn.Sequential(
            nn.Linear(128*3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
    def forward(self, x):
        B = x.size(0); xf = x.view(B, 64, 700)
        feats = [b(xf[:,:,w0:w1]) for (w0,w1),b in zip(WINDOWS, self.branches)]
        return self.proj(torch.cat(feats, dim=-1))

class MIMOAttention(nn.Module):
    def __init__(self, embed_dim=256, n_heads=4):
        super().__init__()
        self.ant_embed = nn.Linear(700, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.tensor([[math.cos(2*math.pi*i/8),
                           math.sin(2*math.pi*i/8)] for i in range(8)],
                          dtype=torch.float32))
        self.pos_proj  = nn.Linear(2, embed_dim)
        el = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
             dim_feedforward=512, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers=2)
        self.pool = nn.Linear(8, 1)
    def forward(self, x):
        B = x.size(0)
        ae = self.ant_embed(x.mean(dim=2)) + self.pos_proj(self.pos_embed).unsqueeze(0)
        return self.pool(self.transformer(ae).transpose(1,2)).squeeze(-1)

def build_model(n_classes=4, embed_dim=256, dropout=0.3):
    class PhysioMIMONet(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal   = PhysicsTemporalEncoder(embed_dim)
            self.antenna    = MIMOAttention(embed_dim)
            self.fusion     = nn.Sequential(
                nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim),
                nn.GELU(), nn.Dropout(dropout))
            self.classifier = nn.Linear(embed_dim, n_classes)
            self.regressor  = nn.Linear(embed_dim, 1)
        def forward(self, x):
            f = self.fusion(torch.cat([self.temporal(x), self.antenna(x)], dim=-1))
            return self.classifier(f), self.regressor(f).squeeze(-1)
    return PhysioMIMONet()

m = build_model(4).to(DEVICE)
p = sum(x.numel() for x in m.parameters())
print(f"PhysioMIMO-Net v1: {p:,} parameters")
with torch.no_grad():
    dummy = torch.randn(2, 8, 8, 700).to(DEVICE)
    logits, rad = m(dummy)
    print(f"Output shapes: logits {logits.shape}, radius {rad.shape}  OK")
del m
"""))

# ── Cell 5: Training helpers ─────────────────────────────────────────────────
cells.append(md("## Cell 5: Training Helpers"))
cells.append(code("""\
def make_loaders(train_rec, val_rec, test_rec, batch=32, binary=False):
    tr = DataLoader(BrainMIMODataset(train_rec, augment=True,  binary=binary),
                    batch_size=batch, shuffle=True,  num_workers=2, pin_memory=True)
    va = DataLoader(BrainMIMODataset(val_rec,   augment=False, binary=binary),
                    batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    te = DataLoader(BrainMIMODataset(test_rec,  augment=False, binary=binary),
                    batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    return tr, va, te

def train_model(model, train_dl, val_dl, epochs=60, lr=1e-3, reg_w=0.1,
                checkpoint='/kaggle/working/best.pt', patience=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    ce_loss   = nn.CrossEntropyLoss(label_smoothing=0.05)
    mse_loss  = nn.MSELoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))
    history   = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    best_val_acc = 0.0; no_improve = 0

    print(f"{'Ep':>4} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} {'VaAcc':>7}")
    print('-'*42)

    for epoch in range(1, epochs+1):
        for phase, dl, train in [('train',train_dl,True),('val',val_dl,False)]:
            model.train(train)
            tot_loss = tot_n = tot_correct = 0
            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx:
                for sigs, labels, radii in dl:
                    sigs   = sigs.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    radii  = radii.to(DEVICE, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=(DEVICE=='cuda')):
                        logits, pred_r = model(sigs)
                        cls_l  = ce_loss(logits, labels)
                        bmask  = labels > 0
                        reg_l  = mse_loss(pred_r[bmask], radii[bmask]) if bmask.any() else torch.tensor(0., device=DEVICE)
                        loss   = cls_l + reg_w * reg_l
                    if train:
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer); scaler.update()
                    tot_loss    += loss.item() * len(labels)
                    tot_correct += (logits.argmax(1) == labels).sum().item()
                    tot_n       += len(labels)
            l = tot_loss/tot_n; a = tot_correct/tot_n
            history[f'{phase}_loss'].append(l); history[f'{phase}_acc'].append(a)
        scheduler.step()
        va_a = history['val_acc'][-1]; tr_a = history['train_acc'][-1]
        va_l = history['val_loss'][-1]; tr_l = history['train_loss'][-1]
        if va_a > best_val_acc:
            best_val_acc = va_a; torch.save(model.state_dict(), checkpoint)
            no_improve = 0; mk = ' *'
        else:
            no_improve += 1; mk = ''
        print(f"{epoch:>4} {tr_l:>8.4f} {tr_a*100:>6.2f}% {va_l:>8.4f} {va_a*100:>6.2f}%{mk}")
        if no_improve >= patience:
            print(f"Early stop at epoch {epoch}"); break

    print(f"\\nBest val acc: {best_val_acc*100:.2f}%")
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    return history, best_val_acc

def evaluate(model, test_dl, class_names):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for sigs, lbl, _ in test_dl:
            logits, _ = model(sigs.to(DEVICE))
            probs.extend(F.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(lbl.numpy())
    preds  = np.array(preds);  labels = np.array(labels)
    probs  = np.array(probs)
    acc    = (preds == labels).mean()
    f1m    = f1_score(labels, preds, average='macro')
    print(f"Test Accuracy: {acc*100:.2f}%  |  Macro F1: {f1m:.4f}")
    print(classification_report(labels, preds, target_names=class_names))
    return preds, labels, probs, acc, f1m

print("Helpers ready.")
"""))

# ── Cell 6: Build dataloaders ─────────────────────────────────────────────────
cells.append(md("## Cell 6: Build Dataloaders (Combined Dataset)"))
cells.append(code("""\
BATCH = 32
tr_rec, val_rec = train_test_split(df_train.to_dict('records'), test_size=0.2,
                                   random_state=42, stratify=df_train['label'].values)
test_rec = df_test.to_dict('records')

train_dl, val_dl, test_dl = make_loaders(tr_rec, val_rec, test_rec, BATCH, binary=False)
print(f"4-class  | Train: {len(tr_rec)}  Val: {len(val_rec)}  Test: {len(test_rec)}")

train_dl_bin, val_dl_bin, test_dl_bin = make_loaders(tr_rec, val_rec, test_rec, BATCH, binary=True)
print(f"Binary   | same splits — label_bin used")
"""))

# ── Cell 7: Train 4-class ─────────────────────────────────────────────────────
cells.append(md("## Cell 7: Train 4-Class Classifier (Full Dataset)"))
cells.append(code("""\
model_4cls = build_model(n_classes=4).to(DEVICE)
history_4cls, best_val_4cls = train_model(
    model_4cls, train_dl, val_dl,
    epochs=60, lr=1e-3, reg_w=0.1,
    checkpoint='/kaggle/working/model_4cls_combined.pt',
    patience=10
)
"""))

# ── Cell 8: Evaluate 4-class ──────────────────────────────────────────────────
cells.append(md("## Cell 8: Evaluate 4-Class (Test Set)"))
cells.append(code("""\
preds_4, labels_4, probs_4, acc_4, f1_4 = evaluate(model_4cls, test_dl, CLASS_NAMES)

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cm = confusion_matrix(labels_4, preds_4)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], cbar_kws={'label':'Proportion'}, annot_kws={'size':13})
axes[0].set_title(f'4-Class Confusion Matrix — {acc_4*100:.1f}%')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

# Per-class AUC
lb = label_binarize(labels_4, classes=[0,1,2,3])
for lbl in range(4):
    fpr, tpr, _ = roc_curve(lb[:,lbl], probs_4[:,lbl])
    axes[1].plot(fpr, tpr, color=PALETTE[lbl], lw=2,
                 label=f'{CLASS_NAMES[lbl]} AUC={auc(fpr,tpr):.3f}')
axes[1].plot([0,1],[0,1],'k--',lw=1)
axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC Curves — 4-Class')
axes[1].legend(loc='lower right')
plt.tight_layout()
plt.savefig('4cls_results.png')
plt.show()
"""))

# ── Cell 9: Training curves ──────────────────────────────────────────────────
cells.append(md("## Cell 9: 4-Class Training Curves"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ep = range(1, len(history_4cls['train_loss'])+1)
axes[0].plot(ep, history_4cls['train_loss'], label='Train', color='steelblue', lw=2)
axes[0].plot(ep, history_4cls['val_loss'],   label='Val',   color='coral',     lw=2)
axes[0].set(title='Loss', xlabel='Epoch', ylabel='Loss'); axes[0].legend()
axes[1].plot(ep, [a*100 for a in history_4cls['train_acc']], label='Train', color='steelblue', lw=2)
axes[1].plot(ep, [a*100 for a in history_4cls['val_acc']],   label='Val',   color='coral',     lw=2)
axes[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy (%)'); axes[1].legend()
plt.suptitle('4-Class Training History (Part1+Part2)', fontweight='bold')
plt.tight_layout(); plt.savefig('4cls_training.png'); plt.show()
"""))

# ── Cell 10: Ablation study ──────────────────────────────────────────────────
cells.append(md("""\
## Cell 10: Ablation Study

Three experiments:
1. **No temporal split** — single global temporal branch (no physics windows)
2. **No Transformer** — replace MIMOAttention with average pooling
3. **No multi-task** — remove regression head (classification only)
"""))
cells.append(code("""\
# ── Ablation 1: No temporal split ────────────────────────────────────────
class AblationNoSplit(nn.Module):
    \"\"\"Single global temporal CNN — no physics window splitting.\"\"\"
    def __init__(self, n_classes=4, embed_dim=256, dropout=0.3):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1), nn.BatchNorm1d(embed_dim), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.antenna    = MIMOAttention(embed_dim)
        self.fusion     = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.regressor  = nn.Linear(embed_dim, 1)
    def forward(self, x):
        B = x.size(0)
        t  = self.temporal(x.view(B, 64, 700)).squeeze(-1)
        f  = self.fusion(torch.cat([t, self.antenna(x)], dim=-1))
        return self.classifier(f), self.regressor(f).squeeze(-1)

# ── Ablation 2: No Transformer ───────────────────────────────────────────
class AblationNoTransformer(nn.Module):
    \"\"\"Full temporal encoder but antenna branch uses mean pooling only.\"\"\"
    def __init__(self, n_classes=4, embed_dim=256, dropout=0.3):
        super().__init__()
        self.temporal   = PhysicsTemporalEncoder(embed_dim)
        self.ant_proj   = nn.Linear(700, embed_dim)
        self.fusion     = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.regressor  = nn.Linear(embed_dim, 1)
    def forward(self, x):
        t  = self.temporal(x)
        a  = self.ant_proj(x.mean(dim=2)).mean(dim=1)  # simple mean pool
        f  = self.fusion(torch.cat([t, a], dim=-1))
        return self.classifier(f), self.regressor(f).squeeze(-1)

# ── Ablation 3: No multi-task ────────────────────────────────────────────
def build_model_no_multitask(n_classes=4, embed_dim=256, dropout=0.3):
    class PhysioNoReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.temporal   = PhysicsTemporalEncoder(embed_dim)
            self.antenna    = MIMOAttention(embed_dim)
            self.fusion     = nn.Sequential(
                nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim),
                nn.GELU(), nn.Dropout(dropout))
            self.classifier = nn.Linear(embed_dim, n_classes)
            self.regressor  = nn.Linear(embed_dim, 1)  # kept for API compat, not trained
        def forward(self, x):
            f = self.fusion(torch.cat([self.temporal(x), self.antenna(x)], dim=-1))
            return self.classifier(f), self.regressor(f).squeeze(-1)
    return PhysioNoReg()

print("Ablation models defined.")
print("  Ablation 1: No temporal split (single global CNN)")
print("  Ablation 2: No Transformer (mean-pool antenna branch)")
print("  Ablation 3: No multi-task (reg_w=0)")
"""))

# ── Cell 11: Run ablations ────────────────────────────────────────────────────
cells.append(md("## Cell 11: Run Ablation Experiments"))
cells.append(code("""\
ablation_results = {}

# ── Ablation 1: No temporal split ────────────────────────────────────────
print("="*55)
print("ABLATION 1: No Temporal Split")
print("="*55)
m1 = AblationNoSplit(n_classes=4).to(DEVICE)
hist1, _ = train_model(m1, train_dl, val_dl,
    epochs=60, checkpoint='/kaggle/working/ablation1.pt', patience=10)
p1, l1, pr1, acc1, f1_1 = evaluate(m1, test_dl, CLASS_NAMES)
ablation_results['No Temporal Split'] = {'acc': acc1, 'f1': f1_1}

# ── Ablation 2: No Transformer ────────────────────────────────────────────
print("\\n" + "="*55)
print("ABLATION 2: No Transformer")
print("="*55)
m2 = AblationNoTransformer(n_classes=4).to(DEVICE)
hist2, _ = train_model(m2, train_dl, val_dl,
    epochs=60, checkpoint='/kaggle/working/ablation2.pt', patience=10)
p2, l2, pr2, acc2, f1_2 = evaluate(m2, test_dl, CLASS_NAMES)
ablation_results['No Transformer'] = {'acc': acc2, 'f1': f1_2}

# ── Ablation 3: No multi-task ─────────────────────────────────────────────
print("\\n" + "="*55)
print("ABLATION 3: No Multi-Task Regression")
print("="*55)
m3 = build_model_no_multitask(n_classes=4).to(DEVICE)
hist3, _ = train_model(m3, train_dl, val_dl,
    epochs=60, reg_w=0.0,  # zero out regression loss
    checkpoint='/kaggle/working/ablation3.pt', patience=10)
p3, l3, pr3, acc3, f1_3 = evaluate(m3, test_dl, CLASS_NAMES)
ablation_results['No Multi-Task'] = {'acc': acc3, 'f1': f1_3}

print("\\nAll ablations complete.")
"""))

# ── Cell 12: Ablation table ───────────────────────────────────────────────────
cells.append(md("## Cell 12: Ablation Summary Table"))
cells.append(code("""\
ablation_results['Full Model (v1)'] = {'acc': acc_4, 'f1': f1_4}

rows = [
    ('Full Model (v1)',     acc_4,  f1_4,  'Baseline'),
    ('No Temporal Split',   acc1,   f1_1,  '–temporal window split'),
    ('No Transformer',      acc2,   f1_2,  '–MIMO Transformer'),
    ('No Multi-Task',       acc3,   f1_3,  '–radius regression'),
]

print(f"{'Model':30} {'Accuracy':>10} {'Macro F1':>10} {'Delta Acc':>10}")
print('-'*62)
base_acc = acc_4
for name, acc, f1, note in rows:
    delta = (acc - base_acc) * 100
    delta_str = f'({delta:+.1f}pp)' if name != 'Full Model (v1)' else 'baseline'
    print(f"{name:30} {acc*100:>9.2f}% {f1:>10.4f} {delta_str:>10}")

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
names = [r[0].replace(' (v1)','') for r in rows]
accs  = [r[1]*100 for r in rows]
f1s   = [r[2] for r in rows]
colors = ['steelblue' if n.startswith('Full') else '#e67e22' for n in names]

axes[0].bar(names, accs, color=colors, edgecolor='black', lw=0.8, width=0.5)
axes[0].axhline(base_acc*100, color='steelblue', linestyle='--', lw=1.5, alpha=0.6)
axes[0].set_title('Ablation Study — Test Accuracy')
axes[0].set_ylabel('Accuracy (%)'); axes[0].set_ylim(0, 105)
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(accs):
    axes[0].text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)

axes[1].bar(names, f1s, color=colors, edgecolor='black', lw=0.8, width=0.5)
axes[1].axhline(f1_4, color='steelblue', linestyle='--', lw=1.5, alpha=0.6)
axes[1].set_title('Ablation Study — Macro F1')
axes[1].set_ylabel('Macro F1'); axes[1].set_ylim(0, 1.1)
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(f1s):
    axes[1].text(i, v+0.005, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)

plt.suptitle('PhysioMIMO-Net Ablation Study — Each Component Contribution',
             fontweight='bold')
plt.tight_layout(); plt.savefig('ablation_study.png'); plt.show()
"""))

# ── Cell 13: Binary classification ────────────────────────────────────────────
cells.append(md("## Cell 13: Binary Classification (Healthy vs Any Haemorrhage)"))
cells.append(code("""\
print("Training binary classifier...")
model_bin = build_model(n_classes=2).to(DEVICE)
history_bin, best_val_bin = train_model(
    model_bin, train_dl_bin, val_dl_bin,
    epochs=60, lr=1e-3, reg_w=0.05,
    checkpoint='/kaggle/working/model_binary_combined.pt',
    patience=10
)
"""))

# ── Cell 14: Evaluate binary ─────────────────────────────────────────────────
cells.append(md("## Cell 14: Binary Evaluation"))
cells.append(code("""\
preds_b, labels_b, probs_b, acc_b, f1_b = evaluate(model_bin, test_dl_bin, BIN_NAMES)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

cm_b = confusion_matrix(labels_b, preds_b)
cm_bn = cm_b.astype(float) / cm_b.sum(axis=1, keepdims=True)
sns.heatmap(cm_bn, annot=cm_b, fmt='d', cmap='Blues',
            xticklabels=BIN_NAMES, yticklabels=BIN_NAMES,
            ax=axes[0], cbar_kws={'label':'Proportion'}, annot_kws={'size':14})
axes[0].set_title(f'Binary Confusion Matrix — {acc_b*100:.1f}%')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

fpr_b, tpr_b, _ = roc_curve(labels_b, probs_b[:,1])
roc_b = auc(fpr_b, tpr_b)
axes[1].plot(fpr_b, tpr_b, color='steelblue', lw=2, label=f'AUC = {roc_b:.4f}')
axes[1].plot([0,1],[0,1],'k--',lw=1)
axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC Curve — Binary')
axes[1].legend()

plt.tight_layout(); plt.savefig('binary_results_combined.png'); plt.show()
"""))

# ── Cell 15: Final summary ────────────────────────────────────────────────────
cells.append(md("## Cell 15: Final Results Summary"))
cells.append(code("""\
from sklearn.metrics import f1_score as _f1

f1_per = _f1(labels_4, preds_4, average=None)
lb4 = label_binarize(labels_4, classes=[0,1,2,3])
aucs = [auc(*roc_curve(lb4[:,i], probs_4[:,i])[:2]) for i in range(4)]

print("=" * 60)
print("WaveForge Combined Dataset — Final Results")
print("=" * 60)
print()
print("4-CLASS RESULTS (Part1 + Part2)")
print(f"  Test Accuracy:   {acc_4*100:.2f}%")
print(f"  Macro F1:        {f1_4:.4f}")
print(f"  Per-class F1 / AUC:")
for i, name in enumerate(CLASS_NAMES):
    print(f"    {name:18} F1={f1_per[i]:.3f}  AUC={aucs[i]:.3f}")
print()
print("BINARY RESULTS (Healthy vs Any Haemorrhage)")
print(f"  Test Accuracy:   {acc_b*100:.2f}%")
print(f"  F1:              {f1_b:.4f}")
print(f"  AUC:             {auc(*roc_curve(labels_b, probs_b[:,1])[:2]):.4f}")
print()
print("ABLATION STUDY")
print(f"{'Component':30} {'Acc':>8} {'F1':>8} {'Delta':>8}")
print('-'*56)
for name, res in ablation_results.items():
    delta = (res['acc'] - acc_4) * 100
    ds = f'({delta:+.1f}pp)' if name != 'Full Model (v1)' else 'baseline'
    print(f"  {name:28} {res['acc']*100:>7.2f}% {res['f1']:>8.4f} {ds:>8}")
print()
print("COMPARISON TO PUBLISHED SYSTEMS")
print("  Alon 2021   (binary, stochastic, no DAS): >94%")
print("  Hossain 2020 (binary, DAS-VGG):           ~93%")
print(f"  Ours binary (raw MIMO):                  {acc_b*100:.1f}%")
print()
print("  Yin 2021 (4-class, ~76%, unconfirmed)")
print(f"  Ours 4-class (raw MIMO, no DAS):         {acc_4*100:.2f}%")
print("=" * 60)
"""))

# ── Build notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"}
    },
    "cells": cells
}

out = pathlib.Path(__file__).parent / 'waveforge_combined_classifier.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)

import json as _j
with open(out) as f: _j.load(f)
print(f"Saved: {out}  ({out.stat().st_size//1024} KB)  {len(cells)} cells  OK")
