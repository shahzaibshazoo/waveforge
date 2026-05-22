"""Build binary_classifier.ipynb — healthy vs any haemorrhage."""
import json, pathlib

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":src}

cells = []

cells.append(md("""# Binary Haemorrhage Detector: Healthy vs Any Haemorrhage
**Architecture:** PhysioMIMO-Net v1 (same as 4-class, output head changed to 2 classes)
**Task:** Binary — 0=Healthy, 1=Any haemorrhage (EDH+SDH+ICH combined)
**Purpose:** Establish upper-bound accuracy; counter "79.7% 4-class looks low" objection
**Dataset:** WaveForge Part 1 — 1,600 train / 374 test (auto-detected)
"""))

# Cell 1: Setup
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
                              roc_curve, auc, accuracy_score)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white',
    'axes.grid':True,'grid.alpha':0.3,'axes.labelsize':12,
    'axes.titlesize':13,'font.size':11,'figure.dpi':100,
    'savefig.dpi':300,'savefig.bbox':'tight',
})
PALETTE = sns.color_palette('Set2', 2)
CLASS_NAMES = ['Healthy', 'Haemorrhage']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""))

# Cell 2: Data loading
cells.append(md("## Cell 2: Load Dataset"))
cells.append(code("""\
# Auto-detect dataset location
print("Searching for dataset...")
all_npz = list(Path('/kaggle/input').rglob('*.npz'))
print(f"Found {len(all_npz)} .npz files")

from collections import Counter
folder_counts = Counter(str(p.parent) for p in all_npz)
for folder, count in sorted(folder_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  {folder}: {count} files")

train_folder = Path(sorted(folder_counts.items(), key=lambda x: -x[1])[0][0])
DATA_ROOT = train_folder.parent
TRAIN_DIR = train_folder
TEST_DIRS = [DATA_ROOT / 'test_gpu0', DATA_ROOT / 'test_gpu1']
print(f"\\nTRAIN: {TRAIN_DIR}")

def load_meta(p):
    try:
        s = np.load(p, allow_pickle=True)
        return {
            'path':       str(p),
            'label_4':    int(s['label']),
            'label_bin':  0 if int(s['label']) == 0 else 1,  # binary
            'bleed_type': str(s['bleed_type']),
            'bleed_age':  str(s['bleed_age']),
            'radius_mm':  float(s['bleed_radius_mm']),
        }
    except: return None

train_recs = [r for r in (load_meta(p) for p in sorted(TRAIN_DIR.glob('*.npz'))) if r]
test_recs  = []
for td in TEST_DIRS:
    if td.exists():
        test_recs += [r for r in (load_meta(p) for p in sorted(td.glob('*.npz'))) if r]

df_train = pd.DataFrame(train_recs)
df_test  = pd.DataFrame(test_recs)
print(f"Train: {len(df_train)}  Test: {len(df_test)}")
print("\\nBinary distribution (train):")
print(df_train['label_bin'].value_counts().rename({0:'Healthy', 1:'Haemorrhage'}))
print("\\n4-class distribution (train):")
print(df_train['label_4'].value_counts().sort_index().rename({0:'Healthy',1:'EDH',2:'SDH',3:'ICH'}))
"""))

# Cell 3: EDA
cells.append(md("## Cell 3: Binary Class Balance"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

counts = df_train['label_bin'].value_counts().sort_index()
axes[0].bar(CLASS_NAMES, counts.values, color=PALETTE, edgecolor='black', lw=0.8, width=0.5)
axes[0].set_title('Binary Class Distribution (Train)')
axes[0].set_ylabel('Samples')
for i, v in enumerate(counts.values):
    axes[0].text(i, v+5, str(v), ha='center', fontweight='bold')

# 4-class breakdown within haemorrhage
h_counts = df_train[df_train['label_4']>0]['label_4'].value_counts().sort_index()
axes[1].bar(['EDH','SDH','ICH'], h_counts.values,
            color=sns.color_palette('Set2',3), edgecolor='black', lw=0.8, width=0.5)
axes[1].set_title('Haemorrhage Subtype Breakdown (Train)')
axes[1].set_ylabel('Samples')
for i, v in enumerate(h_counts.values):
    axes[1].text(i, v+3, str(v), ha='center', fontweight='bold')

plt.suptitle('WaveForge Binary Classification — Data Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig('binary_class_dist.png')
plt.show()
"""))

# Cell 4: Dataset class
cells.append(md("## Cell 4: PyTorch Dataset (Binary Labels)"))
cells.append(code("""\
class BrainMIMOBinaryDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment  = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        s   = np.load(row['path'], allow_pickle=True)
        sig = s['signals_scattered'].astype(np.float32)   # (8,8,700)
        label = row['label_bin']   # 0 = healthy, 1 = any haemorrhage

        mu, sd = sig.mean(), sig.std() + 1e-12
        sig = (sig - mu) / sd

        if self.augment and np.random.random() < 0.5:
            perm = np.random.permutation(8)
            sig = sig[perm][:, perm, :]

        return torch.tensor(sig), torch.tensor(label, dtype=torch.long)


tr_rec, val_rec = train_test_split(df_train.to_dict('records'), test_size=0.2,
                                   random_state=42, stratify=df_train['label_bin'])
test_rec = df_test.to_dict('records')

BATCH = 32
train_dl = DataLoader(BrainMIMOBinaryDataset(tr_rec, augment=True),
                      batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(BrainMIMOBinaryDataset(val_rec), batch_size=BATCH,
                      shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(BrainMIMOBinaryDataset(test_rec), batch_size=BATCH,
                      shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(tr_rec)}  Val: {len(val_rec)}  Test: {len(test_rec)}")
sig, lbl = BrainMIMOBinaryDataset(tr_rec)[0]
print(f"Signal shape: {sig.shape}  Label: {lbl.item()} ({'Healthy' if lbl==0 else 'Haemorrhage'})")
"""))

# Cell 5: Architecture (same as 4-class but 2-output head)
cells.append(md("## Cell 5: PhysioMIMO-Net Binary (same backbone, 2-class head)"))
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
        B = x.size(0); x_flat = x.view(B, 64, 700)
        feats = [b(x_flat[:,:,w0:w1]) for (w0,w1),b in zip(WINDOWS, self.branches)]
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
        ant_emb = self.ant_embed(x.mean(dim=2)) + self.pos_proj(self.pos_embed).unsqueeze(0)
        out = self.transformer(ant_emb)
        return self.pool(out.transpose(1,2)).squeeze(-1)

class PhysioMIMONetBinary(nn.Module):
    \"\"\"Same backbone as 4-class — only the output head changes to 2 classes.\"\"\"
    def __init__(self, embed_dim=256, dropout=0.3):
        super().__init__()
        self.temporal   = PhysicsTemporalEncoder(embed_dim)
        self.antenna    = MIMOAttention(embed_dim)
        self.fusion     = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(embed_dim, 2)   # BINARY: 2 classes

    def forward(self, x):
        fused = self.fusion(torch.cat([self.temporal(x), self.antenna(x)], dim=-1))
        return self.classifier(fused)

model = PhysioMIMONetBinary().to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f"PhysioMIMO-Net Binary: {params:,} parameters")
with torch.no_grad():
    dummy = torch.randn(2, 8, 8, 700).to(DEVICE)
    out = model(dummy)
    print(f"Output shape: {out.shape}  ✓")
"""))

# Cell 6: Training
cells.append(md("## Cell 6: Training"))
cells.append(code("""\
EPOCHS = 60
LR     = 1e-3

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
ce_loss   = nn.CrossEntropyLoss(label_smoothing=0.05)
scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE=='cuda'))

history = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
CHECKPOINT = '/kaggle/working/binary_best.pt'
best_val_acc = 0.0; no_improve = 0; patience = 10

print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
print('-'*55)

for epoch in range(1, EPOCHS+1):
    for phase, dl, train in [('train', train_dl, True), ('val', val_dl, False)]:
        model.train(train)
        tot_loss = tot_correct = tot_n = 0
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
        l = tot_loss/tot_n; a = tot_correct/tot_n
        history[f'{phase}_loss'].append(l); history[f'{phase}_acc'].append(a)

    scheduler.step()
    tr_l, tr_a = history['train_loss'][-1], history['train_acc'][-1]
    va_l, va_a = history['val_loss'][-1],   history['val_acc'][-1]

    if va_a > best_val_acc:
        best_val_acc = va_a
        torch.save(model.state_dict(), CHECKPOINT)
        no_improve = 0; marker = ' *'
    else:
        no_improve += 1; marker = ''

    print(f"{epoch:>6} {tr_l:>11.4f} {tr_a*100:>9.2f}% {va_l:>10.4f} {va_a*100:>8.2f}%{marker}")
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

print(f"\\nBest val accuracy: {best_val_acc*100:.2f}%")
"""))

# Cell 7: Training curves
cells.append(md("## Cell 7: Training Curves"))
cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ep = range(1, len(history['train_loss'])+1)
axes[0].plot(ep, history['train_loss'], label='Train', color='steelblue', lw=2)
axes[0].plot(ep, history['val_loss'],   label='Val',   color='coral',     lw=2)
axes[0].set(title='Loss', xlabel='Epoch', ylabel='Loss'); axes[0].legend()
axes[1].plot(ep, [a*100 for a in history['train_acc']], label='Train', color='steelblue', lw=2)
axes[1].plot(ep, [a*100 for a in history['val_acc']],   label='Val',   color='coral',     lw=2)
axes[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy (%)'); axes[1].legend()
axes[1].set_ylim(50, 102)
plt.suptitle('Binary Classifier Training History', fontweight='bold')
plt.tight_layout(); plt.savefig('binary_training.png'); plt.show()
"""))

# Cell 8: Evaluation
cells.append(md("## Cell 8: Test Evaluation"))
cells.append(code("""\
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for sigs, labels in test_dl:
        logits = model(sigs.to(DEVICE))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds); all_labels.extend(labels.numpy())
        all_probs.extend(probs)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)
acc = (all_preds == all_labels).mean()
print(f"Binary Test Accuracy: {acc*100:.2f}%  ({(all_preds==all_labels).sum()}/{len(all_labels)})")
print()
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], cbar_kws={'label':'Proportion'}, annot_kws={'size':14})
axes[0].set_title(f'Confusion Matrix — {acc*100:.1f}% Accuracy')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.4f}')
axes[1].plot([0,1],[0,1],'k--',lw=1)
axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC Curve — Binary Detection')
axes[1].legend()

plt.tight_layout(); plt.savefig('binary_results.png'); plt.show()
"""))

# Cell 9: Per-subtype analysis
cells.append(md("## Cell 9: Accuracy by Haemorrhage Subtype"))
cells.append(code("""\
# Reload test records with 4-class labels to analyse per-subtype accuracy
test_df_full = df_test.copy().reset_index(drop=True)
test_df_full = test_df_full.iloc[:len(all_preds)].copy()
test_df_full['pred_bin'] = all_preds
test_df_full['correct']  = (all_preds == all_labels)

print("Per-subtype detection rate (all should be close to 100%):")
subtype_map = {0:'Healthy', 1:'Epidural', 2:'Subdural', 3:'Intracerebral'}
for lbl4, name in subtype_map.items():
    subset = test_df_full[test_df_full['label_4'] == lbl4]
    if len(subset) == 0: continue
    correct_pred = subset['label_bin'].values == subset['pred_bin'].values
    rate = correct_pred.mean() * 100
    print(f"  {name:18}  {rate:.1f}%  ({correct_pred.sum()}/{len(subset)})")

fig, ax = plt.subplots(figsize=(8, 4))
rates = []
names = []
for lbl4, name in subtype_map.items():
    subset = test_df_full[test_df_full['label_4']==lbl4]
    if len(subset) == 0: continue
    r = (subset['label_bin'].values == subset['pred_bin'].values).mean() * 100
    rates.append(r); names.append(name)

colors = [PALETTE[0]] + [PALETTE[1]]*3
bars = ax.bar(names, rates, color=colors, edgecolor='black', lw=0.8, width=0.5)
ax.set_ylim(0, 110); ax.set_ylabel('Detection Accuracy (%)')
ax.set_title('Binary Detection Rate per Haemorrhage Subtype')
for bar, v in zip(bars, rates):
    ax.text(bar.get_x()+bar.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig('binary_by_subtype.png'); plt.show()
"""))

# Cell 10: Summary
cells.append(md("## Cell 10: Final Summary"))
cells.append(code("""\
from sklearn.metrics import f1_score

f1_bin  = f1_score(all_labels, all_preds, average='binary')
roc_auc_val = auc(*roc_curve(all_labels, all_probs[:,1])[:2])

print("="*55)
print("PhysioMIMO-Net BINARY — Final Results")
print("="*55)
print(f"  Test Accuracy:    {acc*100:.2f}%")
print(f"  F1 Score:         {f1_bin:.4f}")
print(f"  AUC:              {roc_auc_val:.4f}")
print()
print("Context for paper:")
print(f"  4-class accuracy (same architecture): 79.68%")
print(f"  Binary accuracy  (this notebook):     {acc*100:.2f}%")
print(f"  => Binary is {acc*100 - 79.68:.1f}pp higher — expected because binary is easier")
print()
print("Comparison to published binary microwave systems:")
print("  Alon & Dehkharghani 2021 (stochastic, no DAS): >94%")
print("  Hossain 2020 (DAS-VGG, fixed phantom):          ~93%")
print(f"  Our binary (raw MIMO, varied phantom):          {acc*100:.1f}%")
print("="*55)
"""))

# Build notebook
nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out = pathlib.Path(__file__).parent / 'binary_haemorrhage_classifier.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)

import json as j
with open(out) as f: j.load(f)
print(f"Saved: {out}  ({out.stat().st_size//1024} KB)  {len(cells)} cells  ✓")
