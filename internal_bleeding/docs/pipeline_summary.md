# WaveForge — Brain Haemorrhage Dataset Pipeline
### Summary for Supervisor Review

---

## What We Are Building

A synthetic FDTD (Finite-Difference Time-Domain) simulation dataset for classifying
intracranial haemorrhages from microwave radar signals. The system simulates microwave
pulses propagating through a 3D anatomical brain phantom, records the scattered signals
from 8 receiver antennas, and generates a labelled dataset ready for deep learning.

---

## How the Pipeline Works

```
[1] Randomised Head Phantom
        ↓
[2] UWB Pulse Injection (8 antennas, one at a time)
        ↓
[3] FDTD Simulation (64³ grid, 700 time steps, GPU-accelerated)
        ↓
[4] Background Subtraction (total − healthy reference)
        ↓
[5] DAS Backprojection → 64×64 image
        ↓
[6] Save .npz file  (signals + DAS image + label + metadata)
```

### Step 1 — Anatomically Realistic Head Phantom
Each sample gets a **unique** randomly-generated head geometry drawn from published
MRI population statistics (Ruan 2012, Lynnerup 2005):
- Head radius: 80–110 mm
- Skull thickness: 3–12 mm
- Brain-to-skull ratio: 0.80–0.92

The phantom has 6 concentric tissue shells with frequency-dependent dielectric
properties from the Gabriel 1996 Cole-Cole model:

| Layer | ε_r at 1 GHz | σ (S/m) |
|-------|-------------|---------|
| Scalp | 40.0 | 0.87 |
| Skull (cortical) | 13.1 | 0.10 |
| Dura mater | 44.0 | 0.82 |
| CSF | 68.0 | 2.46 |
| Gray matter | 52.7 | 0.94 |
| White matter | 38.1 | 0.61 |

Bleeds are placed in anatomically constrained zones:
- **Epidural**: between skull inner surface and dura
- **Subdural**: between dura and CSF/brain surface
- **Intracerebral**: inside gray/white matter

Blood dielectric properties vary with age (acute: ε_r=61, subacute: ε_r=51, chronic: ε_r=44).

### Step 2 — UWB Pulse Injection
An **ultra-wideband (UWB) Gaussian monocycle pulse** is injected at each antenna
sequentially (0.5–1.5 GHz, 1 GHz bandwidth). This matches published brain microwave
imaging standards and is the maximum bandwidth achievable on a 3 mm/cell grid
(Nyquist limit ≤1.2 GHz in CSF, ε_r=68).

### Step 3 — 3D FDTD Simulation
Full-vector Maxwell equations solved on a 64×64×64 Yee grid (192 mm cube):
- Cell size: 3 mm (λ/10 at 1.5 GHz in tissue)
- Time steps: 700 (covers full 3.96 ns round-trip)
- Boundary: Mur first-order absorbing boundary condition (6 faces)
- GPU acceleration: PyTorch CUDA kernels (~567 Mcells/s on T4)
- **8 TX simulations per sample** (one per antenna) → full MIMO signal matrix

### Step 4 — Background Subtraction
Each sample is simulated **twice**: once with the haemorrhage, once without (same
geometry, no bleed). Subtracting the reference isolates the scattered signal caused
only by the bleed, removing direct-path and tissue-interface clutter.

### Step 5 — DAS Backprojection
Delay-and-Sum beamforming reconstructs a 2D image of the scatter source:
- Travel times computed via **analytic ray-sphere intersection** through each tissue
  layer (not free-space — correctly accounts for slower propagation in tissue)
- Hilbert envelope detection removes sign ambiguity
- Output: 64×64 power image with hotspot near true bleed location

### Step 6 — Save
Each sample saved as a compressed `.npz` file (~2 MB) containing:
- `signals_scattered`: (8, 8, 700) float32 MIMO signal matrix
- `das_image`: (64, 64) float32 DAS power image
- `label`: 0/1/2/3 class label
- Full geometry metadata (skull radius, bleed location, blood age, etc.)

---

## Dataset Specification

| Property | Value |
|----------|-------|
| Total samples | 2,000 (1,600 train + 400 test) |
| Classes | 4 (healthy, epidural, subdural, intracerebral) |
| Class balance | 25% each (400 per class in train, 100 in test) |
| Frequency | UWB 0.5–1.5 GHz (centre 1.0 GHz) |
| Grid | 64³ at 3mm/cell (192 mm domain) |
| Antennas | 8-element ring, radius 90 mm |
| Signal shape | (8, 8, 700) per sample |
| DAS image shape | (64, 64) per sample |
| Approx. total size | ~4 GB uncompressed |
| Generation time | ~13 hours on Kaggle T4 ×2 GPU |

---

## Quality Rules

All 5 publication-quality dataset rules are enforced:

1. **Cole-Cole tissue physics** — Gabriel 1996 4-pole model at 1 GHz
2. **Anatomically constrained bleed placement** — bleeds placed only in correct anatomical zones
3. **Blood aging physics** — 3 stages: acute, subacute, chronic (different ε_r)
4. **Balanced class distribution** — exactly 25% per class
5. **Independent test phantoms** — test samples use a separate seed space (offset by 10⁷) ensuring no geometric overlap with training data

---

## Platform

- Simulation engine: **WaveForge** (custom PyTorch FDTD, open source)
- Hardware: Kaggle T4 ×2 GPU (16 GB VRAM each)
- Repository: `github.com/shahzaibshazoo/waveforge`
- Current status: Dataset generation running (committed Kaggle version)
