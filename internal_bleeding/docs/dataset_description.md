# WaveForge Brain Haemorrhage FDTD Dataset
### Complete Dataset Description — v1.4

---

## 1. Purpose and Motivation

Intracranial haemorrhage (ICH) is a life-threatening emergency requiring rapid diagnosis.
CT and MRI scanners are the gold standard but are expensive, bulky, and unavailable in
emergency pre-hospital settings. Microwave imaging offers a low-cost, portable alternative:
antennas placed around the head transmit microwave pulses; the dielectric contrast between
blood and normal brain tissue causes measurable backscattering that can be detected and
localised.

Training deep learning classifiers for microwave brain imaging requires large labelled
datasets. Clinical datasets are scarce and ethically restricted. This dataset provides
**2,000 physically accurate 3D FDTD simulations** with ground-truth labels and
full raw signal access.

---

## 2. Dataset Overview

| Property | Value |
|----------|-------|
| Name | WaveForge Brain Haemorrhage FDTD Dataset |
| Version | 1.4 |
| Total samples | 2,000 |
| Training set | 1,600 samples |
| Test set | 400 samples |
| Classes | 4 |
| Class balance | Exactly 25% per class (balanced) |
| Signal format | MIMO radar: (8 TX × 8 RX × 700 time steps) |
| Image format | DAS backprojection: 64×64 pixels |
| File format | NumPy `.npz` (compressed) |
| Approx. file size | ~2 MB per sample (~4 GB total) |

---

## 3. Classes

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Healthy | No haemorrhage — normal brain tissue only |
| 1 | Epidural (EDH) | Bleed between skull inner surface and dura mater |
| 2 | Subdural (SDH) | Bleed between dura and brain surface (CSF space) |
| 3 | Intracerebral (ICH) | Bleed inside gray or white matter parenchyma |

---

## 4. Physical Setup

### 4.1 Head Phantom
A spherical multi-shell anatomical model representing an adult human head:

```
[Air]
  └── Scalp (ε_r=40.0, σ=0.87 S/m)  ← outer surface
       └── Skull cortical bone (ε_r=13.1, σ=0.10 S/m)
            └── Dura mater (ε_r=44.0, σ=0.82 S/m)
                 └── CSF (ε_r=68.0, σ=2.46 S/m)
                      └── Gray matter (ε_r=52.7, σ=0.94 S/m)
                           └── White matter core (ε_r=38.1, σ=0.61 S/m)
```

Dielectric values from Gabriel 1996 (4-pole Cole-Cole model, evaluated at 1.0 GHz).

**Unique geometry per sample:** Each sample uses a randomly drawn head geometry from
the published population distribution:
- Head outer radius: 80–110 mm (mean 94 mm, σ=4 mm)
- Skull thickness: 3–12 mm (mean 7 mm, σ=1.5 mm)
- Brain/skull ratio: 0.80–0.92

This forces the model to learn tissue-contrast physics rather than memorise a fixed geometry.

### 4.2 Haemorrhage Model

| Type | Zone | Radius range | Blood ε_r |
|------|------|-------------|-----------|
| Epidural | skull_inner − dura (3–6 cells = 9–18 mm) | 2–7 cells | 61 (acute) |
| Subdural | dura − CSF (3–6 cells = 9–18 mm) | 2–7 cells | 50.8 (subacute) |
| Intracerebral | inside gray/white matter | 2–12 cells | 43.7 (chronic) |

**Blood aging stages** (3 stages, weighted distribution):
- Acute (40%): ε_r=61.0, σ=1.58 S/m — fresh blood, highest contrast
- Subacute (35%): ε_r=50.8, σ=1.19 S/m — 2–7 days
- Chronic (25%): ε_r=43.7, σ=1.03 S/m — oldest, lowest contrast

**Bleed size distribution:**
- Small (30%): radius 2–4 cells (6–12 mm)
- Medium (50%): radius 4–7 cells (12–21 mm)
- Large (20%): radius 7–12 cells (21–36 mm)

### 4.3 Antenna Array
- 8 monopole antennas uniformly placed on a ring at z=grid_centre
- Ring radius: 90 mm (outside scalp surface at ~82 mm)
- Sequential TX firing: each antenna transmits while all 8 receive
- Full MIMO matrix: 8 TX × 8 RX = 64 signal traces per sample

### 4.4 Source Waveform
Ultra-wideband (UWB) Gaussian monocycle pulse:
- Band: **0.5–1.5 GHz** (1 GHz bandwidth)
- Centre frequency: 1.0 GHz
- Pulse duration: ~1.4 ns (6σ)
- Rationale: Maximum bandwidth safe for 3 mm/cell grid (CSF ε_r=68 limits accuracy to ≤1.2 GHz)

---

## 5. Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Grid | 64 × 64 × 64 cells |
| Cell size (dx) | 3.0 mm |
| Domain size | 192 × 192 × 192 mm |
| Time step (dt) | 5.66 ps (CFL stable) |
| Steps per simulation | 700 (3.96 ns — covers full 2.94 ns round-trip) |
| Total sim per sample | 8 TX × 700 steps × 2 (target + reference) = 11,200 steps |
| Boundary condition | Mur 1st-order ABC (6 faces, 3D) |
| FDTD solver | WaveForge 3D (PyTorch CUDA, full 6-component) |
| GPU throughput | ~567 Mcells/s (T4), ~1,400 Mcells/s (A100) |

---

## 6. Per-Sample File Contents

Each `.npz` file contains:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `signals_total` | (8,8,700) | float32 | Raw MIMO signals (target phantom) |
| `signals_reference` | (8,8,700) | float32 | Raw MIMO signals (healthy reference, same geometry) |
| `signals_scattered` | (8,8,700) | float32 | Differential: total − reference |
| `das_image` | (64,64) | float32 | DAS backprojection power image |
| `label` | scalar | int32 | 0=healthy, 1=EDH, 2=SDH, 3=ICH |
| `bleed_type` | string | — | 'none','epidural','subdural','intracerebral' |
| `bleed_age` | string | — | 'none','acute','subacute','chronic' |
| `bleed_center_cells` | (3,) | int32 | Bleed centre in grid cells (i,j,k) |
| `bleed_center_mm` | (3,) | float32 | Bleed centre in mm |
| `bleed_radius_cells` | scalar | int32 | Bleed radius in cells |
| `bleed_radius_mm` | scalar | float32 | Bleed radius in mm |
| `bleed_volume_ml` | scalar | float32 | Bleed volume in mL |
| `freq_hz` | scalar | float32 | Centre frequency (1.0e9 Hz) |
| `freq_low_hz` | scalar | float32 | Lower band edge (0.5e9 Hz) |
| `freq_high_hz` | scalar | float32 | Upper band edge (1.5e9 Hz) |
| `uwb_mode` | scalar | bool | True (UWB Gaussian monocycle used) |
| `dx_mm` | scalar | float32 | Cell size (3.0 mm) |
| `dt_s` | scalar | float64 | Time step in seconds |
| `n_tx` | scalar | int32 | Number of antennas (8) |
| `n_steps` | scalar | int32 | Time steps (700) |
| `phantom_skull_inner_r` | scalar | int32 | Skull inner radius in cells |
| `phantom_gray_r` | scalar | int32 | Gray matter radius in cells |
| `phantom_scalp_outer_r` | scalar | int32 | Head outer radius in cells |
| `phantom_seed` | scalar | int32 | Seed for exact reproducibility |

---

## 7. Train / Test Split Design

The test set uses a **different random seed space** (offset by 10,000,000) ensuring that
no training phantom geometry is reused in the test set. This guarantees the model is
evaluated on truly unseen anatomy.

```
Training seeds:  0 – 9,999,999   (1,600 samples)
Test seeds:      10,000,000+     (400 samples)
```

---

## 8. How to Load a Sample

```python
import numpy as np

s = np.load('sample_000042.npz', allow_pickle=True)

signals = s['signals_scattered']    # (8, 8, 700) — use this for ML input
das_image = s['das_image']          # (64, 64)    — or use this
label = int(s['label'])             # 0/1/2/3

print(f"Class: {['healthy','EDH','SDH','ICH'][label]}")
print(f"Bleed: {s['bleed_type']} | age: {s['bleed_age']}")
print(f"Radius: {float(s['bleed_radius_mm']):.1f} mm")
print(f"Volume: {float(s['bleed_volume_ml']):.2f} mL")
```

---

## 9. Intended Use

- Binary classification: healthy vs. any haemorrhage
- 4-class classification: healthy / EDH / SDH / ICH
- Regression: bleed localisation from DAS images
- Multi-modal learning: combine raw signals + DAS images
- Transfer learning pre-training for real clinical microwave data

---

## 10. Limitations

1. **2D antenna ring** — antennas placed at z=centre only; full 3D array would improve ICH depth localisation
2. **Spherical phantom** — real heads are not spherical; shape variation is partially addressed by random radii
3. **Single bleed per sample** — multiple simultaneous bleeds not modelled
4. **Dispersive tissue approximated at centre frequency** — full Cole-Cole PLRC not implemented; valid within ±30% of 1.0 GHz
5. **No skull heterogeneity** — real skull has variable thickness and cortical/trabecular structure

---

## 11. References

1. Gabriel, S. et al. (1996). "The dielectric properties of biological tissues: II." *Phys. Med. Biol.* 41:2251.
2. Conceição, R.C. et al. (2016). "Classification of breast tumour malignancy using MAS signals." *Progress in Electromagnetics Research.*
3. Ruan, S. et al. (2012). "Statistical analysis of skull thickness." *J. Biomechanics.*
4. Yee, K.S. (1966). "Numerical solution of initial boundary value problems." *IEEE Trans. Antennas Propag.*
5. Mur, G. (1981). "Absorbing boundary conditions." *IEEE Trans. Electromagn. Compat.*

---

*Generated by WaveForge v1.4 — github.com/shahzaibshazoo/waveforge*
