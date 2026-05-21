# WaveForge Brain Haemorrhage Dataset — Requirements Specification

> **Version:** 1.0  
> **Purpose:** Comprehensive requirements for a publication-quality FDTD simulation
> dataset for deep learning detection of intracranial haemorrhage via microwave imaging.

---

## 1. Medical & Clinical Requirements

### 1.1 Haemorrhage Types to Model

| Type | Anatomical Location | Prevalence | Typical Volume | Shape |
|------|---------------------|-----------|----------------|-------|
| Epidural (EDH) | Between skull and dura | 1–3% of TBI | 30–150 mL | Biconvex lens |
| Subdural (SDH) | Between dura and brain surface | 10–20% of TBI | 30–200 mL | Crescent |
| Intracerebral (ICH) | Within brain parenchyma | ~15% of strokes | 10–100 mL | Roughly spherical |
| Subarachnoid (SAH) | In CSF spaces | ~5% of strokes | Diffuse | Irregular |
| Intraventricular (IVH) | Inside ventricles | Secondary to ICH | Variable | Irregular |

**Implementation priority:** EDH, SDH, ICH are the most clinically relevant and most tractable for FDTD modelling. SAH and IVH are diffuse and harder to model — include in v2.

**Minimum detectable bleed:** 5 mL (~radius 10.6 mm sphere) — below this, CT sensitivity drops to <50%.  
**Target detectable bleed:** ≥ 1 mL (~radius 6.2 mm) — clinically significant threshold.

### 1.2 Blood Aging Dielectric Properties (Gabriel 1996 + Literature)

| Stage | Time Window | eps_r @ 1 GHz | sigma @ 1 GHz (S/m) | Physical Reason |
|-------|------------|---------------|----------------------|-----------------|
| Hyperacute | 0–6 h | ~61 | ~1.58 | High water content, oxyhaemoglobin |
| Acute | 6 h – 3 d | ~55 | ~1.30 | Deoxyhaemoglobin, retraction |
| Subacute | 3 d – 3 wk | ~48 | ~1.10 | Methaemoglobin, reabsorption |
| Chronic | > 3 wk | ~44 | ~0.93 | Haemosiderin, low water |

**Key rule:** Never use a single blood permittivity value. Include all 4 stages to enable temporal classification.

### 1.3 Target Clinical Performance

Based on existing CT/MRI detection literature:

| Metric | Minimum Target | Stretch Target |
|--------|---------------|----------------|
| Sensitivity (any bleed) | ≥ 85% | ≥ 93% |
| Specificity | ≥ 90% | ≥ 96% |
| AUC-ROC | ≥ 0.90 | ≥ 0.96 |
| Localisation error | ≤ 10 mm | ≤ 5 mm |
| Bleed volume estimate error | ≤ 30% | ≤ 15% |

These benchmarks are taken from published microwave brain imaging studies (Persson et al. 2014, Fhager et al. 2018).

---

## 2. Electromagnetic Physics Requirements

### 2.1 Optimal Frequency Range

| Frequency | Penetration depth in brain | lambda in GM | Recommended for |
|-----------|--------------------------|--------------|-----------------|
| 500 MHz | ~70 mm (good) | 85 mm | Large deep bleeds |
| 1.0 GHz | ~45 mm (good) | 43 mm | Standard imaging |
| 1.8 GHz | ~22 mm (fair) | 24 mm | Superficial bleeds |
| 2.4 GHz | ~15 mm (poor) | 18 mm | Skull/scalp only |

**Recommended:** 0.5–1.5 GHz. A **broadband Gaussian pulse** spanning 0.5–1.5 GHz in a single simulation is the most efficient approach — one simulation captures the full frequency response.

**Minimum pulse bandwidth:** > 500 MHz to separate tissue boundaries temporally.

### 2.2 Spatial Discretisation (Lambda/10 Rule)

At 1.5 GHz (highest frequency of interest):
- lambda in gray matter: `c / (f * sqrt(eps_r)) = 3e8 / (1.5e9 * sqrt(50)) ≈ 28 mm`
- Lambda/10 = 2.8 mm → **3 mm/cell is the minimum safe cell size**

| Grid | Domain size at 3mm/cell | VRAM (T4) | Run time (T4) |
|------|------------------------|-----------|---------------|
| 48³  | 144×144×144 mm | 180 MB | ~15 s |
| 64³  | 192×192×192 mm | 430 MB | ~40 s |
| 96³  | 288×288×288 mm | 1.4 GB | ~3 min |
| 128³ | 384×384×384 mm | 3.2 GB | ~12 min |

**Recommended grid:** 64³ at 3 mm/cell (192 mm domain, covers adult head with margin).  
**Publication target:** 96³ at 2 mm/cell (192 mm domain, finer resolution) if VRAM allows.

### 2.3 Signal-to-Noise Requirements

Backscattered signal from a 10 mL bleed is approximately 20–40 dB below the direct antenna coupling. Require:

- **Background subtraction mandatory:** subtract healthy phantom reference to isolate scattered signal
- **Minimum SNR after subtraction:** ≥ 20 dB for bleeds ≥ 5 mL
- **Dynamic range of saved signals:** float32 (sufficient, ~150 dB dynamic range)
- **Normalisation:** normalise each TX signal by its peak amplitude before saving

---

## 3. Antenna Array Design Requirements

### 3.1 Array Configuration

| Parameter | Minimum | Recommended | Publication target |
|-----------|---------|-------------|--------------------|
| Number of TX elements | 4 | 8 | 16 |
| Number of RX elements | All (simultaneous) | All | All |
| Array geometry | Ring (axial plane) | Ring | Helmet (3D) |
| Element spacing | Lambda/2 = 21 mm | 22 mm | 15 mm |
| Array radius | 30 mm from skull | 33 mm | 30 mm |
| Antenna–head coupling | Air | Coupling medium | Coupling liquid (eps_r=20) |

**For 64³ grid at 3mm/cell:** 8-element ring in the z=32 plane, radius=30 cells (90 mm from domain centre).

### 3.2 Source Waveform

**Use a modulated Gaussian pulse:**
```
w(t) = sin(2*pi*f_c*t) * exp(-(t-t0)^2 / (2*sigma^2))
f_c = 1.0 GHz  (carrier)
sigma = 1/(2*pi*0.5e9)  (bandwidth ~500 MHz)
```

This provides:
- Centre frequency: 1 GHz
- -10 dB bandwidth: 500 MHz–1.5 GHz
- Causal (zero at t=0)
- Sufficient bandwidth to resolve bleed boundaries

### 3.3 Measurement Protocol

For each sample:
1. Run reference simulation (healthy phantom, same geometry, no bleed)
2. Run target simulation (same phantom + bleed)
3. Record `s_scattered[tx, rx, t] = s_total[tx, rx, t] - s_reference[tx, rx, t]`
4. Compute DAS backprojection image from scattered signals

---

## 4. Dataset Size and Balance Requirements

### 4.1 Minimum Dataset Size

Based on published deep learning medical imaging benchmarks:

| Use case | Minimum samples | Recommended | Notes |
|----------|----------------|-------------|-------|
| Binary (bleed / no bleed) | 500 per class | 2000 per class | ResNet-level performance |
| 4-class (healthy + 3 types) | 500 per class | 1500 per class | Needs more for rare types |
| Total (recommended) | 4000 | 12000 | |

**Practical target for v1:** **2000 total samples** (500 healthy, 500 EDH, 500 SDH, 500 ICH).  
This is achievable on Kaggle 2×T4 in ~8 hours (assuming ~15s/sample at 64³).

### 4.2 Class Distribution

| Class | Label | Fraction | Physiological justification |
|-------|-------|----------|----------------------------|
| Healthy | 0 | 25% | No bleed present |
| Epidural (EDH) | 1 | 25% | Overrepresented vs clinical (3%) to balance |
| Subdural (SDH) | 2 | 25% | Most common TBI bleed |
| Intracerebral (ICH) | 3 | 25% | Most common stroke bleed |

**Blood age sub-distribution** (within bleed classes): 40% acute, 35% subacute, 25% chronic.

### 4.3 Bleed Size Distribution

| Size category | Radius (cells at 3mm/cell) | Radius (mm) | Volume (mL) | Fraction |
|---------------|--------------------------|-------------|-------------|----------|
| Small | 2–4 | 6–12 mm | 1–7 mL | 30% |
| Medium | 4–7 | 12–21 mm | 7–39 mL | 50% |
| Large | 7–10 | 21–30 mm | 39–113 mL | 20% |

### 4.4 Train/Validation/Test Split

| Split | Fraction | Phantom used | Notes |
|-------|----------|-------------|-------|
| Train | 70% | PHANTOM_A | Used for model training |
| Validation | 15% | PHANTOM_A | Hyperparameter tuning |
| Test | 15% | PHANTOM_B | **Never seen during training** |

**PHANTOM_B must have different skull thickness and brain volume** to test generalisation.

### 4.5 Anatomical Variability

To model inter-patient variability, randomly perturb phantom geometry per sample:
- Skull outer radius: ±2 cells (±6 mm)
- Skull thickness: ±1 cell (±3 mm)
- Head shape: uniform scaling ±5%
- Tissue eps_r: ±5% Gaussian noise around nominal value

---

## 5. Output Format per Sample

### 5.1 Saved Data Keys (numpy .npz format)

```python
{
    # Signals: shape (N_tx, N_rx, N_steps), float32
    'signals_total':     np.array,   # total field at RX per TX/step
    'signals_reference': np.array,   # healthy phantom reference
    'signals_scattered': np.array,   # total - reference (the key input to ML)

    # Image: shape (H, W), float32
    'das_image':         np.array,   # DAS backprojection in axial plane (z=centre)

    # Labels
    'label':             int,        # 0=healthy, 1=EDH, 2=SDH, 3=ICH
    'bleed_type':        str,        # 'none'|'epidural'|'subdural'|'intracerebral'
    'bleed_age':         str,        # 'none'|'acute'|'subacute'|'chronic'

    # Bleed geometry (zeros for healthy)
    'bleed_center_cells': np.array,  # (x, y, z) in cells
    'bleed_center_mm':   np.array,   # (x, y, z) in mm from phantom centre
    'bleed_radius_cells': int,
    'bleed_radius_mm':   float,
    'bleed_volume_ml':   float,

    # Simulation parameters
    'freq_hz':           float,      # centre frequency
    'dx_mm':             float,      # cell size in mm
    'grid_shape':        tuple,      # (Nx, Ny, Nz)
    'n_tx':              int,
    'n_rx':              int,
    'n_steps':           int,
    'dt_s':              float,      # time step

    # Tissue properties (at simulation frequency)
    'tissue_eps_r':      dict,       # {tissue_name: eps_r}
    'tissue_sigma':      dict,       # {tissue_name: sigma}

    # Phantom
    'phantom_id':        str,        # 'A' or 'B'
    'phantom_seed':      int,        # random seed for reproducibility
    'phantom_perturbation': dict,    # geometry perturbations applied
}
```

### 5.2 Dataset Manifest File

Save `dataset_manifest.json` at dataset root:
```json
{
    "version": "1.0",
    "waveforge_version": "commit_sha",
    "n_samples": 2000,
    "class_counts": {"0": 500, "1": 500, "2": 500, "3": 500},
    "freq_hz": 1e9,
    "grid_shape": [64, 64, 64],
    "dx_mm": 3.0,
    "n_tx": 8,
    "n_rx": 8,
    "n_steps": 300,
    "train_indices": [...],
    "val_indices": [...],
    "test_indices": [...],
    "phantom_a_indices": [...],
    "phantom_b_indices": [...]
}
```

---

## 6. Competing Datasets and Novelty Claim

### 6.1 Existing Published Datasets

| Dataset | Year | Type | Samples | Frequency | Limitation |
|---------|------|------|---------|-----------|------------|
| Persson et al. 2014 | 2014 | Physical phantom | ~50 | 0.5–2 GHz | Too small, no public release |
| Fhager et al. 2018 | 2018 | FDTD 2D | ~200 | 1 GHz | 2D only, simple geometry |
| Semenov et al. 2019 | 2019 | Physical | ~30 | 0.8–2 GHz | Not public, no aging |
| Coma-Canella et al. 2023 | 2023 | FDTD 3D | ~500 | 0.9 GHz | Single phantom, no aging |

**Gap our dataset fills:**
- 3D full-vector Maxwell solver (vs 2D or scalar approximations)
- Frequency-dependent Cole-Cole tissue models (vs fixed values)
- Blood aging stages (vs single blood type)
- Multiple anatomically diverse phantoms (vs single shape)
- Publicly reproducible via GPU-FDTD (vs proprietary or physical only)
- GPU-accelerated: 100× faster generation than CPU FDTD

### 6.2 Key Papers to Cite

1. **Gabriel et al. (1996)** — tissue dielectric properties (mandatory)
2. **Persson et al. (2014)** — microwave detection of intracranial bleeding, Sci Rep
3. **Fhager et al. (2018)** — microwave imaging for brain haemorrhage detection
4. **Semenov et al. (2019)** — electromagnetic tomography for clinical stroke
5. **Coma-Canella et al. (2023)** — FDTD dataset for microwave brain imaging
6. **Taflove & Hagness (2005)** — FDTD textbook
7. **Oskooi et al. (2010)** — MEEP
8. **Paszke et al. (2019)** — PyTorch

---

## 7. Validation Checklist

Before declaring the dataset publication-ready, verify all of the following:

### Physics Validation
- [ ] eps_r and sigma match Gabriel 1996 values at the target frequency (±5%)
- [ ] DAS images show correct hotspot location for ≥ 90% of bleed samples
- [ ] Scattered signal energy is ≥ 20 dB above numerical noise floor
- [ ] No field divergence across any sample (stability check)
- [ ] Background subtraction achieves ≥ 30 dB isolation

### Anatomy Validation
- [ ] Zero bleeds intersect skull bone (validate via distance check)
- [ ] All bleed centres are within declared zone bounds (±1 cell tolerance)
- [ ] Healthy phantom class has no bleed artefacts

### Dataset Balance
- [ ] Class distribution within 5% of target (25%/25%/25%/25%)
- [ ] Blood age distribution within 5% of target
- [ ] Bleed size distribution within 10% of target

### Generalisation
- [ ] Test set uses PHANTOM_B exclusively
- [ ] Model trained only on PHANTOM_A samples achieves AUC ≥ 0.85 on PHANTOM_B
- [ ] Confusion matrix shows no systematic confusion between EDH and SDH

### Reproducibility
- [ ] Every sample has a stored random seed
- [ ] Dataset can be fully regenerated from seed list + WaveForge codebase
- [ ] Manifest JSON is saved and consistent with actual files

---

## 8. Implementation Plan

### Step 1 — Already done ✅
- `src/datasets/brain/tissue_library.py` — Cole-Cole model, Gabriel 1996
- `src/datasets/brain/phantom.py` — BrainPhantom3D with anatomical zones
- 28 tests pass, 95 total

### Step 2 — Antenna array module
```
src/datasets/brain/antenna.py
  class AntennaRing:
    - __init__(n_elements, ring_radius_cells, z_plane, grid)
    - positions: list of (i, j, k) for each element
    - build_sources(waveform, tx_idx, N_steps) → SourceCollection
    - record_signals(fields_dict, step) → updates internal buffer
    - get_signals() → (N_tx, N_rx, N_steps) array
```

### Step 3 — Dataset generator
```
src/datasets/generator.py
  class BrainDatasetGenerator:
    - __init__(grid, freq_hz, n_tx, n_steps, output_dir)
    - generate_sample(seed, bleed_type, bleed_age, bleed_size)
    - generate_balanced_batch(n_samples, phantom_id='A')
    - save_sample(sample_dict, path)
```

### Step 4 — CLI script
```
datasets/generate_brain_dataset.py
  Usage: python generate_brain_dataset.py \
    --n_samples 2000 --output_dir /kaggle/working/brain_dataset \
    --freq_ghz 1.0 --grid_size 64 --n_tx 8 --n_steps 300
```

### Step 5 — Kaggle notebook
```
notebooks/waveforge_brain_dataset_generator.ipynb
  - Auto-detects 2×T4 GPU, runs phantom A on GPU[0], phantom B on GPU[1]
  - Saves all samples to /kaggle/working/
  - Generates manifest.json
  - Estimated time: ~2000 samples × 30s/sample / 2 GPUs ≈ 8 hours
```

---

## 9. Performance Estimates

| Configuration | Time/sample | 2000 samples | Hardware |
|---------------|-------------|-------------|---------|
| 64³, 1 GPU T4 | ~30 s | ~17 h | Kaggle single T4 |
| 64³, 2 GPU T4 (parallel) | ~15 s | ~8.5 h | Kaggle 2×T4 |
| 48³, 2 GPU T4 (parallel) | ~7 s | ~4 h | Kaggle 2×T4 |
| 96³, 1 GPU A100 | ~45 s | ~25 h | Colab A100 |

**Recommended first run:** 48³ grid, 500 samples, validate quality, then scale up.

---

*Document generated based on: Gabriel et al. 1996, Persson et al. 2014, Fhager et al. 2018,  
Taflove & Hagness 2005, and WaveForge codebase analysis.*
