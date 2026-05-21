# WaveForge — Brain Haemorrhage Dataset Pipeline
### Supervisor Summary — Plain-Language Technical Overview

---

## 1. Background: What Problem Are We Solving?

**Intracranial haemorrhage** means bleeding *inside the skull*. "Intra" = inside,
"cranial" = relating to the skull/brain. It is one of the most dangerous medical
emergencies — without rapid diagnosis and treatment, blood accumulates inside the rigid
skull, compresses the brain, and causes permanent damage or death within minutes to hours.

The current gold standard for diagnosis is a **CT scanner** (Computed Tomography — an
X-ray machine that takes 3D images). CT scanners are:
- Large and expensive (>$1 million)
- Found only in hospitals
- Unavailable in ambulances, rural clinics, or developing countries

**Our goal** is to build a deep learning classifier that can detect and classify brain
haemorrhages using **microwave radar signals** — the kind of technology in a cheap,
portable device. Microwave signals can safely pass through the skull and reflect off
the blood, similar to how radar detects aircraft.

To train such a classifier we need thousands of labelled examples. Real patient data is
scarce and ethically restricted. So we **simulate** the physics computationally.

---

## 2. The Four Classes We Classify

The brain is surrounded by several protective layers. Bleeds can occur between any
two layers:

```
  SKULL (bone)
    │
    ├── Epidural space  →  Class 1: EPIDURAL haemorrhage (EDH)
    │                      Blood between skull and the dura membrane
    │                      (often from trauma cracking the skull)
    │
    ├── Dura mater (tough protective membrane)
    │
    ├── Subdural space  →  Class 2: SUBDURAL haemorrhage (SDH)
    │                      Blood between dura and brain surface
    │                      (often in elderly patients from minor head bumps)
    │
    ├── CSF (Cerebrospinal Fluid — the fluid cushioning the brain)
    │
    └── Brain tissue    →  Class 3: INTRACEREBRAL haemorrhage (ICH)
                           Blood inside the brain itself
                           (most dangerous, often from high blood pressure)

    Class 0: HEALTHY — no bleeding
```

---

## 3. Why Microwave Signals Can Detect Bleeds

The key physical property is the **dielectric permittivity** (symbol: **ε**, Greek
letter epsilon), which describes how much a material slows down and reflects
electromagnetic waves such as microwaves or light.

- **ε_r** (relative permittivity, also called *dielectric constant*) — a dimensionless
  number. Higher = more the material slows the wave. Air = 1.0. Water ≈ 80.
- **σ** (sigma) — electrical *conductivity* in Siemens per metre (S/m). Higher = more
  energy absorbed as heat as the wave passes through.

Different brain tissues have different ε_r and σ values. Blood has a **distinctly
different** dielectric property from the surrounding tissue — this contrast is what
makes the bleed detectable:

| Tissue | ε_r (at 1 GHz) | σ (S/m) | What it means physically |
|--------|---------------|---------|--------------------------|
| Scalp (skin) | 40.0 | 0.87 | Moderately slows waves |
| Skull bone | 13.1 | 0.10 | Low permittivity, less slowing — good for signal passage |
| Dura mater | 44.0 | 0.82 | Thin membrane, similar to scalp |
| CSF (brain fluid) | 68.0 | 2.46 | High permittivity, highly absorbing |
| Gray matter (brain) | 52.7 | 0.94 | Main brain tissue |
| White matter (core) | 38.1 | 0.61 | Inner brain, less water content |
| **Blood (acute)** | **61.0** | **1.58** | **Higher than brain → strong reflection** |
| **Blood (chronic)** | **43.7** | **1.03** | **Older bleed → weaker contrast** |

When a microwave pulse hits a boundary between two materials with different ε_r,
part of the wave **reflects back**. The bigger the contrast in ε_r, the stronger the
reflection. Blood vs. gray matter: Δε_r ≈ 8 — detectable but subtle, which is why we
need advanced signal processing and machine learning.

The values above come from the **Gabriel 1996 Cole-Cole model** — the most widely
cited database of biological tissue dielectric properties, measured experimentally
across a broad frequency range (10 Hz to 20 GHz) on human and animal tissue samples
(Gabriel, S., Lau, R.W., Gabriel, C., Phys. Med. Biol. 41:2251, 1996).

---

## 4. What is FDTD?

**FDTD** stands for **Finite-Difference Time-Domain**. It is a numerical method for
solving Maxwell's equations — the fundamental equations governing all electromagnetic
phenomena (light, radio, microwave, radar).

- "Finite-difference" means we approximate derivatives with differences on a discrete
  grid (like pixels in an image, but for the electromagnetic field)
- "Time-domain" means we simulate the field evolving step-by-step in time (like a
  movie, not a photograph)

The simulation grid we use is called a **Yee grid** (after Kane Yee, 1966). The 3D
space is divided into 64×64×64 tiny cubic cells, each 3 mm on a side. At each cell we
track six field components: Ex, Ey, Ez (electric field) and Hx, Hy, Hz (magnetic
field). The simulation advances in time steps of 5.66 picoseconds (5.66×10⁻¹² seconds).

Running 700 time steps with 8 antennas requires simulating ~64³ × 700 × 8 × 2 = 
**11.6 billion field updates per sample**. This takes ~30 seconds on a GPU but would
take hours on a CPU.

---

## 5. The Pipeline — Step by Step

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STEP 1: Build a unique randomised head phantom                     │
 │                                                                     │
 │  Every sample gets its own head geometry drawn from population      │
 │  statistics — skull thickness, brain size, and head scale all vary. │
 │  This forces the neural network to learn the physics of blood       │
 │  scattering rather than memorising a fixed skull shape.             │
 └────────────────────────────┬────────────────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────────────────┐
 │  STEP 2: Place the haemorrhage (for classes 1–3)                    │
 │                                                                     │
 │  A sphere of blood dielectric material is placed in the correct     │
 │  anatomical zone (epidural/subdural/intracerebral). Size is random  │
 │  (6–36 mm radius). Age is random (acute/subacute/chronic) affecting │
 │  the ε_r value of the blood.                                        │
 └────────────────────────────┬────────────────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────────────────┐
 │  STEP 3: Fire UWB pulses from each of 8 antennas                    │
 │                                                                     │
 │  A Gaussian monocycle pulse (0.5–1.5 GHz bandwidth, 1 GHz centre)  │
 │  is injected at one antenna at a time. All 8 antennas record the   │
 │  received signal simultaneously → 8 TX × 8 RX = 64 signal traces.  │
 │  This is called a MIMO (Multiple-Input Multiple-Output) radar setup.│
 │  Each antenna fires once = 8 separate FDTD simulations per sample.  │
 └────────────────────────────┬────────────────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────────────────┐
 │  STEP 4: Simulate the healthy reference (same geometry, no bleed)   │
 │                                                                     │
 │  The same head is simulated again without any blood. Subtracting    │
 │  this "background" from the bleed signals isolates the scattered    │
 │  signal caused *only* by the haemorrhage.  Without this step the    │
 │  direct path between antennas and tissue reflections dominate and   │
 │  the bleed signal is invisible.                                     │
 └────────────────────────────┬────────────────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────────────────┐
 │  STEP 5: DAS Backprojection → 2D image                              │
 │                                                                     │
 │  DAS = Delay-And-Sum. This is a classical radar imaging algorithm:  │
 │  for each pixel in the image, we ask "if there were a scatterer at  │
 │  this location, at what time would the echo arrive at each          │
 │  antenna?". We sum up the signal at those exact arrival times for   │
 │  all antenna pairs. Where signals add up coherently → bright pixel  │
 │  (bleed). Elsewhere signals cancel out → dark pixel.               │
 │                                                                     │
 │  Travel times are computed accounting for the slower wave speed     │
 │  inside tissue (speed = c/√ε_r where c=3×10⁸ m/s is light speed). │
 └────────────────────────────┬────────────────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────────────────┐
 │  STEP 6: Save the sample (.npz file ~2 MB)                          │
 │                                                                     │
 │  Saved data: raw MIMO signals (8×8×700), DAS image (64×64),        │
 │  class label, bleed location, size, blood age, phantom geometry.    │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## 6. The Antenna Array

Eight antennas are placed uniformly in a ring around the equator of the head phantom,
at a radius of 90 mm (just outside the scalp surface at ~82 mm). This mimics a
realistic wearable microwave headset.

**Why 8 antennas?** With 8 elements we get 64 TX-RX signal pairs (8×8 MIMO matrix).
This gives sufficient spatial diversity to localise a bleed in 2D via DAS imaging,
while keeping simulation cost practical.

**Why 0.5–1.5 GHz (UWB)?**
- Too low (<0.3 GHz): wavelength too long → poor spatial resolution
- Too high (>1.5 GHz): skull and CSF absorb most of the signal before it reaches
  deep brain tissue (CSF has ε_r=68 which limits the grid accuracy to ≤1.2 GHz
  at 3 mm cell size)
- 0.5–1.5 GHz is the **sweet spot** — good penetration and reasonable resolution,
  matching published brain microwave imaging research

---

## 7. Dataset Numbers at a Glance

| Property | Value |
|----------|-------|
| Total samples | 2,000 |
| Training samples | 1,600 (400 per class) |
| Test samples | 400 (100 per class) |
| Grid size | 64³ cells = 192 mm cube |
| Cell size | 3 mm |
| Frequency band | 0.5–1.5 GHz (UWB, 1 GHz bandwidth) |
| Antennas | 8 (ring at 90 mm from centre) |
| Time steps | 700 per simulation |
| FDTD sims per sample | 16 (8 TX × 2: bleed + healthy reference) |
| Estimated total size | ~4 GB |
| Generation time | ~13 hours on 2× Nvidia T4 GPU (Kaggle) |

---

## 8. What Makes This Dataset Publication-Quality

Five rigorous rules are enforced (matching standards from recent IEEE/TMTT papers
on computational brain microwave imaging):

1. **Frequency-dependent tissue physics** — ε_r and σ computed from the Gabriel 1996
   4-pole Cole-Cole dispersion model at exactly 1.0 GHz. Not constant values.

2. **Anatomically constrained bleed placement** — bleeds can only appear in the
   correct tissue zone for their type. An epidural bleed cannot accidentally be placed
   inside the brain.

3. **Blood aging physics** — three distinct blood dielectric states (acute = fresh,
   subacute = 2–7 days, chronic = >7 days) with different ε_r values, reflecting the
   real biochemical changes in haematoma over time.

4. **Balanced class distribution** — exactly 25% of samples per class (healthy, EDH,
   SDH, ICH) prevents classifier bias from class imbalance.

5. **Independent test anatomy** — training and test samples use separate random seed
   spaces (offset by 10,000,000), guaranteeing no phantom geometry is shared between
   train and test. The model is tested on truly unseen anatomy.

---

## 9. Hardware and Software

- **Simulation engine:** WaveForge — custom-built 3D FDTD solver using PyTorch GPU kernels
- **Repository:** github.com/shahzaibshazoo/waveforge (open source, MIT licence)
- **Generation hardware:** Kaggle cloud platform, 2× Nvidia T4 GPU (16 GB each)
- **GPU throughput:** ~567 million cell-updates per second (T4), enabling the full
  2,000-sample dataset to be generated in approximately 13 hours

---

*Document version 1.4 — WaveForge Brain Haemorrhage Dataset*
