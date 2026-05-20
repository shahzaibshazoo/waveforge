# Phase 6: 3D Simulation Examples

## Overview

This document provides seven canonical examples demonstrating the WaveForge 3D FDTD engine across a range of applications: fundamental propagation, scattering, antenna radiation, and biomedical imaging. Each example includes grid specifications, material configurations, source definitions, and expected physics validation.

## Example 1: 3D Free-Space Pulse

**Purpose**: Validate basic 3D field propagation and PML absorption.

**Problem Description**: A Gaussian pulse source centered in a cubic domain with PML boundaries. No materials present (uniform vacuum, ε_r = 1, σ = 0).

### Configuration

```python
from waveforge import Simulation3D, MaterialMap, PMLBoundary, PointSource

# Grid
nx, ny, nz = 64, 64, 64
sim = Simulation3D(
    domain_size=(0.064, 0.064, 0.064),  # 64 mm cube
    grid_resolution=(nx, ny, nz),
    time_step_safety_factor=0.9
)

# PML boundaries
pml = PMLBoundary(thickness=8)  # PML layer: 8 cells
sim.add_boundary_condition('pml', pml)

# Source: Gaussian pulse at center
source = PointSource(
    position=(0.032, 0.032, 0.032),  # center
    frequency=1e9,                    # 1 GHz
    pulse_width=1e-10,                # 100 ps
    source_type='gaussian',
    polarization='z'                  # E_z polarization
)
sim.add_source(source)

# Monitor planes for visualization
sim.add_monitor('xy_slice', plane='xy', z=0.032, sample_rate=10)
sim.add_monitor('xz_slice', plane='xz', y=0.032, sample_rate=10)
sim.add_monitor('yz_slice', plane='yz', x=0.032, sample_rate=10)

# Run simulation
steps = 500
sim.run(steps)
```

### Expected Output

- Symmetric Gaussian pulse expanding from center in all three orthogonal directions
- Identical field magnitudes in xy, xz, yz slices (three-fold symmetry)
- Smooth amplitude decay due to spherical spreading
- PML boundaries show reflected power < -60 dB at boundaries

### Physics Validation

- Energy decay rate matches spherical divergence: `E ∝ 1/r`
- Pulse group velocity equals phase velocity (vacuum): `v_g = c`
- Spatial wavelength at 1 GHz: λ = c/f = 0.3 m = 300 mm (many wavelengths fit in grid)

### Grid Parameters

- Domain: 64 mm × 64 mm × 64 mm
- Cells: 64 × 64 × 64 = 262,144 cells
- Memory: ~32 MB (field data only)
- Runtime: ~2 seconds on T4 GPU

---

## Example 2: 3D Dielectric Sphere Scattering

**Purpose**: Validate scattering calculations against Mie theory (analytical solution).

**Problem Description**: A dielectric sphere with permittivity ε_r = 9 embedded in vacuum. Plane wave incidence along x-axis, frequency 1 GHz.

### Configuration

```python
from waveforge import Simulation3D, MaterialMap, PlaneWave

# Grid
sim = Simulation3D(
    domain_size=(0.3, 0.3, 0.3),   # 300 mm cube
    grid_resolution=(128, 128, 128),
    time_step_safety_factor=0.9
)

# Material: dielectric sphere
material_map = MaterialMap()
material_map.add_sphere(
    center=(0.15, 0.15, 0.15),      # center of domain
    radius=0.03,                     # 30 mm sphere
    epsilon_r=9.0,
    mu_r=1.0,
    conductivity=0.0
)
sim.set_material_map(material_map)

# PML boundary
pml = PMLBoundary(thickness=16)
sim.add_boundary_condition('pml', pml)

# Source: plane wave
plane_wave = PlaneWave(
    direction=(1.0, 0.0, 0.0),      # propagation along x
    frequency=1e9,                   # 1 GHz
    amplitude=1.0,
    polarization='y',               # E_y polarization
    wavelength_samples_per_period=10
)
sim.add_source(plane_wave)

# Far-field monitor (spherical shell)
sim.add_far_field_monitor(
    center=(0.15, 0.15, 0.15),
    radius=0.1,
    angular_samples=36
)

# Run simulation
steps = 1000
sim.run(steps)

# Post-process: extract RCS (radar cross-section)
rcs = sim.compute_radar_cross_section()
```

### Expected Output

- Scattering pattern exhibits dipole-like characteristics (TE₁⁰ and TM₁⁰ modes dominant)
- Forward scattering amplitude much larger than backscatter
- RCS maximum in the forward direction, minimum in shadow region

### Physics Validation

**Mie Theory Comparison**:

For a dielectric sphere of radius `a` and permittivity `ε_r = 9` at wavelength λ = 0.3 m:

```
Mie size parameter: x = πa/λ = π(0.03)/(0.3) ≈ 0.314 (small sphere limit, Rayleigh scattering)
```

In the Rayleigh limit, RCS ∝ a⁶ (strong size dependence). Expected numerical RCS should match analytical Mie series to within 5%.

### Grid Parameters

- Domain: 300 mm × 300 mm × 300 mm
- Cells: 128 × 128 × 128 = 2,097,152 cells
- Sphere radius: 30 mm (10 cells)
- Memory: ~500 MB
- Runtime: ~10 seconds on T4 GPU

---

## Example 3: 3D Patch Antenna Radiation Pattern

**Purpose**: Validate antenna radiation patterns and near-to-far-field transformation.

**Problem Description**: A microstrip patch antenna on a substrate, fed via a coaxial probe. Extract 3D radiation pattern (far-field pattern in all directions).

### Configuration

```python
from waveforge import Simulation3D, MaterialMap, VoltageSource

# Grid with asymmetric padding (more space in radiation direction)
sim = Simulation3D(
    domain_size=(0.2, 0.2, 0.3),   # 200×200×300 mm
    grid_resolution=(128, 128, 160),
    time_step_safety_factor=0.9
)

# Materials: substrate and ground plane
material_map = MaterialMap()

# Ground plane (metallic, PEC)
material_map.add_box(
    center=(0.1, 0.1, 0.0025),
    half_extents=(0.1, 0.1, 0.0025),
    epsilon_r=1e6,      # approximates perfect conductor
    conductivity=1e6
)

# Dielectric substrate (FR4, ε_r ≈ 4.4)
material_map.add_box(
    center=(0.1, 0.1, 0.025),
    half_extents=(0.1, 0.1, 0.02),
    epsilon_r=4.4,
    mu_r=1.0,
    conductivity=0.01   # small loss in FR4
)

sim.set_material_map(material_map)

# PML boundary
pml = PMLBoundary(thickness=20)
sim.add_boundary_condition('pml', pml)

# Source: coaxial feed (voltage source at antenna feed)
feed = VoltageSource(
    position=(0.1, 0.1, 0.052),    # above substrate
    frequency=2.4e9,                # 2.4 GHz ISM band
    amplitude=1.0,
    source_type='sinusoidal'
)
sim.add_source(feed)

# Patch antenna geometry (conductor on top of substrate)
material_map.add_box(
    center=(0.1, 0.1, 0.062),
    half_extents=(0.03, 0.02, 0.001),
    epsilon_r=1e6,
    conductivity=1e6
)

# Far-field monitor (spherical shell)
sim.add_far_field_monitor(
    center=(0.1, 0.1, 0.1),
    radius=0.15,
    angular_samples=72
)

# Run simulation (wait for steady state)
steps = 2000
sim.run(steps)

# Extract radiation pattern
radiation_pattern = sim.compute_radiation_pattern(theta_samples=91, phi_samples=181)
```

### Expected Output

- Directional radiation pattern with maximum in +z direction (above antenna)
- Gain approximately 6-7 dBi for patch antenna
- Null (or minimum) in -z direction (toward ground plane)
- Side lobes at ~-13 dB relative to main lobe

### Physics Validation

- Resonant frequency: ~2.4 GHz (design frequency achieved)
- Input impedance at port: ~50 Ω (matched for efficient coupling)
- Radiation efficiency: ~85-90% (remainder is dielectric loss)

### Grid Parameters

- Domain: 200 × 200 × 300 mm
- Cells: 128 × 128 × 160 = 2,621,440 cells
- Memory: ~600 MB
- Runtime: ~15 seconds on T4 GPU

---

## Example 4: 3D Breast Tumor Detection

**Purpose**: Simulate realistic biomedical imaging scenario with tissue phantoms and detection array.

**Problem Description**: An ellipsoidal breast phantom with embedded spherical tumor, illuminated by a 3D MIMO antenna array. Extract received signals for tumor localization.

### Configuration

```python
from waveforge import Simulation3D, MaterialMap

# Grid (256³ for high resolution)
sim = Simulation3D(
    domain_size=(0.128, 0.128, 0.064),  # 128×128×64 mm
    grid_resolution=(256, 256, 128),
    time_step_safety_factor=0.9
)

# Material: breast phantom
material_map = MaterialMap()

# Fatty tissue (background)
material_map.add_ellipsoid(
    center=(0.064, 0.064, 0.032),
    semi_axes=(0.040, 0.035, 0.025),   # ellipsoidal breast
    epsilon_r=5.0,
    conductivity=0.1
)

# Glandular tissue regions
material_map.add_sphere(
    center=(0.050, 0.060, 0.028),
    radius=0.012,
    epsilon_r=50.0,
    conductivity=1.5
)

material_map.add_sphere(
    center=(0.078, 0.070, 0.035),
    radius=0.010,
    epsilon_r=50.0,
    conductivity=1.5
)

# Tumor (spherical, high contrast)
material_map.add_sphere(
    center=(0.064, 0.064, 0.032),
    radius=0.008,                       # 8 mm diameter tumor
    epsilon_r=60.0,                     # higher than surrounding tissue
    conductivity=3.0
)

sim.set_material_map(material_map)

# PML boundary (absorbing)
pml = PMLBoundary(thickness=16)
sim.add_boundary_condition('pml', pml)

# MIMO antenna array (4 elements, 1 GHz frequency)
antenna_positions = [
    (0.064, 0.010, 0.032),  # bottom
    (0.064, 0.118, 0.032),  # top
    (0.010, 0.064, 0.032),  # left
    (0.118, 0.064, 0.032),  # right
]

from waveforge import AntennaArray
antenna_array = AntennaArray(frequency=1e9)

for idx, pos in enumerate(antenna_positions):
    antenna_array.add_transmit_element(
        position=pos,
        amplitude=1.0,
        excitation_index=idx
    )
    antenna_array.add_receive_element(
        position=pos,
        monitor_name=f'rx_{idx}'
    )

sim.add_antenna_array(antenna_array)

# Run simulation
steps = 1500
for step in range(steps):
    # Transmit from antenna 0, measure at all RX
    if step == 0:
        antenna_array.set_active_tx(0)
    sim.step()

# Extract S-parameters (reflection and transmission)
s_params = sim.compute_s_parameters()
```

### Expected Output

- S₁₁ (reflection): minimum at ~1 GHz (antenna resonance)
- S₂₁ (transmission TX0 → RX1): amplitude decrease with distance
- S-parameter signature includes strong tumor response at antenna pair closest to tumor
- Tumor localization: Time-delay imaging from S-parameters shows tumor location

### Physics Validation

- Attenuation through tissue: ~0.5 dB/cm at 1 GHz (matches literature)
- Phase velocity in tissue: ~c/√ε ≈ c/√5 ≈ 0.45c
- Tissue dispersion: σ/ωε < 1 at 1 GHz (reasonable for dielectric loss model)

### Grid Parameters

- Domain: 128 × 128 × 64 mm
- Cells: 256 × 256 × 128 = 8,388,608 cells
- Memory: ~600 MB (large grid, single precision)
- Runtime: ~30 seconds on T4 GPU

---

## Example 5: 3D Brain Hemorrhage Imaging

**Purpose**: High-contrast biomedical imaging: detect acute blood clot in brain.

**Problem Description**: Realistic head geometry approximated as concentric spheres (scalp, bone, cerebrospinal fluid, brain). A spherical acute hemorrhage (high water content) embedded in brain tissue. UWB microwave array for detection.

### Configuration

```python
from waveforge import Simulation3D, MaterialMap

# Grid
sim = Simulation3D(
    domain_size=(0.16, 0.16, 0.20),    # 160×160×200 mm head
    grid_resolution=(160, 160, 200),
    time_step_safety_factor=0.9
)

# Material structure (concentric layers)
material_map = MaterialMap()

# Brain (background, 75% water)
material_map.add_sphere(
    center=(0.08, 0.08, 0.10),
    radius=0.060,                       # 60 mm radius
    epsilon_r=50.0,                     # 50% H2O-equivalent
    conductivity=2.0
)

# Cerebrospinal fluid (CSF, high water content)
material_map.add_sphere(
    center=(0.08, 0.08, 0.10),
    radius=0.070,
    epsilon_r=80.0,                     # 80% water
    conductivity=2.5
)

# Bone layer
material_map.add_sphere(
    center=(0.08, 0.08, 0.10),
    radius=0.085,
    epsilon_r=15.0,                     # low-permittivity bone
    conductivity=0.8
)

# Scalp
material_map.add_sphere(
    center=(0.08, 0.08, 0.10),
    radius=0.100,
    epsilon_r=40.0,                     # skin/fat layer
    conductivity=1.2
)

# Acute blood clot (high water content, slightly different from surrounding brain)
material_map.add_sphere(
    center=(0.080, 0.075, 0.095),      # 10 mm offset from center
    radius=0.010,                       # 10 mm diameter clot
    epsilon_r=65.0,                     # higher water content → higher ε_r
    conductivity=2.5
)

sim.set_material_map(material_map)

# PML boundary
pml = PMLBoundary(thickness=16)
sim.add_boundary_condition('pml', pml)

# UWB antenna array around head (8 elements, 500 MHz – 2 GHz bandwidth)
import numpy as np

theta_positions = np.linspace(0, 2*np.pi, 8, endpoint=False)
antenna_positions = [
    (0.08 + 0.12*np.cos(theta), 0.08 + 0.12*np.sin(theta), 0.10)
    for theta in theta_positions
]

antenna_array = AntennaArray(frequency=1.2e9)  # center frequency

for idx, pos in enumerate(antenna_positions):
    antenna_array.add_transmit_element(position=pos, amplitude=1.0, excitation_index=idx)
    antenna_array.add_receive_element(position=pos, monitor_name=f'rx_{idx}')

sim.add_antenna_array(antenna_array)

# Run multi-transmit measurement
steps = 2000
for tx_idx in range(8):
    antenna_array.set_active_tx(tx_idx)
    for _ in range(steps // 8):
        sim.step()

# Reconstruct image via delay-and-sum beamformer
image_3d = sim.compute_delay_sum_image()
```

### Expected Output

- Received signals show amplitude modulation due to tissue heterogeneity
- Clot location recovered via beamforming: localization error < 5 mm
- Clot appears as bright spot in reconstructed image
- Contrast-to-noise ratio: ~10 dB (distinguishable from background)

### Physics Validation

- Attenuation through head: ~2-3 dB/cm at 1 GHz (matches in-vivo measurements)
- Clot permittivity (65) vs brain (50): ~30% change in ε_r (sufficient contrast)
- Propagation delay through head: ~1.6 ns (consistent with group velocity in brain)

### Grid Parameters

- Domain: 160 × 160 × 200 mm
- Cells: 160 × 160 × 200 = 5,120,000 cells
- Memory: ~400 MB
- Runtime: ~20 seconds on T4 GPU

---

## Example 6: 3D Rectangular Waveguide

**Purpose**: Validate modal propagation and cutoff frequency against theory.

**Problem Description**: A hollow rectangular waveguide (a × b × L) with TE₁₀ mode excitation at 10 GHz. Domain extends beyond waveguide to observe near-field.

### Configuration

```python
from waveforge import Simulation3D, MaterialMap

# Grid
sim = Simulation3D(
    domain_size=(0.040, 0.020, 0.150),  # 40×20×150 mm
    grid_resolution=(160, 80, 600),     # high aspect ratio
    time_step_safety_factor=0.9
)

# Material: waveguide walls (PEC)
material_map = MaterialMap()

# Four walls of rectangular waveguide
# Walls parallel to yz-plane (x = 0 and x = a)
material_map.add_box(
    center=(0.0, 0.010, 0.075),
    half_extents=(0.001, 0.010, 0.075),
    epsilon_r=1e6,
    conductivity=1e6
)

material_map.add_box(
    center=(0.040, 0.010, 0.075),
    half_extents=(0.001, 0.010, 0.075),
    epsilon_r=1e6,
    conductivity=1e6
)

# Walls parallel to xz-plane (y = 0 and y = b)
material_map.add_box(
    center=(0.020, 0.0, 0.075),
    half_extents=(0.020, 0.001, 0.075),
    epsilon_r=1e6,
    conductivity=1e6
)

material_map.add_box(
    center=(0.020, 0.020, 0.075),
    half_extents=(0.020, 0.001, 0.075),
    epsilon_r=1e6,
    conductivity=1e6
)

sim.set_material_map(material_map)

# PML boundary (minimal, main absorption at waveguide end)
pml = PMLBoundary(thickness=8)
sim.add_boundary_condition('pml', pml)

# Source: TE10 mode excitation (E_y field)
# Modal field for TE10: E_y = A sin(πx/a)
source = ModalSource(
    position=(0.020, 0.010, 0.025),   # at input
    mode_type='te10',
    waveguide_width=0.040,
    waveguide_height=0.020,
    frequency=10e9,
    amplitude=1.0
)
sim.add_source(source)

# Monitor: E_y field along waveguide centerline
sim.add_line_monitor('centerline', start=(0.020, 0.010, 0.010), 
                      end=(0.020, 0.010, 0.140), samples=200)

# Run simulation
steps = 1500
sim.run(steps)

# Analyze: extract wavelength and phase velocity
wavelength_num = sim.compute_wavelength_from_monitor('centerline')
fc_theoretical = 3e8 / (2 * 0.040)  # c / (2a) for TE10
f_propagation = 10e9
wavelength_theory = 3e8 / np.sqrt(10e9**2 - fc_theoretical**2)
```

### Expected Output

- TE₁₀ mode propagates with distinct field pattern: E_y symmetric across y, sinusoidal across x
- Modal propagation velocity < c (dispersive)
- Wavelength in guide: λ_g = λ₀ / √(1 - (f_c/f)²)
- At 10 GHz and a=40 mm: f_c = 3.75 GHz, λ_g ≈ 46 mm (numerical ~45 mm)

### Physics Validation

- Cutoff frequency: f_c = c / (2a) = 300 MHz / (2×40 mm) = 3.75 GHz (theoretical vs numerical agreement)
- Dispersion relation: k_z = √(k² - (πf_c/c)²) verified
- Attenuation absent (ideal conductor) → steady wave amplitude maintained
- Phase velocity: v_p = ω/k_z > c (superluminal phase velocity, subluminal group velocity)

### Grid Parameters

- Domain: 40 × 20 × 150 mm (rectangular, elongated in propagation direction)
- Cells: 160 × 80 × 600 = 7,680,000 cells
- Memory: ~550 MB
- Runtime: ~25 seconds on T4 GPU

---

## Example 7: 3D Photonic Crystal Bandgap

**Purpose**: Validate photonic bandgap prediction and mode confinement in periodic structures.

**Problem Description**: A periodic 3D photonic crystal (cubic lattice of dielectric spheres) with a defect (missing sphere). Excite defect mode and observe confinement.

### Configuration

```python
from waveforge import Simulation3D, MaterialMap

# Grid
sim = Simulation3D(
    domain_size=(0.105, 0.105, 0.105),  # 105 mm (5 lattice constants)
    grid_resolution=(128, 128, 128),
    time_step_safety_factor=0.9
)

# Material: periodic lattice of dielectric spheres
material_map = MaterialMap()

lattice_constant = 0.021  # 21 mm spacing
sphere_radius = 0.007     # 7 mm sphere radius

# Create 5×5×5 lattice
for i in range(5):
    for j in range(5):
        for k in range(5):
            # Skip center (defect at origin)
            if (i, j, k) == (2, 2, 2):
                continue
            
            x = 0.0105 + i * lattice_constant
            y = 0.0105 + j * lattice_constant
            z = 0.0105 + k * lattice_constant
            
            material_map.add_sphere(
                center=(x, y, z),
                radius=sphere_radius,
                epsilon_r=12.0,           # high-index dielectric (Si @ 1.55 μm)
                mu_r=1.0,
                conductivity=0.0
            )

sim.set_material_map(material_map)

# PML boundary (absorbing background)
pml = PMLBoundary(thickness=12)
sim.add_boundary_condition('pml', pml)

# Source: defect mode excitation (point source at center)
source = PointSource(
    position=(0.0525, 0.0525, 0.0525),  # defect center
    frequency=1.4e14,                    # ~200 nm wavelength in vacuum
    pulse_width=5e-15,                   # short pulse
    source_type='gaussian',
    polarization='z'
)
sim.add_source(source)

# Monitor: near-field at defect and far-field
sim.add_point_monitor('defect_center', position=(0.0525, 0.0525, 0.0525))
sim.add_volume_monitor('near_field', bounds=[(0.045, 0.045, 0.045), (0.060, 0.060, 0.060)])

# Run simulation
steps = 2000
sim.run(steps)

# Analyze: extract mode frequency from power spectrum
power_spectrum = sim.compute_power_spectrum('defect_center')
defect_mode_freq = sim.find_peak_frequency(power_spectrum)

# Expected defect mode within bandgap
lattice_photon_energy = 1.24e-6 / (200e-9) / 1.5  # energy in eV (rough estimate)
```

### Expected Output

- Defect mode frequency lies within photonic bandgap (no bulk propagation)
- Field highly confined to defect region (within ~1 lattice constant)
- Far-field radiation minimal due to defect confinement
- Mode Q-factor high (> 100) indicating weak radiation loss

### Physics Validation

- Bandgap position: ~1.3 to 1.5 eV (1 μm region, typical for Si photonic crystals)
- Defect mode resonance: ~1.4 eV (within theoretical bandgap)
- Field decay outside defect: exponential with decay length ~λ_0/π (evanescent)
- Confinement factor: >90% of energy within 1 lattice constant sphere

### Grid Parameters

- Domain: 105 × 105 × 105 mm (5 lattice constants per side)
- Cells: 128 × 128 × 128 = 2,097,152 cells
- Lattice constant: 21 mm (8 cells per lattice constant)
- Memory: ~500 MB
- Runtime: ~15 seconds on T4 GPU

---

## Summary Comparison Table

| Example | Application | Grid | Cells | Physics Focus |
|---------|-------------|------|-------|---|
| 1 | Free-space pulse | 64³ | 0.26M | Propagation, PML |
| 2 | Mie scattering | 128³ | 2.1M | Scattering, validation |
| 3 | Patch antenna | 128×128×160 | 2.6M | Radiation, far-field |
| 4 | Breast imaging | 256×256×128 | 8.4M | Complex tissue, MIMO |
| 5 | Brain hemorrhage | 160×160×200 | 5.1M | Biomedical, beamforming |
| 6 | Waveguide | 160×80×600 | 7.7M | Modal, dispersion |
| 7 | Photonic crystal | 128³ | 2.1M | Periodic, bandgap |

