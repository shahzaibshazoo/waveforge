# Phase 4: 3D Source Types — Points, Planes, and TFSF

## Overview

This document specifies the framework for injecting electromagnetic energy into 3D FDTD simulations. Phase 4 extends the 2D source infrastructure (PointSource, LineSource, waveform generators) to three dimensions and introduces new source classes tailored to 3D physics: PlaneSource for uniform field injection, TFSF (Total-Field/Scattered-Field) boundaries for clean plane-wave excitation, GaussianBeam for focused wavefront delivery, and Dipole sources for radiation patterns.

The source injection system remains fundamentally soft (additive) using ellipsis indexing to preserve batch safety. All sources share the pre-computed waveform model with three families of temporal envelopes: broadband (GaussianPulse, Ricker), narrowband (ModulatedGaussian, Chirp), and continuous-wave (SinusoidalSource).

---

## Architecture and Design Principles

### Soft Injection via Ellipsis Indexing

All source types inject energy additively into field tensors using PyTorch ellipsis (`...`) indexing. This pattern is batch-safe and maintains device locality:

```python
# Generic soft-source update (batch-compatible)
target_field[..., i, j, k] += waveform_value(t)
```

Where `target_field` may have shape:
- `(Nx, Ny, Nz)` for single-domain
- `(B, Nx, Ny, Nz)` for batched multi-domain

### Pre-Computed Waveforms

All waveform objects follow the same design:
1. Abstract base class `Waveform` with `_compute(t)` method
2. Public `build(N_steps, dt, device, dtype)` pre-computes a 1-D tensor
3. Properties: `peak_time`, `bandwidth`, `amplitude`
4. Method: `is_causal(dt)` checks causality at t=0

Waveforms are cached on the device at source construction time to avoid repeated recomputation.

### Source Collection and Stepping

A `SourceCollection` maintains an ordered list of independent sources and steps them together:

```python
sources = SourceCollection()
sources.add(point_source)
sources.add(plane_source)
sources.add(tfsf_source)
...
sources.step(fields_dict, n)  # Step all sources at time index n
```

Each `step()` call injects into one or more field components at timestep `n`.

---

## Waveform Types

### Broadband Waveforms (Existing, Extend to 3D)

#### GaussianPulse

Gaussian-envelope pulse centered at time `t0` with standard deviation `sigma`.

**Formula:**
$$f(t) = A \exp\left(-\frac{(t - t_0)^2}{2\sigma^2}\right)$$

**Parameters:**
- `amplitude`: Peak amplitude A (non-zero)
- `t0`: Pulse center time (default: 5σ for causality)
- `sigma`: Temporal std deviation (s) — **OR** use `freq` parameter
- `freq`: Alternative parameter; sets `sigma = 1 / (2π freq / 3)` for conservative bandwidth

**Properties:**
- `peak_time`: Returns t0
- `bandwidth`: 1/(2π σ) Hz
- `amplitude`: Returns A

**Causality check:** By default, t0 = 5σ ensures |f(0)|/A < 1e-4.

**3D Usage:** Works unchanged; inject into any of {Ex, Ey, Ez, Hx, Hy, Hz}.

#### RickerWavelet

Mexican-hat (second derivative of Gaussian) for broadband geophysical surveys.

**Formula:**
$$f(t) = A \left(1 - 2\pi^2 f_p^2 (t-t_0)^2\right) \exp\left(-\pi^2 f_p^2 (t-t_0)^2\right)$$

**Parameters:**
- `amplitude`: Peak amplitude A
- `peak_freq`: Peak spectral frequency fp (Hz)
- `t0`: Wavelet center time (default: 1.5 / fp for causality)

**Properties:**
- `peak_time`: Returns t0
- `bandwidth`: Returns fp Hz
- `amplitude`: Returns A

**3D Usage:** Broadband over ~0.5 to ~1.5 × fp. Useful for multi-scale structural imaging.

#### SinusoidalSource

Continuous-wave sinusoid with smooth exponential ramp-on envelope.

**Formula:**
$$f(t) = A \sin(2\pi f_0 t + \phi) \left(1 - \exp\left(-\frac{t}{\tau_{\text{ramp}}}\right)\right)$$

where $\tau_{\text{ramp}} = \frac{3}{2\pi f_0}$.

**Parameters:**
- `amplitude`: Peak amplitude A
- `frequency`: Carrier frequency f0 (Hz)
- `phase`: Initial phase φ (radians, default 0)

**Properties:**
- `peak_time`: Returns -1.0 (sentinel; no single peak)
- `bandwidth`: Returns 0.0 (monochromatic)
- `amplitude`: Returns A

**3D Usage:** Monochromatic excitation for resonator or eigenmode studies.

### Narrowband Waveforms (New for 3D)

#### ModulatedGaussian

Modulates a sinusoidal carrier with a Gaussian envelope. Useful for narrowband frequency sweeps and Q-factor measurements.

**Formula:**
$$f(t) = A \sin(2\pi f_c t + \phi) \exp\left(-\frac{(t - t_0)^2}{2\sigma^2}\right)$$

**Parameters:**
- `amplitude`: Peak amplitude A
- `carrier_freq`: Carrier frequency fc (Hz)
- `phase`: Carrier phase φ (radians)
- `t0`: Envelope center time (s)
- `sigma`: Envelope temporal std deviation (s)

**Properties:**
- `peak_time`: Returns t0
- `bandwidth`: ≈ fc ± 1/(2πσ), narrower than GaussianPulse
- `amplitude`: Returns A

**Design rationale:** Narrower spectral content than GaussianPulse for measuring resonance peaks with minimal off-resonance excitation.

#### Chirp

Linear frequency sweep from f_start to f_end over time interval [t_start, t_end].

**Formula:**
$$f(t) = A \sin\left(2\pi \int_{t_{\text{start}}}^{t} f(t') dt' + \phi\right) \cdot w(t)$$

where the instantaneous frequency varies linearly:
$$f_{\text{inst}}(t) = f_{\text{start}} + \frac{f_{\text{end}} - f_{\text{start}}}{t_{\text{end}} - t_{\text{start}}} (t - t_{\text{start}})$$

and $w(t)$ is a smooth window (Hann or Tukey).

**Parameters:**
- `amplitude`: Peak amplitude A
- `f_start`: Starting frequency (Hz)
- `f_end`: Ending frequency (Hz)
- `t_start`: Chirp start time (s)
- `t_end`: Chirp end time (s)
- `phase`: Initial phase φ (radians)
- `window`: 'hann' or 'tukey' (default 'hann')

**Properties:**
- `peak_time`: Returns (t_start + t_end) / 2
- `bandwidth`: Approximately f_end - f_start Hz
- `amplitude`: Returns A

**Design rationale:** Swept-frequency excitation captures response over a broad band with temporal resolution, enabling inverse scattering and material characterization.

---

## Source Types

### 1. PointSource (Existing, Extends to 3D)

Single-cell soft injection at grid indices (i, j, k) into a specified field component.

**Constructor:**
```python
PointSource(
    waveform: Waveform,
    i: int,
    j: int,
    k: int,
    component: str = "Hz",
    *,
    grid: YeeGrid,
    N_steps: int,
)
```

**Parameters:**
- `waveform`: Any Waveform subclass (GaussianPulse, Ricker, Sinusoidal, Modulated, Chirp)
- `(i, j, k)`: Grid cell indices (0-indexed, must be in-domain)
- `component`: Field component name, one of {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
- `grid`: YeeGrid instance
- `N_steps`: Total simulation steps

**Injection:**
```python
source.step(fields_dict, n)
# target_field[..., i, j, k] += waveform_tensor[n]
```

**3D Valid Components:**
- Electric: Ex, Ey, Ez (face-centered on Yee cell)
- Magnetic: Hx, Hy, Hz (edge-centered on Yee cell)

**Use cases:**
- Dipole approximation (inject Hz at a point)
- Initial condition for standing-wave resonators
- Impulse response measurement

---

### 2. LineSource (Existing, Extends to 3D)

Multi-cell soft injection along a principal grid axis (x, y, or z).

**Constructor:**
```python
LineSource(
    waveform: Waveform,
    axis: str,            # 'x', 'y', or 'z'
    position: tuple[int, int],  # (j, k) if axis='x'; (i, k) if axis='y'; (i, j) if axis='z'
    component: str = "Hz",
    start_idx: int = 0,
    end_idx: int = None,  # Defaults to grid.Nx/Ny/Nz
    *,
    grid: YeeGrid,
    N_steps: int,
)
```

**Parameters:**
- `axis`: Direction of line ('x', 'y', 'z')
- `position`: Indices perpendicular to the line
- `component`: Field component (must align with axis orientation)
- `start_idx`, `end_idx`: Range along the line axis

**Injection:**
```python
# Example: line along x-axis at (j=50, k=75)
source.step(fields_dict, n)
# fields["Ex"][..., :, 50, 75] += waveform_tensor[n]
```

**3D Valid Combinations:**
- axis='x': component ∈ {Ex, Hx} (0-D perpendicular to field)
- axis='y': component ∈ {Ey, Hy}
- axis='z': component ∈ {Ez, Hz}

**Use cases:**
- Waveguide excitation (line antenna along propagation axis)
- Uniformity tests (compare field profiles)
- Linear aperture arrays

---

### 3. PlaneSource (New for 3D)

Uniform field injection on an entire 2-D plane (xy, xz, or yz).

**Constructor:**
```python
PlaneSource(
    waveform: Waveform,
    plane: str,          # 'xy', 'xz', or 'yz'
    position: int,       # k (if plane='xy'), j (if plane='xz'), i (if plane='yz')
    component: str,      # Must be perpendicular to plane
    *,
    grid: YeeGrid,
    N_steps: int,
)
```

**Parameters:**
- `plane`: Orientation ('xy', 'xz', or 'yz')
- `position`: Cell index normal to the plane
  - plane='xy' → position sets k
  - plane='xz' → position sets j
  - plane='yz' → position sets i
- `component`: Field component perpendicular to plane
  - plane='xy' → component ∈ {Ez, Hz}
  - plane='xz' → component ∈ {Ey, Hy}
  - plane='yz' → component ∈ {Ex, Hx}

**Injection:**
```python
# Example: xy-plane at k=100, Hz component
source.step(fields_dict, n)
# fields["Hz"][..., :, :, 100] += waveform_tensor[n]
```

**Memory footprint:**
- Pre-computed waveform: N_steps × float32 (no per-cell memory increase)
- Broadcasting: O(1) in code, O(Nx × Ny) in arithmetic

**Use cases:**
- Wavefront initialization on a plane
- Plane-wave approximation (uniform amplitude across transverse plane)
- Distributed source array (multiple PlaneSource at different positions)

---

### 4. TFSF (Total-Field/Scattered-Field) Boundary

Injects a known plane wave without near-field diffraction by separating total field (inside box) from scattered field (outside box). Uses the equivalence theorem to correct fields at the six faces of a cubic TFSF volume.

#### Theory

**Problem:** Injecting a plane wave directly into the domain causes diffraction at the boundaries, contaminating near-field results.

**Solution:** Define a TFSF box with boundaries at (i_min, i_max) × (j_min, j_max) × (k_min, k_max). Conceptually:
- **Inside:** Total field = incident + scattered
- **Outside:** Scattered field only

The injection corrects tangential E and H on all six faces using the known 1-D incident field.

#### Implementation Strategy

**1. Auxiliary 1-D FDTD**

Maintain a 1-D FDTD solver along the propagation direction (assume z for simplicity):
```python
class AuxiliaryFDTD1D:
    """1-D solver for incident field E_z(z, t) and H_y(z, t)."""
    def __init__(self, propagation_axis, waveform, grid_1d, N_steps):
        self.Ex_1d = torch.zeros(...)  # Along propagation axis
        self.Hz_1d = torch.zeros(...)
    
    def step(self, n):
        """Update 1-D fields at time step n using 1-D FDTD equations."""
        # Standard 1-D curl updates
        self.Hz_1d[n+1] = self.Hz_1d[n] + (dt/dz) * (Ex_1d[n,i+1] - Ex_1d[n,i])
        self.Ex_1d[n+1] = self.Ex_1d[n] + (dt/dz) * (Hz_1d[n,i] - Hz_1d[n,i-1])
```

**2. Field Correction on TFSF Boundaries**

For each of the six faces, compute the curl using incident field values and subtract/add to the 3-D domain:

**Face at k=k_max (top face, normal = +z):**
- Tangential components: Ex, Ey (incident field contributes Ex_inc, Ey_inc)
- H-components: Hx, Hy
- Correction:
  ```
  Ez[..., i, j, k_max] = Ez_target
  Hz[..., i, j, k_max] = Hz_corrected
  ```

**Detailed boundary condition for one face:**

Let incident field propagate along +z with E in x-direction (x-polarized plane wave):

```
E_x^{inc}(z, t) = A sin(k z - ω t)
H_y^{inc}(z, t) = (A / Z_0) sin(k z - ω t)
```

where Z_0 = √(μ_0/ε_0) = 377 Ω is the impedance of free space.

**At boundary k=k_max**, subtract the incident field from the 3-D solution:

```python
# Compute 1-D incident field at timestep n and position k_max
Ex_inc = auxiliary_1d.Ex_1d[n, z_index]
Hy_inc = auxiliary_1d.Hy_1d[n, z_index]

# Correct the 3-D field by imposing scattered field boundary:
# (Total field) = (incident) + (scattered)
# We update: total_field[boundary] = incident + scattered_computed_so_far

# For soft source, add correction:
fields_3d["Ex"][..., :, :, k_max] += Ex_inc * (1.0 - mask_inside)
fields_3d["Hy"][..., :, :, k_max] += Hy_inc * (1.0 - mask_inside)
```

**3. Mask for Selective Correction**

Use a binary mask to apply corrections only outside the TFSF box:

```python
mask_inside = torch.zeros_like(fields_3d["Ex"])
mask_inside[i_min:i_max, j_min:j_max, k_min:k_max] = 1.0

# Correction term (additive):
correction = Ex_inc * (1.0 - mask_inside)
```

#### Constructor and API

```python
class TFSF:
    """Total-Field/Scattered-Field boundary source.
    
    Parameters
    ----------
    waveform : Waveform
        Incident pulse or CW source.
    propagation_axis : str
        Direction of plane wave: 'x', 'y', or 'z'.
    polarization_component : str
        Field component of the E-field (e.g., 'Ex', 'Ey', 'Ez').
    tfsf_box : dict
        Boundaries: {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}.
    amplitude : float
        Incident field amplitude.
    grid : YeeGrid
    N_steps : int
    """
    
    def __init__(
        self,
        waveform: Waveform,
        propagation_axis: str,
        polarization_component: str,
        tfsf_box: dict,
        amplitude: float = 1.0,
        *,
        grid: YeeGrid,
        N_steps: int,
    ):
        # Validate inputs
        if propagation_axis not in ('x', 'y', 'z'):
            raise ValueError("propagation_axis must be 'x', 'y', or 'z'")
        
        # Initialize auxiliary 1-D FDTD
        self._auxiliary_1d = self._setup_auxiliary_fdtd(
            propagation_axis, polarization_component, grid, N_steps
        )
        
        self._waveform = waveform.build(N_steps, grid.dt, grid.device, grid.dtype)
        self._amplitude = amplitude
        self._tfsf_box = tfsf_box
        self._grid = grid
        
        # Pre-compute mask for efficiency
        self._mask_inside = self._create_mask_inside()
    
    def step(self, fields_dict: dict, n: int):
        """Inject incident field at boundary; step auxiliary 1-D FDTD."""
        # Step 1D solver
        self._auxiliary_1d.step(n)
        
        # Step 2: Extract incident field from 1-D solution
        # Step 3: Apply corrections to 3-D boundaries
        ...
```

**Use cases:**
- Plane-wave illumination of scattering structures
- Beam-focused imaging without edge artifacts
- Frequency-domain scattered-field extraction
- Inverse scattering with clean incident reference

---

### 5. GaussianBeam (New for 3D)

Focused beam with Gaussian transverse profile, propagating along a principal axis.

#### Theory (Paraxial Approximation)

A Gaussian beam in the paraxial limit has the form:

$$E(x, y, z, t) = A \frac{w_0}{w(z)} \exp\left(-\frac{x^2 + y^2}{w(z)^2}\right) \exp\left(-i k z + i \phi(z) - i \omega t\right)$$

where:
- $w_0$: waist radius (focus spot size)
- $w(z) = w_0 \sqrt{1 + (z/z_R)^2}$: beam radius at distance z
- $z_R = \pi w_0^2 / \lambda$: Rayleigh range
- $\phi(z) = \arctan(z / z_R)$: Gouy phase

For a beam propagating along +z with real (in-phase) source:

$$E_x(x, y, z, t) = A \frac{w_0}{w(z)} \exp\left(-\frac{x^2 + y^2}{w(z)^2}\right) \cos(k z - \omega t + \phi(z))$$

#### Constructor and API

```python
class GaussianBeam:
    """Focused Gaussian beam with waist and Rayleigh range.
    
    Parameters
    ----------
    waveform : Waveform
        Temporal modulation (GaussianPulse, Ricker, etc.).
    propagation_axis : str
        'x', 'y', or 'z'.
    waist_radius : float
        Beam waist w_0 in meters.
    focal_position : float
        Position of the focus (waist) along propagation axis (meters).
    polarization_component : str
        Ex, Ey, or Ez (must be perpendicular to propagation axis).
    center_position : tuple[float, float]
        Transverse center position in (m, m). For axis='z', (x_center, y_center).
    wavelength : float
        Free-space wavelength λ (meters) — determines Rayleigh range.
    grid : YeeGrid
    N_steps : int
    """
    
    def __init__(
        self,
        waveform: Waveform,
        propagation_axis: str,
        waist_radius: float,
        focal_position: float,
        polarization_component: str,
        center_position: tuple[float, float],
        wavelength: float,
        *,
        grid: YeeGrid,
        N_steps: int,
    ):
        self._waveform = waveform.build(N_steps, grid.dt, grid.device, grid.dtype)
        self._waist = waist_radius
        self._focal_pos = focal_position
        self._wavelength = wavelength
        self._rayleigh = math.pi * waist_radius**2 / wavelength
        self._grid = grid
    
    def step(self, fields_dict: dict, n: int):
        """Inject Gaussian beam at all points on the source plane."""
        # For each point (i, j, k) on the source plane:
        for idx in range(grid.N_source_points):
            r_transverse = self._transverse_distance(idx)
            z_prop = self._propagation_distance(idx)
            
            # Beam envelope
            w_z = self._waist * math.sqrt(1 + (z_prop / self._rayleigh)**2)
            amplitude_profile = math.exp(-(r_transverse / w_z)**2)
            
            # Phase including Gouy
            gouy_phase = math.atan2(z_prop, self._rayleigh)
            spatial_phase = 2 * math.pi * z_prop / self._wavelength + gouy_phase
            
            # Inject
            fields_dict[self._component][..., idx] += (
                amplitude_profile * math.cos(spatial_phase) * self._waveform[n]
            )
```

**Use cases:**
- Tight-focus beam steerer
- Optical trapping simulations
- Laser-material interaction
- Photonic waveguide launching

---

### 6. Dipole Sources (New for 3D)

Hertzian dipole (infinitesimal current element) or magnetic dipole source. Models small antennas and elementary radiators.

#### Theory

**Electric Dipole:**

An oscillating dipole moment **p**(t) at the origin radiates a field. In the near field (r << λ), dipole is approximately a point source. In the far field (r >> λ), it radiates a sin²(θ) pattern.

A z-directed current element (Hertzian dipole aligned to z) is modeled as a source of Hz at a single point (i, j, k).

**Magnetic Dipole:**

An oscillating magnetic moment **m**(t) is equivalent to a small current loop. Injects Hx, Hy, or Hz depending on loop axis.

#### Constructor and API

```python
class HertzianDipole:
    """Electric dipole (current element) — radiates like a point source.
    
    Parameters
    ----------
    waveform : Waveform
        Temporal oscillation.
    position : tuple[int, int, int]
        (i, j, k) grid indices.
    dipole_moment_component : str
        'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz' — direction of dipole moment.
    dipole_strength : float
        Effective dipole moment amplitude (scaling factor, dimensionless).
    grid : YeeGrid
    N_steps : int
    """
    
    def __init__(
        self,
        waveform: Waveform,
        position: tuple[int, int, int],
        dipole_moment_component: str,
        dipole_strength: float = 1.0,
        *,
        grid: YeeGrid,
        N_steps: int,
    ):
        self._waveform = waveform.build(N_steps, grid.dt, grid.device, grid.dtype)
        self._position = position
        self._component = dipole_moment_component
        self._strength = dipole_strength
    
    def step(self, fields_dict: dict, n: int):
        """Inject dipole current at position."""
        i, j, k = self._position
        fields_dict[self._component][..., i, j, k] += (
            self._strength * self._waveform[n]
        )
    
    def far_field_pattern(self, theta: float, phi: float, frequency: float) -> float:
        """Compute far-field magnitude at spherical angle (θ, φ).
        
        For a z-directed dipole, |E_θ| ∝ sin(θ).
        Returns normalized amplitude [0, 1].
        """
        if self._component in ("Ez", "Hx", "Hy"):
            # z-directed dipole
            return abs(math.sin(theta))
        elif self._component in ("Ex", "Hy", "Hz"):
            # x-directed dipole
            return abs(math.sin(theta) * math.cos(phi))
        elif self._component in ("Ey", "Hx", "Hz"):
            # y-directed dipole
            return abs(math.sin(theta) * math.sin(phi))
        else:
            return 0.0
```

**Use cases:**
- Antenna radiation patterns
- Resonator excitation tests
- Hanle effect and coherent control
- Near-field to far-field transformations

---

## Polarization Control

In 3D, sources can excite arbitrary linear, circular, or elliptical polarizations by injecting into multiple components with controlled phase offsets.

### Linear Polarization

**x-polarized plane wave:**
```python
source_ex = PointSource(
    waveform=GaussianPulse(...),
    i=i0, j=j0, k=k0,
    component="Ex",
    grid=grid, N_steps=N
)
```

**y-polarized plane wave:**
```python
source_ey = PointSource(
    waveform=GaussianPulse(...),
    i=i0, j=j0, k=k0,
    component="Ey",
    grid=grid, N_steps=N
)
```

### Circular Polarization

Inject two orthogonal components with 90° temporal phase offset:

```python
# Right-hand circular (RHC) polarization in xy-plane (z-propagating)
waveform_ex = GaussianPulse(amplitude=1.0, sigma=sigma, t0=t0)
waveform_ey_delayed = GaussianPulse(amplitude=1.0, sigma=sigma, t0=t0 + T_delay)
# where T_delay = λ / (4 * c) for 90° phase offset

source_ex = PointSource(waveform_ex, i0, j0, k0, "Ex", grid=grid, N_steps=N)
source_ey = PointSource(waveform_ey_delayed, i0, j0, k0, "Ey", grid=grid, N_steps=N)

sources.add(source_ex)
sources.add(source_ey)
```

### Elliptical Polarization

Generalize by varying amplitude ratio and phase offset:

```python
A_x, A_y = 1.0, 0.7  # Ellipse axes
phase_y = math.pi / 4  # 45° phase lag

source_ex = PointSource(...component="Ex", ...)
source_ey = PointSource(...component="Ey"...)  # With phase delay
```

---

## Near-to-Far-Field Transform (Surface Equivalence)

Convert near-field tangential E and H recorded on a closed surface into far-field radiation pattern via Fourier transform.

### Theory

**Principle:** The surface equivalence theorem states that tangential E and H on a closed surface uniquely determine all fields outside the surface via a 2-D Fourier transform.

**Step 1: Record tangential fields on closed surface**

Maintain a history buffer:
```python
class NearToFarFieldTransform:
    def __init__(self, surface_box, frequency, grid, N_steps):
        self.E_tangential_history = []  # List of (E_x, E_y, E_z) snapshots
        self.H_tangential_history = []  # List of (H_x, H_y, H_z) snapshots
        self.surface_box = surface_box  # {'x': (x_min, x_max), ...}
    
    def record_fields(self, fields_dict, n):
        """Extract tangential E, H on surface at time step n."""
        # For each of the 6 faces, extract tangential components
        # Append to history buffers
        pass
```

**Step 2: FFT along tangential directions**

For each face, compute 2-D FFT to get spectral field components $\tilde{E}(k_x, k_y, f)$ and $\tilde{H}(k_x, k_y, f)$.

**Step 3: Evaluate far-field pattern**

Using the spectral Green's function, compute the far-field E and H at observation point **r** = (r, θ, φ):

$$\mathbf{E}_{\text{far}}(\mathbf{r}, f) = \text{FFT}_{2D}\left[\text{Tangential E on surface}\right] \times G_{\text{spectral}}$$

where the Green's function depends on frequency and wavenumber.

### Implementation Outline

```python
def compute_far_field(self, frequency: float, theta: float, phi: float, distance: float = 1e6) -> tuple[complex, complex]:
    """Compute far-field E and H at (θ, φ, r).
    
    Parameters
    ----------
    frequency : float
        Target frequency (Hz).
    theta : float
        Polar angle (radians).
    phi : float
        Azimuthal angle (radians).
    distance : float
        Observation distance (meters).
    
    Returns
    -------
    E_theta, E_phi : complex
        Far-field E-field components.
    """
    # 1. Extract spectral data at requested frequency
    # 2. Apply spectral Green's function
    # 3. Compute radiation pattern
    
    k = 2 * math.pi * frequency / c_0
    
    # Radiation pattern (simplified):
    pattern = self._compute_spectral_pattern(frequency, theta, phi)
    
    E_magnitude = pattern * math.exp(1j * k * distance) / distance
    return E_magnitude, E_magnitude * 0.5  # Approximate H as E/Z_0
```

**Use cases:**
- Antenna gain and directivity
- Scattering cross-section (RCS)
- Radiator efficiency
- Microwave circuit characterization

---

## Source Collection and Timestep Interface

### SourceCollection Class

```python
class SourceCollection:
    """Manages a collection of sources stepped together."""
    
    def __init__(self):
        self._sources = []
    
    def add(self, source):
        """Add a source to the collection."""
        self._sources.append(source)
    
    def step(self, fields_dict: dict, n: int):
        """Step all sources at timestep n.
        
        Parameters
        ----------
        fields_dict : dict
            Dictionary of field tensors keyed by component name.
        n : int
            Current timestep index.
        """
        for source in self._sources:
            source.step(fields_dict, n)
    
    def __len__(self) -> int:
        """Total number of sources."""
        return len(self._sources)
```

### Integration with FDTD Loop

```python
def run_simulation(fdtd_solver, sources, N_steps):
    """Main FDTD time loop with source injection."""
    for n in range(N_steps):
        # Inject sources
        sources.step(fdtd_solver.fields, n)
        
        # FDTD updates
        fdtd_solver.step()
        
        # Output/analysis (optional)
        if n % output_freq == 0:
            save_snapshot(fdtd_solver.fields, n)
```

---

## Batch Safety and Multi-Domain Simulations

All source injections use ellipsis indexing to support batch processing:

```python
# Single domain: shape (Nx, Ny, Nz)
fields["Ex"][..., i, j, k] += waveform[n]

# Batched: shape (B, Nx, Ny, Nz)
fields["Ex"][..., i, j, k] += waveform[n]  # Broadcasts correctly
```

No explicit `for` loops over batch dimension. PyTorch broadcasting ensures efficiency.

---

## Causality and Stability

### Causality Constraints

All waveforms must satisfy |f(0)| / |amplitude| < 1e-4 to avoid non-physical pre-cursor fields.

**Default settings (pass all checks):**
- GaussianPulse: t0 = 5σ
- RickerWavelet: t0 = 1.5 / fp
- SinusoidalSource: phase = 0
- ModulatedGaussian: t0 ≥ 3σ
- Chirp: smooth window from t_start

### Numerical Stability

Sources are injected **after** field updates in the timestep. This preserves the Courant-Friedrichs-Lewy (CFL) condition. For 3D:

$$\Delta t \leq 0.99 \times \frac{\Delta}{c_0 \sqrt{3}}$$

where Δ is the grid spacing.

---

## Memory and Computational Cost

### Per-Source Memory

| Source Type | Memory | Notes |
|-------------|--------|-------|
| PointSource | N_steps × 4 B | Waveform tensor only |
| LineSource | N_steps × 4 B | Waveform cached |
| PlaneSource | N_steps × 4 B | No per-cell overhead |
| TFSF | N_steps × 8 B + grid_size/6 | 1-D auxiliary solver + mask |
| GaussianBeam | N_steps × 4 B + spatial profile | Lazy profile computation |
| Dipole | N_steps × 4 B | Point source equivalent |

Total for typical simulation: O(N_steps) GB (negligible vs. field tensors).

### Computational Cost per Timestep

| Source Type | FLOPs | Scaling |
|-------------|-------|---------|
| PointSource | O(1) | Constant, no spatial loops |
| LineSource | O(L) | L = line length |
| PlaneSource | O(A) | A = plane area |
| TFSF | O(S) | S = surface area of box (6 faces) |
| GaussianBeam | O(A) | A = illuminated area |
| Dipole | O(1) | Constant |

For typical 256³ domain with PlaneSource: ~65K flops/step (negligible vs. ~4M curl operations).

---

## Implementation Checklist

- [ ] **Waveforms:**
  - [ ] GaussianPulse (existing, test in 3D)
  - [ ] RickerWavelet (existing, test in 3D)
  - [ ] SinusoidalSource (existing, test in 3D)
  - [ ] ModulatedGaussian (new)
  - [ ] Chirp (new)

- [ ] **Sources:**
  - [ ] PointSource (extend to k dimension, test all 6 components)
  - [ ] LineSource (extend to 3D, test xyz axes)
  - [ ] PlaneSource (new)
  - [ ] TFSF (new, implement 1-D auxiliary solver)
  - [ ] GaussianBeam (new)
  - [ ] HertzianDipole (new)

- [ ] **Utilities:**
  - [ ] SourceCollection.add() and .step()
  - [ ] Causality checks for all waveforms
  - [ ] Batch-safe ellipsis indexing (regression test)
  - [ ] NearToFarFieldTransform (new)

- [ ] **Tests:**
  - [ ] Unit tests: each source at each valid component
  - [ ] Polarization tests: RHC, LHC, linear
  - [ ] TFSF plane-wave validation (compare to analytical)
  - [ ] Gaussian beam waist evolution (Rayleigh range)
  - [ ] Batch multi-domain sources (B > 1)
  - [ ] Memory usage profiling

- [ ] **Documentation:**
  - [ ] Docstrings for all classes and methods
  - [ ] Examples: dipole radiation, TFSF wavefront, beam focusing
  - [ ] Jupyter notebook: 3D source tutorial

---

## References

- Taflove, A., & Hagness, S. C. (2005). Computational Electromagnetics: The Finite-Difference Time-Domain Method (3rd ed.). Artech House.
- Umashankar, K. R., & Taflove, A. (1982). "A novel method to analyze electromagnetic scattering of complex objects." IEEE Transactions on Electromagnetic Compatibility, 24(4), 397-405. [TFSF introduction]
- Ramahi, O. M., Chuang, C. C., & Naishadham, L. (1997). "Multiresolution time-domain using CWT." IEEE Transactions on Microwave Theory and Techniques, 51(4), 1269-1277.
- Yariv, A., & Yeh, P. (1984). Optical Waves in Crystals. Wiley. [Gaussian beam theory]
- Jackson, J. D. (1998). Classical Electrodynamics (3rd ed.). Wiley. [Dipole radiation]
- Newman, E. H., & Tulyathan, P. (1987). "Source reconstruction from electromagnetic near-field data." IEEE Transactions on Antennas and Propagation, 36(6), 816-824. [Near-to-far-field]
