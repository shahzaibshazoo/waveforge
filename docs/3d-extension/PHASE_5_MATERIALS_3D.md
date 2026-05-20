# Phase 5: 3D Material System

## Overview

The 3D material system extends MaterialMap to handle volumetric material definitions with full support for isotropic, anisotropic, and dispersive materials. Materials are represented as 3D spatial distributions of electromagnetic properties, enabling complex heterogeneous structures like tissue phantoms, composite materials, and frequency-dependent media.

## 3D Material Representation

### Field Tensor Shapes

The Ca and Cb coefficient tensors expand to 3D:

- **Ca tensor**: shape `(Nx, Ny, Nz)` — one value per Yee cell
- **Cb tensor**: shape `(Nx, Ny, Nz)` — one value per Yee cell
- **Da tensor**: shape `(Nx, Ny, Nz)` — magnetic material coefficient
- **Db tensor**: shape `(Nx, Ny, Nz)` — magnetic material coefficient

Each element represents the electromagnetic property at that voxel location. This per-voxel approach enables:

- Arbitrary material distributions without explicit geometry descriptions
- Direct import of computational data (MRI/CT scans, volumetric models)
- Smooth material transitions via interpolation

### Memory Requirements

For a grid of size `Nx × Ny × Nz = 256 × 256 × 256`:

```
Ca tensor:  256³ × 4 bytes = 67 MB
Cb tensor:  256³ × 4 bytes = 67 MB
Da tensor:  256³ × 4 bytes = 67 MB
Db tensor:  256³ × 4 bytes = 67 MB
─────────────────────────────
Total:      4 × 256³ × 4 = 268 MB
```

For dispersive materials, auxiliary field tensors add proportional storage per pole (see Dispersive Materials section below).

## 3D Geometry Primitives

The material painting API provides methods to add structured geometries to the MaterialMap:

### Basic Shapes

#### Sphere

```python
material_map.add_sphere(
    center=(x0, y0, z0),
    radius=r,
    epsilon_r=eps_r,
    mu_r=1.0,
    conductivity=0.0
)
```

Fills all voxels within the sphere with uniform material properties.

#### Cylinder

```python
material_map.add_cylinder(
    center=(x0, y0, z0),
    axis='z',           # 'x', 'y', or 'z'
    radius=r,
    height=h,
    epsilon_r=eps_r,
    mu_r=1.0,
    conductivity=0.0
)
```

Axis-aligned infinite cylinder (height constraint ignored outside bounds).

#### Box (Rectangular Parallelepiped)

```python
material_map.add_box(
    center=(x0, y0, z0),
    half_extents=(dx, dy, dz),
    epsilon_r=eps_r,
    mu_r=1.0,
    conductivity=0.0
)
```

Axis-aligned rectangular region.

#### Ellipsoid

```python
material_map.add_ellipsoid(
    center=(x0, y0, z0),
    semi_axes=(a, b, c),
    epsilon_r=eps_r,
    mu_r=1.0,
    conductivity=0.0
)
```

Ellipsoid with semi-axes lengths `a`, `b`, `c` along x, y, z respectively.

### Voxel Data Import

For complex tissue models or computational data:

```python
material_map.add_voxel_data(
    data_array,          # shape (Nx, Ny, Nz) with permittivity values
    origin=(x_min, y_min, z_min),
    voxel_size=(dx, dy, dz),
    interpolation='nearest'  # or 'linear'
)
```

Maps an external volumetric dataset (e.g., from MRI imaging) directly onto the simulation grid with optional spatial interpolation.

## Magnetic Materials

For materials with relative permeability `mu_r ≠ 1`, the magnetic field update equation includes an analogous coefficient tensor:

The H-field update in the presence of magnetic materials:

```
H_new = (2 - σ*Δt) / (2 + σ*Δt) × H_old 
        + Δt / (2μ_r*μ_0 + σ*Δt*μ_r*μ_0) × curl(E)
```

This is encoded in the Db coefficient tensor analogous to Cb for electric materials.

**Note**: Most dielectrics have `mu_r ≈ 1`. Magnetic materials (ferrites, iron) are rare in RF/photonic applications but essential for biomedical imaging (MRI applications).

## Dispersive Materials

Dispersive materials exhibit frequency-dependent permittivity: `ε(ω) = ε_r(ω)`. Simulating these in the time domain requires auxiliary differential equations (ADE).

### Auxiliary Differential Equation (ADE) Method

For a material with Lorentz poles:

```
ε(ω) = ε_∞ × [1 + Σ_p (ω_p² / (ω₀,p² - ω² - i*γ_p*ω))]
```

The polarization field P evolves via:

```
d²P/dt² + γ_p * dP/dt + ω₀,p² * P = ε_0 * (ε_∞ - 1) * ω_p² * E
```

In discrete form, this becomes a system of coupled ODEs updated each time step.

### Lorentz Pole Definition

```python
lorentz_pole = {
    'omega_0': 1e15,      # resonance frequency (rad/s)
    'gamma': 1e13,        # damping rate (rad/s)
    'omega_p': 1e15,      # plasma frequency (rad/s)
    'epsilon_inf': 2.0    # background permittivity
}

material_map.add_lorentz_sphere(
    center=(x0, y0, z0),
    radius=r,
    lorentz_poles=[lorentz_pole],
    sigma=0.0
)
```

Each Lorentz pole requires three auxiliary field tensors per voxel:

- **P_x**: polarization in x-direction, shape `(Nx, Ny, Nz)`
- **P_y**: polarization in y-direction, shape `(Nx, Ny, Nz)`
- **P_z**: polarization in z-direction, shape `(Nx, Ny, Nz)`

### Drude Model

A special case with `ω₀ = 0` (no restoring force):

```python
drude_pole = {
    'omega_p': 1e15,      # plasma frequency
    'gamma': 1e13,        # collision rate
}

material_map.add_drude_sphere(
    center=(x0, y0, z0),
    radius=r,
    omega_p=drude_pole['omega_p'],
    gamma=drude_pole['gamma']
)
```

Drude materials are commonly used for metals (gold, silver) and plasmonic structures.

### Debye Relaxation

For materials with relaxation polarization (e.g., lossy dielectrics):

```python
debye_pole = {
    'epsilon_s': 80.0,    # static permittivity
    'epsilon_inf': 5.0,   # high-frequency limit
    'tau': 1e-11          # relaxation time (s)
}

material_map.add_debye_sphere(
    center=(x0, y0, z0),
    radius=r,
    epsilon_s=debye_pole['epsilon_s'],
    epsilon_inf=debye_pole['epsilon_inf'],
    tau=debye_pole['tau']
)
```

### Auxiliary Field Memory

For a simulation with M Lorentz poles, each with 3 field components:

```
Total auxiliary storage = M × 3 × (Nx, Ny, Nz) × 4 bytes
Example: 2 poles in 256³ grid = 2 × 3 × 268 MB = 1.6 GB
```

## Anisotropic Materials

For materials with direction-dependent permittivity (e.g., crystals, aligned fibers):

### Diagonal Anisotropy

```python
material_map.add_anisotropic_sphere(
    center=(x0, y0, z0),
    radius=r,
    epsilon_tensor_diagonal=(eps_xx, eps_yy, eps_zz),
    sigma_tensor_diagonal=(sig_xx, sig_yy, sig_zz)
)
```

Stores three Ca/Cb-like tensors for each field component. The E-field update becomes:

```
E_x_new = Ca_xx * E_x_old + Cb_xx * (dH_z/dy - dH_y/dz) / Δx
E_y_new = Ca_yy * E_y_old + Cb_yy * (dH_x/dz - dH_z/dx) / Δy
E_z_new = Ca_zz * E_z_old + Cb_zz * (dH_y/dx - dH_x/dy) / Δz
```

Each field component samples its own material coefficient tensor.

**Memory impact**: 3× the storage of isotropic case for anisotropic regions.

## 3D Tissue Phantoms

Realistic biomedical imaging simulations require accurate tissue models derived from patient data or standard anatomical models.

### Voxel-Based Import Workflow

1. **Source data**: MRI/CT scan (typically DICOM format, 0.5–2 mm resolution)
2. **Segmentation**: Classify voxels into tissue types (fat, glandular tissue, tumor)
3. **Property mapping**: Assign frequency-dependent ε(ω) and σ(ω) to each tissue type
4. **Downsampling** (if needed): Coarsen high-resolution data to simulation grid

### Example: Breast Phantom

```python
# Load segmented voxel data (output from segmentation pipeline)
tissue_data = np.load('breast_segmentation.npy')  # shape (256, 256, 128)

# Define tissue properties at simulation frequency
tissue_properties = {
    0: {'eps_r': 1.0, 'sigma': 0.0},           # air
    1: {'eps_r': 5.0, 'sigma': 0.1},           # fatty tissue
    2: {'eps_r': 50.0, 'sigma': 2.0},          # glandular tissue
    3: {'eps_r': 60.0, 'sigma': 3.0},          # tumor
}

# Convert segmentation to permittivity field
eps_r_field = np.zeros_like(tissue_data, dtype=np.float32)
sigma_field = np.zeros_like(tissue_data, dtype=np.float32)

for tissue_id, props in tissue_properties.items():
    mask = tissue_data == tissue_id
    eps_r_field[mask] = props['eps_r']
    sigma_field[mask] = props['sigma']

# Load into MaterialMap
material_map.add_voxel_data(
    eps_r_field,
    origin=(-0.064, -0.064, 0.0),  # in meters
    voxel_size=(0.0005, 0.0005, 0.0005),
    interpolation='linear'
)
```

## API Summary

### MaterialMap Methods (3D Extensions)

| Method | Purpose |
|--------|---------|
| `add_sphere()` | Uniform sphere |
| `add_cylinder()` | Axis-aligned cylinder |
| `add_box()` | Rectangular region |
| `add_ellipsoid()` | Ellipsoidal region |
| `add_voxel_data()` | Import volumetric data |
| `add_lorentz_sphere()` | Dispersive Lorentz material in sphere |
| `add_drude_sphere()` | Drude metal in sphere |
| `add_debye_sphere()` | Debye relaxation in sphere |
| `add_anisotropic_sphere()` | Anisotropic tensor material in sphere |
| `get_ca()` | Return Ca coefficient tensor |
| `get_cb()` | Return Cb coefficient tensor |
| `get_da()` | Return Da coefficient tensor |
| `get_db()` | Return Db coefficient tensor |
| `get_auxiliary_fields()` | Return dict of auxiliary P tensors for dispersive materials |

## Design Considerations

### Per-Voxel vs. Parametric Storage

Storing material properties per voxel (rather than as geometric parameters) provides:

- **Flexibility**: Arbitrary material distributions without shape limitations
- **Data integration**: Direct import from external tools and measurements
- **Simplicity**: Single code path for all geometries

Trade-off: Higher memory footprint for simple geometries (e.g., single sphere).

### Interpolation for Non-Aligned Grids

When importing voxel data, the source and simulation grids may not align:

- **Nearest-neighbor**: Fast, preserves discontinuities
- **Linear interpolation**: Smoother transitions, reduces Gibbs artifacts

Anisotropic materials and dispersive poles (ADE) both require accurate field interpolation to minimize dispersion errors.

### Conductivity and Loss

For lossy materials, conductivity σ modifies the update coefficients:

```
Ca = (2 - σ*Δt/ε) / (2 + σ*Δt/ε)
Cb = 2*Δt / ((2 + σ*Δt/ε)*ε*Δs)
```

Stability condition: `σ*Δt/ε < 1` (enforced at initialization).

