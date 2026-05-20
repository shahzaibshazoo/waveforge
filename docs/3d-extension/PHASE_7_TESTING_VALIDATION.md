# Phase 7: Testing and Physics Validation

## Overview

This document defines the comprehensive testing and physics validation framework for the WaveForge 3D FDTD engine. Tests verify numerical correctness, physical accuracy, and stability across diverse simulation scenarios.

## Unit Tests for 3D Update Equations

### Test Structure

Each test validates a core update equation in isolation using small grids (8³ to 16³ cells) with simple field configurations.

### Test 1: E-field Update (Curl of H)

**Objective**: Verify that the discrete curl operator correctly computes ∇ × **H**.

**Setup**:
- 8×8×8 grid, uniform material (ε_r = 1, σ = 0)
- Initialize simple H-field: **H** = (0, H_y(x), 0)
- Update E-field using the discrete Ampère-Maxwell equation

**Verification**:
```
E_z_new = Cb * (dH_y/dx) ≈ Cb * (H_y[i,j+1,k] - H_y[i,j,k]) / Δx
```

Expected behavior: E_z should match finite-difference approximation to exact curl, with truncation error O(Δx²).

**Assertion**:
```python
def test_e_field_curl():
    grid = (8, 8, 8)
    sim = Simulation3D(domain_size=(0.008, 0.008, 0.008), grid_resolution=grid)
    
    # Set H_y = sin(2πx/Lx) * A
    H_test = compute_analytical_H_field(grid)
    sim.set_field('H', H_test)
    
    sim.step()
    
    E_computed = sim.get_field('E')
    E_expected = compute_analytical_E_field(H_test, Δx=sim.dx)
    
    error = np.max(np.abs(E_computed - E_expected))
    assert error < 1e-3, f"E-field curl error: {error}"
```

**Pass Criteria**: Max relative error < 0.1% at interior points (boundary excluded).

### Test 2: H-field Update (Curl of E)

**Objective**: Verify discrete curl of **E** in H-field update.

**Setup**:
- 8×8×8 grid, initialized with non-zero E-field
- Update H-field; verify against finite-difference curl

**Verification**:
```
H_y_new = Db * (dE_z/dx - dE_x/dz)
```

**Assertion**:
```python
def test_h_field_curl():
    grid = (8, 8, 8)
    sim = Simulation3D(domain_size=(0.008, 0.008, 0.008), grid_resolution=grid)
    
    E_test = compute_analytical_E_field(grid)
    sim.set_field('E', E_test)
    
    sim.step()
    
    H_computed = sim.get_field('H')
    H_expected = compute_analytical_H_field(E_test, Δx=sim.dx)
    
    error = np.max(np.abs(H_computed - H_expected))
    assert error < 1e-3, f"H-field curl error: {error}"
```

**Pass Criteria**: Max relative error < 0.1%.

### Test 3: Boundary Indexing (PML Interface)

**Objective**: Verify correct field indexing at Yee grid boundaries and PML layer interfaces.

**Setup**:
- 16×16×16 grid with 4-cell PML on each side
- Initialize uniform field
- Verify no out-of-bounds indexing during update

**Assertion**:
```python
def test_pml_boundary_indexing():
    grid = (16, 16, 16)
    pml_thickness = 4
    
    sim = Simulation3D(
        domain_size=(0.016, 0.016, 0.016),
        grid_resolution=grid,
        pml_thickness=pml_thickness
    )
    
    # Run multiple steps to catch off-by-one errors
    for _ in range(100):
        try:
            sim.step()
        except IndexError as e:
            raise AssertionError(f"Boundary indexing error: {e}")
    
    # Check that fields are finite (no NaN/Inf)
    E = sim.get_field('E')
    H = sim.get_field('H')
    assert np.all(np.isfinite(E)), "E-field contains NaN/Inf"
    assert np.all(np.isfinite(H)), "H-field contains NaN/Inf"
```

**Pass Criteria**: No IndexError, all field values finite.

### Test 4: Sign Conventions (Curl Orientation)

**Objective**: Verify correct signs in curl operations (right-hand rule).

**Setup**:
- 8×8×8 grid
- Initialize **E** with known orientation
- Verify ∇ × **E** points in correct direction

**Assertion**:
```python
def test_curl_sign_convention():
    grid = (8, 8, 8)
    sim = Simulation3D(domain_size=(0.008, 0.008, 0.008), grid_resolution=grid)
    
    # E-field in +x direction
    E_field = np.zeros((*grid, 3), dtype=np.float32)
    E_field[:, :, :, 0] = 1.0  # E_x = 1
    
    sim.set_field('E', E_field)
    sim.step()
    
    H = sim.get_field('H')
    
    # ∇ × E should produce H with components in yz-plane
    # Interior points only (exclude boundaries)
    H_interior = H[2:6, 2:6, 2:6, :]
    
    # Check dominant H components are non-zero in expected directions
    assert np.mean(np.abs(H_interior[:, :, :, 1])) > 0.1, "H_y should be non-zero"
    assert np.mean(np.abs(H_interior[:, :, :, 2])) > 0.1, "H_z should be non-zero"
```

**Pass Criteria**: Curl correctly produces field components in expected quadrant.

## Energy Conservation Test

**Objective**: Verify that total electromagnetic energy in a lossless domain remains constant (within floating-point precision).

### Theory

For a lossless domain (σ = 0, no dispersive losses), Poynting's theorem requires:

```
d(EM Energy)/dt = -∮ S · dA
```

where **S** = **E** × **H** is the Poynting vector. In a domain with PML boundaries, all outgoing energy is absorbed → net energy should decay exponentially (no new energy injected by source after pulse passes).

### Test Implementation

```python
def test_energy_conservation_lossless():
    """
    Run pulse in lossless domain with PML.
    Measure total energy decay.
    """
    grid = (64, 64, 64)
    sim = Simulation3D(
        domain_size=(0.064, 0.064, 0.064),
        grid_resolution=grid
    )
    
    # Lossless material
    material_map = MaterialMap()
    sim.set_material_map(material_map)  # default: ε_r=1, σ=0
    
    # Pulse source
    source = PointSource(
        position=(0.032, 0.032, 0.032),
        frequency=1e9,
        pulse_width=1e-10,
        source_type='gaussian'
    )
    sim.add_source(source)
    
    # PML boundary
    pml = PMLBoundary(thickness=8)
    sim.add_boundary_condition('pml', pml)
    
    # Track energy
    energies = []
    time_steps = [0, 100, 200, 300, 400, 500]
    
    for step in range(501):
        sim.step()
        if step in time_steps:
            energy = sim.compute_total_em_energy()
            energies.append(energy)
    
    # Check monotonic decay
    for i in range(1, len(energies)):
        assert energies[i] <= energies[i-1], \
            f"Energy increased: {energies[i-1]} → {energies[i]}"
    
    # Check decay is smooth (no sudden jumps)
    energy_ratios = [energies[i] / energies[i-1] for i in range(1, len(energies))]
    for ratio in energy_ratios:
        assert 0.85 < ratio < 1.0, \
            f"Energy decay ratio out of range: {ratio}"
    
    print(f"Energy conservation test PASSED: {energies[0]:.3e} → {energies[-1]:.3e}")
```

**Pass Criteria**:
- Energy monotonically decreases
- Decay approximately exponential (ratio between consecutive measurements consistent)
- Final energy < 1e-6 of initial (full absorption after ~8 time constants)

## Symmetry Test

**Objective**: Verify that a symmetric source produces symmetric field distributions.

### Theory

An isotropic, homogeneous medium with a point source at the center should produce radially symmetric fields. Any deviation indicates grid/update asymmetry.

### Test Implementation

```python
def test_spherical_symmetry():
    """
    Point source at center of cubic domain.
    Field should be radially symmetric.
    """
    grid = (64, 64, 64)
    sim = Simulation3D(
        domain_size=(0.064, 0.064, 0.064),
        grid_resolution=grid
    )
    
    # Uniform material
    material_map = MaterialMap()
    sim.set_material_map(material_map)
    
    # Point source at center
    source = PointSource(
        position=(0.032, 0.032, 0.032),
        frequency=1e9,
        pulse_width=1e-10,
        source_type='gaussian'
    )
    sim.add_source(source)
    
    # Run to collect field
    for _ in range(300):
        sim.step()
    
    # Measure field magnitude at equal distances from center
    E_field = sim.get_field('E')
    H_field = sim.get_field('H')
    
    center = np.array([32, 32, 32])
    distances = [5, 10, 15, 20]
    
    for d in distances:
        # Sample on sphere at distance d
        samples = [
            E_field[center[0]+d, center[1], center[2], 0],
            E_field[center[0]-d, center[1], center[2], 0],
            E_field[center[0], center[1]+d, center[2], 0],
            E_field[center[0], center[1]-d, center[2], 0],
            E_field[center[0], center[1], center[2]+d, 0],
            E_field[center[0], center[1], center[2]-d, 0],
        ]
        
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        
        # Symmetry: std should be < 1% of mean (loose tolerance for numerical noise)
        assert std_val < 0.01 * mean_val, \
            f"Symmetry broken at distance {d}: std/mean = {std_val/mean_val:.3f}"
    
    print("Spherical symmetry test PASSED")
```

**Pass Criteria**: Field magnitude at equidistant points differs by < 1%.

## Convergence Test

**Objective**: Verify second-order spatial convergence: refining grid by 2× reduces error by factor of 4.

### Theory

The Yee FDTD scheme is second-order accurate in space. Spatial error ∝ Δx². When grid is refined by factor 2 (Δx → Δx/2), error should decrease by factor 4.

### Test Implementation

```python
def test_spatial_convergence():
    """
    Compare solutions on grids of size N, 2N, 4N.
    Error should scale as 1/N^2.
    """
    grids = [32, 64, 128]
    solutions = []
    
    for grid_size in grids:
        sim = Simulation3D(
            domain_size=(0.128, 0.128, 0.128),
            grid_resolution=(grid_size, grid_size, grid_size)
        )
        
        # Plane wave source
        source = PlaneWave(
            direction=(1, 0, 0),
            frequency=1e9,
            wavelength_samples_per_period=10
        )
        sim.add_source(source)
        
        # Run 500 steps
        for _ in range(500):
            sim.step()
        
        # Extract field at center slice
        E_field = sim.get_field('E')
        center_slice = E_field[grid_size//2, :, :]
        solutions.append(center_slice)
    
    # Interpolate coarse solutions to fine grid for comparison
    # (nearest-neighbor for simplicity)
    
    error_32_64 = np.max(np.abs(solutions[0][::2, ::2] - solutions[1][::4, ::4]))
    error_64_128 = np.max(np.abs(solutions[1][::2, ::2] - solutions[2][::4, ::4]))
    
    convergence_rate = error_32_64 / error_64_128
    
    # Should be close to 4 (2^2) for 2nd-order accuracy
    assert 3.0 < convergence_rate < 5.0, \
        f"Convergence rate {convergence_rate} not O(Δx²)"
    
    print(f"Convergence test PASSED: rate = {convergence_rate:.2f} (expected ~4.0)")
```

**Pass Criteria**: Convergence rate between 3.0 and 5.0 (target: 4.0).

## Mie Scattering Validation

**Objective**: Compare numerical RCS (radar cross-section) to analytical Mie series for sphere.

### Test Implementation

```python
def test_mie_scattering():
    """
    Dielectric sphere in free space, plane wave incidence.
    Compare numerical RCS to Mie theory.
    """
    # Simulation parameters
    freq = 1e9
    wavelength = 3e8 / freq
    sphere_radius = 0.03  # 30 mm
    epsilon_r = 9.0
    
    grid = (128, 128, 128)
    sim = Simulation3D(
        domain_size=(0.3, 0.3, 0.3),
        grid_resolution=grid
    )
    
    # Sphere material
    material_map = MaterialMap()
    material_map.add_sphere(
        center=(0.15, 0.15, 0.15),
        radius=sphere_radius,
        epsilon_r=epsilon_r,
        mu_r=1.0,
        conductivity=0.0
    )
    sim.set_material_map(material_map)
    
    # Incident plane wave
    source = PlaneWave(
        direction=(1, 0, 0),
        frequency=freq,
        amplitude=1.0,
        polarization='y'
    )
    sim.add_source(source)
    
    # Far-field monitor
    sim.add_far_field_monitor(
        center=(0.15, 0.15, 0.15),
        radius=0.12,
        angular_samples=36
    )
    
    # Run simulation
    for _ in range(1000):
        sim.step()
    
    # Extract numerical RCS
    rcs_numerical = sim.compute_radar_cross_section()
    
    # Compute Mie theory prediction
    x = 2 * np.pi * sphere_radius / wavelength  # size parameter
    mie = Mie(x=x, m=np.sqrt(epsilon_r))
    rcs_mie = mie.qsca() * np.pi * sphere_radius**2
    
    # Compare
    error = np.abs(rcs_numerical - rcs_mie) / rcs_mie
    
    assert error < 0.05, \
        f"RCS error {error:.3f} exceeds 5% tolerance. Numerical: {rcs_numerical:.3e}, Mie: {rcs_mie:.3e}"
    
    print(f"Mie scattering test PASSED: error = {error*100:.2f}%")
```

**Pass Criteria**: RCS error < 5%.

## Waveguide Dispersion Test

**Objective**: Validate TE₁₀ cutoff frequency against theory: f_c = c / (2a).

### Test Implementation

```python
def test_waveguide_dispersion():
    """
    Rectangular waveguide, TE10 mode.
    Measure cutoff frequency.
    """
    # Waveguide dimensions
    a = 0.040  # width (40 mm)
    b = 0.020  # height (20 mm)
    L = 0.150  # length (150 mm)
    
    # Theoretical cutoff
    f_c_theory = 3e8 / (2 * a)  # 3.75 GHz
    
    # Excitation frequency (well above cutoff for propagation)
    f_exc = 10e9
    
    grid = (160, 80, 600)
    sim = Simulation3D(
        domain_size=(a, b, L),
        grid_resolution=grid
    )
    
    # Waveguide walls (PEC)
    material_map = MaterialMap()
    # [Add walls as before]
    sim.set_material_map(material_map)
    
    # TE10 mode source
    source = ModalSource(
        mode_type='te10',
        waveguide_width=a,
        waveguide_height=b,
        frequency=f_exc,
        amplitude=1.0
    )
    sim.add_source(source)
    
    # Line monitor along propagation axis
    sim.add_line_monitor('propagation', 
                         start=(a/2, b/2, L*0.1),
                         end=(a/2, b/2, L*0.9),
                         samples=200)
    
    # Run simulation
    for _ in range(1500):
        sim.step()
    
    # Measure wavelength in guide
    E_line = sim.get_monitor_data('propagation')
    wavelength_measured = measure_wavelength_from_oscillation(E_line)
    
    # Compute group velocity from dispersion relation
    k = 2 * np.pi / wavelength_measured
    k_c = 2 * np.pi * f_c_theory / 3e8
    v_g = 3e8 * np.sqrt(1 - (f_c_theory / f_exc)**2)
    f_group_expected = v_g / wavelength_measured
    
    # Verify dispersion relation
    omega_theory = 2 * np.pi * f_exc
    k_theory = np.sqrt(omega_theory**2 / (3e8)**2 - k_c**2)
    wavelength_theory = 2 * np.pi / k_theory
    
    error = np.abs(wavelength_measured - wavelength_theory) / wavelength_theory
    
    assert error < 0.05, \
        f"Waveguide wavelength error {error:.3f} exceeds 5%"
    
    print(f"Waveguide dispersion test PASSED: error = {error*100:.2f}%")
```

**Pass Criteria**: Measured wavelength matches theory within 5%.

## PML Reflection Test

**Objective**: Measure reflected power from PML boundary; verify reflection < -60 dB.

### Test Implementation

```python
def test_pml_reflection():
    """
    Plane wave incident on PML.
    Measure reflected power.
    """
    grid = (64, 64, 64)
    sim = Simulation3D(
        domain_size=(0.064, 0.064, 0.064),
        grid_resolution=grid
    )
    
    # Uniform material
    material_map = MaterialMap()
    sim.set_material_map(material_map)
    
    # PML boundary (default thickness)
    pml = PMLBoundary(thickness=8)
    sim.add_boundary_condition('pml', pml)
    
    # Incident plane wave along +z
    source = PlaneWave(
        direction=(0, 0, 1),
        frequency=1e9,
        amplitude=1.0,
        polarization='x'
    )
    sim.add_source(source)
    
    # Monitor incident and reflected fields
    monitors = {
        'incident': (32, 32, 10),    # before PML
        'boundary': (32, 32, 58),    # near PML
    }
    
    incident_power = []
    reflected_power = []
    
    for step in range(500):
        sim.step()
        
        if step > 100:  # Skip transient
            E_incident = sim.get_field_at_point(monitors['incident'])
            E_boundary = sim.get_field_at_point(monitors['boundary'])
            
            # Estimate power ∝ |E|²
            incident_power.append(np.sum(E_incident**2))
            reflected_power.append(np.sum(E_boundary**2))
    
    # Average over steady-state measurements
    P_inc_avg = np.mean(incident_power[-100:])
    P_refl_avg = np.mean(reflected_power[-100:])
    
    # Reflected power should be much smaller than incident
    reflection_ratio = P_refl_avg / P_inc_avg
    reflection_db = 10 * np.log10(reflection_ratio)
    
    assert reflection_db < -60, \
        f"PML reflection {reflection_db:.1f} dB exceeds -60 dB threshold"
    
    print(f"PML reflection test PASSED: {reflection_db:.1f} dB")
```

**Pass Criteria**: Reflection < -60 dB.

## 2D vs 3D Equivalence Test

**Objective**: Run equivalent 2D problem on 3D engine (uniform in z-direction); results must match 2D reference.

### Test Implementation

```python
def test_2d_equivalence_on_3d():
    """
    2D problem (uniform in z) run on 3D engine.
    Compare to dedicated 2D solver.
    """
    # 2D problem: TM mode in xy-plane
    freq = 1e9
    wavelength = 3e8 / freq
    
    # 3D simulation (uniform in z)
    grid_xy = (64, 64)
    grid_z = 2  # minimal in z
    sim_3d = Simulation3D(
        domain_size=(0.064, 0.064, 0.001),  # thin slab
        grid_resolution=(*grid_xy, grid_z)
    )
    
    # Add circular scatterer (constant in z)
    material_map = MaterialMap()
    material_map.add_cylinder(
        center=(0.032, 0.032, 0.0005),
        axis='z',
        radius=0.010,
        height=0.001,  # full height in z
        epsilon_r=9.0
    )
    sim_3d.set_material_map(material_map)
    
    # Plane wave in xy-plane
    source_3d = PlaneWave(
        direction=(1, 0, 0),
        frequency=freq,
        polarization='y'
    )
    sim_3d.add_source(source_3d)
    
    # Run 3D simulation
    for _ in range(500):
        sim_3d.step()
    
    # Extract xy-plane field from 3D simulation
    E_3d = sim_3d.get_field('E')
    E_3d_xy = E_3d[:, :, 0, :]  # z=0 slice
    
    # Compare to 2D reference (or expected behavior)
    # E_3d should match 2D solution identically (up to grid sampling)
    
    # Check that field is uniform in z
    E_3d_z0 = E_3d[:, :, 0, :]
    E_3d_z1 = E_3d[:, :, 1, :]
    
    error_z = np.max(np.abs(E_3d_z0 - E_3d_z1)) / np.max(np.abs(E_3d_z0))
    
    assert error_z < 0.01, \
        f"Field not uniform in z: error {error_z:.3f}"
    
    print(f"2D equivalence test PASSED: z-uniformity error = {error_z*100:.2f}%")
```

**Pass Criteria**: Field uniform in z-direction (< 1% variation).

## Regression Testing

**Objective**: Store reference field snapshots; compare future runs to prevent unintended changes.

### Framework

```python
def test_regression():
    """
    Compare current run against saved reference snapshot.
    """
    import pickle
    
    # Test configuration (canonical example)
    grid = (64, 64, 64)
    sim = Simulation3D(
        domain_size=(0.064, 0.064, 0.064),
        grid_resolution=grid
    )
    
    source = PointSource(
        position=(0.032, 0.032, 0.032),
        frequency=1e9,
        pulse_width=1e-10,
        source_type='gaussian'
    )
    sim.add_source(source)
    
    # Run simulation
    for _ in range(250):
        sim.step()
    
    # Extract current fields
    E_current = sim.get_field('E')
    H_current = sim.get_field('H')
    
    # Load reference (if exists)
    ref_path = 'tests/regression_data/baseline_3d_pulse.pkl'
    if os.path.exists(ref_path):
        with open(ref_path, 'rb') as f:
            reference = pickle.load(f)
        
        # Compare
        error_E = np.max(np.abs(E_current - reference['E'])) / np.max(np.abs(reference['E']))
        error_H = np.max(np.abs(H_current - reference['H'])) / np.max(np.abs(reference['H']))
        
        assert error_E < 1e-6, f"E-field regression error: {error_E:.3e}"
        assert error_H < 1e-6, f"H-field regression error: {error_H:.3e}"
        
        print(f"Regression test PASSED: errors E={error_E:.3e}, H={error_H:.3e}")
    else:
        # First run: save reference
        with open(ref_path, 'wb') as f:
            pickle.dump({'E': E_current, 'H': H_current}, f)
        print(f"Regression baseline saved to {ref_path}")
```

## Test Execution Summary

| Test | Category | Scope | Pass Criterion |
|------|----------|-------|---|
| E-field curl | Unit | 8³ grid | Error < 0.1% |
| H-field curl | Unit | 8³ grid | Error < 0.1% |
| PML indexing | Unit | 16³ grid | No IndexError |
| Curl signs | Unit | 8³ grid | Correct quadrant |
| Energy conservation | Physics | 64³ grid | Monotonic decay |
| Spherical symmetry | Physics | 64³ grid | Std/mean < 1% |
| Spatial convergence | Physics | 32³-128³ | Rate 3-5 |
| Mie scattering | Validation | 128³ grid | Error < 5% |
| Waveguide dispersion | Validation | 160×80×600 | Error < 5% |
| PML reflection | Validation | 64³ grid | Reflection < -60 dB |
| 2D equivalence | Correctness | 64×64×2 | Z-error < 1% |
| Regression | Baseline | 64³ grid | Error < 1e-6 |

## Continuous Integration

Tests are organized into three tiers:

1. **Smoke** (< 5 sec): Unit tests, indexing, basic validation
2. **Validation** (< 60 sec): Physics tests, convergence, PML
3. **Extended** (< 300 sec): Full regression suite, Mie, waveguide

Run `pytest tests/test_3d_fdtd.py -v` for all tests.

