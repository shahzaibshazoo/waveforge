# WaveForge 3D — Phase Transition Criteria

## Phase Completion Gates

Each phase must pass its gate before the next phase begins. This prevents building on incomplete foundations.

---

## Gate 1 → 2: Equations Verified

**From:** Phase 1 (Maxwell 3D documentation)  
**To:** Phase 2 (Engine implementation)

**Criteria:**
- [ ] All 6 update equations written in discrete form
- [ ] Index ranges verified (which cells are updated by each equation)
- [ ] Sign conventions consistent across all 6 equations
- [ ] CFL formula derived and verified for 3D case
- [ ] Material coefficient formulas documented (Ca, Cb, Da, Db)

---

## Gate 2 → 3: Free-Space Engine Works

**From:** Phase 2 (fdtd3d.py core)  
**To:** Phase 3 (Boundaries)

**Criteria:**
- [ ] `FDTD3D` class instantiates without error on 64³ grid
- [ ] Gaussian pulse at center produces expanding spherical wavefront
- [ ] Energy is conserved in interior (away from boundaries) for 100 steps
- [ ] No NaN/Inf after 1000 steps with proper CFL dt
- [ ] Fields show correct symmetry (octant symmetry for centered point source)
- [ ] `FDTD3D` with Nz=1 produces identical results to `FDTD2D`

---

## Gate 3 → 4: Boundaries Absorb

**From:** Phase 3 (MurABC3D / CPML3D)  
**To:** Phase 4 (Sources)

**Criteria:**
- [ ] MurABC3D: outgoing pulse exits domain without visible reflection
- [ ] MurABC3D: reflection amplitude < -15 dB (visual inspection sufficient)
- [ ] CPML3D: reflection amplitude < -60 dB (quantitative test)
- [ ] No instability at edges/corners after 5000 steps
- [ ] Boundary application adds < 10% overhead to step time

---

## Gate 4 → 5: Sources Inject Cleanly

**From:** Phase 4 (3D Sources)  
**To:** Phase 5 (Materials)

**Criteria:**
- [ ] PointSource injects into any of 6 field components correctly
- [ ] PlaneSource creates uniform wavefront propagating in correct direction
- [ ] TFSF box produces clean plane wave inside, zero field outside
- [ ] No source artifacts (no DC offset, no edge diffraction from soft source)
- [ ] All waveforms tested: Gaussian, Ricker, Sinusoidal, ModulatedGaussian

---

## Gate 5 → 6: Materials Scatter

**From:** Phase 5 (3D Materials)  
**To:** Phase 6 (Examples)

**Criteria:**
- [ ] Dielectric sphere scatters incident plane wave (visual)
- [ ] Ca/Cb tensors have correct shape (Nx, Ny, Nz) for any geometry
- [ ] Conductivity causes exponential field decay (skin depth check)
- [ ] Material interfaces don't introduce instability
- [ ] Geometry primitives: sphere, cylinder, box all produce correct shapes
- [ ] Painter's algorithm: later add_*() calls override earlier ones

---

## Gate 6 → 7: Examples Run End-to-End

**From:** Phase 6 (3D Examples)  
**To:** Phase 7 (Validation)

**Criteria:**
- [ ] All example scripts run without error on both CPU and GPU
- [ ] Output PNG files are generated and show physically plausible results
- [ ] No NaN/Inf in any simulation output
- [ ] Computation completes in reasonable time (<5 min for largest example on T4)
- [ ] At least 3 distinct physics scenarios demonstrated

---

## Gate 7 → 8: Physics Validated

**From:** Phase 7 (Testing)  
**To:** Phase 8 (Optimization)

**Criteria:**
- [ ] Energy conservation test passes (lossless case, |ΔU/U| < 1e-5)
- [ ] Spatial convergence is 2nd order (4× error reduction on 2× refinement)
- [ ] Mie scattering RCS agrees with analytical within 1 dB for ka=1
- [ ] TE₁₀ cutoff frequency matches theory within 2%
- [ ] PML reflection < -55 dB (allows some margin from theoretical -60)
- [ ] Symmetry test passes (max asymmetry < 1e-10 relative)

---

## Gate 8 → Release: Performance Meets Targets

**From:** Phase 8 (Optimization)  
**To:** Release / Production

**Criteria:**
- [ ] 128³ grid: ≥ 200 Mcells/s on T4
- [ ] 256³ grid: ≥ 100 Mcells/s on T4
- [ ] Peak VRAM usage documented for all grid sizes
- [ ] No performance regression in 2D examples
- [ ] Benchmark reproducible on Colab/Kaggle T4 instances
- [ ] torch.compile() integration tested and documented
- [ ] Profiling shows no obvious remaining bottlenecks (>80% of peak memory bandwidth utilized)

---

## Failure Recovery

If a gate check fails:

1. **Identify root cause** — Is it a sign error? Index error? Memory layout issue?
2. **Fix in the current phase** — Do not proceed to next phase
3. **Re-run ALL gate checks** — A fix may break something else
4. **Document the failure** — What went wrong, why, how it was fixed (in git commit messages)

## Parallel Work

Phases 3, 4, and 5 are partially independent once Phase 2 is complete:
- Phase 3 (Boundaries) only depends on the field tensor interface
- Phase 4 (Sources) only depends on the field tensor interface  
- Phase 5 (Materials) only depends on the Ca/Cb coefficient interface

These three can proceed in parallel after Phase 2's gate is passed.
