"""
test_fdtd2d.py — Comprehensive pytest suite for WaveForge FDTD2D solver.

Covers: grid stability, field initialisation, physics correctness (Faraday /
Ampere sign checks), pulse propagation, Mur ABC absorption, telemetry, and
source injection correctness.

All tests are independent: no shared mutable state leaks between them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import math

import pytest
import torch

from core.grid import YeeGrid, compute_stable_dt
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import (
    GaussianPulse,
    SinusoidalSource,
    RickerWavelet,
    PointSource,
    SourceCollection,
)
from core.fdtd2d import FDTD2D, SimulationDivergedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_grid():
    return YeeGrid(32, 32, dx=1e-3, dy=1e-3, device="cpu")


@pytest.fixture
def small_sim(small_grid):
    fields = FieldSet(small_grid)
    boundary = MurABC(small_grid, fields.Hz)
    sim = FDTD2D(small_grid, fields, boundary, n_check=10)
    return sim


# ---------------------------------------------------------------------------
# Grid tests
# ---------------------------------------------------------------------------


def test_cfl_dt_is_stable(small_grid):
    """dt computed by YeeGrid satisfies CFL: dt <= 1/(c0*sqrt(1/dx^2+1/dy^2))."""
    c0 = 299_792_458.0
    dt_max = 1.0 / (c0 * math.sqrt(1 / small_grid.dx**2 + 1 / small_grid.dy**2))
    # Small tolerance of 0.1 % accommodates floating-point arithmetic in the
    # Courant formula while still catching a genuinely unstable dt.
    assert small_grid.dt <= dt_max * 1.001


def test_grid_coord_staggering(small_grid):
    """E-field coords at half-integer offsets, H-field at integer offsets.

    Coordinate tensors are stored as float32, which has ~7 significant digits.
    For dx=1e-3, float32 rounding on 0.5*dx is ~2.4e-11; we use a tolerance
    of 1e-9 (1 ppm of dx) to be robust to float32 rounding while still
    detecting a mis-implemented stagger.
    """
    # float32 machine epsilon ~1.2e-7; error on 0.5*1e-3 is ~2.4e-11, i.e. ~5e-8
    # relative to dx.  Use 1e-7 * dx as the absolute tolerance.
    tol = 1e-7 * small_grid.dx  # ~1e-10 absolute for dx=1e-3; safe for float32
    # E-field: first position = 0.5 * dx  (half-integer stagger)
    assert abs(float(small_grid.xs[0]) - 0.5 * small_grid.dx) < tol
    # H-field: first position = 0 * dx = 0.0  (integer stagger, exact zero)
    assert abs(float(small_grid.xs_h[0])) < tol


def test_mur_coefficient_stable(small_grid):
    """Mur coefficient must be in (-1, 0) for stable absorption."""
    # MurABC accepts any tensor with the correct shape (Nx, Ny, 1).
    Hz = torch.zeros(32, 32, 1)
    mur = MurABC(small_grid, Hz)
    assert -1.0 < mur.C_mur_x < 0.0
    assert -1.0 < mur.C_mur_y < 0.0


# ---------------------------------------------------------------------------
# Field tests
# ---------------------------------------------------------------------------


def test_fields_initialize_zero(small_grid):
    """All field components start at zero."""
    fields = FieldSet(small_grid)
    for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
        assert getattr(fields, comp).abs().max().item() == 0.0


def test_fields_total_energy_zero(small_grid):
    """Zero fields -> zero total energy."""
    fields = FieldSet(small_grid)
    assert fields.total_energy() == 0.0


def test_fields_total_energy_units(small_grid):
    """Energy = eps0/2 * Ex^2 * Nx*Ny*dV for uniform Ex=1.

    FieldSet uses eps0 = 8.854e-12 F/m (module-level constant).  The 2-D grid
    defaults dz = dx = 1e-3, so cell_volume = 1e-3^3 = 1e-9 m^3.
    """
    fields = FieldSet(small_grid)
    fields.Ex[:] = 1.0
    eps0 = 8.854e-12  # must match fields.py _EPS0 constant
    expected = (
        eps0 / 2 * small_grid.Nx * small_grid.Ny * 1 * small_grid.cell_volume
    )
    assert abs(fields.total_energy() - expected) / expected < 1e-3


# ---------------------------------------------------------------------------
# Stability tests
# ---------------------------------------------------------------------------


def test_free_space_stable_500_steps(small_sim):
    """Free-space simulation with small source stays bounded for 500 steps."""
    g = small_sim.grid
    # Amplitude 0.01 keeps the peak field well below any reasonable threshold.
    src = PointSource(
        GaussianPulse(0.01, sigma=20 * g.dt), i=16, j=16,
        component="Hz", grid=g, N_steps=600,
    )
    small_sim._sources = SourceCollection([src])
    small_sim.run(500)
    assert small_sim.last_field_max < 1e6


def test_divergence_error_raised():
    """SimulationDivergedError raised when fields exceed threshold.

    With n_check=1, the stability check fires after every step.
    Hz is pre-set to 2.0 (> threshold=1.0) so the very first step triggers
    the error in _check_stability().
    """
    g = YeeGrid(8, 8, dx=1e-3, dy=1e-3, device="cpu")
    f = FieldSet(g)
    b = MurABC(g, f.Hz)
    sim = FDTD2D(g, f, b, stability_threshold=1.0, n_check=1)
    f.Hz[:] = 2.0
    with pytest.raises(SimulationDivergedError):
        sim.step()


def test_reset_clears_fields(small_sim):
    """reset() zeros all fields and resets step counter."""
    g = small_sim.grid
    src = PointSource(
        GaussianPulse(1.0, sigma=20 * g.dt), i=16, j=16,
        component="Hz", grid=g, N_steps=200,
    )
    small_sim._sources = SourceCollection([src])
    small_sim.run(50)
    assert small_sim.steps_completed == 50
    small_sim.reset()
    assert small_sim.steps_completed == 0
    assert small_sim.fields.Hz.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Physics correctness tests
# ---------------------------------------------------------------------------


def test_faraday_sign_hz_increases_with_dEx_dy():
    """With Ex[i,j+1]=1 and all else zero, Hz should increase (correct Faraday sign).

    Faraday update:
        Hz[i,j] += Dh * ((Ex[i,j+1] - Ex[i,j]) / dy - (Ey[i+1,j] - Ey[i,j]) / dx)

    Setting Ex[:, 4, 0] = 1.0 means at j=3:
        dEx/dy = (Ex[i,4] - Ex[i,3]) / dy = (1 - 0) / dy > 0
    so Hz[i, 3] must increase.
    """
    g = YeeGrid(8, 8, dx=1e-3, dy=1e-3, device="cpu")
    f = FieldSet(g)
    b = MurABC(g, f.Hz)
    # n_check=1000 prevents stability interruption during the single test step.
    sim = FDTD2D(g, f, b, n_check=1000)
    f.Ex[:, 4, 0] = 1.0
    b.snapshot()
    sim.step()
    assert float(f.Hz[3, 3, 0]) > 0.0, "Hz must increase when dEx/dy > 0"


def test_ampere_ex_increases_with_dHz_dy():
    """With Hz[i,j]=1 and Hz[i,j-1]=0, Ex should increase.

    Ampere-x update:
        Ex[i, j] += De * (Hz[i, j] - Hz[i, j-1]) / dy

    Setting Hz[:, 4, 0] = 1.0 means at j=4:
        dHz/dy = (Hz[i,4] - Hz[i,3]) / dy = (1 - 0) / dy > 0
    so Ex[i, 4] must increase.
    """
    g = YeeGrid(8, 8, dx=1e-3, dy=1e-3, device="cpu")
    f = FieldSet(g)
    b = MurABC(g, f.Hz)
    sim = FDTD2D(g, f, b, n_check=1000)
    f.Hz[:, 4, 0] = 1.0
    b.snapshot()
    sim.step()
    assert float(f.Ex[3, 4, 0]) > 0.0, "Ex must increase when dHz/dy > 0"


def test_ampere_ey_decreases_with_dHz_dx():
    """With Hz[i,j]=1 and Hz[i-1,j]=0, Ey should decrease.

    Ampere-y update:
        Ey[i, j] -= De * (Hz[i, j] - Hz[i-1, j]) / dx

    Setting Hz[4, :, 0] = 1.0 means at i=4:
        dHz/dx = (Hz[4,j] - Hz[3,j]) / dx = (1 - 0) / dx > 0
    so Ey[4, j] must decrease.
    """
    g = YeeGrid(8, 8, dx=1e-3, dy=1e-3, device="cpu")
    f = FieldSet(g)
    b = MurABC(g, f.Hz)
    sim = FDTD2D(g, f, b, n_check=1000)
    f.Hz[4, :, 0] = 1.0
    b.snapshot()
    sim.step()
    assert float(f.Ey[4, 3, 0]) < 0.0, "Ey must decrease when dHz/dx > 0"


def test_energy_increases_with_source():
    """Total EM energy should increase while source is active."""
    g = YeeGrid(32, 32, dx=1e-3, dy=1e-3, device="cpu")
    f = FieldSet(g)
    b = MurABC(g, f.Hz)
    # sigma=30*dt keeps the pulse active well beyond 50 steps
    # (peak at t0=5*sigma=150*dt); N_steps=200 covers the full run.
    src = PointSource(
        GaussianPulse(1.0, sigma=30 * g.dt), i=16, j=16,
        component="Hz", grid=g, N_steps=200,
    )
    sim = FDTD2D(g, f, b, SourceCollection([src]), n_check=1000)
    e0 = f.total_energy()
    sim.run(50)
    assert f.total_energy() > e0, "Energy must increase when source is injecting"


# ---------------------------------------------------------------------------
# Propagation tests
# ---------------------------------------------------------------------------


def test_pulse_propagates_outward():
    """After N steps, energy should exist away from source (propagation occurred)."""
    g = YeeGrid(64, 64, dx=1e-3, dy=1e-3, device="cpu")
    f = FieldSet(g)
    b = MurABC(g, f.Hz)
    src = PointSource(
        GaussianPulse(1.0, sigma=20 * g.dt), i=32, j=32,
        component="Hz", grid=g, N_steps=300,
    )
    sim = FDTD2D(g, f, b, SourceCollection([src]), n_check=1000)
    sim.run(100)
    # After 100 steps the wave front has travelled ~100*dt*c0 cells.
    # Check a ring 18-22 cells left of centre; at least one cell must be non-zero.
    ring_energy = f.Hz[32 - 22:32 - 18, 32, 0].abs().sum().item()
    assert ring_energy > 0.0, "Pulse must have propagated 20 cells from source"


def test_mur_abc_absorbs_outgoing():
    """With Mur ABC, field max should decrease after source stops injecting."""
    g = YeeGrid(64, 64, dx=1e-3, dy=1e-3, device="cpu")
    f = FieldSet(g)
    b = MurABC(g, f.Hz)
    # sigma=15*dt -> peak at t0=75*dt; source effectively done by step ~75.
    # N_steps=600 is ample headroom for both run() calls (80 + 300 = 380 < 600).
    src = PointSource(
        GaussianPulse(1.0, sigma=15 * g.dt), i=32, j=32,
        component="Hz", grid=g, N_steps=600,
    )
    sim = FDTD2D(g, f, b, SourceCollection([src]), n_check=1000)
    sim.run(80)   # source is done; wave fills the grid
    peak_after_source = f.Hz.abs().max().item()
    sim.run(300)  # let wave reach boundaries and be absorbed by Mur ABC
    peak_after_absorption = f.Hz.abs().max().item()
    assert peak_after_absorption < peak_after_source, (
        "Mur ABC must reduce peak field after wave reaches boundary"
    )


# ---------------------------------------------------------------------------
# Telemetry tests
# ---------------------------------------------------------------------------


def test_steps_completed_increments(small_sim):
    """steps_completed increments correctly."""
    small_sim.run(25)
    assert small_sim.steps_completed == 25
    small_sim.run(25)
    assert small_sim.steps_completed == 50


def test_simulation_time_matches_steps(small_sim):
    """sim.time == steps_completed * dt."""
    small_sim.run(100)
    expected_time = 100 * small_sim.grid.dt
    assert abs(small_sim.time - expected_time) < 1e-30


def test_mcells_per_second_positive(small_sim):
    """mcells_per_second > 0 after run()."""
    small_sim.run(50)
    assert small_sim.mcells_per_second > 0.0


# ---------------------------------------------------------------------------
# Source tests
# ---------------------------------------------------------------------------


def test_gaussian_pulse_causal(small_grid):
    """GaussianPulse default is causal (|f(0)|/A < 1e-4).

    Default t0 = 5*sigma.  At t=0: exp(-(5*sigma)^2 / (2*sigma^2)) = exp(-12.5)
    ~ 3.7e-6, which is well below the 1e-4 causality threshold.
    """
    p = GaussianPulse(amplitude=1.0, sigma=1e-10)
    assert p.is_causal(small_grid.dt)


def test_ricker_wavelet_zero_crossings(small_grid):
    """RickerWavelet has both positive and negative values (oscillatory)."""
    rw = RickerWavelet(amplitude=1.0, peak_freq=1e9)
    w = rw.build(1000, small_grid.dt, torch.device("cpu"), torch.float32)
    assert w.max().item() > 0 and w.min().item() < 0


def test_source_injects_at_correct_cell(small_grid):
    """PointSource injects only at specified cell, not neighbours.

    We call src.step() directly (bypassing the FDTD stepper) so only the
    injection logic is exercised.  The waveform index is chosen to be near
    the pulse peak so the injected value is clearly non-zero, while
    N_steps=200 covers both the build() call and the step index used.
    """
    f = FieldSet(small_grid)
    b = MurABC(small_grid, f.Hz)
    waveform = GaussianPulse(1.0, sigma=50 * small_grid.dt)
    src = PointSource(
        waveform, i=10, j=10, component="Hz", grid=small_grid, N_steps=200,
    )
    sim = FDTD2D(small_grid, f, b, SourceCollection([src]), n_check=1000)  # noqa: F841

    # Choose the step closest to the pulse peak, capped at N_steps-1=199
    # so the waveform tensor index is in-bounds.
    n_peak = int(round(waveform.peak_time / small_grid.dt))
    step_idx = min(n_peak, 199)

    fields_dict = {"Ex": f.Ex, "Ey": f.Ey, "Hz": f.Hz}
    src.step(fields_dict, step_idx)

    # The injection cell must be non-zero.
    assert f.Hz[10, 10, 0].item() != 0.0
    # The neighbour cell must still be zero (source only touches [10, 10]).
    assert f.Hz[10, 11, 0].item() == 0.0, "neighbour untouched by injection"
