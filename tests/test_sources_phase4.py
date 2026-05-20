"""
test_sources_phase4.py — Tests for Phase 4 source additions.

Covers ModulatedGaussian, Chirp, and PlaneSource, including
construction, validation, waveform shape, injection correctness,
and integration with SourceCollection and FDTD3D.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.sources import (
    ModulatedGaussian,
    Chirp,
    PlaneSource,
    GaussianPulse,
    SourceCollection,
    PointSource,
    VALID_COMPONENTS,
)
from core.grid import YeeGrid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def make_grid(N: int = 16, dx: float = 1e-3) -> YeeGrid:
    """Return a 3-D YeeGrid with N^3 cells at uniform spacing dx."""
    return YeeGrid(N, N, dx=dx, dy=dx, Nz=N, dz=dx, device=DEVICE)


def _zero_fields(Nx: int, Ny: int, Nz: int) -> dict[str, torch.Tensor]:
    """Return a dict of six zero field tensors with shape (Nx, Ny, Nz)."""
    return {
        comp: torch.zeros(Nx, Ny, Nz, device=DEVICE)
        for comp in VALID_COMPONENTS
    }


# ===========================================================================
# ModulatedGaussian tests
# ===========================================================================


class TestModulatedGaussian:
    """Tests for ModulatedGaussian(Waveform)."""

    # ------------------------------------------------------------------
    # 1. Smoke — basic construction
    # ------------------------------------------------------------------

    def test_modulated_gaussian_smoke(self):
        """should verify peak_time == t0, amplitude, and bandwidth == 1/(2*pi*sigma)."""
        fc = 2.4e9
        sigma = 5e-10
        t0 = 5 * sigma
        amp = 3.7

        src = ModulatedGaussian(amplitude=amp, fc=fc, sigma=sigma, t0=t0)

        assert src.peak_time == pytest.approx(t0)
        assert src.amplitude == pytest.approx(amp)
        expected_bw = 1.0 / (2.0 * math.pi * sigma)
        assert src.bandwidth == pytest.approx(expected_bw, rel=1e-5)

    # ------------------------------------------------------------------
    # 2. Default t0
    # ------------------------------------------------------------------

    def test_modulated_gaussian_default_t0(self):
        """should default t0 to 5*sigma when t0 is not supplied."""
        sigma = 3e-10
        src = ModulatedGaussian(amplitude=1.0, fc=1e9, sigma=sigma)

        assert src.peak_time == pytest.approx(5.0 * sigma, rel=1e-6)

    # ------------------------------------------------------------------
    # 3. Causality — default params with phi=0
    # ------------------------------------------------------------------

    def test_modulated_gaussian_is_causal_default(self):
        """should return True from is_causal when phi=0 and t0=5*sigma."""
        sigma = 5e-10
        src = ModulatedGaussian(amplitude=1.0, fc=2e9, sigma=sigma, phi=0.0)

        assert src.is_causal(dt=1e-12) is True

    # ------------------------------------------------------------------
    # 4. Causality warning when phi != 0 and t0 is small
    # ------------------------------------------------------------------

    def test_modulated_gaussian_causality_warning(self):
        """should emit UserWarning when phi=pi/2 and t0=sigma (carrier not zero at t=0)."""
        sigma = 1e-10
        # phi=pi/2 means cos(2*pi*fc*0 + pi/2) = cos(pi/2) = 0, but
        # sin(2*pi*fc*0 + pi/2) = 1, so f(0) = A * envelope(0) * 1.
        # With t0=sigma the Gaussian envelope at t=0 is exp(-0.5) ≈ 0.607,
        # which is well above the 1e-4 causality threshold.
        with pytest.warns(UserWarning, match="causal"):
            ModulatedGaussian(
                amplitude=1.0,
                fc=2e9,
                sigma=sigma,
                t0=sigma,       # intentionally small → large f(0)
                phi=math.pi / 2,
            )

    # ------------------------------------------------------------------
    # 5. Build tensor shape and magnitude
    # ------------------------------------------------------------------

    def test_modulated_gaussian_build_tensor(self):
        """should produce shape (N,), all finite, peak abs value within 2x amplitude."""
        N = 200
        dt = 1e-12
        sigma = 40 * dt
        amp = 2.5

        src = ModulatedGaussian(amplitude=amp, fc=5e9, sigma=sigma)
        wave = src.build(N_steps=N, dt=dt, device=DEVICE)

        assert wave.shape == (N,)
        assert torch.isfinite(wave).all()
        # Peak should be bounded by amplitude (envelope peak) times a cosine
        # factor — not exceed 2x amplitude
        assert float(wave.abs().max()) <= 2.0 * abs(amp)

    # ------------------------------------------------------------------
    # 6. Invalid fc
    # ------------------------------------------------------------------

    def test_modulated_gaussian_invalid_fc_zero(self):
        """should raise ValueError when fc=0."""
        with pytest.raises(ValueError, match="fc"):
            ModulatedGaussian(amplitude=1.0, fc=0, sigma=1e-10)

    def test_modulated_gaussian_invalid_fc_negative(self):
        """should raise ValueError when fc is negative."""
        with pytest.raises(ValueError, match="fc"):
            ModulatedGaussian(amplitude=1.0, fc=-1e9, sigma=1e-10)

    # ------------------------------------------------------------------
    # 7. Invalid sigma
    # ------------------------------------------------------------------

    def test_modulated_gaussian_invalid_sigma_none(self):
        """should raise ValueError when sigma is None (not supplied)."""
        with pytest.raises((ValueError, TypeError)):
            ModulatedGaussian(amplitude=1.0, fc=1e9, sigma=None)

    def test_modulated_gaussian_invalid_sigma_zero(self):
        """should raise ValueError when sigma=0."""
        with pytest.raises(ValueError, match="sigma"):
            ModulatedGaussian(amplitude=1.0, fc=1e9, sigma=0)

    def test_modulated_gaussian_invalid_sigma_negative(self):
        """should raise ValueError when sigma is negative."""
        with pytest.raises(ValueError, match="sigma"):
            ModulatedGaussian(amplitude=1.0, fc=1e9, sigma=-1e-10)


# ===========================================================================
# Chirp tests
# ===========================================================================


class TestChirp:
    """Tests for Chirp(Waveform)."""

    # ------------------------------------------------------------------
    # 1. Smoke — basic construction
    # ------------------------------------------------------------------

    def test_chirp_smoke(self):
        """should verify peak_time, bandwidth, and amplitude on a well-formed Chirp."""
        f_start = 1e9
        f_end = 5e9
        t_start = 1e-9
        t_end = 6e-9
        amp = 1.5

        src = Chirp(amplitude=amp, f_start=f_start, f_end=f_end,
                    t_start=t_start, t_end=t_end)

        expected_peak = (t_start + t_end) / 2.0
        assert src.peak_time == pytest.approx(expected_peak, rel=1e-6)
        assert src.bandwidth == pytest.approx(abs(f_end - f_start), rel=1e-6)
        assert src.amplitude == pytest.approx(amp)

    # ------------------------------------------------------------------
    # 2. Default t_end
    # ------------------------------------------------------------------

    def test_chirp_default_t_end(self):
        """should default t_end to t_start + 10/f_start when t_end is not supplied."""
        f_start = 1e9
        t_start = 2e-9

        src = Chirp(amplitude=1.0, f_start=f_start, f_end=3e9, t_start=t_start)

        expected_t_end = t_start + 10.0 / f_start
        # Access through peak_time which equals (t_start + t_end) / 2
        expected_peak = (t_start + expected_t_end) / 2.0
        assert src.peak_time == pytest.approx(expected_peak, rel=1e-5)

    # ------------------------------------------------------------------
    # 3. Chirp is always causal
    # ------------------------------------------------------------------

    def test_chirp_is_always_causal(self):
        """should return True from is_causal regardless of params (window zeroes t=0)."""
        src = Chirp(amplitude=1.0, f_start=1e9, f_end=4e9,
                    t_start=5e-9, t_end=20e-9)

        assert src.is_causal(dt=1e-12) is True

    # ------------------------------------------------------------------
    # 4. Zero outside sweep window
    # ------------------------------------------------------------------

    def test_chirp_zero_outside_window(self):
        """should produce ~0 values for t < t_start and t > t_end."""
        dt = 1e-12
        t_start = 10e-12   # starts at step 10
        t_end = 90e-12     # ends at step 90
        N = 120

        src = Chirp(amplitude=1.0, f_start=1e9, f_end=4e9,
                    t_start=t_start, t_end=t_end)
        wave = src.build(N_steps=N, dt=dt, device=DEVICE)

        # Steps 0..9 are before t_start — torch.where guarantees bit-exact zero
        pre_window = wave[:9]
        assert float(pre_window.abs().max()) == 0.0, (
            f"Pre-window must be exactly zero, got {float(pre_window.abs().max()):.3e}"
        )

        # Steps 92..119 are after t_end — torch.where guarantees bit-exact zero
        post_window = wave[92:]
        assert float(post_window.abs().max()) == 0.0, (
            f"Post-window must be exactly zero, got {float(post_window.abs().max()):.3e}"
        )

    # ------------------------------------------------------------------
    # 5. Hann window zeroes exactly at endpoints
    # ------------------------------------------------------------------

    def test_chirp_hann_zero_at_endpoints(self):
        """should evaluate to exactly 0 at t=t_start and t=t_end (Hann endpoints)."""
        # Use a fine time axis so the start/end samples are effectively
        # captured.  Build at dt=1ps, window [1ns, 3ns], N=4001 steps →
        # step 1000 is t=1ns (t_start), step 3000 is t=3ns (t_end).
        dt = 1e-12
        t_start = 1e-9
        t_end = 3e-9
        N = 4001

        src = Chirp(amplitude=2.0, f_start=0.5e9, f_end=2e9,
                    t_start=t_start, t_end=t_end)
        wave = src.build(N_steps=N, dt=dt, device=DEVICE)

        # Sample at exact start index
        start_idx = round(t_start / dt)   # 1000
        end_idx = round(t_end / dt)       # 3000

        val_start = float(wave[start_idx].item())
        val_end = float(wave[end_idx].item())

        # Hann window is w(0) = 0.5*(1 - cos(0)) = 0 and w(1) = 0.5*(1 - cos(2pi)) = 0
        assert abs(val_start) < 1e-5, (
            f"Chirp must be 0 at t=t_start, got {val_start:.4e}"
        )
        assert abs(val_end) < 1e-5, (
            f"Chirp must be 0 at t=t_end, got {val_end:.4e}"
        )

    # ------------------------------------------------------------------
    # 6. Degenerate CW (f_start == f_end, bandwidth == 0)
    # ------------------------------------------------------------------

    def test_chirp_degenerate_cw(self):
        """should build without error when f_start == f_end (zero bandwidth)."""
        src = Chirp(amplitude=1.0, f_start=2e9, f_end=2e9,
                    t_start=1e-9, t_end=5e-9)

        assert src.bandwidth == pytest.approx(0.0)

        wave = src.build(N_steps=100, dt=1e-12, device=DEVICE)
        assert wave.shape == (100,)
        assert torch.isfinite(wave).all()

    # ------------------------------------------------------------------
    # 7. Invalid arguments
    # ------------------------------------------------------------------

    def test_chirp_invalid_amplitude_zero(self):
        """should raise ValueError when amplitude=0."""
        with pytest.raises(ValueError, match="amplitude"):
            Chirp(amplitude=0.0, f_start=1e9, f_end=3e9,
                  t_start=1e-9, t_end=5e-9)

    def test_chirp_invalid_f_start_zero(self):
        """should raise ValueError when f_start=0."""
        with pytest.raises(ValueError, match="f_start"):
            Chirp(amplitude=1.0, f_start=0.0, f_end=3e9,
                  t_start=1e-9, t_end=5e-9)

    def test_chirp_invalid_f_end_negative(self):
        """should raise ValueError when f_end is negative."""
        with pytest.raises(ValueError, match="f_end"):
            Chirp(amplitude=1.0, f_start=1e9, f_end=-1.0,
                  t_start=1e-9, t_end=5e-9)

    def test_chirp_invalid_t_start_negative(self):
        """should raise ValueError when t_start is negative."""
        with pytest.raises(ValueError, match="t_start"):
            Chirp(amplitude=1.0, f_start=1e9, f_end=3e9,
                  t_start=-1e-9, t_end=5e-9)

    def test_chirp_invalid_t_end_le_t_start(self):
        """should raise ValueError when t_end <= t_start."""
        with pytest.raises(ValueError, match="t_end"):
            Chirp(amplitude=1.0, f_start=1e9, f_end=3e9,
                  t_start=5e-9, t_end=5e-9)


# ===========================================================================
# PlaneSource tests
# ===========================================================================


class TestPlaneSource:
    """Tests for PlaneSource — uniform injection on a full Yee-grid plane."""

    # ------------------------------------------------------------------
    # 1. Smoke xy
    # ------------------------------------------------------------------

    def test_planesource_xy_smoke(self):
        """should construct on 'xy' plane and expose correct component, plane, position."""
        grid = make_grid(16)
        k0 = grid.Nz // 2
        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)

        src = PlaneSource(
            waveform=pulse,
            plane="xy",
            position=k0,
            component="Ez",
            grid=grid,
            N_steps=100,
        )

        assert src.component == "Ez"
        assert src.plane == "xy"
        assert src.position == k0

    # ------------------------------------------------------------------
    # 2. Smoke xz
    # ------------------------------------------------------------------

    def test_planesource_xz_smoke(self):
        """should construct on 'xz' plane with Ey component."""
        grid = make_grid(16)
        j0 = grid.Ny // 2
        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)

        src = PlaneSource(
            waveform=pulse,
            plane="xz",
            position=j0,
            component="Ey",
            grid=grid,
            N_steps=100,
        )

        assert src.component == "Ey"
        assert src.plane == "xz"
        assert src.position == j0

    # ------------------------------------------------------------------
    # 3. Smoke yz
    # ------------------------------------------------------------------

    def test_planesource_yz_smoke(self):
        """should construct on 'yz' plane with Ex component."""
        grid = make_grid(16)
        i0 = grid.Nx // 2
        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)

        src = PlaneSource(
            waveform=pulse,
            plane="yz",
            position=i0,
            component="Ex",
            grid=grid,
            N_steps=100,
        )

        assert src.component == "Ex"
        assert src.plane == "yz"
        assert src.position == i0

    # ------------------------------------------------------------------
    # 4. n_cells
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("plane,expected_fn", [
        ("xy", lambda g: g.Nx * g.Ny),
        ("xz", lambda g: g.Nx * g.Nz),
        ("yz", lambda g: g.Ny * g.Nz),
    ])
    def test_planesource_n_cells(self, plane: str, expected_fn):
        """should report n_cells equal to the product of the two in-plane dimensions."""
        grid = make_grid(16)
        comp_map = {"xy": "Ez", "xz": "Ey", "yz": "Ex"}
        pos_map = {"xy": grid.Nz // 2, "xz": grid.Ny // 2, "yz": grid.Nx // 2}

        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)
        src = PlaneSource(
            waveform=pulse,
            plane=plane,
            position=pos_map[plane],
            component=comp_map[plane],
            grid=grid,
            N_steps=100,
        )

        assert src.n_cells == expected_fn(grid)

    # ------------------------------------------------------------------
    # 5. Injection correctness — xy plane
    # ------------------------------------------------------------------

    def test_planesource_injects_correct_plane(self):
        """should inject non-zero values only into Ez[..., :, :, k0] for xy plane."""
        N = 16
        grid = make_grid(N)
        k0 = N // 2
        N_steps = 100

        # Choose step 50 — pulse is near its peak
        sigma = 40 * grid.dt
        pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
        src = PlaneSource(
            waveform=pulse,
            plane="xy",
            position=k0,
            component="Ez",
            grid=grid,
            N_steps=N_steps,
        )

        fields = _zero_fields(N, N, N)
        src.step(fields, n=50)

        Ez = fields["Ez"]

        # The injected k-slice must be uniformly non-zero
        injected_slice = Ez[:, :, k0]
        assert float(injected_slice.abs().max()) > 0.0, (
            "Ez at k=k0 must be non-zero after injection"
        )

        # Every cell in the plane must have the same value (broadcast scalar)
        first_val = float(injected_slice[0, 0])
        assert torch.allclose(
            injected_slice,
            torch.full_like(injected_slice, first_val),
        ), "All cells in the injected xy plane must have equal value"

        # All other z-slices must still be zero
        for k in range(N):
            if k == k0:
                continue
            slice_k = Ez[:, :, k]
            assert float(slice_k.abs().max()) == 0.0, (
                f"Ez at k={k} must remain zero (only k0={k0} was injected)"
            )

    # ------------------------------------------------------------------
    # 6. Injection correctness — xz plane
    # ------------------------------------------------------------------

    def test_planesource_injects_xz_plane(self):
        """should inject non-zero values only into Ey[:, j0, :] for xz plane."""
        N = 16
        grid = make_grid(N)
        j0 = N // 2
        N_steps = 100

        sigma = 40 * grid.dt
        pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
        src = PlaneSource(
            waveform=pulse,
            plane="xz",
            position=j0,
            component="Ey",
            grid=grid,
            N_steps=N_steps,
        )

        fields = _zero_fields(N, N, N)
        src.step(fields, n=50)

        Ey = fields["Ey"]

        # The injected j-slice must be non-zero
        assert float(Ey[:, j0, :].abs().max()) > 0.0, (
            "Ey at j=j0 must be non-zero after xz injection"
        )

        # All other j-slices must remain exactly zero
        for j in range(N):
            if j == j0:
                continue
            assert float(Ey[:, j, :].abs().max()) == 0.0, (
                f"Ey at j={j} must remain zero after xz injection"
            )

    # ------------------------------------------------------------------
    # 6b. Injection correctness — yz plane
    # ------------------------------------------------------------------

    def test_planesource_injects_yz_plane(self):
        """should inject non-zero values only into Ex[i0, :, :] for yz plane."""
        N = 16
        grid = make_grid(N)
        i0 = N // 2
        N_steps = 100

        sigma = 40 * grid.dt
        pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
        src = PlaneSource(
            waveform=pulse,
            plane="yz",
            position=i0,
            component="Ex",
            grid=grid,
            N_steps=N_steps,
        )

        fields = _zero_fields(N, N, N)
        src.step(fields, n=50)

        Ex = fields["Ex"]

        assert float(Ex[i0, :, :].abs().max()) > 0.0, (
            "Ex at i=i0 must be non-zero after yz injection"
        )

        for i in range(N):
            if i == i0:
                continue
            assert float(Ex[i, :, :].abs().max()) == 0.0, (
                f"Ex at i={i} must remain zero after yz injection"
            )

    # ------------------------------------------------------------------
    # 7. Additive (soft source) injection
    # ------------------------------------------------------------------

    def test_planesource_additive(self):
        """should double the field when step is called twice at the same n."""
        N = 16
        grid = make_grid(N)
        k0 = N // 2
        N_steps = 100

        sigma = 40 * grid.dt
        pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
        src = PlaneSource(
            waveform=pulse,
            plane="xy",
            position=k0,
            component="Ez",
            grid=grid,
            N_steps=N_steps,
        )

        fields = _zero_fields(N, N, N)

        # First injection at n=50
        src.step(fields, n=50)
        val_single = float(fields["Ez"][:, :, k0].abs().max())

        # Second injection at the same n=50
        src.step(fields, n=50)
        val_double = float(fields["Ez"][:, :, k0].abs().max())

        assert val_double == pytest.approx(2.0 * val_single, rel=1e-5), (
            "Soft source must add; two calls must double the field"
        )

    # ------------------------------------------------------------------
    # 8. Invalid plane string
    # ------------------------------------------------------------------

    def test_planesource_invalid_plane(self):
        """should raise ValueError when plane is not 'xy', 'xz', or 'yz'."""
        grid = make_grid(16)
        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)

        with pytest.raises(ValueError, match="plane"):
            PlaneSource(
                waveform=pulse,
                plane="xy_bad",
                position=8,
                component="Ez",
                grid=grid,
                N_steps=100,
            )

    # ------------------------------------------------------------------
    # 9. Position out of range
    # ------------------------------------------------------------------

    def test_planesource_position_out_of_range_high(self):
        """should raise ValueError when position equals Nz for 'xy' plane."""
        grid = make_grid(16)
        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)

        with pytest.raises(ValueError, match="position"):
            PlaneSource(
                waveform=pulse,
                plane="xy",
                position=grid.Nz,   # out of range: valid range is [0, Nz)
                component="Ez",
                grid=grid,
                N_steps=100,
            )

    def test_planesource_position_out_of_range_negative(self):
        """should raise ValueError when position=-1 for any plane."""
        grid = make_grid(16)
        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)

        with pytest.raises(ValueError, match="position"):
            PlaneSource(
                waveform=pulse,
                plane="xy",
                position=-1,
                component="Ez",
                grid=grid,
                N_steps=100,
            )

    # ------------------------------------------------------------------
    # 10. Integration with SourceCollection
    # ------------------------------------------------------------------

    def test_planesource_in_source_collection(self):
        """should work alongside PointSource in a SourceCollection and report combined n_cells."""
        N = 16
        grid = make_grid(N)
        N_steps = 100

        sigma = 40 * grid.dt
        pulse_plane = GaussianPulse(amplitude=1.0, sigma=sigma)
        plane_src = PlaneSource(
            waveform=pulse_plane,
            plane="xy",
            position=N // 2,
            component="Ez",
            grid=grid,
            N_steps=N_steps,
        )

        pulse_pt = GaussianPulse(amplitude=1.0, sigma=sigma)
        pt_src = PointSource(
            waveform=pulse_pt,
            i=4, j=4,
            component="Ez",
            k=4,
            grid=grid,
            N_steps=N_steps,
        )

        collection = SourceCollection([plane_src, pt_src])

        # n_injection_cells_total = Nx*Ny + 1
        expected_total = N * N + 1
        assert collection.n_injection_cells_total == expected_total, (
            f"Expected {expected_total} total injection cells, "
            f"got {collection.n_injection_cells_total}"
        )

        fields = _zero_fields(N, N, N)
        collection.step(fields, n=50)

        # Both Ez planes should have been modified
        plane_val = float(fields["Ez"][:, :, N // 2].abs().max())
        assert plane_val > 0.0, "PlaneSource injection must produce non-zero Ez"

        pt_val = float(fields["Ez"][4, 4, 4])
        assert pt_val != 0.0, "PointSource injection must produce non-zero Ez at (4,4,4)"


# ===========================================================================
# Integration test — PlaneSource inside a full FDTD3D simulation
# ===========================================================================


class TestPlaneSourceFDTD3DIntegration:
    """End-to-end integration: PlaneSource drives a real 3-D FDTD simulation."""

    def test_planesource_3d_sim_runs(self):
        """should produce finite, non-zero Ez at z=8 after 50 FDTD steps."""
        from core.fields import FieldSet
        from core.fdtd3d import FDTD3D
        from core.boundaries import MurABC3D

        N = 16
        N_steps = 50
        grid = YeeGrid(N, N, dx=1e-3, dy=1e-3, Nz=N, dz=1e-3, device=DEVICE)

        fields = FieldSet(grid)
        boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

        # Plane at z=8 injects Ez uniformly
        k0 = 8
        sigma = 20 * grid.dt
        pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
        src = PlaneSource(
            waveform=pulse,
            plane="xy",
            position=k0,
            component="Ez",
            grid=grid,
            N_steps=N_steps,
        )
        sources = SourceCollection([src])

        sim = FDTD3D(grid, fields, boundary, sources)
        sim.run(N_steps, verbose=False)

        Ez = fields.Ez

        # Fields must be globally finite
        assert torch.isfinite(Ez).all(), "Ez must be finite after 50 steps"
        assert torch.isfinite(fields.Hx).all(), "Hx must be finite after 50 steps"
        assert torch.isfinite(fields.Hy).all(), "Hy must be finite after 50 steps"
        assert torch.isfinite(fields.Hz).all(), "Hz must be finite after 50 steps"

        # The z=k0 plane must have non-zero Ez (source fired)
        ez_at_source_plane = Ez[:, :, k0]
        assert float(ez_at_source_plane.abs().max()) > 0.0, (
            "Ez at the source plane z=k0 must be non-zero after 50 steps"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
