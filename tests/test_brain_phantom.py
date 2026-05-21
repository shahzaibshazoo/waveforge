"""
test_brain_phantom.py — Tests for the brain phantom and tissue library.

Covers all 5 dataset generation rules:
  Rule 1: Cole-Cole frequency dependence
  Rule 2: Anatomical bleed placement constraints
  Rule 3: Blood aging physics
  Rule 4a: Labels for class balance
  Rule 4b: Independent phantoms
"""

import math
import sys
import pytest
import torch

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

from core.grid import YeeGrid
from datasets.brain import (
    BrainPhantom3D,
    PHANTOM_A,
    PHANTOM_B,
    TISSUES,
    BLOOD_AGING,
    tissue_at_freq,
    all_tissues_at_freq,
    cole_cole_to_fdtd,
    cole_cole_spectrum,
)


@pytest.fixture
def grid():
    return YeeGrid(64, 64, dx=3e-3, dy=3e-3, Nz=64, dz=3e-3, device='cpu')


# ---------------------------------------------------------------------------
# Rule 1: Cole-Cole frequency dependence
# ---------------------------------------------------------------------------

class TestColeColeDispersion:
    def test_gray_matter_changes_with_frequency(self, grid):
        """eps_r and sigma must change with frequency (Rule 1)."""
        eps_1ghz, sig_1ghz = tissue_at_freq('gray_matter', 1e9)
        eps_3ghz, sig_3ghz = tissue_at_freq('gray_matter', 3e9)
        # eps_r decreases with frequency (typical for biological tissue)
        assert eps_1ghz > eps_3ghz, "eps_r should decrease with frequency"
        # sigma increases with frequency
        assert sig_3ghz > sig_1ghz, "sigma should increase with frequency"

    def test_all_tissues_at_1ghz(self):
        """All tissues have physically reasonable values at 1 GHz."""
        values = all_tissues_at_freq(1e9)
        for name, (eps_r, sigma) in values.items():
            assert eps_r >= 1.0, f"{name}: eps_r must be >= 1"
            assert sigma >= 0.0, f"{name}: sigma must be >= 0"
            assert eps_r < 200.0, f"{name}: eps_r={eps_r} unreasonably large"
            assert sigma < 10.0, f"{name}: sigma={sigma} unreasonably large"

    def test_skull_has_lower_eps_than_brain(self):
        """Skull should be less conductive than brain at any frequency."""
        for freq in [0.5e9, 1e9, 2.4e9]:
            skull_eps, _ = tissue_at_freq('skull_cortical', freq)
            brain_eps, _ = tissue_at_freq('gray_matter', freq)
            assert skull_eps < brain_eps, f"Skull eps should be < brain at {freq/1e9} GHz"

    def test_blood_has_high_permittivity(self):
        """Fresh blood has high eps_r (~60) at 1 GHz (Rule 3)."""
        eps_r, sigma = tissue_at_freq('blood', 1e9)
        assert 55 < eps_r < 70, f"Fresh blood eps_r={eps_r} at 1 GHz should be ~60"

    def test_phantom_builds_at_different_frequencies(self, grid):
        """Phantom builds correctly at 500 MHz, 1 GHz, 2.4 GHz."""
        for freq in [0.5e9, 1e9, 2.4e9]:
            p = BrainPhantom3D(grid, freq_hz=freq)
            Ca, Cb = p.build()
            assert Ca.shape == (64, 64, 64)
            assert torch.isfinite(Ca).all()
            assert torch.isfinite(Cb).all()

    def test_spectrum_monotone(self):
        """eps_r should decrease monotonically for brain tissue."""
        freqs, eps_r, sigma = cole_cole_spectrum(TISSUES['gray_matter'], 1e8, 5e9, 50)
        # Not necessarily strictly monotone at every point but overall trend
        assert eps_r[0] > eps_r[-1], "eps_r should decrease from low to high frequency"
        assert sigma[-1] > sigma[0], "sigma should increase from low to high frequency"

    def test_unknown_tissue_raises(self):
        """Requesting unknown tissue raises KeyError."""
        with pytest.raises(KeyError):
            tissue_at_freq('nonexistent_tissue', 1e9)


# ---------------------------------------------------------------------------
# Rule 2: Anatomical bleed placement
# ---------------------------------------------------------------------------

class TestAnatomicalPlacement:
    def test_intracerebral_stays_inside_brain(self, grid):
        """Intracerebral bleed centre + radius must be inside brain."""
        for seed in range(10):
            p = BrainPhantom3D(grid, freq_hz=1e9, seed=seed)
            b = p.add_random_bleed('intracerebral', 'acute', 'small')
            if b is None:
                continue
            cx, cy, cz = PHANTOM_A.center
            d = math.sqrt(sum((b.center[i] - [cx, cy, cz][i])**2 for i in range(3)))
            assert d + b.radius <= PHANTOM_A.gray_matter_r, \
                f"Intracerebral bleed extends outside brain: d={d:.1f}, r={b.radius}"

    def test_bone_placement_rejected(self, grid):
        """Placing bleed inside skull bone should raise ValueError."""
        p = BrainPhantom3D(grid, freq_hz=1e9)
        # Outside head entirely
        with pytest.raises(ValueError):
            p.add_bleed_at((32, 32, 62), radius=3, bleed_type='intracerebral')
        # Inside bone (skull_inner=23 to skull_outer=26, z=32 → r~1 from edge)
        with pytest.raises(ValueError):
            p.add_bleed_at((32, 32, 5), radius=3, bleed_type='intracerebral')

    def test_epidural_placement_succeeds(self, grid):
        """Epidural small bleeds should succeed with multiple seeds."""
        n_ok = sum(
            1 for s in range(30)
            if BrainPhantom3D(grid, freq_hz=1e9, seed=s)
               .add_random_bleed('epidural', 'acute', 'small') is not None
        )
        assert n_ok >= 5, f"Only {n_ok}/30 epidural placements succeeded"

    def test_subdural_placement_succeeds(self, grid):
        """Subdural small bleeds should succeed with multiple seeds."""
        n_ok = sum(
            1 for s in range(30)
            if BrainPhantom3D(grid, freq_hz=1e9, seed=s)
               .add_random_bleed('subdural', 'acute', 'small') is not None
        )
        assert n_ok >= 10, f"Only {n_ok}/30 subdural placements succeeded"

    def test_intracerebral_placement_succeeds(self, grid):
        """Intracerebral bleeds should succeed reliably."""
        n_ok = sum(
            1 for s in range(20)
            if BrainPhantom3D(grid, freq_hz=1e9, seed=s)
               .add_random_bleed('intracerebral', 'acute', 'medium') is not None
        )
        assert n_ok >= 18, f"Only {n_ok}/20 intracerebral placements succeeded"

    def test_bleed_position_in_valid_zone(self, grid):
        """For each successful placement, centre must be in the correct zone."""
        for btype, (inner_r, outer_r) in [
            ('subdural', (PHANTOM_A.csf_inner_r, PHANTOM_A.dura_inner_r)),
            ('intracerebral', (0, PHANTOM_A.gray_matter_r)),
        ]:
            for seed in range(20):
                p = BrainPhantom3D(grid, freq_hz=1e9, seed=seed)
                b = p.add_random_bleed(btype, 'acute', 'small')
                if b is None:
                    continue
                cx, cy, cz = PHANTOM_A.center
                d = math.sqrt(sum((b.center[i]-[cx,cy,cz][i])**2 for i in range(3)))
                # Allow 1 cell tolerance for integer discretisation
                assert inner_r - 1 <= d <= outer_r + 1, \
                    f"{btype} centre d={d:.1f} not in zone [{inner_r},{outer_r}]"

    def test_invalid_bleed_type_raises(self, grid):
        """Requesting invalid bleed type raises ValueError."""
        p = BrainPhantom3D(grid, freq_hz=1e9)
        with pytest.raises(ValueError):
            p.add_random_bleed('epidural_wrong', 'acute', 'small')


# ---------------------------------------------------------------------------
# Rule 3: Blood aging physics
# ---------------------------------------------------------------------------

class TestBloodAging:
    def test_aging_stages_present(self):
        """All three blood aging stages must be in BLOOD_AGING."""
        assert 'acute' in BLOOD_AGING
        assert 'subacute' in BLOOD_AGING
        assert 'chronic' in BLOOD_AGING

    def test_acute_has_higher_eps_than_chronic(self):
        """Fresh blood has higher permittivity than clotted blood."""
        eps_acute, _ = tissue_at_freq('acute', 1e9)
        eps_chronic, _ = tissue_at_freq('chronic', 1e9)
        assert eps_acute > eps_chronic, \
            "Acute blood should have higher eps_r than chronic (clotted)"

    def test_blood_aging_conductivity_decreases(self):
        """Blood conductivity should decrease as it ages (water reabsorption)."""
        _, sig_acute = tissue_at_freq('acute', 1e9)
        _, sig_chronic = tissue_at_freq('chronic', 1e9)
        assert sig_acute > sig_chronic, \
            "Acute blood should have higher sigma than chronic"

    def test_bleed_age_sets_correct_material(self, grid):
        """BleedConfig records the correct eps_r for each age stage."""
        for age in ('acute', 'subacute', 'chronic'):
            p = BrainPhantom3D(grid, freq_hz=1e9, seed=0)
            # Force placement at known valid position
            b = p.add_bleed_at(
                position=(32, 32, 24), radius=2,
                bleed_type='intracerebral', age=age
            )
            expected_eps, _ = tissue_at_freq(age, 1e9)
            assert abs(b.eps_r - expected_eps) < 0.01, \
                f"BleedConfig has wrong eps_r for age={age}"


# ---------------------------------------------------------------------------
# Rule 4a: Class labels
# ---------------------------------------------------------------------------

class TestClassLabels:
    def test_healthy_label_is_zero(self, grid):
        p = BrainPhantom3D(grid, freq_hz=1e9)
        assert p.get_label() == 0

    def test_epidural_label_is_one(self, grid):
        for seed in range(20):
            p = BrainPhantom3D(grid, freq_hz=1e9, seed=seed)
            b = p.add_random_bleed('epidural', 'acute', 'small')
            if b is not None:
                assert p.get_label() == 1
                break

    def test_subdural_label_is_two(self, grid):
        p = BrainPhantom3D(grid, freq_hz=1e9, seed=0)
        p.add_bleed_at((32, 32, 18), radius=2, bleed_type='subdural', age='acute')
        assert p.get_label() == 2

    def test_intracerebral_label_is_three(self, grid):
        p = BrainPhantom3D(grid, freq_hz=1e9, seed=0)
        p.add_bleed_at((32, 32, 24), radius=2, bleed_type='intracerebral', age='acute')
        assert p.get_label() == 3

    def test_metadata_keys_complete(self, grid):
        p = BrainPhantom3D(grid, freq_hz=1e9, seed=0)
        p.add_bleed_at((32, 32, 24), radius=2, bleed_type='intracerebral', age='subacute')
        meta = p.get_metadata()
        for key in ('freq_hz', 'has_bleed', 'label', 'n_bleeds', 'phantom_type',
                    'bleeds', 'tissue_params'):
            assert key in meta, f"Metadata missing key: {key}"
        bleed_meta = meta['bleeds'][0]
        for key in ('type', 'age', 'center_cells', 'radius_cells',
                    'center_mm', 'radius_mm', 'eps_r', 'sigma'):
            assert key in bleed_meta, f"Bleed metadata missing key: {key}"


# ---------------------------------------------------------------------------
# Rule 4b: Independent test phantom
# ---------------------------------------------------------------------------

class TestIndependentPhantom:
    def test_phantom_a_and_b_are_different(self, grid):
        """The two phantom geometries must differ in skull thickness."""
        assert PHANTOM_A.skull_inner_r != PHANTOM_B.skull_inner_r, \
            "PHANTOM_A and PHANTOM_B must have different skull inner radius"
        assert PHANTOM_A.white_matter_r != PHANTOM_B.white_matter_r, \
            "PHANTOM_A and PHANTOM_B must have different white matter core"

    def test_both_phantoms_build_successfully(self, grid):
        """Both phantoms should build Ca/Cb without error."""
        for geom in (PHANTOM_A, PHANTOM_B):
            p = BrainPhantom3D(grid, freq_hz=1e9, geometry=geom)
            Ca, Cb = p.build()
            assert Ca.shape == (64, 64, 64)
            assert Cb.shape == (64, 64, 64)
            assert torch.isfinite(Ca).all()

    def test_phantoms_produce_different_tissue_distributions(self, grid):
        """Different phantoms must produce different Ca tensors."""
        pA = BrainPhantom3D(grid, freq_hz=1e9, geometry=PHANTOM_A)
        pB = BrainPhantom3D(grid, freq_hz=1e9, geometry=PHANTOM_B)
        Ca_A, _ = pA.build()
        Ca_B, _ = pB.build()
        assert not torch.allclose(Ca_A, Ca_B), \
            "PHANTOM_A and PHANTOM_B produced identical Ca tensors"

    def test_phantom_type_recorded_in_metadata(self, grid):
        """Metadata must record which phantom was used."""
        pA = BrainPhantom3D(grid, freq_hz=1e9, geometry=PHANTOM_A)
        assert pA.get_metadata()['phantom_type'] == 'A'

        pB = BrainPhantom3D(grid, freq_hz=1e9, geometry=PHANTOM_B)
        assert pB.get_metadata()['phantom_type'] == 'B'


# ---------------------------------------------------------------------------
# Integration: FDTD simulation with phantom
# ---------------------------------------------------------------------------

class TestFDTDIntegration:
    def test_phantom_ca_cb_run_fdtd(self, grid):
        """Ca/Cb from phantom can drive a full FDTD simulation."""
        from core.fields import FieldSet
        from core.boundaries import MurABC3D
        from core.sources import GaussianPulse, PointSource, SourceCollection
        from core.fdtd3d import FDTD3D

        p = BrainPhantom3D(grid, freq_hz=1e9, seed=42)
        p.add_bleed_at((32, 32, 24), radius=2,
                       bleed_type='intracerebral', age='acute')
        Ca, Cb = p.build()

        fields = FieldSet(grid)
        boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)
        pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)
        src = PointSource(pulse, 5, 32, 'Ez', k=32, grid=grid, N_steps=50)
        sim = FDTD3D(grid, fields, boundary, SourceCollection([src]),
                     Ca=Ca, Cb=Cb, n_check=100)

        with torch.no_grad():
            sim.run(30)

        assert sim.steps_completed == 30
        assert torch.isfinite(fields.Ez).all(), "Ez has NaN/Inf"
        assert torch.isfinite(fields.Hx).all(), "Hx has NaN/Inf"
