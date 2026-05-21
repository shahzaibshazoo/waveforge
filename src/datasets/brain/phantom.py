"""
phantom.py — 3D anatomical brain phantom with constrained haemorrhage placement.

Builds a multi-shell spherical head model with tissue layers, and provides
methods to place bleeds only in anatomically valid zones.

Anatomy (concentric shells from outside in):
    Scalp → Skull → Dura → CSF → Gray matter → White matter

Bleed placement rules:
    - Epidural: between skull inner surface and dura outer surface
    - Subdural: between dura inner surface and brain (CSF space)
    - Intracerebral: inside gray or white matter

Usage:
    from datasets.brain.phantom import BrainPhantom3D

    phantom = BrainPhantom3D(grid, freq_hz=1e9)
    phantom.add_random_bleed(bleed_type='subdural', age='acute', size='medium')
    Ca, Cb = phantom.build()
"""

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from core.grid import YeeGrid
from core.materials import Material, MaterialMap3D
from .tissue_library import tissue_at_freq, TISSUES, BLOOD_AGING


@dataclass
class BleedConfig:
    """Configuration for a single haemorrhage."""
    bleed_type: str        # 'epidural', 'subdural', 'intracerebral'
    age: str               # 'acute', 'subacute', 'chronic'
    center: tuple[int, int, int]  # (i, j, k) cell coordinates
    radius: int            # radius in cells
    eps_r: float           # computed from Cole-Cole at target freq
    sigma: float           # computed from Cole-Cole at target freq


@dataclass
class PhantomGeometry:
    """Head phantom structural parameters (in cells)."""
    center: tuple[int, int, int]
    scalp_outer_r: int
    skull_outer_r: int
    skull_inner_r: int
    dura_inner_r: int
    csf_inner_r: int
    gray_matter_r: int
    white_matter_r: int


# Standard adult male proportions (Phantom A)
# At 3mm/cell: scalp 6mm, skull 9mm, epidural space 3mm, dura 3mm,
# subdural/CSF 6mm, gray matter shell 21mm, white matter core r=42mm
PHANTOM_A = PhantomGeometry(
    center=(32, 32, 32),
    scalp_outer_r=28,    # 84mm outer radius
    skull_outer_r=26,    # skull thickness = 2 cells (6mm)
    skull_inner_r=23,    # skull inner surface
    dura_inner_r=20,     # dura = 3 cells thick (epidural = skull_inner to dura_outer = 3 cells)
    csf_inner_r=17,      # subdural/CSF space = 3 cells (9mm) — subdural bleeds fit here
    gray_matter_r=17,    # gray matter starts at CSF boundary
    white_matter_r=10,   # white matter inner core
)

# Alternative proportions for test split (Phantom B) — thicker skull, smaller brain
PHANTOM_B = PhantomGeometry(
    center=(32, 32, 32),
    scalp_outer_r=27,    # slightly smaller head
    skull_outer_r=25,
    skull_inner_r=21,    # thicker skull (4 cells = 12mm)
    dura_inner_r=18,
    csf_inner_r=15,      # wider CSF space
    gray_matter_r=15,
    white_matter_r=9,
)


# Bleed size categories (radius in cells)
BLEED_SIZES = {
    'small':  (2, 4),
    'medium': (4, 7),
    'large':  (7, 12),
}


class BrainPhantom3D:
    """3D anatomical brain phantom with frequency-dependent tissue properties.

    Parameters
    ----------
    grid : YeeGrid
        Simulation grid (must be 3D, typically 64x64x64).
    freq_hz : float
        Centre frequency in Hz. All tissue properties are computed at this
        frequency using the 4-pole Cole-Cole model.
    geometry : PhantomGeometry, optional
        Head dimensions. Defaults to PHANTOM_A (adult male).
    seed : int, optional
        Random seed for reproducible bleed placement.
    """

    def __init__(
        self,
        grid: YeeGrid,
        freq_hz: float,
        geometry: Optional[PhantomGeometry] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._grid = grid
        self._freq_hz = freq_hz
        self._geom = geometry or PHANTOM_A
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._bleeds: list[BleedConfig] = []

        # Pre-compute tissue FDTD params at target frequency
        self._tissue_params: dict[str, tuple[float, float]] = {}
        for name in ['scalp', 'skull_cortical', 'csf', 'gray_matter',
                     'white_matter', 'dura']:
            self._tissue_params[name] = tissue_at_freq(name, freq_hz)

    @property
    def freq_hz(self) -> float:
        return self._freq_hz

    @property
    def geometry(self) -> PhantomGeometry:
        return self._geom

    @property
    def bleeds(self) -> list[BleedConfig]:
        return list(self._bleeds)

    @property
    def has_bleed(self) -> bool:
        return len(self._bleeds) > 0

    def _distance_from_center(self, i: int, j: int, k: int) -> float:
        """Euclidean distance from a cell to the phantom centre."""
        cx, cy, cz = self._geom.center
        return math.sqrt((i - cx)**2 + (j - cy)**2 + (k - cz)**2)

    def _get_zone_bounds(self, bleed_type: str) -> tuple[float, float]:
        """Get the (inner_r, outer_r) shell bounds for a bleed zone.

        Anatomy from outside-in:
          skull_outer_r ... skull_inner_r → skull bone (no bleeds)
          skull_inner_r ... dura_inner_r  → epidural space
          dura_inner_r  ... csf_inner_r   → subdural/CSF space
          csf_inner_r (= gray_matter_r)  → brain surface
          0 ... gray_matter_r             → intracerebral
        """
        g = self._geom
        if bleed_type == 'epidural':
            return (g.dura_inner_r, g.skull_inner_r)
        elif bleed_type == 'subdural':
            return (g.csf_inner_r, g.dura_inner_r)
        elif bleed_type == 'intracerebral':
            return (0, g.gray_matter_r)
        else:
            raise ValueError(
                f"bleed_type must be 'epidural', 'subdural', or 'intracerebral', "
                f"got '{bleed_type}'"
            )

    def _is_in_zone(self, i: int, j: int, k: int, bleed_type: str) -> bool:
        """Check if (i,j,k) is a valid centre for a bleed of the given type."""
        d = self._distance_from_center(i, j, k)
        inner, outer = self._get_zone_bounds(bleed_type)
        return inner < d < outer

    def _bleed_fits(self, i: int, j: int, k: int, radius: int,
                    bleed_type: str) -> bool:
        """Check that a bleed doesn't extend into bone.

        Epidural/subdural bleeds are permitted to extend into adjacent soft
        tissue (CSF, brain) but MUST NOT enter bone. Intracerebral bleeds
        must stay inside the brain surface.
        """
        d_center = self._distance_from_center(i, j, k)
        g = self._geom
        if bleed_type == 'intracerebral':
            return d_center + radius <= g.gray_matter_r
        # For epidural/subdural: must not extend outward into bone
        return d_center + radius <= g.skull_inner_r

    def sample_bleed_position(
        self,
        bleed_type: str,
        radius: int,
        max_attempts: int = 1000,
    ) -> Optional[tuple[int, int, int]]:
        """Sample a random valid bleed centre position.

        Parameters
        ----------
        bleed_type : str
            'epidural', 'subdural', or 'intracerebral'.
        radius : int
            Bleed radius in cells.
        max_attempts : int
            Maximum rejection-sampling attempts.

        Returns
        -------
        tuple[int, int, int] or None
            Valid (i, j, k) position, or None if no valid position found.
        """
        cx, cy, cz = self._geom.center
        g = self._geom
        Nx, Ny, Nz = self._grid.Nx, self._grid.Ny, self._grid.Nz

        inner, outer = self._get_zone_bounds(bleed_type)
        if bleed_type == 'intracerebral':
            r_min, r_max = 0, outer - radius
        else:
            # Centre must be in the zone; bleed can extend into adjacent soft tissue
            # but _bleed_fits enforces it doesn't enter bone
            r_min, r_max = inner, outer

        if r_min >= r_max:
            return None

        if r_min >= r_max:
            return None

        for _ in range(max_attempts):
            # Sample a point uniformly within the valid shell
            r = self._rng.uniform(r_min, r_max)
            theta = self._rng.uniform(0, math.pi)
            phi = self._rng.uniform(0, 2 * math.pi)

            i = int(cx + r * math.sin(theta) * math.cos(phi))
            j = int(cy + r * math.sin(theta) * math.sin(phi))
            k = int(cz + r * math.cos(theta))

            # Bounds check
            if not (radius < i < Nx - radius and
                    radius < j < Ny - radius and
                    radius < k < Nz - radius):
                continue

            # Anatomical check
            if self._bleed_fits(i, j, k, radius, bleed_type):
                return (i, j, k)

        return None

    def add_random_bleed(
        self,
        bleed_type: str = 'intracerebral',
        age: str = 'acute',
        size: str = 'medium',
    ) -> Optional[BleedConfig]:
        """Add a randomly placed bleed with anatomical constraints.

        Parameters
        ----------
        bleed_type : str
            'epidural', 'subdural', or 'intracerebral'.
        age : str
            'acute', 'subacute', or 'chronic'.
        size : str
            'small', 'medium', or 'large'.

        Returns
        -------
        BleedConfig or None
            The bleed configuration, or None if placement failed.
        """
        if age not in ('acute', 'subacute', 'chronic'):
            raise ValueError(f"age must be 'acute', 'subacute', or 'chronic', got '{age}'")
        if size not in BLEED_SIZES:
            raise ValueError(f"size must be one of {list(BLEED_SIZES.keys())}, got '{size}'")

        r_min, r_max = BLEED_SIZES[size]
        radius = self._rng.randint(r_min, r_max)

        pos = self.sample_bleed_position(bleed_type, radius)
        if pos is None:
            return None

        eps_r, sigma = tissue_at_freq(age, self._freq_hz)
        bleed = BleedConfig(
            bleed_type=bleed_type,
            age=age,
            center=pos,
            radius=radius,
            eps_r=eps_r,
            sigma=sigma,
        )
        self._bleeds.append(bleed)
        return bleed

    def add_bleed_at(
        self,
        position: tuple[int, int, int],
        radius: int,
        bleed_type: str = 'intracerebral',
        age: str = 'acute',
    ) -> BleedConfig:
        """Add a bleed at a specific position (with validation).

        Raises ValueError if the position is anatomically invalid.
        """
        if not self._bleed_fits(position[0], position[1], position[2],
                                radius, bleed_type):
            raise ValueError(
                f"Position {position} with radius {radius} does not fit in "
                f"'{bleed_type}' zone for the current phantom geometry."
            )

        eps_r, sigma = tissue_at_freq(age, self._freq_hz)
        bleed = BleedConfig(
            bleed_type=bleed_type,
            age=age,
            center=position,
            radius=radius,
            eps_r=eps_r,
            sigma=sigma,
        )
        self._bleeds.append(bleed)
        return bleed

    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build Ca/Cb coefficient tensors for the full phantom.

        Applies tissue layers (painter's algorithm: outside-in) then
        overlays any bleeds on top.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (Ca, Cb) tensors of shape (Nx, Ny, Nz) on grid.device.
        """
        mm = MaterialMap3D(self._grid)
        g = self._geom

        # Tissue layers: outside-in (later overwrites earlier)
        layers = [
            ('scalp', g.scalp_outer_r),
            ('skull_cortical', g.skull_outer_r),
            ('dura', g.skull_inner_r),
            ('csf', g.csf_inner_r),
            ('gray_matter', g.gray_matter_r),
            ('white_matter', g.white_matter_r),
        ]

        for tissue_name, radius in layers:
            eps_r, sigma = self._tissue_params[tissue_name]
            mat = Material(tissue_name, eps_r=eps_r, sigma=sigma)
            mm.add_sphere(center=g.center, radius=radius, material=mat)

        # Overlay bleeds
        for bleed in self._bleeds:
            mat = Material(
                f'bleed_{bleed.bleed_type}_{bleed.age}',
                eps_r=bleed.eps_r,
                sigma=bleed.sigma,
            )
            mm.add_sphere(center=bleed.center, radius=bleed.radius, material=mat)

        return mm.build3d()

    def get_label(self) -> int:
        """Get classification label for this phantom.

        Returns
        -------
        int
            0=healthy, 1=epidural, 2=subdural, 3=intracerebral
        """
        if not self._bleeds:
            return 0
        type_map = {'epidural': 1, 'subdural': 2, 'intracerebral': 3}
        return type_map.get(self._bleeds[0].bleed_type, 0)

    def get_metadata(self) -> dict:
        """Get metadata dict for saving to the dataset .npz."""
        return {
            'freq_hz': self._freq_hz,
            'has_bleed': self.has_bleed,
            'label': self.get_label(),
            'n_bleeds': len(self._bleeds),
            'phantom_type': 'A' if self._geom is PHANTOM_A else 'B',
            'bleeds': [
                {
                    'type': b.bleed_type,
                    'age': b.age,
                    'center_cells': b.center,
                    'radius_cells': b.radius,
                    'center_mm': tuple(c * self._grid.dx * 1e3 for c in b.center),
                    'radius_mm': b.radius * self._grid.dx * 1e3,
                    'eps_r': b.eps_r,
                    'sigma': b.sigma,
                }
                for b in self._bleeds
            ],
            'tissue_params': {
                name: {'eps_r': p[0], 'sigma': p[1]}
                for name, p in self._tissue_params.items()
            },
        }

    def __repr__(self) -> str:
        bleed_str = f', bleeds={len(self._bleeds)}' if self._bleeds else ''
        return (
            f"BrainPhantom3D(freq={self._freq_hz/1e9:.2f}GHz, "
            f"grid={self._grid.Nx}x{self._grid.Ny}x{self._grid.Nz}"
            f"{bleed_str})"
        )
