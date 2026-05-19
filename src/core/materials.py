"""
materials.py — Per-cell material coefficient tensors for GPU-MEEP.

Provides:
  - Material: dataclass holding eps_r, sigma for a single medium
  - TISSUE_LIBRARY: pre-defined brain imaging materials at 1 GHz
  - MaterialMap: builds Ca/Cb coefficient tensors from geometric region assignments
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable

import torch

from .grid import YeeGrid

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

EPS0: float = 8.8541878128e-12   # F/m
MU0:  float = 1.2566370614e-6    # H/m


# ---------------------------------------------------------------------------
# Material dataclass
# ---------------------------------------------------------------------------

@dataclass
class Material:
    """Electromagnetic material properties for a single homogeneous region.

    Parameters
    ----------
    name : str
        Human-readable label.
    eps_r : float
        Relative permittivity (dimensionless, >= 1).
    sigma : float
        Electric conductivity (S/m, >= 0).
    mu_r : float
        Relative permeability (dimensionless). Default 1 (non-magnetic).
    """
    name:  str
    eps_r: float
    sigma: float
    mu_r:  float = 1.0

    def __post_init__(self) -> None:
        if self.eps_r < 1.0:
            raise ValueError(f"eps_r must be >= 1, got {self.eps_r} for '{self.name}'")
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma} for '{self.name}'")
        if self.mu_r <= 0.0:
            raise ValueError(f"mu_r must be > 0, got {self.mu_r} for '{self.name}'")

    def ca(self, dt: float) -> float:
        """Multiplicative E-field decay coefficient (Crank-Nicolson semi-implicit).

        Returns
        -------
        float
            Ca = (1 - alpha) / (1 + alpha),  alpha = sigma*dt / (2*eps0*eps_r).
            Equals 1.0 for lossless media (sigma=0). In (0, 1] for passive media.
        """
        alpha = self.sigma * dt / (2.0 * EPS0 * self.eps_r)
        return (1.0 - alpha) / (1.0 + alpha)

    def cb(self, dt: float) -> float:
        """Curl-to-E scaling coefficient.

        Returns
        -------
        float
            Cb = (dt / (eps0*eps_r)) / (1 + alpha).
        """
        alpha = self.sigma * dt / (2.0 * EPS0 * self.eps_r)
        return (dt / (EPS0 * self.eps_r)) / (1.0 + alpha)

    def skin_depth(self, freq: float) -> float:
        """Approximate skin depth (m) at given frequency (Hz)."""
        if self.sigma == 0.0:
            return float('inf')
        omega = 2.0 * math.pi * freq
        return math.sqrt(2.0 / (omega * MU0 * self.sigma))

    def wavelength(self, freq: float) -> float:
        """Wavelength (m) in this medium at given frequency (Hz)."""
        c0 = 1.0 / math.sqrt(EPS0 * MU0)
        return c0 / (freq * math.sqrt(self.eps_r * self.mu_r))


# ---------------------------------------------------------------------------
# Pre-defined tissue library at 1 GHz
# ---------------------------------------------------------------------------

TISSUE_LIBRARY: dict[str, Material] = {
    'free_space': Material('free_space', eps_r=1.0,  sigma=0.000, mu_r=1.0),
    'skull':      Material('skull',      eps_r=8.0,  sigma=0.100, mu_r=1.0),
    'brain':      Material('brain',      eps_r=40.0, sigma=1.500, mu_r=1.0),
    'tumor':      Material('tumor',      eps_r=55.0, sigma=2.100, mu_r=1.0),
    'blood':      Material('blood',      eps_r=55.0, sigma=2.100, mu_r=1.0),
    'saline':     Material('saline',     eps_r=70.0, sigma=2.000, mu_r=1.0),
}


# ---------------------------------------------------------------------------
# MaterialMap
# ---------------------------------------------------------------------------

class MaterialMap:
    """Assigns materials to grid cells and builds Ca/Cb coefficient tensors.

    Usage
    -----
    .. code-block:: python

        mat_map = MaterialMap(grid, default=TISSUE_LIBRARY['free_space'])
        mat_map.add_circle(center=(75,75), radius=55, material=TISSUE_LIBRARY['skull'])
        mat_map.add_circle(center=(75,75), radius=51, material=TISSUE_LIBRARY['brain'])
        mat_map.add_circle(center=(95,75), radius=6,  material=TISSUE_LIBRARY['tumor'])
        Ca, Cb = mat_map.build()

    Regions are applied in the order they are added. Later regions overwrite earlier ones
    (painter's algorithm), so add larger background regions first, specific features last.
    """

    def __init__(self, grid: YeeGrid, default: Material | None = None) -> None:
        self._grid = grid
        self._default = default or TISSUE_LIBRARY['free_space']
        # Each entry: callable (i, j) -> Material | None (None = skip)
        self._regions: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = []
        self._materials: list[Material] = []

    # ------------------------------------------------------------------
    # Region adders
    # ------------------------------------------------------------------

    def add_circle(self, center: tuple[float, float], radius: float,
                   material: Material) -> 'MaterialMap':
        """Fill a circular region with a material.

        Parameters
        ----------
        center : (ci, cj) in cell indices (float allowed for sub-cell precision).
        radius : float, in cell units.
        material : Material to assign.
        """
        ci, cj = float(center[0]), float(center[1])
        r2 = radius * radius

        def mask(I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
            return ((I - ci) ** 2 + (J - cj) ** 2) <= r2

        self._regions.append(mask)
        self._materials.append(material)
        return self

    def add_rectangle(self, i_range: tuple[int, int], j_range: tuple[int, int],
                      material: Material) -> 'MaterialMap':
        """Fill a rectangular region with a material.

        Parameters
        ----------
        i_range : (i_min, i_max) inclusive cell indices in x.
        j_range : (j_min, j_max) inclusive cell indices in y.
        material : Material to assign.
        """
        i0, i1 = i_range
        j0, j1 = j_range

        def mask(I: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
            return (I >= i0) & (I <= i1) & (J >= j0) & (J <= j1)

        self._regions.append(mask)
        self._materials.append(material)
        return self

    def add_custom(self, mask_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                   material: Material) -> 'MaterialMap':
        """Add a region defined by an arbitrary mask function.

        Parameters
        ----------
        mask_fn : callable (I, J) -> bool tensor, shapes (Nx, Ny).
        material : Material to assign where mask is True.
        """
        self._regions.append(mask_fn)
        self._materials.append(material)
        return self

    # ------------------------------------------------------------------
    # Build coefficient tensors
    # ------------------------------------------------------------------

    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Ca and Cb coefficient tensors from the material assignments.

        Returns
        -------
        (Ca, Cb) : tuple of Tensors, each shape (Nx, Ny, 1), float32, on grid.device.

        Raises
        ------
        ValueError
            If any Ca value is <= 0 (unphysical conductivity or dt).
        """
        Nx, Ny = self._grid.Nx, self._grid.Ny
        dt = self._grid.dt
        device = self._grid.device
        dtype = self._grid.dtype

        # Start with default material everywhere
        ca_arr = torch.full((Nx, Ny), self._default.ca(dt), dtype=torch.float32)
        cb_arr = torch.full((Nx, Ny), self._default.cb(dt), dtype=torch.float32)

        # Build integer index grids on CPU for mask evaluation
        I = torch.arange(Nx, dtype=torch.float32).unsqueeze(1).expand(Nx, Ny)  # (Nx,Ny)
        J = torch.arange(Ny, dtype=torch.float32).unsqueeze(0).expand(Nx, Ny)  # (Nx,Ny)

        # Apply regions in order (painter's algorithm)
        for mask_fn, mat in zip(self._regions, self._materials):
            mask = mask_fn(I, J)  # bool tensor (Nx, Ny)
            ca_arr[mask] = mat.ca(dt)
            cb_arr[mask] = mat.cb(dt)

        # Validate Ca > 0 everywhere
        ca_min = float(ca_arr.min().item())
        if ca_min <= 0.0:
            raise ValueError(
                f"Ca <= 0 detected (min={ca_min:.6f}). "
                "Check material conductivities or reduce dt. "
                "Physical passive media require Ca in (0, 1]."
            )

        # Warn if any Ca is close to 0 (highly lossy)
        if ca_min < 0.5:
            warnings.warn(
                f"Some Ca values are below 0.5 (min={ca_min:.4f}). "
                "Very high conductivity — verify material parameters.",
                UserWarning, stacklevel=2
            )

        # Reshape to (Nx, Ny, 1) and move to device
        Ca = ca_arr.unsqueeze(2).to(device=device, dtype=dtype)
        Cb = cb_arr.unsqueeze(2).to(device=device, dtype=dtype)

        return Ca, Cb

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the material map."""
        dt = self._grid.dt
        lines = [
            f"MaterialMap — grid ({self._grid.Nx}×{self._grid.Ny}), dt={dt:.3e} s",
            f"  Default: {self._default.name} (eps_r={self._default.eps_r}, σ={self._default.sigma})",
        ]
        for i, (mat) in enumerate(self._materials):
            lines.append(
                f"  Region {i}: {mat.name} — eps_r={mat.eps_r}, σ={mat.sigma} S/m, "
                f"Ca={mat.ca(dt):.6f}, Cb={mat.cb(dt):.3e}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"MaterialMap(grid=({self._grid.Nx},{self._grid.Ny}), "
                f"n_regions={len(self._regions)}, default={self._default.name})")
