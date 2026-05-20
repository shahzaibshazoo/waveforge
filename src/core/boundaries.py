"""
boundaries.py — Absorbing boundary conditions for WaveForge.

Provides:
  - _compute_pml_coeffs: helper for polynomial-graded PML coefficient arrays
  - MurABC: first-order Mur absorbing boundary condition for 2D TM fields,
    with optional PML scaffold coefficient storage
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from .grid import YeeGrid

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

C0: float = 299_792_458.0       # speed of light in vacuum, m/s
EPS0: float = 8.8541878128e-12  # permittivity of free space, F/m
ETA0: float = 376.730313        # impedance of free space, Ohm


# ---------------------------------------------------------------------------
# PML helper
# ---------------------------------------------------------------------------

def _pml_bsigma_at_rho(
    rho: float,
    sigma_max: float,
    kappa_max: float,
    alpha_max: float,
    m: int,
    dt: float,
) -> Tuple[float, float, float]:
    """Return (sigma, b, c) at a single normalised PML depth *rho* in [0,1]."""
    rho_m = rho ** m
    sigma_k = sigma_max * rho_m
    kappa_k = 1.0 + (kappa_max - 1.0) * rho_m
    alpha_k = alpha_max * (1.0 - rho)
    b_k = math.exp(-(sigma_k / kappa_k + alpha_k) * dt / EPS0)
    denom = sigma_k * kappa_k + kappa_k ** 2 * alpha_k
    c_k = 0.0 if (sigma_k == 0.0 and alpha_k == 0.0) else sigma_k / denom * (b_k - 1.0)
    return sigma_k, b_k, c_k


def _compute_pml_coeffs(
    D: int,
    dx: float,
    dt: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dual-staggered polynomial-graded CPML coefficients of length *D*.

    The Yee grid staggers E and H fields by half a cell, so rigorous CPML
    requires two separate coefficient sets evaluated at different sample points:

    - **E-field coefficients** (``*_E``): evaluated at integer-cell positions
      ``rho = k / D`` (cell edges).
    - **H-field coefficients** (``*_H``): evaluated at half-integer positions
      ``rho = (k + 0.5) / D`` (cell centres).

    Using a single set for both introduces a discrete-divergence mismatch that
    limits PML reflection to ~10⁻⁴.  Dual sets achieve the theoretical
    −80 dB target.

    Parameters
    ----------
    D : int
        PML depth in cells.
    dx : float
        Cell spacing (metres) for the graded direction.
    dt : float
        Time step (seconds).
    device : torch.device
        Target device.
    dtype : torch.dtype
        Floating-point dtype.

    Returns
    -------
    tuple of six Tensors, each shape ``(D,)``:
        ``(b_E, c_E, b_H, c_H, sigma_E, sigma_H)``
    """
    m: int = 3
    R_target: float = 1e-8
    kappa_max: float = 5.0
    alpha_max: float = 0.05
    sigma_max: float = (m + 1) * math.log(1.0 / R_target) / (2.0 * ETA0 * D * dx)

    b_E_list, c_E_list, s_E_list = [], [], []
    b_H_list, c_H_list, s_H_list = [], [], []

    for k in range(D):
        # E-field: integer-cell edge (k/D)
        rho_e = k / D
        s_e, b_e, c_e = _pml_bsigma_at_rho(rho_e, sigma_max, kappa_max, alpha_max, m, dt)
        b_E_list.append(b_e); c_E_list.append(c_e); s_E_list.append(s_e)

        # H-field: half-integer cell centre ((k+0.5)/D)
        rho_h = (k + 0.5) / D
        s_h, b_h, c_h = _pml_bsigma_at_rho(rho_h, sigma_max, kappa_max, alpha_max, m, dt)
        b_H_list.append(b_h); c_H_list.append(c_h); s_H_list.append(s_h)

    to_t = lambda lst: torch.tensor(lst, device=device, dtype=dtype)
    return to_t(b_E_list), to_t(c_E_list), to_t(b_H_list), to_t(c_H_list), to_t(s_E_list), to_t(s_H_list)


# ---------------------------------------------------------------------------
# MurABC
# ---------------------------------------------------------------------------

class MurABC:
    """First-order Mur absorbing boundary condition for 2D TM FDTD.

    Applies the Mur ABC update to all four edges of the Hz field after each
    FDTD field update.  Optionally pre-computes and stores PML scalar
    coefficient arrays (sigma, b, c) for use by an external convolutional
    PML implementation — no psi tensors or curl operators are managed here.

    The update formulas for Hz are::

        Left  (i=0):    Hz[0,   :, 0] = Hz_prev[1,    :, 0]
                                       + C_x*(Hz[1,    :, 0] - Hz_prev[0,   :, 0])
        Right (i=Nx-1): Hz[Nx-1,:, 0] = Hz_prev[Nx-2, :, 0]
                                       + C_x*(Hz[Nx-2, :, 0] - Hz_prev[Nx-1,:, 0])
        Bottom(j=0):    Hz[:, 0,   0] = Hz_prev[:, 1,    0]
                                       + C_y*(Hz[:, 1,    0] - Hz_prev[:, 0,  0])
        Top   (j=Ny-1): Hz[:, Ny-1,0] = Hz_prev[:, Ny-2,  0]
                                       + C_y*(Hz[:, Ny-2,  0] - Hz_prev[:,Ny-1,0])

    Corner cells are overwritten twice (x-boundaries first, then y-boundaries),
    which is acceptable for first-order Mur.

    Parameters
    ----------
    grid : YeeGrid
        Yee-lattice geometry providing Nx, Ny, dx, dy, dt, device, and dtype.
    Hz_field : torch.Tensor
        The live Hz tensor of shape ``(Nx, Ny, 1)`` that the FDTD solver owns.
        This class holds a reference; it does **not** take ownership.
    pml_depth : int
        Depth of the PML region in cells (default 0 means no PML).  When
        greater than zero, sigma/b/c coefficient arrays are pre-computed and
        stored on ``grid.device``.

    Raises
    ------
    ValueError
        If either Mur coefficient falls outside the open interval ``(-1, 0)``,
        which would indicate a CFL violation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        grid: YeeGrid,
        Hz_field: torch.Tensor,
        *,
        pml_depth: int = 0,
    ) -> None:
        """Initialise Mur coefficients, snapshot buffer, and optional PML arrays."""
        dt: float = grid.dt
        dx: float = grid.dx
        dy: float = grid.dy

        # Mur coefficients — Python floats, computed once
        self.C_mur_x: float = (C0 * dt - dx) / (C0 * dt + dx)
        self.C_mur_y: float = (C0 * dt - dy) / (C0 * dt + dy)

        # Stability validation: CFL guarantees C_mur in (-1, 0)
        if not (-1.0 < self.C_mur_x < 0.0):
            raise ValueError(
                f"C_mur_x={self.C_mur_x!r} is not in the open interval (-1, 0). "
                "This indicates a CFL violation — reduce dt or increase dx."
            )
        if not (-1.0 < self.C_mur_y < 0.0):
            raise ValueError(
                f"C_mur_y={self.C_mur_y!r} is not in the open interval (-1, 0). "
                "This indicates a CFL violation — reduce dt or increase dy."
            )

        # Snapshot buffer — same shape, dtype, and device as Hz_field
        self.Hz_prev: torch.Tensor = torch.zeros_like(Hz_field)
        # Live reference; boundary class does NOT own Hz
        self.Hz_curr: torch.Tensor = Hz_field

        self._Nx: int = grid.Nx
        self._Ny: int = grid.Ny
        self._pml_depth: int = pml_depth
        self._has_pml: bool = pml_depth > 0

        # PML coefficient arrays — dual staggered sets for E and H fields.
        # b_*_E / c_*_E: evaluated at integer-cell edges (for E-field CPML updates).
        # b_*_H / c_*_H: evaluated at half-integer centres (for H-field CPML updates).
        # sigma_* retained for diagnostics and Phase-5 use.
        if pml_depth > 0:
            (self.b_x_E, self.c_x_E,
             self.b_x_H, self.c_x_H,
             self.sigma_x_E, self.sigma_x_H) = _compute_pml_coeffs(
                pml_depth, dx, dt, grid.device, grid.dtype
            )
            (self.b_y_E, self.c_y_E,
             self.b_y_H, self.c_y_H,
             self.sigma_y_E, self.sigma_y_H) = _compute_pml_coeffs(
                pml_depth, dy, dt, grid.device, grid.dtype
            )
        else:
            # Size-0 tensors share no data — aliasing all to the same object is
            # safe (no elements to mutate) and avoids redundant allocations.
            _empty = torch.empty(0, device=grid.device, dtype=grid.dtype)
            (self.b_x_E, self.c_x_E, self.b_x_H, self.c_x_H,
             self.sigma_x_E, self.sigma_x_H) = (_empty,) * 6
            (self.b_y_E, self.c_y_E, self.b_y_H, self.c_y_H,
             self.sigma_y_E, self.sigma_y_H) = (_empty,) * 6

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def snapshot(self) -> None:
        """Capture the current Hz state before the FDTD field update.

        Must be called **before** the FDTD curl-update each time step so that
        :py:meth:`apply` can compare the post-update Hz against the saved
        pre-update values.
        """
        self.Hz_prev.copy_(self.Hz_curr)

    def apply(self, Hz: torch.Tensor) -> None:
        """Apply Mur ABC in-place to *Hz* after the FDTD field update.

        Overwrites the four boundary edges of *Hz* using the saved snapshot.
        Corner cells are overwritten twice (x-first, then y), which is
        acceptable for first-order Mur.

        Parameters
        ----------
        Hz : torch.Tensor
            The magnetic field z-component tensor of shape ``(Nx, Ny, 1)``.
            Modified in-place; no allocation is performed.
        """
        Nx, Ny = self._Nx, self._Ny
        cx, cy = self.C_mur_x, self.C_mur_y

        # Ellipsis (...) absorbs any leading batch dimension (B, Nx, Ny, 1) as well as
        # the unbatched case (Nx, Ny, 1), so both FieldSet modes work correctly.

        # Left edge (i = 0) — x-direction Mur
        Hz[..., 0, :, 0] = self.Hz_prev[..., 1, :, 0] + cx * (Hz[..., 1, :, 0] - self.Hz_prev[..., 0, :, 0])

        # Right edge (i = Nx-1) — x-direction Mur
        Hz[..., Nx - 1, :, 0] = (
            self.Hz_prev[..., Nx - 2, :, 0]
            + cx * (Hz[..., Nx - 2, :, 0] - self.Hz_prev[..., Nx - 1, :, 0])
        )

        # Bottom edge (j = 0) — y-direction Mur
        Hz[..., :, 0, 0] = self.Hz_prev[..., :, 1, 0] + cy * (Hz[..., :, 1, 0] - self.Hz_prev[..., :, 0, 0])

        # Top edge (j = Ny-1) — y-direction Mur
        Hz[..., :, Ny - 1, 0] = (
            self.Hz_prev[..., :, Ny - 2, 0]
            + cy * (Hz[..., :, Ny - 2, 0] - self.Hz_prev[..., :, Ny - 1, 0])
        )

    # ------------------------------------------------------------------
    # Properties (read-only)
    # ------------------------------------------------------------------

    @property
    def n_boundary_cells(self) -> int:
        """Total number of boundary cells across all four edges.

        Returns
        -------
        int
            ``2 * (Nx + Ny)``.
        """
        return 2 * (self._Nx + self._Ny)

    @property
    def mur_coefficient_x(self) -> float:
        """Mur ABC coefficient for x-directed boundaries.

        Returns
        -------
        float
            ``C_mur_x = (c0*dt - dx) / (c0*dt + dx)``.
        """
        return self.C_mur_x

    @property
    def mur_coefficient_y(self) -> float:
        """Mur ABC coefficient for y-directed boundaries.

        Returns
        -------
        float
            ``C_mur_y = (c0*dt - dy) / (c0*dt + dy)``.
        """
        return self.C_mur_y

    @property
    def has_pml(self) -> bool:
        """``True`` when PML coefficient arrays have been computed.

        Returns
        -------
        bool
        """
        return self._has_pml

    @property
    def pml_depth(self) -> int:
        """PML depth in cells (0 means no PML).

        Returns
        -------
        int
        """
        return self._pml_depth

    @property
    def pml_coeffs_x(self) -> dict[str, torch.Tensor]:
        """Dual-staggered PML coefficient arrays for the x-direction.

        Returns separate E-field and H-field coefficient sets to account for
        the half-cell Yee stagger, enabling −80 dB theoretical PML reflection.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys: ``"b_E"``, ``"c_E"`` (E-field, integer-cell edges),
            ``"b_H"``, ``"c_H"`` (H-field, half-integer centres),
            ``"sigma_E"``, ``"sigma_H"`` (conductivity profiles).
            All tensors have shape ``(pml_depth,)`` or ``(0,)`` when inactive.
        """
        return {
            "b_E": self.b_x_E, "c_E": self.c_x_E,
            "b_H": self.b_x_H, "c_H": self.c_x_H,
            "sigma_E": self.sigma_x_E, "sigma_H": self.sigma_x_H,
        }

    @property
    def pml_coeffs_y(self) -> dict[str, torch.Tensor]:
        """Dual-staggered PML coefficient arrays for the y-direction.

        Returns
        -------
        dict[str, torch.Tensor]
            Same keys as :attr:`pml_coeffs_x`, graded along y.
        """
        return {
            "b_E": self.b_y_E, "c_E": self.c_y_E,
            "b_H": self.b_y_H, "c_H": self.c_y_H,
            "sigma_E": self.sigma_y_E, "sigma_H": self.sigma_y_H,
        }

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a human-readable summary of the MurABC configuration."""
        return (
            f"MurABC("
            f"shape=({self._Nx}, {self._Ny}, 1), "
            f"C_mur_x={self.C_mur_x:.6f}, "
            f"C_mur_y={self.C_mur_y:.6f}, "
            f"has_pml={self._has_pml}, "
            f"pml_depth={self._pml_depth}, "
            f"n_boundary_cells={self.n_boundary_cells}"
            f")"
        )
