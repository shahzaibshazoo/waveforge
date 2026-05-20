"""
grid.py — Yee grid geometry and metadata for WaveForge.

Provides:
  - C0: speed of light constant
  - GridMetadata: dataclass summarising grid dimensions, spacing, and time step
  - compute_stable_dt: CFL-compliant time-step calculator
  - YeeGrid: full Yee-lattice object with coordinate tensors and field helpers
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

C0: float = 299_792_458.0  # speed of light in vacuum, m/s


# ---------------------------------------------------------------------------
# GridMetadata
# ---------------------------------------------------------------------------

@dataclass
class GridMetadata:
    """Immutable summary of grid dimensions, spacing, time step, and device.

    Attributes
    ----------
    Nx, Ny, Nz : int
        Number of cells along each axis.
    dx, dy, dz : float
        Cell spacing (metres) along each axis.
    dt : float
        Time step (seconds).
    courant : float
        Courant number used to derive *dt*.
    device : torch.device
        PyTorch device on which field tensors will be allocated.
    dtype : torch.dtype
        Floating-point dtype for field tensors (default ``torch.float32``).
    is_3d : bool
        ``True`` when ``Nz > 1``, ``False`` for 2-D simulations.
    """

    Nx: int
    Ny: int
    Nz: int
    dx: float
    dy: float
    dz: float
    dt: float
    courant: float
    device: torch.device
    dtype: torch.dtype = field(default=torch.float32)
    is_3d: bool = field(default=False)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def cells_total(self) -> int:
        """Return the total number of cells in the grid.

        Returns
        -------
        int
            ``Nx * Ny * Nz``.
        """
        return self.Nx * self.Ny * self.Nz

    def vram_estimate_bytes(self, n_field_tensors: int = 6) -> int:
        """Estimate VRAM required for a given number of field tensors.

        Each tensor has shape ``(Nx, Ny, Nz)`` and uses *dtype* element width.

        Parameters
        ----------
        n_field_tensors : int
            Number of full-grid tensors to account for (default 6, covering
            Ex/Ey/Ez/Hx/Hy/Hz).

        Returns
        -------
        int
            Estimated bytes.
        """
        element_bytes: int = torch.finfo(self.dtype).bits // 8
        return n_field_tensors * self.cells_total() * element_bytes

    def __repr__(self) -> str:
        """Return a human-readable summary of the grid metadata."""
        mode = "3D" if self.is_3d else "2D"
        vram_mb = self.vram_estimate_bytes() / (1024 ** 2)
        return (
            f"GridMetadata("
            f"mode={mode}, "
            f"shape=({self.Nx}, {self.Ny}, {self.Nz}), "
            f"spacing=({self.dx:.3g}, {self.dy:.3g}, {self.dz:.3g}) m, "
            f"dt={self.dt:.6e} s, "
            f"courant={self.courant}, "
            f"device={self.device}, "
            f"dtype={self.dtype}, "
            f"cells={self.cells_total():,}, "
            f"VRAM~{vram_mb:.2f} MB"
            f")"
        )


# ---------------------------------------------------------------------------
# CFL helper
# ---------------------------------------------------------------------------

def compute_stable_dt(
    dx: float,
    dy: float,
    dz: float | None = None,
    courant: float = 0.98,
) -> float:
    """Compute the CFL-stable time step for a given grid spacing.

    Implements the standard FDTD Courant–Friedrichs–Lewy condition::

        dt = courant / (C0 * sqrt(1/dx² + 1/dy² + [1/dz²]))

    Parameters
    ----------
    dx, dy : float
        Cell spacing along x and y (metres). Must be strictly positive.
    dz : float | None
        Cell spacing along z (metres). ``None`` for 2-D simulations (z term
        is omitted from the sum).  Must be strictly positive when provided.
    courant : float
        Courant number. Must satisfy ``0 < courant < 1``.

    Returns
    -------
    float
        Maximum stable time step in seconds.

    Raises
    ------
    ValueError
        If *courant* is not in the open interval ``(0, 1)``.
    ValueError
        If any provided spacing is ``<= 0``.
    """
    if not (0.0 < courant < 1.0):
        raise ValueError(
            f"courant must satisfy 0 < courant < 1, got {courant!r}"
        )
    if dx <= 0.0:
        raise ValueError(f"dx must be > 0, got {dx!r}")
    if dy <= 0.0:
        raise ValueError(f"dy must be > 0, got {dy!r}")
    if dz is not None and dz <= 0.0:
        raise ValueError(f"dz must be > 0 when provided, got {dz!r}")

    inv_sq_sum = (1.0 / dx) ** 2 + (1.0 / dy) ** 2
    if dz is not None:
        inv_sq_sum += (1.0 / dz) ** 2

    return courant / (C0 * math.sqrt(inv_sq_sum))


# ---------------------------------------------------------------------------
# YeeGrid
# ---------------------------------------------------------------------------

class YeeGrid:
    """Yee-lattice grid geometry for FDTD electromagnetic simulation.

    Manages cell dimensions, CFL time stepping, and coordinate tensors for
    both E-field (half-integer) and H-field (integer) sampling positions on
    the Yee staggered grid.

    Parameters
    ----------
    Nx, Ny : int
        Number of cells along x and y.  Each must be >= 2.
    dx, dy : float
        Cell spacing (metres) along x and y.  Must be > 0.
    Nz : int
        Number of cells along z (default 1 → 2-D mode).  Must be >= 2 when
        set to a value > 1.
    dz : float | None
        Cell spacing along z.  Required (and must be > 0) when ``Nz > 1``.
        Ignored in 2-D mode for CFL purposes; defaults internally to *dx* for
        cell-volume consistency.
    courant : float
        Courant number in ``(0, 1)`` (default 0.98).
    dt : float | None
        Override the time step.  When provided, must satisfy
        ``dt <= compute_stable_dt(...)``; a warning is issued when *dt* is
        less than 10 % of the stable value (suspiciously small).  When
        ``None``, the stable value is computed automatically.
    device : str | torch.device
        Target device.  Accepts ``"cuda"``, ``"cpu"``, or a
        ``torch.device`` instance.  If ``"cuda"`` is requested but
        unavailable the grid silently falls back to ``"cpu"`` with a
        ``warnings.warn`` message.
    dtype : torch.dtype
        Floating-point dtype for all tensors (default ``torch.float32``).

    Attributes
    ----------
    meta : GridMetadata
        Read-only metadata snapshot.
    xs, ys, zs : torch.Tensor
        E-field coordinate arrays at half-integer cell offsets (shape
        ``(Nx,)``, ``(Ny,)``, ``(Nz,)`` respectively).
    xs_h, ys_h, zs_h : torch.Tensor
        H-field coordinate arrays at integer cell offsets.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        Nx: int,
        Ny: int,
        dx: float,
        dy: float,
        *,
        Nz: int = 1,
        dz: float | None = None,
        courant: float = 0.98,
        dt: float | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialise and validate the Yee grid."""
        # --- validate integer dimensions --------------------------------
        # Nx and Ny must each be >= 2.  Nz == 1 selects 2-D mode; when
        # Nz > 1 it must also be >= 2 (minimum viable 3-D grid).
        for name, val in (("Nx", Nx), ("Ny", Ny)):
            if not isinstance(val, int) or val < 2:
                raise ValueError(
                    f"{name} must be an integer >= 2, got {val!r}"
                )
        if not isinstance(Nz, int) or Nz < 1:
            raise ValueError(
                f"Nz must be an integer >= 1 (use Nz=1 for 2-D mode), got {Nz!r}"
            )
        if Nz > 1 and Nz < 2:  # logically unreachable but kept for clarity
            raise ValueError(
                f"Nz must be >= 2 in 3-D mode, got {Nz!r}"
            )

        # --- validate spacings ------------------------------------------
        if dx <= 0.0:
            raise ValueError(f"dx must be > 0, got {dx!r}")
        if dy <= 0.0:
            raise ValueError(f"dy must be > 0, got {dy!r}")

        is_3d: bool = Nz > 1

        if is_3d:
            if dz is None:
                raise ValueError(
                    "dz is required when Nz > 1 (3-D mode)"
                )
            if dz <= 0.0:
                raise ValueError(f"dz must be > 0, got {dz!r}")
        else:
            # 2-D mode: keep dz consistent with dx for cell_volume
            if dz is None:
                dz = dx

        # --- validate courant -------------------------------------------
        if not (0.0 < courant < 1.0):
            raise ValueError(
                f"courant must satisfy 0 < courant < 1, got {courant!r}"
            )

        # --- resolve device, fall back from cuda if needed --------------
        resolved_device = self._resolve_device(device)

        # --- compute / validate dt --------------------------------------
        dz_for_cfl: float | None = dz if is_3d else None
        stable_dt = compute_stable_dt(dx, dy, dz_for_cfl, courant)

        if dt is None:
            dt = stable_dt
        else:
            if dt > stable_dt:
                raise ValueError(
                    f"Provided dt={dt:.6e} exceeds CFL-stable limit "
                    f"{stable_dt:.6e}. Simulation will be unstable."
                )
            threshold = 0.10 * stable_dt
            if dt < threshold:
                warnings.warn(
                    f"Provided dt={dt:.6e} is less than 10 % of the "
                    f"CFL-stable dt={stable_dt:.6e}. This is suspiciously "
                    "small and may indicate a unit error.",
                    UserWarning,
                    stacklevel=2,
                )

        # --- build metadata ---------------------------------------------
        self.meta = GridMetadata(
            Nx=Nx, Ny=Ny, Nz=Nz,
            dx=dx, dy=dy, dz=dz,
            dt=dt,
            courant=courant,
            device=resolved_device,
            dtype=dtype,
            is_3d=is_3d,
        )

        # --- build coordinate tensors -----------------------------------
        self.xs: torch.Tensor = self._make_e_coords(Nx, dx, resolved_device, dtype)
        self.ys: torch.Tensor = self._make_e_coords(Ny, dy, resolved_device, dtype)
        self.zs: torch.Tensor = self._make_e_coords(Nz, dz, resolved_device, dtype)

        self.xs_h: torch.Tensor = self._make_h_coords(Nx, dx, resolved_device, dtype)
        self.ys_h: torch.Tensor = self._make_h_coords(Ny, dy, resolved_device, dtype)
        self.zs_h: torch.Tensor = self._make_h_coords(Nz, dz, resolved_device, dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        """Resolve a device string or object, falling back to CPU if needed.

        Parameters
        ----------
        device : str | torch.device
            Requested device.

        Returns
        -------
        torch.device
            Resolved device (may differ from requested if CUDA unavailable).
        """
        if isinstance(device, str):
            device = torch.device(device)

        if device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but not available. Falling back to CPU.",
                UserWarning,
                stacklevel=3,
            )
            return torch.device("cpu")
        return device

    @staticmethod
    def _make_e_coords(
        N: int, spacing: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build E-field coordinate array at half-integer cell offsets.

        Positions are ``(0.5, 1.5, ..., N-0.5) * spacing``.

        Parameters
        ----------
        N : int
            Number of cells.
        spacing : float
            Cell spacing in metres.
        device : torch.device
            Target device.
        dtype : torch.dtype
            Floating-point dtype.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)``.
        """
        indices = torch.arange(N, device=device, dtype=dtype) + 0.5
        return indices * spacing

    @staticmethod
    def _make_h_coords(
        N: int, spacing: float, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build H-field coordinate array at integer cell offsets.

        Positions are ``(0, 1, ..., N-1) * spacing``.

        Parameters
        ----------
        N : int
            Number of cells.
        spacing : float
            Cell spacing in metres.
        device : torch.device
            Target device.
        dtype : torch.dtype
            Floating-point dtype.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)``.
        """
        indices = torch.arange(N, device=device, dtype=dtype)
        return indices * spacing

    # ------------------------------------------------------------------
    # Properties (read-only, delegating to meta)
    # ------------------------------------------------------------------

    @property
    def Nx(self) -> int:
        """Number of cells along x."""
        return self.meta.Nx

    @property
    def Ny(self) -> int:
        """Number of cells along y."""
        return self.meta.Ny

    @property
    def Nz(self) -> int:
        """Number of cells along z."""
        return self.meta.Nz

    @property
    def dx(self) -> float:
        """Cell spacing along x (metres)."""
        return self.meta.dx

    @property
    def dy(self) -> float:
        """Cell spacing along y (metres)."""
        return self.meta.dy

    @property
    def dz(self) -> float:
        """Cell spacing along z (metres)."""
        return self.meta.dz

    @property
    def dt(self) -> float:
        """Time step (seconds)."""
        return self.meta.dt

    @property
    def courant(self) -> float:
        """Courant number."""
        return self.meta.courant

    @property
    def device(self) -> torch.device:
        """PyTorch device for field tensors."""
        return self.meta.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype for field tensors."""
        return self.meta.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape as ``(Nx, Ny, Nz)``."""
        return (self.meta.Nx, self.meta.Ny, self.meta.Nz)

    @property
    def cell_volume(self) -> float:
        """Volume of a single cell in cubic metres (``dx * dy * dz``)."""
        return self.meta.dx * self.meta.dy * self.meta.dz

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def allocate_field(self, fill: float = 0.0) -> torch.Tensor:
        """Allocate a zero-filled field tensor with grid shape.

        Parameters
        ----------
        fill : float
            Fill value (default 0.0).  The tensor is initialised to zeros
            and then filled, so this is equivalent to
            ``torch.full(shape, fill, ...)``.

        Returns
        -------
        torch.Tensor
            Shape ``(Nx, Ny, Nz)`` on :py:attr:`device` with :py:attr:`dtype`.
        """
        return torch.full(
            self.shape,
            fill,
            device=self.meta.device,
            dtype=self.meta.dtype,
        )

    def allocate_material_coeff(self, fill: float = 1.0) -> torch.Tensor:
        """Allocate a ones-filled material coefficient tensor.

        Initialises all cells to *fill* (default 1.0, representing vacuum).

        Parameters
        ----------
        fill : float
            Fill value (default 1.0).

        Returns
        -------
        torch.Tensor
            Shape ``(Nx, Ny, Nz)`` on :py:attr:`device` with :py:attr:`dtype`.
        """
        return torch.full(
            self.shape,
            fill,
            device=self.meta.device,
            dtype=self.meta.dtype,
        )

    def cell_index(self, i: int, j: int, k: int = 0) -> int:
        """Compute the flat (row-major) cell index for cell ``(i, j, k)``.

        The indexing order is ``i * Ny * Nz + j * Nz + k``.

        Parameters
        ----------
        i, j : int
            Cell indices along x and y.
        k : int
            Cell index along z (default 0).

        Returns
        -------
        int
            Flat index into a linearised ``(Nx, Ny, Nz)`` array.
        """
        return i * self.meta.Ny * self.meta.Nz + j * self.meta.Nz + k

    def physical_size(self) -> tuple[float, float, float]:
        """Return the physical extent of the simulation domain.

        Returns
        -------
        tuple[float, float, float]
            ``(Nx*dx, Ny*dy, Nz*dz)`` in metres.
        """
        return (
            self.meta.Nx * self.meta.dx,
            self.meta.Ny * self.meta.dy,
            self.meta.Nz * self.meta.dz,
        )

    def wavelength_resolution(self, freq_hz: float) -> float:
        """Compute the number of cells per wavelength at a given frequency.

        Uses the finest grid spacing as the reference cell size.

        Parameters
        ----------
        freq_hz : float
            Frequency in Hertz.

        Returns
        -------
        float
            Cells per wavelength: ``lambda / min(dx, dy, dz)``.
        """
        wavelength = C0 / freq_hz
        min_spacing = min(self.meta.dx, self.meta.dy, self.meta.dz)
        return wavelength / min_spacing

    def validate_resolution(
        self,
        freq_hz: float,
        min_cells_per_wavelength: float = 10,
    ) -> None:
        """Assert that the grid resolves *freq_hz* with sufficient density.

        Parameters
        ----------
        freq_hz : float
            Frequency to validate (Hz).
        min_cells_per_wavelength : float
            Minimum acceptable cells per wavelength (default 10).

        Raises
        ------
        ValueError
            When the computed resolution is below *min_cells_per_wavelength*.
        """
        cpw = self.wavelength_resolution(freq_hz)
        if cpw < min_cells_per_wavelength:
            raise ValueError(
                f"Grid resolution at {freq_hz:.3e} Hz is {cpw:.2f} cells "
                f"per wavelength, below the required minimum of "
                f"{min_cells_per_wavelength}. Reduce cell spacing or "
                "lower the target frequency."
            )

    def to(self, device: str | torch.device) -> "YeeGrid":
        """Return a new :class:`YeeGrid` with all tensors on *device*.

        The original grid is not modified.

        Parameters
        ----------
        device : str | torch.device
            Target device.

        Returns
        -------
        YeeGrid
            New grid instance on the requested device.
        """
        resolved = self._resolve_device(
            torch.device(device) if isinstance(device, str) else device
        )

        new_grid = YeeGrid.__new__(YeeGrid)

        new_grid.meta = GridMetadata(
            Nx=self.meta.Nx,
            Ny=self.meta.Ny,
            Nz=self.meta.Nz,
            dx=self.meta.dx,
            dy=self.meta.dy,
            dz=self.meta.dz,
            dt=self.meta.dt,
            courant=self.meta.courant,
            device=resolved,
            dtype=self.meta.dtype,
            is_3d=self.meta.is_3d,
        )

        new_grid.xs = self.xs.to(resolved)
        new_grid.ys = self.ys.to(resolved)
        new_grid.zs = self.zs.to(resolved)
        new_grid.xs_h = self.xs_h.to(resolved)
        new_grid.ys_h = self.ys_h.to(resolved)
        new_grid.zs_h = self.zs_h.to(resolved)

        return new_grid

    def __repr__(self) -> str:
        """Return a detailed string representation of the grid."""
        mode = "3D" if self.meta.is_3d else "2D"
        Lx, Ly, Lz = self.physical_size()
        return (
            f"YeeGrid("
            f"mode={mode}, "
            f"shape={self.shape}, "
            f"physical=({Lx:.3g} m, {Ly:.3g} m, {Lz:.3g} m), "
            f"spacing=(dx={self.meta.dx:.3g}, dy={self.meta.dy:.3g}, "
            f"dz={self.meta.dz:.3g}) m, "
            f"dt={self.meta.dt:.6e} s, "
            f"courant={self.meta.courant}, "
            f"device={self.meta.device}, "
            f"dtype={self.meta.dtype}"
            f")"
        )
