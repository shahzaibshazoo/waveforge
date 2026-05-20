"""
fields.py — Yee-staggered field storage for WaveForge.

Provides:
  - FIELD_COMPONENTS: ordered tuple of the six standard field component names
  - StaggerOffset: dataclass encoding each component's sub-cell sampling offset
  - YEE_OFFSETS: pre-built mapping from component name to StaggerOffset
  - FieldSet: container that owns six zero-initialised tensors on a YeeGrid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .grid import YeeGrid

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

FIELD_COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
"""Ordered tuple of all six Yee field component names."""


# ---------------------------------------------------------------------------
# StaggerOffset
# ---------------------------------------------------------------------------

@dataclass
class StaggerOffset:
    """Sub-cell sampling offset for a single Yee field component.

    All offsets are expressed in units of the local cell spacing (dx, dy, dz)
    and take a value of either 0.0 (cell-face / cell-edge centre) or 0.5
    (half-cell shift in that direction).

    Attributes
    ----------
    component : str
        Component name, e.g. ``"Ex"``.
    ox : float
        Offset along x in units of dx (0.0 or 0.5).
    oy : float
        Offset along y in units of dy (0.0 or 0.5).
    oz : float
        Offset along z in units of dz (0.0 or 0.5).
    """

    component: str
    ox: float  # x stagger offset in units of dx (0.0 or 0.5)
    oy: float  # y stagger offset in units of dy (0.0 or 0.5)
    oz: float  # z stagger offset in units of dz (0.0 or 0.5)


# ---------------------------------------------------------------------------
# YEE_OFFSETS
# ---------------------------------------------------------------------------

YEE_OFFSETS: dict[str, StaggerOffset] = {
    "Ex": StaggerOffset("Ex", ox=0.5, oy=0.0, oz=0.0),
    "Ey": StaggerOffset("Ey", ox=0.0, oy=0.5, oz=0.0),
    "Ez": StaggerOffset("Ez", ox=0.0, oy=0.0, oz=0.5),
    "Hx": StaggerOffset("Hx", ox=0.0, oy=0.5, oz=0.5),
    "Hy": StaggerOffset("Hy", ox=0.5, oy=0.0, oz=0.5),
    "Hz": StaggerOffset("Hz", ox=0.5, oy=0.5, oz=0.0),
}
"""Canonical Yee-lattice stagger offsets for all six field components."""


# ---------------------------------------------------------------------------
# Physical constants used in energy calculations
# ---------------------------------------------------------------------------

_EPS0: float = 8.854e-12   # permittivity of free space, F/m
_MU0: float = 1.2566e-6    # permeability of free space, H/m


# ---------------------------------------------------------------------------
# FieldSet
# ---------------------------------------------------------------------------

class FieldSet:
    """Six-component Yee field storage backed by PyTorch tensors.

    Allocates one zero tensor per field component on the device and with the
    dtype reported by the supplied :class:`~waveforge.core.grid.YeeGrid`.
    Provides named properties, in-place zeroing, cloning, device migration,
    energy diagnostics, and serialisation helpers.

    Parameters
    ----------
    grid : YeeGrid
        Yee-lattice geometry that drives shape, device, and dtype defaults.
    batch_size : int
        Number of independent simulations in a single batch.  Use 1 (the
        default) for un-batched single-simulation storage; any value ``>= 2``
        prepends a batch dimension so that every tensor has shape
        ``(batch_size, Nx, Ny, Nz)``.
    dtype : torch.dtype | None
        Floating-point dtype for all tensors.  Defaults to ``grid.dtype``
        when ``None``.

    Raises
    ------
    ValueError
        If *batch_size* is less than 1.

    Attributes
    ----------
    grid : YeeGrid
    batch_size : int
    dtype : torch.dtype
    offsets : dict[str, StaggerOffset]
        Reference to the module-level :data:`YEE_OFFSETS` mapping.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        grid: YeeGrid,
        *,
        batch_size: int = 1,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialise FieldSet and allocate zero field tensors."""
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {batch_size!r}"
            )

        self.grid: YeeGrid = grid
        self.batch_size: int = batch_size
        self.dtype: torch.dtype = grid.dtype if dtype is None else dtype
        self.offsets: dict[str, StaggerOffset] = YEE_OFFSETS

        tensor_shape = self._tensor_shape()
        self._fields: dict[str, torch.Tensor] = {
            comp: torch.zeros(
                tensor_shape,
                device=grid.device,
                dtype=self.dtype,
            )
            for comp in FIELD_COMPONENTS
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tensor_shape(self) -> tuple[int, ...]:
        """Return the expected shape for a single field tensor.

        Returns
        -------
        tuple[int, ...]
            ``(Nx, Ny, Nz)`` when unbatched or ``(B, Nx, Ny, Nz)`` when
            batched.
        """
        Nx, Ny, Nz = self.grid.Nx, self.grid.Ny, self.grid.Nz
        if self.batch_size == 1:
            return (Nx, Ny, Nz)
        return (self.batch_size, Nx, Ny, Nz)

    def _validate_tensor(self, name: str, t: torch.Tensor) -> None:
        """Validate that *t* matches expected shape and dtype.

        Parameters
        ----------
        name : str
            Component name used in error messages.
        t : torch.Tensor
            Tensor to validate.

        Raises
        ------
        ValueError
            If shape or dtype does not match expectations.
        """
        expected_shape = self._tensor_shape()
        if tuple(t.shape) != expected_shape:
            raise ValueError(
                f"Component '{name}' expected shape {expected_shape}, "
                f"got {tuple(t.shape)}"
            )
        if t.dtype != self.dtype:
            raise ValueError(
                f"Component '{name}' expected dtype {self.dtype}, "
                f"got {t.dtype}"
            )

    # ------------------------------------------------------------------
    # Properties — named component access
    # ------------------------------------------------------------------

    @property
    def Ex(self) -> torch.Tensor:
        """Electric field x-component tensor."""
        return self._fields["Ex"]

    @Ex.setter
    def Ex(self, t: torch.Tensor) -> None:
        """Set Ex tensor, validating shape and dtype."""
        self._validate_tensor("Ex", t)
        self._fields["Ex"] = t

    @property
    def Ey(self) -> torch.Tensor:
        """Electric field y-component tensor."""
        return self._fields["Ey"]

    @Ey.setter
    def Ey(self, t: torch.Tensor) -> None:
        """Set Ey tensor, validating shape and dtype."""
        self._validate_tensor("Ey", t)
        self._fields["Ey"] = t

    @property
    def Ez(self) -> torch.Tensor:
        """Electric field z-component tensor."""
        return self._fields["Ez"]

    @Ez.setter
    def Ez(self, t: torch.Tensor) -> None:
        """Set Ez tensor, validating shape and dtype."""
        self._validate_tensor("Ez", t)
        self._fields["Ez"] = t

    @property
    def Hx(self) -> torch.Tensor:
        """Magnetic field x-component tensor."""
        return self._fields["Hx"]

    @Hx.setter
    def Hx(self, t: torch.Tensor) -> None:
        """Set Hx tensor, validating shape and dtype."""
        self._validate_tensor("Hx", t)
        self._fields["Hx"] = t

    @property
    def Hy(self) -> torch.Tensor:
        """Magnetic field y-component tensor."""
        return self._fields["Hy"]

    @Hy.setter
    def Hy(self, t: torch.Tensor) -> None:
        """Set Hy tensor, validating shape and dtype."""
        self._validate_tensor("Hy", t)
        self._fields["Hy"] = t

    @property
    def Hz(self) -> torch.Tensor:
        """Magnetic field z-component tensor."""
        return self._fields["Hz"]

    @Hz.setter
    def Hz(self, t: torch.Tensor) -> None:
        """Set Hz tensor, validating shape and dtype."""
        self._validate_tensor("Hz", t)
        self._fields["Hz"] = t

    # ------------------------------------------------------------------
    # Properties — metadata
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of each individual field tensor.

        Returns
        -------
        tuple[int, ...]
            ``(Nx, Ny, Nz)`` when unbatched; ``(batch_size, Nx, Ny, Nz)``
            when batched.
        """
        return self._tensor_shape()

    @property
    def is_batched(self) -> bool:
        """``True`` when batch_size > 1, ``False`` otherwise."""
        return self.batch_size > 1

    @property
    def device(self) -> torch.device:
        """PyTorch device derived from the underlying :attr:`grid`."""
        return self.grid.device

    # ------------------------------------------------------------------
    # Mutation and copying
    # ------------------------------------------------------------------

    def zero_(self) -> "FieldSet":
        """Zero all six field tensors in-place.

        Returns
        -------
        FieldSet
            *self*, allowing method chaining.
        """
        for comp in FIELD_COMPONENTS:
            self._fields[comp].zero_()
        return self

    def clone(self) -> "FieldSet":
        """Return a deep copy with fully independent tensors.

        The new :class:`FieldSet` references the same :attr:`grid` object
        (grids are treated as immutable geometry) but owns freshly cloned
        field tensors.

        Returns
        -------
        FieldSet
            New instance with cloned tensors.
        """
        new_fs = FieldSet(self.grid, batch_size=self.batch_size, dtype=self.dtype)
        for comp in FIELD_COMPONENTS:
            new_fs._fields[comp] = self._fields[comp].clone()
        return new_fs

    def to(self, device: str | torch.device) -> "FieldSet":
        """Return a new :class:`FieldSet` with all tensors moved to *device*.

        The original :class:`FieldSet` is not modified.  The grid is also
        migrated via its own :py:meth:`~waveforge.core.grid.YeeGrid.to` method.

        Parameters
        ----------
        device : str | torch.device
            Target device.

        Returns
        -------
        FieldSet
            New instance whose tensors and grid reside on *device*.
        """
        new_grid = self.grid.to(device)
        resolved = new_grid.device
        new_fs = FieldSet(new_grid, batch_size=self.batch_size, dtype=self.dtype)
        for comp in FIELD_COMPONENTS:
            new_fs._fields[comp] = self._fields[comp].to(resolved)
        return new_fs

    def copy_from_(self, other: "FieldSet") -> None:
        """Copy all field tensors from *other* into *self* in-place.

        Uses :py:meth:`torch.Tensor.copy_` so data is transferred even across
        devices, provided shapes match.

        Parameters
        ----------
        other : FieldSet
            Source field set.  Must have the same :attr:`shape` as *self*.

        Raises
        ------
        ValueError
            If *other* has a different shape.
        """
        if other.shape != self.shape:
            raise ValueError(
                f"Cannot copy_from_: shape mismatch — "
                f"self.shape={self.shape}, other.shape={other.shape}"
            )
        for comp in FIELD_COMPONENTS:
            self._fields[comp].copy_(other._fields[comp])

    # ------------------------------------------------------------------
    # Dict / tensor serialisation
    # ------------------------------------------------------------------

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return a shallow copy of the internal field storage mapping.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys are component names; values are the live tensors (not copies).
        """
        return dict(self._fields)

    def as_tensor(self) -> torch.Tensor:
        """Stack all six components into a single contiguous tensor.

        The stacking dimension is 0 for unbatched fields and 1 for batched
        fields, so that the component axis is always immediately after the
        optional batch axis.

        Returns
        -------
        torch.Tensor
            Shape ``(6, Nx, Ny, Nz)`` (unbatched) or
            ``(batch_size, 6, Nx, Ny, Nz)`` (batched).
        """
        tensors = [self._fields[comp] for comp in FIELD_COMPONENTS]
        dim = 1 if self.is_batched else 0
        return torch.stack(tensors, dim=dim)

    def from_tensor(self, t: torch.Tensor) -> None:
        """Unpack a stacked tensor back into the six internal field tensors.

        This is the inverse of :py:meth:`as_tensor`.  The supplied tensor must
        match one of the two canonical shapes produced by that method.

        Parameters
        ----------
        t : torch.Tensor
            Stacked tensor.  Expected shape is ``(6, Nx, Ny, Nz)`` for
            unbatched storage or ``(B, 6, Nx, Ny, Nz)`` for batched storage.

        Raises
        ------
        ValueError
            If *t* has an unexpected shape.
        """
        Nx, Ny, Nz = self.grid.Nx, self.grid.Ny, self.grid.Nz
        unbatched_shape = (6, Nx, Ny, Nz)
        batched_shape = (self.batch_size, 6, Nx, Ny, Nz)

        if tuple(t.shape) == unbatched_shape:
            for i, comp in enumerate(FIELD_COMPONENTS):
                self._fields[comp] = t[i]
        elif tuple(t.shape) == batched_shape:
            for i, comp in enumerate(FIELD_COMPONENTS):
                self._fields[comp] = t[:, i, :, :, :]
        else:
            raise ValueError(
                f"from_tensor: unexpected shape {tuple(t.shape)}. "
                f"Expected {unbatched_shape} (unbatched) or "
                f"{batched_shape} (batched)."
            )

    # ------------------------------------------------------------------
    # Component group accessors
    # ------------------------------------------------------------------

    def electric(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the three electric field component tensors.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(Ex, Ey, Ez)``.
        """
        return self._fields["Ex"], self._fields["Ey"], self._fields["Ez"]

    def magnetic(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the three magnetic field component tensors.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(Hx, Hy, Hz)``.
        """
        return self._fields["Hx"], self._fields["Hy"], self._fields["Hz"]

    # ------------------------------------------------------------------
    # Energy and diagnostics
    # ------------------------------------------------------------------

    def energy_density(self) -> torch.Tensor:
        """Compute the electromagnetic energy density at each grid cell.

        Uses SI physical constants::

            u = eps0/2 * (Ex² + Ey² + Ez²) + mu0/2 * (Hx² + Hy² + Hz²)

        Returns
        -------
        torch.Tensor
            Shape ``(Nx, Ny, Nz)`` (unbatched) or ``(batch_size, Nx, Ny, Nz)``
            (batched).
        """
        acc = torch.zeros_like(self._fields["Ex"])
        for comp in ("Ex", "Ey", "Ez"):
            acc = acc + self._fields[comp] ** 2
        acc = acc * (_EPS0 * 0.5)

        mag_acc = torch.zeros_like(self._fields["Hx"])
        for comp in ("Hx", "Hy", "Hz"):
            mag_acc = mag_acc + self._fields[comp] ** 2
        mag_acc = mag_acc * (_MU0 * 0.5)

        return acc + mag_acc

    def total_energy(self) -> float:
        """Return the total electromagnetic energy integrated over the grid.

        Multiplies the summed energy density by the physical cell volume
        (dx * dy * dz) so the result is in Joules, not J/m³.

        Returns
        -------
        float
            ∭ u dV ≈ Σ u[i,j,k] · dx·dy·dz  (units: Joules)
        """
        return float(self.energy_density().sum().item()) * self.grid.cell_volume

    def max_field(self) -> float:
        """Return the maximum absolute field value across all six components.

        Useful as a simple instability monitor — a rapidly growing value
        indicates a CFL violation or ill-posed source.

        Returns
        -------
        float
            ``max(|Ex|, |Ey|, |Ez|, |Hx|, |Hy|, |Hz|)`` over all cells.
        """
        max_val = 0.0
        for comp in FIELD_COMPONENTS:
            candidate = float(self._fields[comp].abs().max().item())
            if candidate > max_val:
                max_val = candidate
        return max_val

    def norm(self, component: str) -> float:
        """Compute the L2 norm of the named field component.

        Parameters
        ----------
        component : str
            One of the six component names in :data:`FIELD_COMPONENTS`.

        Returns
        -------
        float
            ``‖field‖₂``.

        Raises
        ------
        KeyError
            If *component* is not a recognised field component name.
        """
        if component not in self._fields:
            raise KeyError(
                f"Unknown field component {component!r}. "
                f"Valid names: {FIELD_COMPONENTS}"
            )
        return float(torch.linalg.norm(self._fields[component].float()).item())

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a human-readable summary of the FieldSet.

        Intentionally excludes total_energy() to avoid a GPU→CPU sync
        (via .item()) when the object is printed inside a simulation loop.
        """
        return (
            f"FieldSet("
            f"shape={self.shape}, "
            f"batch_size={self.batch_size}, "
            f"dtype={self.dtype}, "
            f"device={self.device}"
            f")"
        )
