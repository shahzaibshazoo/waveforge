"""
fdtd2d.py — 2D TM-mode FDTD time-stepper for GPU-MEEP.

Implements the Yee leapfrog update for the TM-polarised Maxwell equations::

    mu0  * dHz/dt =  dEy/dx - dEx/dy    (Faraday)
    eps0 * dEx/dt = +dHz/dy              (Ampere-x)
    eps0 * dEy/dt = -dHz/dx              (Ampere-y)

All curl operations are vectorised tensor slices; no Python loops over
spatial indices appear anywhere in the hot path.

Classes
-------
SimulationDivergedError
    RuntimeError subclass raised when field magnitudes exceed a stability
    threshold, indicating a CFL violation or ill-posed source.
FDTD2D
    The main time-stepper.  Owns references to a YeeGrid, FieldSet,
    MurABC boundary, and an optional SourceCollection.
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import torch

from .grid import YeeGrid
from .fields import FieldSet
from .boundaries import MurABC
from .sources import PointSource, LineSource, SourceCollection

# ---------------------------------------------------------------------------
# Module-level physical constants
# ---------------------------------------------------------------------------

EPS0: float = 8.8541878128e-12  # permittivity of free space, F/m
MU0: float = 1.2566370614e-6    # permeability of free space, H/m


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class SimulationDivergedError(RuntimeError):
    """Raised when a field component exceeds the stability threshold.

    Attributes
    ----------
    step : int
        Time-step index at which divergence was detected.
    component : str
        Name of the offending field component (``"Ex"``, ``"Ey"``, or ``"Hz"``).
    field_max : float
        Observed maximum absolute field value.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# FDTD2D
# ---------------------------------------------------------------------------

class FDTD2D:
    """2D TM FDTD time-stepper using Yee leapfrog integration.

    Advances the three non-zero TM field components — ``Ex``, ``Ey``, and
    ``Hz`` — by one time step per call to :meth:`step`.  All curl operations
    are implemented as in-place tensor slice assignments; no Python loops over
    spatial indices and no new tensor allocations occur during time-stepping.

    Parameters
    ----------
    grid : YeeGrid
        Yee-lattice geometry.  Must satisfy ``Nx >= 4`` and ``Ny >= 4``.
    fields : FieldSet
        Container for all six field component tensors.  Only ``Ex``, ``Ey``,
        and ``Hz`` are written during stepping.
    boundary : MurABC
        First-order Mur absorbing boundary condition constructed from the same
        grid (Nx, Ny must match).
    sources : SourceCollection | None
        Optional collection of soft sources injected before the H-field
        update each step.  ``None`` means no sources.
    stability_threshold : float
        Maximum allowed field magnitude (default ``1e10``).  If any of
        ``Ex``, ``Ey``, or ``Hz`` exceeds this value during a stability check,
        :class:`SimulationDivergedError` is raised.
    n_check : int
        Number of time steps between stability checks (default 100).

    Raises
    ------
    ValueError
        If ``grid.Nx < 4`` or ``grid.Ny < 4``.
    ValueError
        If ``boundary`` was constructed with a grid whose ``Nx`` or ``Ny``
        differs from ``grid.Nx`` / ``grid.Ny``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        grid: YeeGrid,
        fields: FieldSet,
        boundary: MurABC,
        sources: Optional[SourceCollection] = None,
        *,
        Ca: Optional[torch.Tensor] = None,
        Cb: Optional[torch.Tensor] = None,
        stability_threshold: float = 1e10,
        n_check: int = 100,
    ) -> None:
        """Initialise the FDTD2D time-stepper and pre-compute coefficients.

        Parameters
        ----------
        Ca : optional (Nx, Ny, 1) tensor
            Per-cell multiplicative E-field decay.  ``None`` → free-space
            scalar fast-path (Ca = 1 everywhere, no tensor load per step).
        Cb : optional (Nx, Ny, 1) tensor
            Per-cell curl-to-E scaling.  ``None`` → free-space scalar De.
            Must be provided together with Ca when materials are present.
        """
        # --- grid size validation (minimum stencil requirement) ----------
        if grid.Nx < 4:
            raise ValueError(
                f"grid.Nx must be >= 4 for a valid Yee stencil, got {grid.Nx}"
            )
        if grid.Ny < 4:
            raise ValueError(
                f"grid.Ny must be >= 4 for a valid Yee stencil, got {grid.Ny}"
            )

        # --- boundary / grid consistency check ---------------------------
        if boundary._Nx != grid.Nx or boundary._Ny != grid.Ny:
            raise ValueError(
                f"boundary was built for a grid of shape "
                f"({boundary._Nx}, {boundary._Ny}) but grid has shape "
                f"({grid.Nx}, {grid.Ny}). They must match."
            )

        # --- store references (no copies) --------------------------------
        self._grid: YeeGrid = grid
        self._fields: FieldSet = fields
        self._boundary: MurABC = boundary
        self._sources: Optional[SourceCollection] = sources

        # --- pre-compute leapfrog coefficients as Python floats ----------
        self._Dh: float = grid.dt / MU0
        self._De: float = grid.dt / EPS0
        self._dx: float = grid.dx
        self._dy: float = grid.dy

        # --- per-cell material coefficient tensors (None = free-space) ---
        # Ca=None activates the scalar fast-path: Ex += De*curl (no DRAM
        # load for Ca/Cb), saving 2×Nx×Ny×4 bytes of bandwidth per step.
        if (Ca is None) != (Cb is None):
            raise ValueError("Ca and Cb must be provided together or both omitted.")
        self._Ca: Optional[torch.Tensor] = Ca
        self._Cb: Optional[torch.Tensor] = Cb
        self._has_materials: bool = Ca is not None

        # --- stability settings ------------------------------------------
        self._threshold: float = stability_threshold
        self._n_check: int = n_check

        # --- telemetry ---------------------------------------------------
        self.steps_completed: int = 0
        self.elapsed_time: float = 0.0
        self.mcells_per_second: float = 0.0
        self.last_field_max: float = -1.0
        self.n_stability_checks: int = 0

    # ------------------------------------------------------------------
    # Core stepping methods
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance all fields by one time step in-place.

        The full leapfrog sequence is::

            1. boundary.snapshot()        — save Hz before H-update
            2. sources.step(fields, n)    — inject sources at step n
            3. Hz Faraday curl update     — Hz^{n-1/2} -> Hz^{n+1/2}
            4. boundary.apply(Hz)         — Mur ABC on Hz edges
            5. Ex Ampere-x curl update    — Ex^n -> Ex^{n+1}
            6. Ey Ampere-y curl update    — Ey^n -> Ey^{n+1}
            7. Telemetry update

        All curl operations are vectorised tensor slice additions; no Python
        loops over spatial indices, no ``torch.roll``, and no new tensor
        allocations occur here.

        Returns
        -------
        None
        """
        n = self.steps_completed

        # Unpack local references — views, not copies
        Ex = self._fields.Ex
        Ey = self._fields.Ey
        Hz = self._fields.Hz
        Dh = self._Dh
        De = self._De
        dx = self._dx
        dy = self._dy

        # Step 1: snapshot Hz before the H-field update
        self._boundary.snapshot()

        # Step 2: inject sources at step n
        if self._sources is not None:
            self._sources.step({"Ex": Ex, "Ey": Ey, "Hz": Hz}, n)

        # Step 3: Hz Faraday curl update (interior cells only)
        # Faraday: mu0 * dHz/dt = dEx/dy - dEy/dx   (corrected sign)
        # Hz^{n+1/2}[i,j] = Hz^{n-1/2}[i,j]
        #   + Dh * ( (Ex[i,j+1] - Ex[i,j]) / dy
        #          - (Ey[i+1,j] - Ey[i,j]) / dx )
        # Ellipsis absorbs optional leading batch dimension (B, Nx, Ny, 1).
        Hz[..., :-1, :-1, :] += Dh * (
            (Ex[..., :-1, 1:, :] - Ex[..., :-1, :-1, :]) / dy
            - (Ey[..., 1:, :-1, :] - Ey[..., :-1, :-1, :]) / dx
        )

        # Step 4: apply Mur ABC to Hz boundary edges
        self._boundary.apply(Hz)

        # Step 5 & 6: E-field Ampere curl updates.
        # Two paths:
        #  - Free-space (Ca=None): scalar De, in-place +=/-=, no Ca/Cb DRAM loads.
        #  - Material (Ca tensor): full Ca*E + Cb*curl form, element-wise.
        dHz_dy = (Hz[..., :, 1:, :] - Hz[..., :, :-1, :]) / dy
        dHz_dx = (Hz[..., 1:, :, :] - Hz[..., :-1, :, :]) / dx

        if not self._has_materials:
            # Fast path — free space, scalar coefficients
            Ex[..., :, 1:, :] += De * dHz_dy
            Ey[..., 1:, :, :] -= De * dHz_dx
        else:
            # Material path — per-cell Ca / Cb tensors
            # Shape: Ca/Cb are (Nx,Ny,1); slices match Ex/Ey interior shapes.
            Ex[..., :, 1:, :] = (self._Ca[..., :, 1:, :] * Ex[..., :, 1:, :]
                                  + self._Cb[..., :, 1:, :] * dHz_dy)
            Ey[..., 1:, :, :] = (self._Ca[..., 1:, :, :] * Ey[..., 1:, :, :]
                                  - self._Cb[..., 1:, :, :] * dHz_dx)

        # Step 7: update step counter (timing is measured in run(), not here,
        # because CUDA kernels execute asynchronously — perf_counter() here
        # only captures dispatch latency, not actual GPU execution time).
        self.steps_completed += 1

        # Stability check (every n_check steps — never every step)
        if self.steps_completed % self._n_check == 0:
            self._check_stability()

    def _check_stability(self) -> None:
        """Check field magnitudes and raise if the threshold is exceeded.

        Iterates over ``Ex``, ``Ey``, and ``Hz`` only — the zero-valued
        components ``Ez``, ``Hx``, ``Hy`` are not checked.

        Raises
        ------
        SimulationDivergedError
            When ``tensor.abs().max() > stability_threshold``.
        """
        self.n_stability_checks += 1
        for name, tensor in (
            ("Ex", self._fields.Ex),
            ("Ey", self._fields.Ey),
            ("Hz", self._fields.Hz),
        ):
            val = float(tensor.abs().max().item())
            if val > self.last_field_max:
                self.last_field_max = val
            if val > self._threshold:
                raise SimulationDivergedError(
                    f"Simulation diverged at step {self.steps_completed}: "
                    f"{name} max = {val:.3e} exceeds threshold "
                    f"{self._threshold:.3e}. "
                    f"Check CFL condition (dt={self._grid.dt:.3e}) and "
                    f"source amplitude."
                )

    # ------------------------------------------------------------------
    # High-level driver
    # ------------------------------------------------------------------

    def run(self, n_steps: int, *, verbose: bool = False) -> None:
        """Run *n_steps* time steps, optionally printing progress.

        Timing is measured around macro-blocks of steps with
        ``torch.cuda.synchronize()`` to account for asynchronous CUDA
        kernel dispatch.  Without synchronization, ``perf_counter()``
        only measures CPU dispatch latency (~μs), not GPU execution time
        (~ms), producing wildly optimistic throughput numbers.

        Parameters
        ----------
        n_steps : int
            Number of time steps to advance.
        verbose : bool
            When ``True``, print a progress line every ``n_steps // 10``
            steps showing percentage, step count, peak field, and
            throughput.
        """
        is_cuda = str(self._grid.device).startswith("cuda")
        report_interval = max(1, n_steps // 10)

        # Flush any pending GPU work before starting the clock.
        if is_cuda:
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        start_steps = self.steps_completed

        for i in range(n_steps):
            self.step()

            if verbose and (i + 1) % report_interval == 0:
                # Force GPU completion so the timer reflects real compute time.
                if is_cuda:
                    torch.cuda.synchronize()
                t_now = time.perf_counter()
                elapsed = t_now - t_start
                steps_done = self.steps_completed - start_steps
                if elapsed > 0.0:
                    self.mcells_per_second = (
                        steps_done * self._grid.Nx * self._grid.Ny
                    ) / elapsed / 1e6
                self.elapsed_time = elapsed
                pct = 100 * (i + 1) // n_steps
                print(
                    f"  {pct:3d}%  step {self.steps_completed}  "
                    f"field_max={self.last_field_max:.3e}  "
                    f"{self.mcells_per_second:.1f} Mcells/s"
                )

        # Final sync + telemetry update after all steps complete.
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        self.elapsed_time += elapsed
        steps_done = self.steps_completed - start_steps
        if self.elapsed_time > 0.0:
            self.mcells_per_second = (
                steps_done * self._grid.Nx * self._grid.Ny
            ) / elapsed / 1e6

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero all fields and reset telemetry to initial state.

        Keeps the grid, boundary, and sources unchanged.  After calling
        ``reset()``, the simulation is in an identical state to immediately
        after construction (except that the pre-computed coefficients are
        preserved).
        """
        self._fields.zero_()
        self.steps_completed = 0
        self.elapsed_time = 0.0
        self.mcells_per_second = 0.0
        self.last_field_max = -1.0
        self.n_stability_checks = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid(self) -> YeeGrid:
        """The Yee grid geometry used by this simulation."""
        return self._grid

    @property
    def fields(self) -> FieldSet:
        """The field storage container."""
        return self._fields

    @property
    def dt(self) -> float:
        """Time step in seconds (``grid.dt``)."""
        return self._grid.dt

    @property
    def time(self) -> float:
        """Elapsed simulation time in seconds (``steps_completed * dt``)."""
        return self.steps_completed * self._grid.dt

    @property
    def throughput(self) -> float:
        """Computational throughput alias: Mcells/s (same as ``mcells_per_second``)."""
        return self.mcells_per_second

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a human-readable summary of the FDTD2D simulation state."""
        return (
            f"FDTD2D("
            f"shape=({self._grid.Nx}, {self._grid.Ny}, {self._grid.Nz}), "
            f"steps={self.steps_completed}, "
            f"time={self.time:.4e} s, "
            f"throughput={self.mcells_per_second:.2f} Mcells/s, "
            f"device={self._grid.device}"
            f")"
        )
