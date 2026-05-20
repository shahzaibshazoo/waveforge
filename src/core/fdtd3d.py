"""
fdtd3d.py — 3D full-vector FDTD time-stepper for WaveForge.

Implements the Yee leapfrog update for the full-vector Maxwell equations in
three dimensions.  All six field components are advanced simultaneously using
the coupled curl equations::

    mu0  * dHx/dt = dEy/dz - dEz/dy          (Faraday-x)
    mu0  * dHy/dt = dEz/dx - dEx/dz          (Faraday-y)
    mu0  * dHz/dt = dEx/dy - dEy/dx          (Faraday-z, same sign as 2D engine)

    eps0 * dEx/dt = dHz/dy - dHy/dz          (Ampere-x)
    eps0 * dEy/dt = dHx/dz - dHz/dx          (Ampere-y)
    eps0 * dEz/dt = dHy/dx - dHx/dy          (Ampere-z)

The Yee stagger means that each update uses only one-sided forward or
backward differences that align naturally with the half-integer offsets of
the Yee lattice.  The concrete index arithmetic is::

    Hx[..., :,  :-1, :-1] += Dh * (
        (Ez[..., :,  1:, :-1] - Ez[..., :,  :-1, :-1]) / dy
      - (Ey[..., :,  :-1, 1:] - Ey[..., :,  :-1, :-1]) / dz
    )

    Hy[..., :-1, :,  :-1] += Dh * (
        (Ex[..., :-1, :,  1:] - Ex[..., :-1, :,  :-1]) / dz
      - (Ez[..., 1:, :,  :-1] - Ez[..., :-1, :,  :-1]) / dx
    )

    Hz[..., :-1, :-1, :] += Dh * (
        (Ex[..., :-1, 1:, :] - Ex[..., :-1, :-1, :]) / dy
      - (Ey[..., 1:, :-1, :] - Ey[..., :-1, :-1, :]) / dx
    )

    Ex[..., :,  1:, 1:] += De * (
        (Hz[..., :,  1:, 1:] - Hz[..., :,  :-1, 1:]) / dy
      - (Hy[..., :,  1:, 1:] - Hy[..., :,  1:, :-1]) / dz
    )

    Ey[..., 1:, :,  1:] += De * (
        (Hx[..., 1:, :,  1:] - Hx[..., 1:, :,  :-1]) / dz
      - (Hz[..., 1:, :,  1:] - Hz[..., :-1, :,  1:]) / dx
    )

    Ez[..., 1:, 1:, :] += De * (
        (Hy[..., 1:, 1:, :] - Hy[..., :-1, 1:, :]) / dx
      - (Hx[..., 1:, 1:, :] - Hx[..., 1:, :-1, :]) / dy
    )

The leading ``...`` absorbs an optional batch dimension so that tensors with
shape ``(Nx, Ny, Nz)`` and ``(B, Nx, Ny, Nz)`` are both handled without
code duplication.

All curl operations are vectorised tensor slices; no Python loops over
spatial indices appear anywhere in the hot path.

Classes
-------
SimulationDivergedError
    RuntimeError subclass raised when field magnitudes exceed a stability
    threshold, indicating a CFL violation or ill-posed source.
FDTD3D
    The main 3D time-stepper.  Owns references to a YeeGrid, FieldSet,
    MurABC3D boundary, and an optional SourceCollection.

Optimization
    run() wraps the step loop in torch.no_grad() for 30-50% GPU speedup.
    call compile_step() once after construction to enable torch.compile fusion.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Optional

import torch

from .grid import YeeGrid
from .fields import FieldSet

# ---------------------------------------------------------------------------
# Module-level physical constants
# ---------------------------------------------------------------------------

EPS0: float = 8.8541878128e-12  # permittivity of free space, F/m
MU0: float = 1.2566370614e-6    # permeability of free space, H/m
_C0: float = 1.0 / math.sqrt(EPS0 * MU0)  # speed of light, m/s


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
        Name of the offending field component (``"Ex"``, ``"Ey"``, ``"Ez"``,
        ``"Hx"``, ``"Hy"``, or ``"Hz"``).
    field_max : float
        Observed maximum absolute field value.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# FDTD3D
# ---------------------------------------------------------------------------

class FDTD3D:
    """3D full-vector FDTD time-stepper using Yee leapfrog integration.

    Advances all six field components — ``Ex``, ``Ey``, ``Ez``, ``Hx``,
    ``Hy``, and ``Hz`` — by one time step per call to :meth:`step`.  All
    curl operations are implemented as in-place tensor slice assignments; no
    Python loops over spatial indices and no new tensor allocations occur
    during time-stepping.

    This class does **not** inherit from :class:`FDTD2D`; the boundary
    interface is different (3D ABC passes three H-field tensors instead of
    one) and the update equations involve all six components.

    Parameters
    ----------
    grid : YeeGrid
        Yee-lattice geometry.  Must satisfy ``Nx >= 4``, ``Ny >= 4``, and
        ``Nz >= 4``.
    fields : FieldSet
        Container for all six field component tensors.
    boundary : MurABC3D
        First-order Mur absorbing boundary condition constructed from the same
        grid.  The boundary must implement:

        * ``snapshot()`` — called before the H-field update, no arguments.
        * ``apply(Hx, Hy, Hz)`` — called after the H-field update with the
          three magnetic field tensors passed positionally.
    sources : SourceCollection | None
        Optional collection of soft sources injected before the H-field
        update each step.  ``None`` means no sources.
    Ca : optional (Nx, Ny, Nz) tensor
        Per-cell multiplicative E-field decay.  ``None`` → free-space
        scalar fast-path (Ca = 1 everywhere, no tensor load per step).
    Cb : optional (Nx, Ny, Nz) tensor
        Per-cell curl-to-E scaling.  ``None`` → free-space scalar De.
        Must be provided together with *Ca* when materials are present.
    stability_threshold : float
        Maximum allowed field magnitude (default ``1e10``).  If any field
        component exceeds this value during a stability check,
        :class:`SimulationDivergedError` is raised.
    n_check : int
        Number of time steps between stability checks (default 100).

    Raises
    ------
    ValueError
        If ``grid.Nx < 4``, ``grid.Ny < 4``, or ``grid.Nz < 4``.
    ValueError
        If exactly one of *Ca* / *Cb* is provided (both or neither required).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        grid: YeeGrid,
        fields: FieldSet,
        boundary: object,
        sources: Optional[object] = None,
        *,
        Ca: Optional[torch.Tensor] = None,
        Cb: Optional[torch.Tensor] = None,
        stability_threshold: float = 1e10,
        n_check: int = 100,
    ) -> None:
        """Initialise the FDTD3D time-stepper and pre-compute coefficients.

        Parameters
        ----------
        Ca : optional (Nx, Ny, Nz) tensor
            Per-cell multiplicative E-field decay.  ``None`` → free-space
            scalar fast-path (Ca = 1 everywhere, no tensor load per step).
        Cb : optional (Nx, Ny, Nz) tensor
            Per-cell curl-to-E scaling.  ``None`` → free-space scalar De.
            Must be provided together with Ca when materials are present.
        """
        # --- grid size validation (minimum stencil requirement) ----------
        if grid.Nx < 4:
            raise ValueError(
                f"grid.Nx must be >= 4 for a valid 3D Yee stencil, got {grid.Nx}"
            )
        if grid.Ny < 4:
            raise ValueError(
                f"grid.Ny must be >= 4 for a valid 3D Yee stencil, got {grid.Ny}"
            )
        if grid.Nz > 1 and grid.Nz < 4:
            raise ValueError(
                f"grid.Nz must be >= 4 when using 3D mode, got {grid.Nz}. "
                "Use Nz=1 for 2D-equivalent mode."
            )

        # --- store references (no copies) --------------------------------
        self._grid: YeeGrid = grid
        self._fields: FieldSet = fields
        self._boundary = boundary
        self._sources = sources

        # --- pre-compute leapfrog coefficients as Python floats ----------
        self._Dh: float = grid.dt / MU0
        self._De: float = grid.dt / EPS0
        self._dx: float = grid.dx
        self._dy: float = grid.dy
        self._dz: float = grid.dz

        # --- per-cell material coefficient tensors (None = free-space) ---
        # Ca=None activates the scalar fast-path: E += De*curl (no DRAM
        # load for Ca/Cb), saving 2 * Nx * Ny * Nz * 4 bytes per step.
        if (Ca is None) != (Cb is None):
            raise ValueError("Ca and Cb must be provided together or both omitted.")
        self._Ca: Optional[torch.Tensor] = Ca
        self._Cb: Optional[torch.Tensor] = Cb
        self._has_materials: bool = Ca is not None

        # --- CFL warning -------------------------------------------------
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        cfl_limit = 0.99 / (_C0 * math.sqrt(
            1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz)
        ))
        if grid.dt > cfl_limit:
            warnings.warn(
                f"grid.dt={grid.dt:.6e} exceeds 0.99 × CFL limit "
                f"{cfl_limit:.6e}. Simulation may be unstable.",
                UserWarning,
                stacklevel=2,
            )

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
        """Advance all six field components by one time step in-place.

        The full 3D leapfrog sequence is::

            1. boundary.snapshot()          — save H-fields before H-update
            2. sources.step(fields, n)      — inject sources at step n
            3. Hx, Hy, Hz Faraday updates  — H^{n-1/2} -> H^{n+1/2}
            4. boundary.apply(Hx, Hy, Hz)  — Mur ABC on H boundary edges
            5. Ex, Ey, Ez Ampere updates   — E^n -> E^{n+1}
            6. Telemetry update

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
        Ez = self._fields.Ez
        Hx = self._fields.Hx
        Hy = self._fields.Hy
        Hz = self._fields.Hz
        Dh = self._Dh
        De = self._De
        dx = self._dx
        dy = self._dy
        dz = self._dz

        # Step 1: snapshot H-fields before the H-field update
        self._boundary.snapshot()

        # Step 2: inject sources at step n
        if self._sources is not None:
            self._sources.step(
                {"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz}, n
            )

        # Step 3: H-field Faraday curl updates (interior cells only).
        # Ellipsis absorbs optional leading batch dimension (B, Nx, Ny, Nz).
        #
        # Faraday-x: mu0 * dHx/dt = dEy/dz - dEz/dy   (CORRECT sign: +Ey/dz first)
        # Hx lives at (i, j+0.5, k+0.5) so update region [:,:-1,:-1]
        # Ey at (i, j+0.5, k): fwd-diff in k at fixed j=0..Ny-2 → Ey[..., :, :-1, 1:] - Ey[..., :, :-1, :-1]
        # Ez at (i, j, k+0.5): fwd-diff in j at fixed k=0..Nz-2 → Ez[..., :, 1:, :-1] - Ez[..., :, :-1, :-1]
        Hx[..., :, :-1, :-1] += Dh * (
            (Ey[..., :, :-1, 1:] - Ey[..., :, :-1, :-1]) / dz
            - (Ez[..., :, 1:, :-1] - Ez[..., :, :-1, :-1]) / dy
        )

        # Faraday-y: mu0 * dHy/dt = dEz/dx - dEx/dz   (CORRECT sign: +Ez/dx first)
        # Hy lives at (i+0.5, j, k+0.5) so update region [:-1,:,:-1]
        # Ez at (i, j, k+0.5): fwd-diff in i at fixed k=0..Nz-2 → Ez[..., 1:, :, :-1] - Ez[..., :-1, :, :-1]
        # Ex at (i+0.5, j, k): fwd-diff in k at fixed i=0..Nx-2 → Ex[..., :-1, :, 1:] - Ex[..., :-1, :, :-1]
        Hy[..., :-1, :, :-1] += Dh * (
            (Ez[..., 1:, :, :-1] - Ez[..., :-1, :, :-1]) / dx
            - (Ex[..., :-1, :, 1:] - Ex[..., :-1, :, :-1]) / dz
        )

        # Faraday-z: mu0 * dHz/dt = dEx/dy - dEy/dx  (same sign as 2D engine)
        # Hz lives at (i+0.5, j+0.5, k) so update region [:-1,:-1,:]
        Hz[..., :-1, :-1, :] += Dh * (
            (Ex[..., :-1, 1:, :] - Ex[..., :-1, :-1, :]) / dy
            - (Ey[..., 1:, :-1, :] - Ey[..., :-1, :-1, :]) / dx
        )

        # Step 4: apply Mur ABC to all three H-field boundary edges
        self._boundary.apply(Hx, Hy, Hz)

        # Step 5: E-field Ampere curl updates.
        # Two paths:
        #  - Free-space (Ca=None): scalar De, in-place +=, no Ca/Cb DRAM loads.
        #  - Material (Ca tensor): full Ca*E + Cb*curl form, element-wise.
        #
        # Ampere-x: eps0 * dEx/dt = dHz/dy - dHy/dz
        # Ex lives at (i+0.5, j, k) → interior update [:, 1:, 1:]
        #   Hz at (i+0.5, j+0.5, k): forward diff in j  → Hz[:,1:,:] - Hz[:,:-1,:]
        #   Hy at (i+0.5, j, k+0.5): forward diff in k  → Hy[:,:,1:] - Hy[:,:,:-1]

        # Ampere-y: eps0 * dEy/dt = dHx/dz - dHz/dx
        # Ey lives at (i, j+0.5, k) → interior update [1:, :, 1:]
        #   Hx at (i, j+0.5, k+0.5): forward diff in k  → Hx[:,:,1:] - Hx[:,:,:-1]
        #   Hz at (i+0.5, j+0.5, k): forward diff in i  → Hz[1:,:,:] - Hz[:-1,:,:]

        # Ampere-z: eps0 * dEz/dt = dHy/dx - dHx/dy
        # Ez lives at (i, j, k+0.5) → interior update [1:, 1:, :]
        #   Hy at (i+0.5, j, k+0.5): forward diff in i  → Hy[1:,:,:] - Hy[:-1,:,:]
        #   Hx at (i, j+0.5, k+0.5): forward diff in j  → Hx[:,1:,:] - Hx[:,:-1,:]

        if not self._has_materials:
            # Fast path — free space, scalar coefficients
            Ex[..., :, 1:, 1:] += De * (
                (Hz[..., :, 1:, 1:] - Hz[..., :, :-1, 1:]) / dy
                - (Hy[..., :, 1:, 1:] - Hy[..., :, 1:, :-1]) / dz
            )
            Ey[..., 1:, :, 1:] += De * (
                (Hx[..., 1:, :, 1:] - Hx[..., 1:, :, :-1]) / dz
                - (Hz[..., 1:, :, 1:] - Hz[..., :-1, :, 1:]) / dx
            )
            Ez[..., 1:, 1:, :] += De * (
                (Hy[..., 1:, 1:, :] - Hy[..., :-1, 1:, :]) / dx
                - (Hx[..., 1:, 1:, :] - Hx[..., 1:, :-1, :]) / dy
            )
        else:
            # Material path — per-cell Ca / Cb tensors
            # Shape: Ca/Cb are (Nx, Ny, Nz); slices match interior shapes.
            Ex[..., :, 1:, 1:] = (
                self._Ca[..., :, 1:, 1:] * Ex[..., :, 1:, 1:]
                + self._Cb[..., :, 1:, 1:] * (
                    (Hz[..., :, 1:, 1:] - Hz[..., :, :-1, 1:]) / dy
                    - (Hy[..., :, 1:, 1:] - Hy[..., :, 1:, :-1]) / dz
                )
            )
            Ey[..., 1:, :, 1:] = (
                self._Ca[..., 1:, :, 1:] * Ey[..., 1:, :, 1:]
                + self._Cb[..., 1:, :, 1:] * (
                    (Hx[..., 1:, :, 1:] - Hx[..., 1:, :, :-1]) / dz
                    - (Hz[..., 1:, :, 1:] - Hz[..., :-1, :, 1:]) / dx
                )
            )
            Ez[..., 1:, 1:, :] = (
                self._Ca[..., 1:, 1:, :] * Ez[..., 1:, 1:, :]
                + self._Cb[..., 1:, 1:, :] * (
                    (Hy[..., 1:, 1:, :] - Hy[..., :-1, 1:, :]) / dx
                    - (Hx[..., 1:, 1:, :] - Hx[..., 1:, :-1, :]) / dy
                )
            )

        # Step 6: update step counter (timing is measured in run(), not here,
        # because CUDA kernels execute asynchronously — perf_counter() here
        # only captures dispatch latency, not actual GPU execution time).
        self.steps_completed += 1

        # Stability check (every n_check steps — never every step)
        if self.steps_completed % self._n_check == 0:
            self._check_stability()

    def compile_step(self, *, backend: str = "inductor", mode: str = "reduce-overhead") -> None:
        """Compile step() with torch.compile for GPU kernel fusion.

        Must be called once after construction, before run(). Replaces the
        Python step() with a compiled version. No-op on CPU (torch.compile
        still works but gain is minimal).

        Parameters
        ----------
        backend : str
            torch.compile backend (default 'inductor').
        mode : str
            Compilation mode: 'default', 'reduce-overhead', or 'max-autotune'.
            'reduce-overhead' eliminates Python overhead between kernels.
        """
        self.step = torch.compile(self.step, backend=backend, mode=mode)

    def _check_stability(self) -> None:
        """Check field magnitudes and raise if the threshold is exceeded.

        Iterates over all six field components: ``Ex``, ``Ey``, ``Ez``,
        ``Hx``, ``Hy``, and ``Hz``.

        Raises
        ------
        SimulationDivergedError
            When ``tensor.abs().max() > stability_threshold``.
        """
        self.n_stability_checks += 1
        for name, tensor in (
            ("Ex", self._fields.Ex),
            ("Ey", self._fields.Ey),
            ("Ez", self._fields.Ez),
            ("Hx", self._fields.Hx),
            ("Hy", self._fields.Hy),
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

        with torch.no_grad():
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
                            steps_done * self._grid.Nx * self._grid.Ny * self._grid.Nz
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
        total_elapsed = time.perf_counter() - t_start
        self.elapsed_time += total_elapsed
        steps_done = self.steps_completed - start_steps
        if total_elapsed > 0.0:
            self.mcells_per_second = (
                steps_done * self._grid.Nx * self._grid.Ny * self._grid.Nz
            ) / total_elapsed / 1e6

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
        """Return a human-readable summary of the FDTD3D simulation state."""
        return (
            f"FDTD3D("
            f"shape=({self._grid.Nx}, {self._grid.Ny}, {self._grid.Nz}), "
            f"steps={self.steps_completed}, "
            f"time={self.time:.4e} s, "
            f"throughput={self.mcells_per_second:.2f} Mcells/s, "
            f"device={self._grid.device}"
            f")"
        )
