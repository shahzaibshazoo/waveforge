"""
sources.py — Electromagnetic source injection for WaveForge.

Provides waveform generators and source injectors that operate on raw PyTorch
field tensors using soft (additive) injection, fully compatible with the Yee
grid defined in grid.py.  All tensor operations remain on the device owned by
the grid; no .item(), .cpu(), or Python arithmetic appears in the hot path.

Classes
-------
Waveform (ABC)
    Abstract base for all waveform generators.
GaussianPulse
    Gaussian-envelope broadband pulse.
SinusoidalSource
    Continuous-wave sinusoidal source with smooth ramp-on envelope.
RickerWavelet
    Ricker (Mexican-hat) wavelet — second derivative of a Gaussian.
PointSource
    Single-cell soft source injector.
LineSource
    Multi-cell soft source injector along a grid axis.
SourceCollection
    Ordered collection of sources, stepped together.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Union

import torch

from .grid import YeeGrid

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

VALID_COMPONENTS: tuple[str, ...] = ("Ex", "Ey", "Hz")
"""Field components that can be targeted by a 2-D TE-mode (TE-to-z) soft source.

In 2-D TE-to-z notation the non-zero components are {Ex, Ey, Hz}, where the
electric field lies entirely in the transverse (xy) plane and Hz points along z.
(Some texts call this "TM-to-z" — the convention varies by author; the field set
{Ex, Ey, Hz} is unambiguous regardless of label.)
"""

# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class Waveform(ABC):
    """Abstract base for pre-computed FDTD source waveforms.

    Subclasses must implement :meth:`_compute`, :attr:`peak_time`, and
    :attr:`bandwidth`.  The concrete :meth:`build` and :meth:`is_causal`
    methods are provided here so that all waveforms share identical
    pre-computation and causality-check semantics.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the waveform at every time in *t*.

        Parameters
        ----------
        t : torch.Tensor
            1-D float32 time vector (seconds).

        Returns
        -------
        torch.Tensor
            Shape equal to *t*; same device and dtype.
        """

    @property
    @abstractmethod
    def peak_time(self) -> float | None:
        """Time of the waveform's peak amplitude (seconds), or ``-1.0`` when
        no single peak exists (e.g. CW sinusoidal)."""

    @property
    @abstractmethod
    def bandwidth(self) -> float:
        """Spectral bandwidth in Hz (1/e radius for Gaussian, 0 for CW)."""

    @property
    @abstractmethod
    def amplitude(self) -> float:
        """Peak signed amplitude of the waveform."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def build(
        self,
        N_steps: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Pre-compute the waveform as a 1-D tensor of length *N_steps*.

        Parameters
        ----------
        N_steps : int
            Total number of time steps in the simulation.
        dt : float
            Time step size in seconds.
        device : torch.device
            Target device for the returned tensor.
        dtype : torch.dtype
            Floating-point dtype (default ``torch.float32``).

        Returns
        -------
        torch.Tensor
            Shape ``(N_steps,)`` on *device* with *dtype*.
        """
        t = torch.arange(N_steps, device=device, dtype=dtype) * dt
        return self._compute(t)

    def is_causal(self, dt: float) -> bool:
        """Return ``True`` when |f(0)| / |amplitude| < 1e-4.

        Parameters
        ----------
        dt : float
            Time step in seconds (unused except to hint at precision; f(0) is
            evaluated at exactly t=0).

        Returns
        -------
        bool
        """
        t0_tensor = torch.tensor([0.0], dtype=torch.float32)
        f0 = float(self._compute(t0_tensor).item())
        return abs(f0) / abs(self.amplitude) < 1e-4


# ---------------------------------------------------------------------------
# GaussianPulse
# ---------------------------------------------------------------------------


class GaussianPulse(Waveform):
    """Gaussian-envelope broadband pulse.

    f(t) = A * exp(-(t - t0)^2 / (2 * sigma^2))

    Parameters
    ----------
    amplitude : float
        Peak amplitude A.  Must be non-zero.
    t0 : float | None
        Pulse centre time in seconds.  Defaults to ``4 * sigma`` when
        not given.
    sigma : float | None
        Temporal standard deviation in seconds.  Must be > 0.
        Required unless *freq* is supplied.
    freq : float | None
        Keyword-only.  Convenience: set ``sigma = 1 / (2*pi*freq/3)``
        (conservative bandwidth for a given maximum frequency).
        Mutually exclusive with *sigma*.

    Raises
    ------
    ValueError
        If neither *sigma* nor *freq* is supplied, or if *sigma* <= 0,
        or *t0* <= 0, or *amplitude* == 0.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        t0: float | None = None,
        sigma: float | None = None,
        *,
        freq: float | None = None,
    ) -> None:
        if amplitude == 0:
            raise ValueError("amplitude must be non-zero.")

        if freq is not None and sigma is not None:
            raise ValueError("Provide either freq or sigma, not both.")

        if freq is not None:
            if freq <= 0:
                raise ValueError(f"freq must be > 0, got {freq!r}")
            sigma = 1.0 / (2.0 * math.pi * freq / 3.0)

        if sigma is None:
            raise ValueError("Either sigma or freq must be supplied.")

        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma!r}")

        if t0 is None:
            t0 = 5.0 * sigma  # exp(-25/2)≈3.7e-6 < 1e-4 — satisfies is_causal

        if t0 <= 0:
            raise ValueError(f"t0 must be > 0, got {t0!r}")

        self._amplitude = float(amplitude)
        self._sigma = float(sigma)
        self._t0 = float(t0)

        if not self.is_causal(0.0):
            warnings.warn(
                f"GaussianPulse: |f(0)|/|A| >= 1e-4 — waveform is not causal. "
                f"Consider increasing t0 (currently {self._t0:.4g} s) relative "
                f"to sigma ({self._sigma:.4g} s).",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Abstract implementation
    # ------------------------------------------------------------------

    def _compute(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate f(t) = A * exp(-(t - t0)^2 / (2 * sigma^2))."""
        exponent = -((t - self._t0) ** 2) / (2.0 * self._sigma ** 2)
        return self._amplitude * torch.exp(exponent)

    @property
    def peak_time(self) -> float:
        """Return the pulse centre time *t0*."""
        return self._t0

    @property
    def bandwidth(self) -> float:
        """Return the 1/e spectral radius: 1 / (2*pi*sigma) Hz."""
        return 1.0 / (2.0 * math.pi * self._sigma)

    @property
    def amplitude(self) -> float:
        """Peak amplitude A."""
        return self._amplitude


# ---------------------------------------------------------------------------
# SinusoidalSource
# ---------------------------------------------------------------------------


class SinusoidalSource(Waveform):
    """Continuous-wave sinusoidal source with smooth ramp-on envelope.

    f(t) = A * sin(2*pi*f0*t + phi) * (1 - exp(-t / tau_ramp))

    where ``tau_ramp = 3 / (2*pi*f0)``.

    Parameters
    ----------
    amplitude : float
        Peak amplitude A.  Must be non-zero.
    frequency : float
        Carrier frequency f0 in Hz.  Must be > 0.
    phase : float
        Initial phase phi in radians (default 0.0).  A warning is issued when
        ``|sin(phi)| >= 1e-4`` because the source is then not causal.

    Raises
    ------
    ValueError
        If *frequency* <= 0 or *amplitude* == 0.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 1e9,
        phase: float = 0.0,
    ) -> None:
        if amplitude == 0:
            raise ValueError("amplitude must be non-zero.")
        if frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {frequency!r}")

        self._amplitude = float(amplitude)
        self._frequency = float(frequency)
        self._phase = float(phase)
        self._tau_ramp = 3.0 / (2.0 * math.pi * self._frequency)

        if phase != 0.0 and abs(math.sin(phase)) >= 1e-4:
            warnings.warn(
                f"SinusoidalSource: phase={phase:.4g} rad gives "
                f"|sin(phi)|={abs(math.sin(phase)):.4g} >= 1e-4 — "
                "source is not causal at t=0.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Abstract implementation
    # ------------------------------------------------------------------

    def _compute(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate f(t) = A * sin(2*pi*f0*t + phi) * (1 - exp(-t/tau_ramp))."""
        omega = 2.0 * math.pi * self._frequency
        carrier = torch.sin(omega * t + self._phase)
        ramp = 1.0 - torch.exp(-t / self._tau_ramp)
        return self._amplitude * carrier * ramp

    @property
    def peak_time(self) -> float:
        """Return sentinel -1.0 (no single peak for a CW source)."""
        return -1.0

    @property
    def bandwidth(self) -> float:
        """Return 0.0 (monochromatic CW source)."""
        return 0.0

    @property
    def amplitude(self) -> float:
        """Peak amplitude A."""
        return self._amplitude


# ---------------------------------------------------------------------------
# RickerWavelet
# ---------------------------------------------------------------------------


class RickerWavelet(Waveform):
    """Ricker (Mexican-hat) wavelet — second derivative of a Gaussian.

    f(t) = A * (1 - 2*pi^2*fp^2*(t-t0)^2) * exp(-pi^2*fp^2*(t-t0)^2)

    Parameters
    ----------
    amplitude : float
        Peak amplitude A.  Must be non-zero.
    peak_freq : float
        Peak spectral frequency fp in Hz.  Must be > 0.
    t0 : float | None
        Wavelet centre time in seconds.  Defaults to ``1.5 / peak_freq``,
        which satisfies ``|f(0)|/A < 1e-7``.

    Raises
    ------
    ValueError
        If *peak_freq* <= 0, *amplitude* == 0, or *t0* <= 0.
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        peak_freq: float = 1e9,
        t0: float | None = None,
    ) -> None:
        if amplitude == 0:
            raise ValueError("amplitude must be non-zero.")
        if peak_freq <= 0:
            raise ValueError(f"peak_freq must be > 0, got {peak_freq!r}")

        if t0 is None:
            t0 = 1.5 / peak_freq

        if t0 <= 0:
            raise ValueError(f"t0 must be > 0, got {t0!r}")

        self._amplitude = float(amplitude)
        self._peak_freq = float(peak_freq)
        self._t0 = float(t0)

        if not self.is_causal(0.0):
            warnings.warn(
                f"RickerWavelet: |f(0)|/|A| >= 1e-4 — waveform is not causal. "
                f"Consider increasing t0 (currently {self._t0:.4g} s) relative "
                f"to 1/peak_freq ({1.0/self._peak_freq:.4g} s).",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Abstract implementation
    # ------------------------------------------------------------------

    def _compute(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate f(t) = A*(1 - 2*pi^2*fp^2*(t-t0)^2)*exp(-pi^2*fp^2*(t-t0)^2)."""
        pi2_fp2 = (math.pi * self._peak_freq) ** 2
        u = pi2_fp2 * (t - self._t0) ** 2
        return self._amplitude * (1.0 - 2.0 * u) * torch.exp(-u)

    @property
    def peak_time(self) -> float:
        """Return the wavelet centre time *t0*."""
        return self._t0

    @property
    def bandwidth(self) -> float:
        """Return the peak spectral frequency fp."""
        return self._peak_freq

    @property
    def amplitude(self) -> float:
        """Peak amplitude A."""
        return self._amplitude


# ---------------------------------------------------------------------------
# PointSource
# ---------------------------------------------------------------------------


class PointSource:
    """Single-cell soft source that additively injects a waveform.

    Parameters
    ----------
    waveform : Waveform
        Pre-built waveform object.  :meth:`build` is called during
        ``__init__`` and the result is cached on the grid's device.
    i : int
        Cell index along x.  Must satisfy ``0 <= i < grid.Nx``.
    j : int
        Cell index along y.  Must satisfy ``0 <= j < grid.Ny``.
    component : str
        Target field component.  Must be one of :data:`VALID_COMPONENTS`.
    grid : YeeGrid
        Yee grid providing device, dtype, and time-step information.
    N_steps : int
        Total number of simulation time steps (used to pre-compute waveform).
    """

    def __init__(
        self,
        waveform: Waveform,
        i: int,
        j: int,
        component: str = "Hz",
        *,
        grid: YeeGrid,
        N_steps: int,
    ) -> None:
        if component not in VALID_COMPONENTS:
            raise ValueError(
                f"component must be one of {VALID_COMPONENTS}, got {component!r}"
            )
        if not (0 <= i < grid.Nx):
            raise ValueError(
                f"i={i} out of range [0, {grid.Nx})"
            )
        if not (0 <= j < grid.Ny):
            raise ValueError(
                f"j={j} out of range [0, {grid.Ny})"
            )

        self._component = component
        self._N_steps = N_steps

        self._waveform_tensor: torch.Tensor = waveform.build(
            N_steps, grid.dt, grid.device, grid.dtype
        )

        dev = grid.device
        self._i_idx = torch.tensor([i], dtype=torch.long, device=dev)
        self._j_idx = torch.tensor([j], dtype=torch.long, device=dev)
        self._k_idx = torch.tensor([0], dtype=torch.long, device=dev)
        self._amp = torch.tensor([1.0], dtype=grid.dtype, device=dev)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Number of injection cells (always 1 for a point source)."""
        return 1

    @property
    def waveform_memory_bytes(self) -> int:
        """VRAM consumed by the pre-computed waveform tensor (bytes)."""
        return self._N_steps * self._waveform_tensor.element_size()

    @property
    def is_precomputed(self) -> bool:
        """Always ``True``; waveform is pre-computed at construction time."""
        return True

    @property
    def component(self) -> str:
        """Target field component name."""
        return self._component

    # ------------------------------------------------------------------
    # Hot-path injection
    # ------------------------------------------------------------------

    def step(self, fields_dict: dict[str, torch.Tensor], n: int) -> None:
        """Inject the source value at time step *n* into *fields_dict*.

        Performs a soft (additive) injection:
        ``field[i, j, 0] += waveform[n]``

        Parameters
        ----------
        fields_dict : dict[str, torch.Tensor]
            Mapping of component name to field tensor, e.g.
            ``{"Ex": ..., "Ey": ..., "Hz": ...}``.
        n : int
            Current time step index.

        Notes
        -----
        All operations remain on the grid's device.  No ``.item()`` or
        ``.cpu()`` calls are made.
        """
        w_now = self._waveform_tensor[n : n + 1]
        values = w_now * self._amp
        target = fields_dict[self._component]
        # Ellipsis absorbs any leading batch dimension (B, Nx, Ny, Nz) as well
        # as the unbatched (Nx, Ny, Nz) case; values broadcasts over B.
        target[..., self._i_idx, self._j_idx, self._k_idx] += values


# ---------------------------------------------------------------------------
# LineSource
# ---------------------------------------------------------------------------


class LineSource:
    """Multi-cell soft source injecting along a grid axis.

    Parameters
    ----------
    waveform : Waveform
        Pre-built waveform object.
    axis : str
        ``"x"`` — vary i from *start* to *stop*-1, fix j=*position*; or
        ``"y"`` — vary j from *start* to *stop*-1, fix i=*position*.
    position : int
        Fixed index along the perpendicular axis.
    start : int
        First index along the injection axis (inclusive).
    stop : int
        One-past-the-last index along the injection axis (exclusive).
    component : str
        Target field component (one of :data:`VALID_COMPONENTS`).
    grid : YeeGrid
        Yee grid providing device, dtype, and time-step information.
    N_steps : int
        Total number of simulation time steps.
    amplitude_profile : array-like | torch.Tensor | None
        Optional per-cell amplitude weights of length ``stop - start``.
        Uniform weight 1.0 is used when ``None``.

    Raises
    ------
    ValueError
        If *axis* is not ``"x"`` or ``"y"``, *component* is invalid,
        indices are out of range, or *start* >= *stop*.
    """

    def __init__(
        self,
        waveform: Waveform,
        axis: str,
        position: int,
        start: int,
        stop: int,
        component: str = "Hz",
        *,
        grid: YeeGrid,
        N_steps: int,
        amplitude_profile: Union[torch.Tensor, list, None] = None,
    ) -> None:
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        if component not in VALID_COMPONENTS:
            raise ValueError(
                f"component must be one of {VALID_COMPONENTS}, got {component!r}"
            )
        if start >= stop:
            raise ValueError(
                f"start ({start}) must be less than stop ({stop})"
            )

        n_cells = stop - start
        dev = grid.device

        if axis == "x":
            _validate_range("i", start, stop, grid.Nx)
            _validate_scalar_index("j (position)", position, grid.Ny)
            i_vals = torch.arange(start, stop, dtype=torch.long, device=dev)
            j_vals = torch.full((n_cells,), position, dtype=torch.long, device=dev)
        else:  # axis == "y"
            _validate_range("j", start, stop, grid.Ny)
            _validate_scalar_index("i (position)", position, grid.Nx)
            i_vals = torch.full((n_cells,), position, dtype=torch.long, device=dev)
            j_vals = torch.arange(start, stop, dtype=torch.long, device=dev)

        self._component = component
        self._N_steps = N_steps
        self._n_cells = n_cells

        self._waveform_tensor: torch.Tensor = waveform.build(
            N_steps, grid.dt, grid.device, grid.dtype
        )

        self._i_idx = i_vals
        self._j_idx = j_vals
        self._k_idx = torch.zeros(n_cells, dtype=torch.long, device=dev)

        if amplitude_profile is None:
            self._amp = torch.ones(n_cells, dtype=grid.dtype, device=dev)
        else:
            amp_t = torch.as_tensor(amplitude_profile, dtype=grid.dtype, device=dev)
            if amp_t.shape != (n_cells,):
                raise ValueError(
                    f"amplitude_profile must have length {n_cells}, "
                    f"got shape {tuple(amp_t.shape)}"
                )
            self._amp = amp_t

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Number of injection cells along the source line."""
        return self._n_cells

    @property
    def waveform_memory_bytes(self) -> int:
        """VRAM consumed by the pre-computed waveform tensor (bytes)."""
        return self._N_steps * self._waveform_tensor.element_size()

    @property
    def is_precomputed(self) -> bool:
        """Always ``True``; waveform is pre-computed at construction time."""
        return True

    @property
    def component(self) -> str:
        """Target field component name."""
        return self._component

    # ------------------------------------------------------------------
    # Hot-path injection
    # ------------------------------------------------------------------

    def step(self, fields_dict: dict[str, torch.Tensor], n: int) -> None:
        """Inject the source values at time step *n* into *fields_dict*.

        Each cell in the line receives ``waveform[n] * amplitude_profile[cell]``
        additively.

        Parameters
        ----------
        fields_dict : dict[str, torch.Tensor]
            Mapping of component name to field tensor.
        n : int
            Current time step index.
        """
        w_now = self._waveform_tensor[n : n + 1]
        values = w_now * self._amp
        target = fields_dict[self._component]
        # Ellipsis absorbs any leading batch dimension (B, Nx, Ny, Nz) as well
        # as the unbatched (Nx, Ny, Nz) case; values broadcasts over B.
        target[..., self._i_idx, self._j_idx, self._k_idx] += values


# ---------------------------------------------------------------------------
# SourceCollection
# ---------------------------------------------------------------------------


class SourceCollection:
    """Ordered container that steps multiple sources together.

    Parameters
    ----------
    sources : list[PointSource | LineSource]
        List of source objects to manage.  Mixed component types are
        permitted; each source is stepped individually in order.
    """

    def __init__(self, sources: list[Union[PointSource, LineSource]]) -> None:
        self._sources: list[Union[PointSource, LineSource]] = list(sources)

    # ------------------------------------------------------------------
    # Hot-path
    # ------------------------------------------------------------------

    def step(self, fields_dict: dict[str, torch.Tensor], n: int) -> None:
        """Call :meth:`step` on every source in sequence.

        Parameters
        ----------
        fields_dict : dict[str, torch.Tensor]
            Mapping of component name to field tensor.
        n : int
            Current time step index.
        """
        for src in self._sources:
            src.step(fields_dict, n)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_vram_bytes(self) -> int:
        """Total VRAM consumed by all pre-computed waveform tensors (bytes)."""
        return sum(src.waveform_memory_bytes for src in self._sources)

    @property
    def sources_graph_compatible(self) -> bool:
        """``True`` when every source is pre-computed (always True in Phase 3)."""
        return all(src.is_precomputed for src in self._sources)

    @property
    def n_injection_cells_total(self) -> int:
        """Total number of injection cells across all sources."""
        return sum(src.n_cells for src in self._sources)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _validate_range(name: str, start: int, stop: int, limit: int) -> None:
    """Raise ValueError if the half-open interval [start, stop) is out of bounds.

    Parameters
    ----------
    name : str
        Axis/index name for error messages.
    start : int
        Inclusive lower bound.
    stop : int
        Exclusive upper bound.
    limit : int
        Grid extent along this axis (valid indices are 0 to limit-1).
    """
    if start < 0 or stop > limit:
        raise ValueError(
            f"{name} range [{start}, {stop}) is out of grid bounds [0, {limit})"
        )


def _validate_scalar_index(name: str, index: int, limit: int) -> None:
    """Raise ValueError if *index* is outside [0, limit).

    Parameters
    ----------
    name : str
        Index name for error messages.
    index : int
        Index value to check.
    limit : int
        Grid extent along this axis.
    """
    if not (0 <= index < limit):
        raise ValueError(
            f"{name}={index} is out of grid bounds [0, {limit})"
        )
