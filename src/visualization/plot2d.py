"""
plot2d.py — 2D field visualisation for GPU-MEEP.

Provides two public exports:

- ``plot_field``: one-shot static plot of any TM field component.
- ``FieldAnimator``: live or file-based animation driven by an FDTD2D sim.

Design constraints
------------------
- ``matplotlib.use()`` is NEVER called here.  The caller controls the backend.
- ``torch`` is imported lazily inside ``_extract_field_numpy`` only, so this
  module can be imported in headless environments that lack CUDA.
- No tensor data is stored between render calls.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Component metadata
# ---------------------------------------------------------------------------

_COMPONENT_META: dict[str, tuple[str, str]] = {
    "Hz": ("Hz field", "Hz (A/m)"),
    "Ex": ("Ex field", "Ex (V/m)"),
    "Ey": ("Ey field", "Ey (V/m)"),
}

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_field_numpy(fields: Any, component: str) -> np.ndarray:
    """Extract a 2-D ``(Ny, Nx)`` numpy array from a FieldSet or tensor dict.

    Works for both GPU and CPU tensors.  The slice ``[:, :, 0]`` drops the
    trailing Nz=1 dimension, yielding shape ``(Nx, Ny)``.  A subsequent
    transpose produces ``(Ny, Nx)`` — the row-major order expected by
    ``imshow`` with ``origin='lower'``.

    Parameters
    ----------
    fields : FieldSet | dict[str, Tensor]
        Field storage object or plain dictionary mapping component names to
        PyTorch tensors.
    component : str
        One of the keys in ``_COMPONENT_META`` (``"Hz"``, ``"Ex"``, ``"Ey"``).

    Returns
    -------
    np.ndarray
        Shape ``(Ny, Nx)``, dtype float32 or float64.

    Raises
    ------
    ValueError
        If *component* is not in ``_COMPONENT_META``.
    """
    if component not in _COMPONENT_META:
        raise ValueError(
            f"Unknown component {component!r}. "
            f"Valid choices: {list(_COMPONENT_META)}"
        )
    import torch  # noqa: F401  — lazy import, no CUDA required at module level

    if isinstance(fields, dict):
        tensor = fields[component]
    else:
        tensor = getattr(fields, component)

    # shape: (Nx, Ny, Nz=1) → drop z → (Nx, Ny) → transpose → (Ny, Nx)
    arr: np.ndarray = tensor.detach().cpu().contiguous().numpy()[:, :, 0]
    return arr.T


def _compute_vmax(arr: np.ndarray) -> float:
    """Return the maximum absolute value of *arr*, floored at 1e-12.

    Parameters
    ----------
    arr : np.ndarray
        2-D field array.

    Returns
    -------
    float
        ``max(|arr|.max(), 1e-12)``.
    """
    return max(float(np.abs(arr).max()), 1e-12)


def _get_extent(grid: Any) -> list[float] | None:
    """Return the physical extent ``[x0, x1, y0, y1]`` in millimetres.

    Parameters
    ----------
    grid : YeeGrid | None
        Yee grid providing ``Nx``, ``Ny``, ``dx``, ``dy``.  Returns ``None``
        when *grid* is ``None``.

    Returns
    -------
    list[float] | None
        ``[0.0, Nx*dx*1e3, 0.0, Ny*dy*1e3]`` or ``None``.
    """
    if grid is None:
        return None
    return [0.0, grid.Nx * grid.dx * 1e3, 0.0, grid.Ny * grid.dy * 1e3]


# ---------------------------------------------------------------------------
# Public function: plot_field
# ---------------------------------------------------------------------------


def plot_field(
    fields: Any,
    component: str = "Hz",
    ax: Any = None,
    grid: Any = None,
    sim: Any = None,
    show_energy: bool = False,
    show_grid: bool = False,
    show_source: bool = False,
    source_pos: Any = None,
    colormap: str = "RdBu_r",
    scale_mode: str = "auto",
    figsize: tuple[int, int] = (8, 6),
    title: str | None = None,
) -> tuple:
    """Render a static 2-D image of a single TM field component.

    Parameters
    ----------
    fields : FieldSet | dict[str, Tensor]
        Field storage containing the component to plot.
    component : str
        Field component to visualise.  Must be one of ``"Hz"``, ``"Ex"``,
        ``"Ey"`` (default ``"Hz"``).
    ax : matplotlib.axes.Axes | None
        Axes to draw into.  A new figure is created when ``None``.
    grid : YeeGrid | None
        Yee grid for physical axis labels and extent.  Cell indices are used
        when ``None``.
    sim : FDTD2D | None
        Simulation object.  When provided, the auto-generated title includes
        the current simulation time and step count.
    show_energy : bool
        Not implemented in Phase 6.  Emits a ``UserWarning`` when ``True``.
    show_grid : bool
        Overlay thin grey grid lines at cell boundaries (only when a *grid*
        is supplied and ``Nx <= 64``).
    show_source : bool
        Draw dashed white crosshairs at *source_pos* when ``True``.
    source_pos : sequence[float] | None
        Physical source position ``(x_m, y_m)`` in metres.
    colormap : str
        Matplotlib colourmap name (default ``"RdBu_r"``).
    scale_mode : str
        Colour scale strategy (reserved; currently only ``"auto"`` is used).
    figsize : tuple[int, int]
        Figure size in inches when a new figure is created.
    title : str | None
        Override the automatically generated axes title.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axes objects.

    Raises
    ------
    ValueError
        If *component* is not in ``_COMPONENT_META``.
    """
    if component not in _COMPONENT_META:
        raise ValueError(
            f"Unknown component {component!r}. "
            f"Valid choices: {list(_COMPONENT_META)}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    arr = _extract_field_numpy(fields, component)   # (Ny, Nx)
    vmax = _compute_vmax(arr)
    extent = _get_extent(grid)

    im = ax.imshow(
        arr,
        origin="lower",
        cmap=colormap,
        vmin=-vmax,
        vmax=vmax,
        extent=extent if extent is not None else None,
        aspect="auto",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(_COMPONENT_META[component][1])

    if grid is not None:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    else:
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("y (cells)")
        warnings.warn(
            "grid=None: axes in cell indices, not physical units.",
            UserWarning,
            stacklevel=2,
        )

    if title is not None:
        ax.set_title(title)
    elif sim is not None:
        ax.set_title(
            f"{_COMPONENT_META[component][0]}  —  "
            f"t = {sim.time * 1e9:.2f} ns  (step {sim.steps_completed})"
        )
    else:
        ax.set_title(_COMPONENT_META[component][0])

    if show_grid and grid is not None and grid.Nx <= 64:
        ax.set_xticks(
            [i * grid.dx * 1e3 for i in range(grid.Nx + 1)], minor=True
        )
        ax.set_yticks(
            [j * grid.dy * 1e3 for j in range(grid.Ny + 1)], minor=True
        )
        ax.grid(which="minor", color="grey", alpha=0.3, linewidth=0.5)

    if show_source and source_pos is not None:
        x_mm = source_pos[0] * 1e3
        y_mm = source_pos[1] * 1e3
        ax.axvline(x_mm, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(y_mm, color="white", linestyle="--", linewidth=0.8, alpha=0.7)

    if show_energy:
        warnings.warn(
            "show_energy not yet implemented in Phase 6.",
            UserWarning,
            stacklevel=2,
        )

    return fig, ax


# ---------------------------------------------------------------------------
# Public class: FieldAnimator
# ---------------------------------------------------------------------------


class FieldAnimator:
    """Live or file-based animation of a running FDTD2D simulation.

    Parameters
    ----------
    sim : FDTD2D
        Simulation to drive.  ``sim.step()`` is called *update_interval* times
        per animation frame.
    component : str
        Field component to display (default ``"Hz"``).
    update_interval : int
        Number of FDTD time steps to advance per animation frame.
    scale_mode : str
        ``"auto"`` — rescale colour limits each frame.
        ``"fixed"`` — lock colour limits to the initial field amplitude.
    colormap : str
        Matplotlib colourmap name.
    figsize : tuple[int, int]
        Figure size in inches.
    show_grid : bool
        Draw cell-boundary grid lines (``Nx <= 64`` only).
    show_source : bool
        Draw source crosshairs (requires *source_pos*).
    source_pos : sequence[float] | None
        Physical source position ``(x_m, y_m)`` in metres.
    """

    def __init__(
        self,
        sim: Any,
        component: str = "Hz",
        update_interval: int = 10,
        scale_mode: str = "auto",
        colormap: str = "RdBu_r",
        figsize: tuple[int, int] = (8, 6),
        show_grid: bool = False,
        show_source: bool = False,
        source_pos: Any = None,
    ) -> None:
        if component not in _COMPONENT_META:
            raise ValueError(
                f"Unknown component {component!r}. "
                f"Valid choices: {list(_COMPONENT_META)}"
            )

        self._sim = sim
        self._component = component
        self._update_interval = update_interval
        self._scale_mode = scale_mode
        self._colormap = colormap
        self._figsize = figsize
        self._show_grid = show_grid
        self._show_source = show_source
        self._source_pos = source_pos
        self._anim = None

        # --- build initial figure ---
        fig, ax = plt.subplots(figsize=figsize)
        arr = _extract_field_numpy(sim.fields, component)
        vmax = _compute_vmax(arr)
        grid = sim.grid if hasattr(sim, "grid") else None
        extent = _get_extent(grid)

        im = ax.imshow(
            arr,
            origin="lower",
            cmap=colormap,
            vmin=-vmax,
            vmax=vmax,
            extent=extent,
            aspect="auto",
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(_COMPONENT_META[component][1])

        if grid is not None:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
        else:
            ax.set_xlabel("x (cells)")
            ax.set_ylabel("y (cells)")

        self._fig = fig
        self._ax = ax
        self._im = im
        self._cbar = cbar
        self._vmax_fixed: float | None = vmax if scale_mode == "fixed" else None

    # ------------------------------------------------------------------
    # Animation frame callback
    # ------------------------------------------------------------------

    def _update_frame(self, frame_idx: int) -> list:
        """Advance the simulation and refresh the displayed array.

        Parameters
        ----------
        frame_idx : int
            Current frame index supplied by ``FuncAnimation``.

        Returns
        -------
        list
            List of updated artists (for ``blit`` protocol compatibility).
        """
        for _ in range(self._update_interval):
            self._sim.step()

        arr = _extract_field_numpy(self._sim.fields, self._component)

        if self._vmax_fixed is None:
            vmax = _compute_vmax(arr)
            self._im.set_clim(-vmax, vmax)
            self._cbar.update_normal(self._im)

        self._im.set_data(arr)

        if hasattr(self._sim, "time"):
            self._ax.set_title(
                f"{_COMPONENT_META[self._component][0]}  —  "
                f"t = {self._sim.time * 1e9:.2f} ns  "
                f"(step {self._sim.steps_completed})"
            )

        self._fig.canvas.draw_idle()
        return [self._im]

    # ------------------------------------------------------------------
    # Public drivers
    # ------------------------------------------------------------------

    def show(self, n_frames: int | None = None, interval_ms: int = 50) -> None:
        """Start a live interactive animation, blocking until the window closes.

        Parameters
        ----------
        n_frames : int | None
            Total number of frames to render.  ``None`` means infinite repeat.
        interval_ms : int
            Delay between frames in milliseconds.
        """
        import matplotlib.animation as animation

        self._anim = animation.FuncAnimation(
            self._fig,
            self._update_frame,
            frames=n_frames if n_frames is not None else None,
            interval=interval_ms,
            blit=False,
            repeat=n_frames is None,
        )
        plt.show()

    def save(
        self,
        filename: str,
        n_frames: int = 100,
        fps: int = 20,
        dpi: int = 150,
    ) -> None:
        """Save the animation to a file.

        Works headless — no display is required.

        Parameters
        ----------
        filename : str
            Output file path.  Suffix determines the container format:
            ``.gif`` uses PillowWriter; anything else uses FFMpegWriter.
        n_frames : int
            Total number of frames to render (default 100).
        fps : int
            Frames per second (default 20).
        dpi : int
            Dots per inch for rasterisation (default 150).
        """
        import matplotlib.animation as animation

        anim = animation.FuncAnimation(
            self._fig,
            self._update_frame,
            frames=n_frames,
            interval=1000 // fps,
            blit=False,
        )
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else "mp4"
        if suffix == "gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)

        print(f"Saving {n_frames} frames to {filename} ...")
        anim.save(filename, writer=writer, dpi=dpi)
        print(f"Saved: {filename}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fig(self) -> "plt.Figure":
        """The underlying matplotlib Figure."""
        return self._fig

    @property
    def ax(self) -> "plt.Axes":
        """The underlying matplotlib Axes."""
        return self._ax

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FieldAnimator("
            f"component={self._component!r}, "
            f"steps={self._sim.steps_completed}, "
            f"update_interval={self._update_interval}, "
            f"scale_mode={self._scale_mode!r}"
            f")"
        )
