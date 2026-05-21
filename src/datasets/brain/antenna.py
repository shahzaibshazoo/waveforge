"""
antenna.py — Antenna ring array for microwave brain haemorrhage imaging.

Places N transmit/receive elements uniformly on a ring in the axial plane,
manages sequential TX firing with simultaneous RX recording, and collects
the scattered signal matrix.

Usage:
    from datasets.brain.antenna import AntennaRing
    from core.sources import ModulatedGaussian

    ring = AntennaRing(n_elements=8, ring_radius_cells=30,
                       z_plane=32, grid=grid)
    waveform = ModulatedGaussian(amplitude=1.0, fc=1e9,
                                 sigma=3.18e-10, grid=grid)

    # For TX index 0:
    sources = ring.build_sources(waveform, tx_idx=0, N_steps=300)
    # ... run simulation ...
    # After each step n:
    ring.record(fields.Ez, step_n, tx_idx=0)

    signals = ring.get_signals()  # shape (N_tx, N_rx, N_steps)
"""

import math
from typing import Optional

import numpy as np
import torch

from core.grid import YeeGrid
from core.sources import Waveform, PointSource, SourceCollection


class AntennaRing:
    """Uniform circular array of N antenna elements in the z=const plane.

    Each element acts as both transmitter (sequential) and receiver
    (simultaneous with all other elements).  Signal recording stores
    the Ez component at each receive position at every time step.

    Parameters
    ----------
    n_elements : int
        Number of antenna elements (typically 8 or 16).
    ring_radius_cells : int
        Ring radius measured in grid cells from the phantom centre.
    z_plane : int
        z-index of the ring plane (typically Nz//2).
    grid : YeeGrid
        Simulation grid.
    centre : tuple[int, int], optional
        (cx, cy) centre of the ring in cells.  Defaults to (Nx//2, Ny//2).
    component : str
        Field component to inject and record (default 'Ez').
    """

    def __init__(
        self,
        n_elements: int,
        ring_radius_cells: int,
        z_plane: int,
        grid: YeeGrid,
        centre: Optional[tuple[int, int]] = None,
        component: str = 'Ez',
    ) -> None:
        if n_elements < 2:
            raise ValueError(f"n_elements must be >= 2, got {n_elements}")
        if not (0 < ring_radius_cells < min(grid.Nx, grid.Ny) // 2):
            raise ValueError(
                f"ring_radius_cells={ring_radius_cells} out of range for "
                f"grid ({grid.Nx}x{grid.Ny})"
            )
        if not (0 <= z_plane < grid.Nz):
            raise ValueError(f"z_plane={z_plane} out of range [0, {grid.Nz})")

        self._n = n_elements
        self._r = ring_radius_cells
        self._z = z_plane
        self._grid = grid
        self._component = component

        cx = centre[0] if centre else grid.Nx // 2
        cy = centre[1] if centre else grid.Ny // 2
        self._cx = cx
        self._cy = cy

        # Compute element positions
        self._positions: list[tuple[int, int, int]] = []
        for k in range(n_elements):
            angle = 2.0 * math.pi * k / n_elements
            i = int(round(cx + ring_radius_cells * math.cos(angle)))
            j = int(round(cy + ring_radius_cells * math.sin(angle)))
            i = max(1, min(grid.Nx - 2, i))
            j = max(1, min(grid.Ny - 2, j))
            self._positions.append((i, j, z_plane))

        # Signal buffer: (N_tx, N_rx, N_steps) — filled during generation
        self._N_steps: int = 0
        self._signals: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_elements(self) -> int:
        return self._n

    @property
    def positions(self) -> list[tuple[int, int, int]]:
        """List of (i, j, k) antenna positions in cell coordinates."""
        return list(self._positions)

    @property
    def positions_mm(self) -> list[tuple[float, float, float]]:
        """List of (x, y, z) antenna positions in mm."""
        dx = self._grid.dx * 1e3
        return [(i * dx, j * dx, k * dx) for i, j, k in self._positions]

    # ------------------------------------------------------------------
    # Source construction
    # ------------------------------------------------------------------

    def build_sources(
        self,
        waveform: Waveform,
        tx_idx: int,
        N_steps: int,
    ) -> SourceCollection:
        """Build a SourceCollection for a single TX element.

        Parameters
        ----------
        waveform : ModulatedGaussian
            Pre-constructed waveform object (carrier + Gaussian envelope).
        tx_idx : int
            Index of the transmitting element [0, n_elements).
        N_steps : int
            Total simulation steps (used to pre-compute the waveform tensor).

        Returns
        -------
        SourceCollection
            Collection containing one PointSource at the TX position.
        """
        if not (0 <= tx_idx < self._n):
            raise ValueError(f"tx_idx={tx_idx} out of range [0, {self._n})")

        self._N_steps = N_steps
        if self._signals is None:
            self._signals = np.zeros(
                (self._n, self._n, N_steps), dtype=np.float32
            )

        i, j, k = self._positions[tx_idx]
        src = PointSource(
            waveform, i, j, self._component, k=k,
            grid=self._grid, N_steps=N_steps
        )
        return SourceCollection([src])

    # ------------------------------------------------------------------
    # Signal recording
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the signal buffer for a new simulation set."""
        if self._signals is not None:
            self._signals[:] = 0.0

    def record(
        self,
        field: torch.Tensor,
        step: int,
        tx_idx: int,
    ) -> None:
        """Record the field value at all RX positions for the current step.

        Must be called once per simulation step inside the run loop.

        Parameters
        ----------
        field : torch.Tensor
            The full field tensor (Nx, Ny, Nz) for the recorded component.
        step : int
            Current time step index.
        tx_idx : int
            Index of the active transmitting element.
        """
        if self._signals is None:
            raise RuntimeError("Call build_sources() before record().")
        if step >= self._N_steps:
            return

        for rx_idx, (i, j, k) in enumerate(self._positions):
            self._signals[tx_idx, rx_idx, step] = field[i, j, k].item()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_signals(self) -> np.ndarray:
        """Return recorded signal matrix.

        Returns
        -------
        np.ndarray
            Shape (N_tx, N_rx, N_steps), dtype float32.
        """
        if self._signals is None:
            raise RuntimeError("No signals recorded yet.")
        return self._signals.copy()

    def compute_das_image(
        self,
        signals: np.ndarray,
        image_size: int = 40,
        c0: float = 3e8,
    ) -> np.ndarray:
        """Delay-and-sum backprojection in the xy-plane at z=z_plane.

        Parameters
        ----------
        signals : np.ndarray
            Shape (N_tx, N_rx, N_steps) scattered signal (background-subtracted).
        image_size : int
            Number of pixels per side in the reconstructed image.
        c0 : float
            Wave speed in medium (m/s). Use effective speed if available.

        Returns
        -------
        np.ndarray
            DAS power image, shape (image_size, image_size), dtype float32.
        """
        dt = self._grid.dt
        dx = self._grid.dx
        Nx, Ny = self._grid.Nx, self._grid.Ny

        # Image spans the full grid in x and y
        img = np.zeros((image_size, image_size), dtype=np.float64)
        x_px = np.linspace(0, (Nx - 1) * dx, image_size)
        y_px = np.linspace(0, (Ny - 1) * dx, image_size)

        # Pre-compute antenna positions in metres
        ant_xy = np.array(
            [(i * dx, j * dx) for i, j, _ in self._positions],
            dtype=np.float64
        )

        for iy, py in enumerate(y_px):
            for ix, px in enumerate(x_px):
                pix = np.array([px, py])
                acc = 0.0
                for tx in range(self._n):
                    d_tx = np.linalg.norm(pix - ant_xy[tx])
                    for rx in range(self._n):
                        d_rx = np.linalg.norm(pix - ant_xy[rx])
                        delay_idx = int((d_tx + d_rx) / c0 / dt)
                        if 0 <= delay_idx < self._N_steps:
                            acc += float(signals[tx, rx, delay_idx])
                img[iy, ix] = acc

        # Squared envelope for power image
        img = img ** 2
        return img.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"AntennaRing(n={self._n}, r={self._r} cells, "
            f"z={self._z}, component='{self._component}')"
        )
