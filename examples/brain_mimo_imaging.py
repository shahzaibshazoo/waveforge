"""
brain_mimo_imaging.py — Full MIMO circular-array brain tumor detection simulation.

Builds a 2-D brain cross-section (skull + brain + optional tumor), runs 16 TX
simulations each collecting signals at 16 RX antennas, computes the scattered
field (S_tumor - S_healthy), applies delay-and-sum backprojection, and saves a
four-panel comparison plot.
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import RickerWavelet, PointSource, SourceCollection
from core.materials import MaterialMap, TISSUE_LIBRARY
from core.fdtd2d import FDTD2D

# ---------------------------------------------------------------------------
# Module-level simulation parameters
# ---------------------------------------------------------------------------

NX: int = 150
NY: int = 150
DX: float = 2e-3          # 2 mm cell size
DY: float = 2e-3
N_STEPS: int = 1000       # steps per TX simulation
N_ANTENNAS: int = 16
ARRAY_RADIUS: int = 65    # cells
CENTER: tuple[int, int] = (75, 75)
SKULL_OUTER_R: int = 55   # cells
SKULL_INNER_R: int = 51   # cells
BRAIN_R: int = 51         # = SKULL_INNER_R
TUMOR_CENTER: tuple[int, int] = (95, 75)   # offset 20 cells from centre
TUMOR_R: int = 6          # cells
FREQ: float = 1e9         # Hz
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helper: build material map
# ---------------------------------------------------------------------------

def build_brain_map(grid: YeeGrid, include_tumor: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a MaterialMap with skull, brain, and optional tumor.

    Regions are painted in order (painter's algorithm): outer skull first,
    then inner brain overwrites its interior, then the tumor.

    Parameters
    ----------
    grid : YeeGrid
    include_tumor : bool

    Returns
    -------
    (Ca, Cb) : tuple of (Nx, Ny, 1) tensors on grid.device
    """
    mat_map = MaterialMap(grid, default=TISSUE_LIBRARY['free_space'])
    mat_map.add_circle(
        center=CENTER,
        radius=SKULL_OUTER_R,
        material=TISSUE_LIBRARY['skull'],
    )
    mat_map.add_circle(
        center=CENTER,
        radius=SKULL_INNER_R,
        material=TISSUE_LIBRARY['brain'],
    )
    if include_tumor:
        mat_map.add_circle(
            center=TUMOR_CENTER,
            radius=TUMOR_R,
            material=TISSUE_LIBRARY['tumor'],
        )
    print(mat_map.summary())
    Ca, Cb = mat_map.build()
    return Ca, Cb


# ---------------------------------------------------------------------------
# Helper: antenna positions
# ---------------------------------------------------------------------------

def get_antenna_positions(
    n: int,
    radius: float,
    center: tuple[int, int],
    grid: YeeGrid,
) -> list[tuple[int, int]]:
    """Return n (i, j) integer cell positions on a circle of given radius.

    Parameters
    ----------
    n : int
        Number of antennas.
    radius : float
        Radius in cell units.
    center : (ci, cj)
        Centre in cell indices.
    grid : YeeGrid
        Used only for clamping.

    Returns
    -------
    list of (i, j) tuples
    """
    positions: list[tuple[int, int]] = []
    for k in range(n):
        angle = 2.0 * math.pi * k / n
        i_raw = center[0] + radius * math.cos(angle)
        j_raw = center[1] + radius * math.sin(angle)
        i = int(round(i_raw))
        j = int(round(j_raw))
        i = max(0, min(i, grid.Nx - 1))
        j = max(0, min(j, grid.Ny - 1))
        positions.append((i, j))
    return positions


# ---------------------------------------------------------------------------
# MIMO run: all TX simulations
# ---------------------------------------------------------------------------

def run_mimo(
    grid: YeeGrid,
    Ca: torch.Tensor,
    Cb: torch.Tensor,
    antenna_positions: list[tuple[int, int]],
    n_steps: int,
) -> np.ndarray:
    """Run all N_tx simulations, return signals array of shape (N_tx, N_rx, n_steps).

    Parameters
    ----------
    grid : YeeGrid
    Ca, Cb : (Nx, Ny, 1) material coefficient tensors
    antenna_positions : list of (i, j)
    n_steps : int

    Returns
    -------
    np.ndarray, float32, shape (N, N, n_steps)
    """
    N = len(antenna_positions)
    all_signals = np.zeros((N, N, n_steps), dtype=np.float32)

    # Pre-build receiver index tensors (reused across TX runs)
    rx_i = torch.tensor([p[0] for p in antenna_positions], dtype=torch.long, device=grid.device)
    rx_j = torch.tensor([p[1] for p in antenna_positions], dtype=torch.long, device=grid.device)

    for tx_idx in range(N):
        print(f"  TX {tx_idx + 1}/{N}", end='\r')

        # Fresh fields and boundary for each TX
        fields = FieldSet(grid)
        boundary = MurABC(grid, fields.Hz)

        # Ricker wavelet source at TX antenna position
        t0 = 1.5 / FREQ
        waveform = RickerWavelet(amplitude=1.0, peak_freq=FREQ, t0=t0)
        ti, tj = antenna_positions[tx_idx]
        src = PointSource(waveform, ti, tj, 'Hz', grid=grid, N_steps=n_steps)
        sources = SourceCollection([src])

        sim = FDTD2D(grid, fields, boundary, sources, Ca=Ca, Cb=Cb, n_check=500)

        # Signal buffer on device: (N_rx, n_steps)
        sig_buf = torch.zeros(N, n_steps, device=grid.device)

        # Step loop — record all RX at every step
        for step in range(n_steps):
            sim.step()
            sig_buf[:, step] = fields.Hz[rx_i, rx_j, 0]

        # Transfer once to CPU numpy
        all_signals[tx_idx] = sig_buf.cpu().numpy()

    print(f"  TX {N}/{N} -- done")
    return all_signals


# ---------------------------------------------------------------------------
# Delay-and-sum backprojection
# ---------------------------------------------------------------------------

def das_backprojection(
    delta_S: np.ndarray,
    antenna_positions: list[tuple[int, int]],
    grid: YeeGrid,
    n_steps: int,
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Delay-and-sum backprojection of scattered field difference.

    Parameters
    ----------
    delta_S : (N_tx, N_rx, n_steps) numpy array
        Scattered field difference.
    antenna_positions : list of (i, j) cell positions
    grid : YeeGrid
    n_steps : int
    image_shape : (Nx_img, Ny_img) or None
        Output image resolution. Defaults to (grid.Nx, grid.Ny).

    Returns
    -------
    np.ndarray, shape (Nx_img, Ny_img)
        Power image (squared coherent sum).
    """
    if image_shape is None:
        Nx_img, Ny_img = grid.Nx, grid.Ny
    else:
        Nx_img, Ny_img = image_shape

    N_tx = N_rx = len(antenna_positions)
    c0 = 299_792_458.0
    v_medium = c0 / math.sqrt(40.0)   # effective speed in brain (eps_r = 40)
    dt = grid.dt
    dx = grid.dx

    # Convert antenna positions to physical coordinates (metres)
    ant_x = np.array([p[0] * dx for p in antenna_positions])
    ant_y = np.array([p[1] * dx for p in antenna_positions])

    # Image pixel physical coordinates
    px = np.arange(Nx_img) * (grid.Nx * dx / Nx_img)
    py = np.arange(Ny_img) * (grid.Ny * dx / Ny_img)
    PX, PY = np.meshgrid(px, py, indexing='ij')   # (Nx_img, Ny_img)

    image = np.zeros((Nx_img, Ny_img), dtype=np.float64)

    for tx in range(N_tx):
        for rx in range(N_rx):
            d_tx = np.sqrt((PX - ant_x[tx]) ** 2 + (PY - ant_y[tx]) ** 2)
            d_rx = np.sqrt((PX - ant_x[rx]) ** 2 + (PY - ant_y[rx]) ** 2)
            tau = (d_tx + d_rx) / v_medium
            n_tau = np.round(tau / dt).astype(np.int32)
            n_tau = np.clip(n_tau, 0, n_steps - 1)
            image += delta_S[tx, rx, n_tau]

    return image ** 2   # power image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full MIMO brain tumor detection pipeline."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_total_start = time.perf_counter()

    print("=" * 60)
    print("GPU-MEEP: MIMO Brain Tumor Detection Simulation")
    print(f"Grid: {NX}x{NY}, dx={DX*1e3:.1f} mm, N_steps={N_STEPS}")
    print(f"Antennas: {N_ANTENNAS}, freq={FREQ/1e9:.1f} GHz, device={DEVICE}")
    print("=" * 60)

    # Build grid
    grid = YeeGrid(NX, NY, DX, DY, device=DEVICE)
    print(f"Grid dt = {grid.dt:.4e} s")

    # Antenna positions
    antenna_positions = get_antenna_positions(N_ANTENNAS, ARRAY_RADIUS, CENTER, grid)
    print(f"Antenna positions computed: {len(antenna_positions)} antennas")

    # ----------------------------------------------------------------
    # Healthy brain simulations (no tumor)
    # ----------------------------------------------------------------
    print("\nRunning healthy brain simulations (no tumor)...")
    Ca_healthy, Cb_healthy = build_brain_map(grid, include_tumor=False)
    S_healthy = run_mimo(grid, Ca_healthy, Cb_healthy, antenna_positions, N_STEPS)
    print(f"  Healthy simulations done. Shape: {S_healthy.shape}")

    # ----------------------------------------------------------------
    # Tumor brain simulations
    # ----------------------------------------------------------------
    print("\nRunning tumor brain simulations...")
    Ca_tumor, Cb_tumor = build_brain_map(grid, include_tumor=True)
    S_tumor = run_mimo(grid, Ca_tumor, Cb_tumor, antenna_positions, N_STEPS)
    print(f"  Tumor simulations done. Shape: {S_tumor.shape}")

    # ----------------------------------------------------------------
    # Scattered field difference
    # ----------------------------------------------------------------
    delta_S = S_tumor - S_healthy
    print(f"Max scattered signal: {np.abs(delta_S).max():.3e}")

    # ----------------------------------------------------------------
    # Delay-and-sum backprojection
    # ----------------------------------------------------------------
    print("Running backprojection...")
    image = das_backprojection(delta_S, antenna_positions, grid, N_STEPS)
    print(f"  DAS image computed. Shape: {image.shape}, max={image.max():.3e}")

    # ----------------------------------------------------------------
    # Four-panel plot
    # ----------------------------------------------------------------
    OUTPUT_DIR = Path(__file__).parent / 'output'
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('GPU-MEEP: MIMO Brain Tumor Detection (1 GHz)', fontsize=14)

    # Panel 1: material map (Ca shows eps contrast)
    ca_np = Ca_tumor[:, :, 0].cpu().numpy().T
    im0 = axes[0, 0].imshow(
        ca_np, origin='lower', cmap='viridis',
        extent=[0, NX * DX * 1e3, 0, NY * DY * 1e3],
    )
    axes[0, 0].set_title('Material Map (Ca coefficient)')
    axes[0, 0].set_xlabel('x (mm)')
    axes[0, 0].set_ylabel('y (mm)')
    plt.colorbar(im0, ax=axes[0, 0])
    for ai, aj in antenna_positions:
        axes[0, 0].plot(ai * DX * 1e3, aj * DY * 1e3, 'w^', markersize=4)

    # Panel 2: DAS backprojection image
    vmax = float(np.abs(image).max())
    im1 = axes[0, 1].imshow(
        image.T, origin='lower', cmap='hot',
        extent=[0, NX * DX * 1e3, 0, NY * DY * 1e3],
        vmin=0, vmax=vmax,
    )
    axes[0, 1].set_title('DAS Backprojection (Tumor Scatter Power)')
    axes[0, 1].set_xlabel('x (mm)')
    axes[0, 1].set_ylabel('y (mm)')
    plt.colorbar(im1, ax=axes[0, 1])
    tc_x = TUMOR_CENTER[0] * DX * 1e3
    tc_y = TUMOR_CENTER[1] * DY * 1e3
    circle = plt.Circle((tc_x, tc_y), TUMOR_R * DX * 1e3, fill=False, color='cyan', linewidth=2)
    axes[0, 1].add_patch(circle)
    axes[0, 1].plot(tc_x, tc_y, 'c+', markersize=10, label='True tumor')
    axes[0, 1].legend(fontsize=8)

    # Panel 3: scattered signal energy over time (sum over all TX-RX pairs)
    delta_sum = np.abs(delta_S).sum(axis=(0, 1))   # (n_steps,)
    t_axis = np.arange(N_STEPS) * grid.dt * 1e9    # ns
    axes[1, 0].plot(t_axis, delta_sum)
    axes[1, 0].set_title('Scattered Signal Energy (sum over all TX-RX)')
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].set_ylabel('|dS| sum')
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: signal comparison TX0 -> RX(N/2)
    rx_opp = N_ANTENNAS // 2
    t_axis_ns = np.arange(N_STEPS) * grid.dt * 1e9
    axes[1, 1].plot(t_axis_ns, S_healthy[0, rx_opp, :], label='Healthy', alpha=0.8)
    axes[1, 1].plot(t_axis_ns, S_tumor[0, rx_opp, :], label='With tumor', alpha=0.8)
    axes[1, 1].plot(t_axis_ns, delta_S[0, rx_opp, :] * 10, label='10x dS', linestyle='--')
    axes[1, 1].set_title(f'TX0 -> RX{rx_opp} Signal Comparison')
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].set_ylabel('Hz (A/m)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'brain_mimo_imaging.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    t_elapsed = time.perf_counter() - t_total_start
    n_tx_total = 2 * N_ANTENNAS  # healthy + tumor
    print("\n" + "=" * 60)
    print(f"Simulation complete in {t_elapsed:.1f} s")
    print(f"Total TX simulations: {n_tx_total} ({N_ANTENNAS} healthy + {N_ANTENNAS} tumor)")
    print(f"Total time steps: {n_tx_total * N_STEPS:,}")
    print(f"Peak scattered signal: {np.abs(delta_S).max():.3e}")
    print(f"DAS image peak: {image.max():.3e}")
    print(f"Output saved to: {out_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
