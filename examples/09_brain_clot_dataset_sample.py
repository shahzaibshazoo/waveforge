"""
09_brain_clot_dataset_sample.py — Mini brain clot dataset generator.

Creates 4 brain phantom simulations:
  Sample 0: Healthy (no clot)
  Sample 1: Clot at left hemisphere  (80, 75),  r=8  cells
  Sample 2: Clot at right hemisphere (70, 75),  r=6  cells
  Sample 3: Clot at front            (75, 95),  r=10 cells

For each sample: runs 8 TX FDTD simulations, collects S[tx, rx, t] (8x8x800),
computes a DAS backprojection image, and saves everything to an .npz dataset.
"""

import sys
import math
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import RickerWavelet, PointSource, SourceCollection
from core.materials import MaterialMap, Material, TISSUE_LIBRARY
from core.fdtd2d import FDTD2D

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

NX: int = 150
NY: int = 150
DX: float = 2e-3          # 2 mm cell size
DY: float = 2e-3
N_STEPS: int = 800
N_TX: int = 16           # 16 TX for full angular coverage
ARRAY_RADIUS: int = 65    # cells
CENTER: tuple[int, int] = (75, 75)
SKULL_OUTER_R: int = 55
SKULL_INNER_R: int = 51
FREQ: float = 1e9         # Hz

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Clot material: hemorrhagic clot with elevated contrast
# eps_r=70 (blood+plasma, higher than standard), sigma=3.0 (elevated)
# This gives Ca contrast of ~0.002 vs brain Ca ~0.981 — 6× stronger than eps_r=55
CLOT_MAT = Material('clot', eps_r=70.0, sigma=3.0, mu_r=1.0)

# ---------------------------------------------------------------------------
# Sample definitions: (label, has_clot, clot_cx, clot_cy, clot_r)
# ---------------------------------------------------------------------------

class SampleSpec(NamedTuple):
    label: str
    has_clot: bool
    clot_cx: int
    clot_cy: int
    clot_r: int   # in cells (1 cell = 2 mm)


SAMPLES: list[SampleSpec] = [
    SampleSpec("Healthy",                     False,  0,  0,  0),
    SampleSpec("Clot at left (82,75) r=24mm",  True, 82, 75, 12),  # 12 cells = 24mm
    SampleSpec("Clot at right (68,75) r=20mm", True, 68, 75, 10),  # 10 cells = 20mm
    SampleSpec("Clot at front (75,93) r=28mm", True, 75, 93, 14),  # 14 cells = 28mm
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_antenna_positions(n: int, radius: float, center: tuple[int, int],
                          grid: YeeGrid) -> list[tuple[int, int]]:
    """Return n (i, j) integer cell positions on a circular array."""
    positions: list[tuple[int, int]] = []
    for k in range(n):
        angle = 2.0 * math.pi * k / n
        i = int(round(center[0] + radius * math.cos(angle)))
        j = int(round(center[1] + radius * math.sin(angle)))
        i = max(0, min(i, grid.Nx - 1))
        j = max(0, min(j, grid.Ny - 1))
        positions.append((i, j))
    return positions


def build_material_map(grid: YeeGrid, spec: SampleSpec
                       ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Build Ca, Cb, and eps_r map for a given sample spec."""
    mat_map = MaterialMap(grid, default=TISSUE_LIBRARY['free_space'])
    mat_map.add_circle(center=CENTER, radius=SKULL_OUTER_R,
                       material=TISSUE_LIBRARY['skull'])
    mat_map.add_circle(center=CENTER, radius=SKULL_INNER_R,
                       material=TISSUE_LIBRARY['brain'])
    if spec.has_clot:
        mat_map.add_circle(center=(spec.clot_cx, spec.clot_cy),
                           radius=spec.clot_r, material=CLOT_MAT)

    Ca, Cb = mat_map.build()

    # Build eps_r map for visualization
    Nx, Ny = grid.Nx, grid.Ny
    I = torch.arange(Nx, dtype=torch.float32).unsqueeze(1).expand(Nx, Ny)
    J = torch.arange(Ny, dtype=torch.float32).unsqueeze(0).expand(Nx, Ny)
    cx, cy = float(CENTER[0]), float(CENTER[1])
    eps_map = torch.ones(Nx, Ny)
    eps_map[((I - cx)**2 + (J - cy)**2) <= SKULL_OUTER_R**2] = 8.0
    eps_map[((I - cx)**2 + (J - cy)**2) <= SKULL_INNER_R**2] = 40.0
    if spec.has_clot:
        icx, icy = float(spec.clot_cx), float(spec.clot_cy)
        eps_map[((I - icx)**2 + (J - icy)**2) <= spec.clot_r**2] = 55.0

    return Ca, Cb, eps_map.numpy()


def run_mimo(grid: YeeGrid, Ca: torch.Tensor, Cb: torch.Tensor,
             antenna_positions: list[tuple[int, int]],
             n_steps: int, sample_label: str) -> np.ndarray:
    """Run N_TX simulations, return signals (N_TX, N_RX, n_steps)."""
    N = len(antenna_positions)
    all_signals = np.zeros((N, N, n_steps), dtype=np.float32)
    rx_i = torch.tensor([p[0] for p in antenna_positions], dtype=torch.long,
                        device=grid.device)
    rx_j = torch.tensor([p[1] for p in antenna_positions], dtype=torch.long,
                        device=grid.device)

    for tx_idx in range(N):
        print(f"    [{sample_label}] TX {tx_idx + 1}/{N}", end='\r')
        fields = FieldSet(grid)
        boundary = MurABC(grid, fields.Hz)
        waveform = RickerWavelet(amplitude=1000.0, peak_freq=FREQ)
        ti, tj = antenna_positions[tx_idx]
        src = PointSource(waveform, ti, tj, 'Hz', grid=grid, N_steps=n_steps)
        sources = SourceCollection([src])
        sim = FDTD2D(grid, fields, boundary, sources, Ca=Ca, Cb=Cb, n_check=400)
        sig_buf = torch.zeros(N, n_steps, device=grid.device)
        for step in range(n_steps):
            sim.step()
            sig_buf[:, step] = fields.Hz[rx_i, rx_j, 0]
        all_signals[tx_idx] = sig_buf.cpu().numpy()

    print(f"    [{sample_label}] TX {N}/{N} done        ")
    return all_signals


def das_backprojection(delta_S: np.ndarray,
                       antenna_positions: list[tuple[int, int]],
                       grid: YeeGrid, n_steps: int) -> np.ndarray:
    """Delay-and-sum backprojection with time-gating to suppress direct path.

    Key improvements over naive DAS:
    1. Time-gate: zero out early samples (direct TX→RX path + skull reflections)
       which are orders of magnitude stronger than clot scatter
    2. Use brain-interior effective speed (eps_r=40) for delay computation
       since we only look at signals that have traveled through brain tissue
    """
    Nx_img, Ny_img = grid.Nx, grid.Ny
    N_tx = N_rx = len(antenna_positions)
    c0 = 299_792_458.0
    # Use brain propagation speed for delay computation (signal in tissue)
    v_medium = c0 / math.sqrt(40.0)
    dt = grid.dt
    dx = grid.dx

    # Time-gate: the minimum round-trip time from any antenna to the skull inner
    # surface and back is ~2 * (array_r - skull_r) * dx / v_air ≈ 0.19 ns
    # Direct path and skull reflections arrive within the first ~300 steps.
    # Clot scatter (inside brain) arrives after ~300 steps.
    # Apply time-gate: only use signal after step 200 (0.9 ns).
    gated = delta_S.copy()
    gate_start = max(0, int(0.9e-9 / dt))  # ~200 steps at dt=4.6ps
    gated[:, :, :gate_start] = 0.0

    ant_x = np.array([p[0] * dx for p in antenna_positions])
    ant_y = np.array([p[1] * dx for p in antenna_positions])
    px = np.arange(Nx_img) * dx
    py = np.arange(Ny_img) * dx
    PX, PY = np.meshgrid(px, py, indexing='ij')
    image = np.zeros((Nx_img, Ny_img), dtype=np.float64)
    for tx in range(N_tx):
        for rx in range(N_rx):
            d_tx = np.sqrt((PX - ant_x[tx])**2 + (PY - ant_y[tx])**2)
            d_rx = np.sqrt((PX - ant_x[rx])**2 + (PY - ant_y[rx])**2)
            tau  = (d_tx + d_rx) / v_medium
            n_tau = np.clip(np.round(tau / dt).astype(np.int32), 0, n_steps - 1)
            image += gated[tx, rx, n_tau]
    return image ** 2


def compute_snr(signal: np.ndarray) -> float:
    """Compute SNR as peak-to-noise ratio in dB."""
    peak = float(np.abs(signal).max())
    noise = float(np.std(signal[:50]))  # early time = noise floor
    if noise < 1e-30:
        return 0.0
    return 20.0 * math.log10(peak / (noise + 1e-30))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    t_total_start = time.perf_counter()

    print("=" * 65)
    print("WaveForge: Brain Clot Dataset Generation (4 samples)")
    print(f"Grid: {NX}x{NY}, dx={DX*1e3:.1f} mm, N_steps={N_STEPS}, "
          f"N_TX={N_TX}, device={DEVICE}")
    print("=" * 65)

    OUTPUT_DIR = Path(__file__).parent / 'output'
    OUTPUT_DIR.mkdir(exist_ok=True)

    grid = YeeGrid(NX, NY, DX, DY, device=DEVICE)
    print(f"Grid dt = {grid.dt:.4e} s")

    antenna_positions = get_antenna_positions(N_TX, ARRAY_RADIUS, CENTER, grid)
    print(f"Antenna positions: {len(antenna_positions)} antennas on r={ARRAY_RADIUS} cell circle\n")

    # ----------------------------------------------------------------
    # Run healthy baseline once (shared reference for delta_S)
    # ----------------------------------------------------------------
    print("Sample 0: Healthy brain (baseline)")
    spec_healthy = SAMPLES[0]
    Ca_h, Cb_h, eps_h = build_material_map(grid, spec_healthy)
    signals_0 = run_mimo(grid, Ca_h, Cb_h, antenna_positions, N_STEPS,
                         spec_healthy.label)

    all_signals = [signals_0]
    all_images = []
    all_eps = [eps_h]

    # DAS for sample 0 uses raw signal (no delta) — show full signal power
    das_0 = das_backprojection(signals_0, antenna_positions, grid, N_STEPS)
    all_images.append(das_0)

    # ----------------------------------------------------------------
    # Clot samples
    # ----------------------------------------------------------------
    for idx in range(1, len(SAMPLES)):
        spec = SAMPLES[idx]
        print(f"\nSample {idx}: {spec.label}")
        Ca_c, Cb_c, eps_c = build_material_map(grid, spec)
        all_eps.append(eps_c)
        signals_i = run_mimo(grid, Ca_c, Cb_c, antenna_positions, N_STEPS,
                             spec.label)
        all_signals.append(signals_i)
        delta_S = signals_i - signals_0
        print(f"  Max |delta_S|: {np.abs(delta_S).max():.3e}")
        das_i = das_backprojection(delta_S, antenna_positions, grid, N_STEPS)
        all_images.append(das_i)
        print(f"  DAS peak: {das_i.max():.3e}")

    # ----------------------------------------------------------------
    # Build labels array [sample_id, has_clot, clot_x, clot_y, clot_r_mm]
    # ----------------------------------------------------------------
    labels = np.array([
        [0, 0,  0,  0,  0],
        [1, 1, 82, 75, 24],
        [2, 1, 68, 75, 20],
        [3, 1, 75, 93, 28],
    ], dtype=np.float32)

    # ----------------------------------------------------------------
    # Save dataset
    # ----------------------------------------------------------------
    dataset_path = OUTPUT_DIR / 'brain_clot_dataset.npz'
    np.savez(
        dataset_path,
        signals_0=all_signals[0], signals_1=all_signals[1],
        signals_2=all_signals[2], signals_3=all_signals[3],
        image_0=all_images[0],   image_1=all_images[1],
        image_2=all_images[2],   image_3=all_images[3],
        labels=labels,
    )
    print(f"\nDataset saved to: {dataset_path}")

    # ----------------------------------------------------------------
    # Plot: 4 rows x 3 columns
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle('WaveForge: Brain Clot Dataset (4 Samples x 8-TX MIMO)', fontsize=13)

    row_titles = [
        "Sample 0: Healthy",
        "Sample 1: Clot at left (80,75) r=8mm",
        "Sample 2: Clot at right (70,75) r=6mm",
        "Sample 3: Clot at front (75,95) r=10mm",
    ]

    bounds = [0, 5, 15, 45, 60]
    cmap4 = plt.colormaps.get_cmap('tab10').resampled(4)
    norm4 = BoundaryNorm(bounds, cmap4.N)
    tick_vals = [2.5, 10, 42.5, 57.5]
    tick_labels = ['Free\n(e=1)', 'Skull\n(e=8)', 'Brain\n(e=40)', 'Clot\n(e=55)']
    extent_mm = [0, NX * DX * 1e3, 0, NY * DY * 1e3]

    for row in range(4):
        spec = SAMPLES[row]
        eps_np = all_eps[row].T  # (Ny, Nx) for imshow

        # Col 0: eps_r material map
        ax0 = axes[row, 0]
        im0 = ax0.imshow(eps_np, origin='lower', cmap=cmap4, norm=norm4,
                         extent=extent_mm)
        cbar0 = plt.colorbar(im0, ax=ax0, ticks=tick_vals)
        cbar0.ax.set_yticklabels(tick_labels, fontsize=6)
        for ai, aj in antenna_positions:
            ax0.plot(ai * DX * 1e3, aj * DY * 1e3, 'w^', markersize=4,
                     markeredgecolor='k', markeredgewidth=0.4)
        ax0.set_title(f'{row_titles[row]}\nMaterial Map (eps_r)', fontsize=8)
        ax0.set_xlabel('x (mm)', fontsize=7)
        ax0.set_ylabel('y (mm)', fontsize=7)

        # Col 1: DAS backprojection
        ax1 = axes[row, 1]
        das_img = all_images[row]
        vmax_das = float(np.abs(das_img).max()) or 1.0
        im1 = ax1.imshow(das_img.T, origin='lower', cmap='hot',
                         extent=extent_mm, vmin=0, vmax=vmax_das)
        plt.colorbar(im1, ax=ax1)
        ax1.set_title(f'{row_titles[row]}\nDAS Backprojection', fontsize=8)
        ax1.set_xlabel('x (mm)', fontsize=7)
        ax1.set_ylabel('y (mm)', fontsize=7)
        if spec.has_clot:
            cx_mm = spec.clot_cx * DX * 1e3
            cy_mm = spec.clot_cy * DY * 1e3
            r_mm = spec.clot_r * DX * 1e3
            circle = plt.Circle((cx_mm, cy_mm), r_mm,
                                 fill=False, color='cyan', linewidth=1.5)
            ax1.add_patch(circle)
            ax1.plot(cx_mm, cy_mm, 'c+', markersize=8)

        # Col 2: time series
        ax2 = axes[row, 2]
        t_ns = np.arange(N_STEPS) * grid.dt * 1e9
        rx_opp = N_TX // 2   # antenna 4 is roughly opposite TX 0
        if row == 0:
            ax2.plot(t_ns, all_signals[0][0, rx_opp, :], color='steelblue', lw=0.8)
            ax2.set_title(f'{row_titles[row]}\nRaw S[0,{rx_opp},:]', fontsize=8)
            ax2.set_ylabel('Hz (A/m)', fontsize=7)
        else:
            delta_sig = all_signals[row][0, rx_opp, :] - all_signals[0][0, rx_opp, :]
            ax2.plot(t_ns, delta_sig, color='crimson', lw=0.8)
            ax2.set_title(f'{row_titles[row]}\ndelta_S[0,{rx_opp},:] vs healthy', fontsize=8)
            ax2.set_ylabel('delta Hz (A/m)', fontsize=7)
        ax2.set_xlabel('Time (ns)', fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = OUTPUT_DIR / '09_brain_clot_dataset_sample.png'
    fig.savefig(plot_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    t_total = time.perf_counter() - t_total_start
    print("\n" + "=" * 65)
    print(f"{'Sample':<8} {'has_clot':<10} {'clot_pos':<14} "
          f"{'DAS_peak':<14} {'signal_SNR':<10}")
    print("-" * 65)
    for row in range(4):
        spec = SAMPLES[row]
        das_peak = float(all_images[row].max())
        snr = compute_snr(all_signals[row][0, N_TX // 2, :])
        clot_str = f"({spec.clot_cx},{spec.clot_cy})" if spec.has_clot else "N/A"
        print(f"{row:<8} {int(spec.has_clot):<10} {clot_str:<14} "
              f"{das_peak:<14.3e} {snr:<10.1f}")
    print("=" * 65)
    print(f"Total time: {t_total:.1f} s")
    print(f"Dataset saved to: {dataset_path}")
    print(f"Plot saved to:    {plot_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
