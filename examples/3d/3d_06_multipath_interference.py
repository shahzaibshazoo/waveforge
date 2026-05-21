"""
3d_06_multipath_interference.py — 3D two-source coherent interference pattern.

Two coherent 2.4 GHz point sources (Ez) at symmetric positions create
constructive/destructive interference. Shows instantaneous Ez and time-averaged
|Ez|^2 standing-wave pattern across three orthogonal cut planes.

Physics:
  - Frequency: 2.4 GHz (WiFi band)
  - Wavelength: lambda = c / f = 124.9 mm
  - Fringe spacing: lambda / 2 = 62.5 mm
  - Source separation: (44-20) * 1.5 mm = 36 mm along y-axis

Run:  python examples/3d/3d_06_multipath_interference.py
Out:  examples/output/3d_06_multipath_interference.png
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.fdtd3d import FDTD3D

# ── Configuration ─────────────────────────────────────────────────────────────
NX = NY = NZ = 64
DX = 1.5e-3          # 1.5 mm cell spacing → 96 mm cube domain
FREQ = 2.4e9          # 2.4 GHz (WiFi)
N_STEPS = 400
AVG_WIN = 200         # Time-averaging window: last 200 steps
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Source positions — symmetric about y=32, along the y-axis
SRC1_POS = (32, 20, 32)   # (i, j, k)
SRC2_POS = (32, 44, 32)   # (i, j, k)

# Physical constants
C0 = 299_792_458.0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_total_start = time.perf_counter()

    # ── Derived quantities ────────────────────────────────────────────────────
    lam = C0 / FREQ                          # wavelength in metres
    fringe_spacing = lam / 2.0               # standing-wave fringe spacing
    source_sep = (SRC2_POS[1] - SRC1_POS[1]) * DX  # physical separation

    print("=" * 65)
    print("WaveForge 3D — Two-Source Coherent Interference")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm, device={DEVICE}")
    print(f"Domain: {NX*DX*1e3:.0f}x{NY*DX*1e3:.0f}x{NZ*DX*1e3:.0f} mm")
    print(f"Frequency: {FREQ/1e9:.1f} GHz, lambda={lam*1e3:.1f} mm")
    print(f"Source 1: i={SRC1_POS[0]}, j={SRC1_POS[1]}, k={SRC1_POS[2]}")
    print(f"Source 2: i={SRC2_POS[0]}, j={SRC2_POS[1]}, k={SRC2_POS[2]}")
    print(f"Source separation: {source_sep*1e3:.1f} mm = {source_sep/lam:.2f} lambda")
    print("=" * 65)

    # ── Grid and fields ───────────────────────────────────────────────────────
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    print(f"dt = {grid.dt:.4e} s")

    # ── Boundary ──────────────────────────────────────────────────────────────
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    # ── Sources: two coherent GaussianPulse Ez point sources ──────────────────
    # Use the same waveform (same sigma/t0) for both sources → coherent
    sigma = 20 * grid.dt
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)

    src1 = PointSource(pulse, SRC1_POS[0], SRC1_POS[1], "Ez",
                       k=SRC1_POS[2], grid=grid, N_steps=N_STEPS)
    src2 = PointSource(pulse, SRC2_POS[0], SRC2_POS[1], "Ez",
                       k=SRC2_POS[2], grid=grid, N_steps=N_STEPS)
    sources = SourceCollection([src1, src2])

    # ── Simulation ────────────────────────────────────────────────────────────
    sim = FDTD3D(grid, fields, boundary, sources, n_check=100)

    # Accumulators for time-averaged |Ez|^2
    ez_sq_xy = np.zeros((NX, NY), dtype=np.float64)   # z=32 plane
    ez_sq_xz = np.zeros((NX, NZ), dtype=np.float64)   # y=32 plane
    ez_sq_yz = np.zeros((NY, NZ), dtype=np.float64)   # x=32 plane

    print(f"\nRunning {N_STEPS} steps (averaging last {AVG_WIN})...")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0_bench = time.perf_counter()

    with torch.no_grad():
        for n in range(N_STEPS):
            sim.step()

            # Accumulate |Ez|^2 for time averaging over last AVG_WIN steps
            if n >= N_STEPS - AVG_WIN:
                Ez_np = fields.Ez.detach().cpu().numpy()
                ez_sq_xy += (Ez_np[:, :, 32].astype(np.float64)) ** 2
                ez_sq_xz += (Ez_np[:, 32, :].astype(np.float64)) ** 2
                ez_sq_yz += (Ez_np[32, :, :].astype(np.float64)) ** 2

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0_bench
    mcells = N_STEPS * NX * NY * NZ / max(elapsed, 1e-9) / 1e6

    print(f"Done: {elapsed:.2f}s | field_max={sim.last_field_max:.3e}")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # ── Final snapshot ────────────────────────────────────────────────────────
    Ez_final = fields.Ez.detach().cpu().numpy()
    snap_xy = Ez_final[:, :, 32]   # xy plane at z=32
    snap_xz = Ez_final[:, 32, :]   # xz plane at y=32
    snap_yz = Ez_final[32, :, :]   # yz plane at x=32

    # Normalise time-averaged fields
    ez_avg_xy = (ez_sq_xy / AVG_WIN).astype(np.float32)
    ez_avg_xz = (ez_sq_xz / AVG_WIN).astype(np.float32)
    ez_avg_yz = (ez_sq_yz / AVG_WIN).astype(np.float32)

    # ── Physics analysis: fringe spacing ──────────────────────────────────────
    # Measure fringe spacing from the time-averaged pattern along y at x=32, z=32
    profile_y = ez_avg_xy[32, :]
    # Find peaks (local maxima)
    peaks = []
    for idx in range(1, len(profile_y) - 1):
        if profile_y[idx] > profile_y[idx - 1] and profile_y[idx] > profile_y[idx + 1]:
            if profile_y[idx] > 0.1 * profile_y.max():  # threshold
                peaks.append(idx)

    if len(peaks) >= 2:
        spacings = np.diff(peaks) * DX * 1e3  # in mm
        measured_fringe = float(np.mean(spacings))
    else:
        measured_fringe = float("nan")

    expected_fringe = fringe_spacing * 1e3  # mm

    print(f"\n{'─'*45}")
    print(f"Physics:")
    print(f"  Frequency:          {FREQ/1e9:.1f} GHz")
    print(f"  Wavelength:         {lam*1e3:.1f} mm")
    print(f"  Expected lambda/2:  {expected_fringe:.1f} mm")
    if not math.isnan(measured_fringe):
        print(f"  Measured fringe:    {measured_fringe:.1f} mm")
        print(f"  Error:              {abs(measured_fringe - expected_fringe):.1f} mm")
    else:
        print(f"  Measured fringe:    (insufficient peaks for measurement)")
    print(f"  Source separation:  {source_sep*1e3:.0f} mm = {source_sep/lam:.2f} lambda")
    print(f"{'─'*45}")

    # ── Plot: 6-panel figure ──────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"WaveForge 3D — Two-Source Interference at {FREQ/1e9:.1f} GHz\n"
        f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm, "
        f"lambda={lam*1e3:.1f} mm, separation={source_sep*1e3:.0f} mm",
        fontsize=12, fontweight="bold"
    )

    ext_mm = [0, NX * DX * 1e3, 0, NY * DX * 1e3]

    # Row 1: Instantaneous Ez at step 400
    vmax = max(float(np.percentile(np.abs(snap_xy), 99)),
               float(np.percentile(np.abs(snap_xz), 99)),
               float(np.percentile(np.abs(snap_yz), 99)), 1e-12)

    # XY plane (z=32)
    ax = axes[0, 0]
    im = ax.imshow(snap_xy.T, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, extent=ext_mm, aspect="auto")
    ax.scatter([SRC1_POS[0] * DX * 1e3, SRC2_POS[0] * DX * 1e3],
              [SRC1_POS[1] * DX * 1e3, SRC2_POS[1] * DX * 1e3],
              c="white", s=80, marker="+", linewidths=2, zorder=5)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="Ez instant — XY (z=32)")
    plt.colorbar(im, ax=ax, label="Ez (V/m)")

    # XZ plane (y=32)
    ax = axes[0, 1]
    ext_xz = [0, NX * DX * 1e3, 0, NZ * DX * 1e3]
    im = ax.imshow(snap_xz.T, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, extent=ext_xz, aspect="auto")
    ax.scatter([SRC1_POS[0] * DX * 1e3], [SRC1_POS[2] * DX * 1e3],
              c="white", s=80, marker="+", linewidths=2, zorder=5)
    ax.set(xlabel="x (mm)", ylabel="z (mm)", title="Ez instant — XZ (y=32)")
    plt.colorbar(im, ax=ax, label="Ez (V/m)")

    # YZ plane (x=32)
    ax = axes[0, 2]
    ext_yz = [0, NY * DX * 1e3, 0, NZ * DX * 1e3]
    im = ax.imshow(snap_yz.T, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, extent=ext_yz, aspect="auto")
    ax.scatter([SRC1_POS[1] * DX * 1e3, SRC2_POS[1] * DX * 1e3],
              [SRC1_POS[2] * DX * 1e3, SRC2_POS[2] * DX * 1e3],
              c="white", s=80, marker="+", linewidths=2, zorder=5)
    ax.set(xlabel="y (mm)", ylabel="z (mm)", title="Ez instant — YZ (x=32)")
    plt.colorbar(im, ax=ax, label="Ez (V/m)")

    # Row 2: Time-averaged |Ez|^2
    avg_vmax = max(float(ez_avg_xy.max()),
                   float(ez_avg_xz.max()),
                   float(ez_avg_yz.max()), 1e-20)

    # XY plane (z=32)
    ax = axes[1, 0]
    im = ax.imshow(ez_avg_xy.T, origin="lower", cmap="hot",
                   vmin=0, vmax=avg_vmax, extent=ext_mm, aspect="auto")
    ax.scatter([SRC1_POS[0] * DX * 1e3, SRC2_POS[0] * DX * 1e3],
              [SRC1_POS[1] * DX * 1e3, SRC2_POS[1] * DX * 1e3],
              c="cyan", s=80, marker="+", linewidths=2, zorder=5)
    ax.set(xlabel="x (mm)", ylabel="y (mm)",
           title="|Ez|$^2$ avg — XY (z=32)")
    plt.colorbar(im, ax=ax, label="|Ez|$^2$ (V$^2$/m$^2$)")

    # XZ plane (y=32)
    ax = axes[1, 1]
    im = ax.imshow(ez_avg_xz.T, origin="lower", cmap="hot",
                   vmin=0, vmax=avg_vmax, extent=ext_xz, aspect="auto")
    ax.scatter([SRC1_POS[0] * DX * 1e3], [SRC1_POS[2] * DX * 1e3],
              c="cyan", s=80, marker="+", linewidths=2, zorder=5)
    ax.set(xlabel="x (mm)", ylabel="z (mm)",
           title="|Ez|$^2$ avg — XZ (y=32)")
    plt.colorbar(im, ax=ax, label="|Ez|$^2$ (V$^2$/m$^2$)")

    # YZ plane (x=32)
    ax = axes[1, 2]
    im = ax.imshow(ez_avg_yz.T, origin="lower", cmap="hot",
                   vmin=0, vmax=avg_vmax, extent=ext_yz, aspect="auto")
    ax.scatter([SRC1_POS[1] * DX * 1e3, SRC2_POS[1] * DX * 1e3],
              [SRC1_POS[2] * DX * 1e3, SRC2_POS[2] * DX * 1e3],
              c="cyan", s=80, marker="+", linewidths=2, zorder=5)
    ax.set(xlabel="y (mm)", ylabel="z (mm)",
           title="|Ez|$^2$ avg — YZ (x=32)")
    plt.colorbar(im, ax=ax, label="|Ez|$^2$ (V$^2$/m$^2$)")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "3d_06_multipath_interference.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed_total = time.perf_counter() - t_total_start
    print(f"\nSaved: {out_path}")
    print(f"Total wall time: {elapsed_total:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
