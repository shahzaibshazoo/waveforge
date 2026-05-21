"""
3d_03_waveguide.py — 3D rectangular waveguide with PEC walls (TE10 mode).

3D extension of examples/03_waveguide.py.  A rectangular waveguide is formed
by enforcing PEC boundary conditions at y=0, y=31, z=0, z=31 (tangential E
zeroed after every FDTD step).  A broadband Gaussian pulse is injected via a
PlaneSource in the yz plane at x=5, exciting the Ez component.

The TE10 mode cutoff frequency is fc = c0 / (2*a) where a = 31*dx is the
waveguide width in the y-direction.  The pulse bandwidth straddles fc so that
only frequencies above cutoff propagate as guided modes, while those below
cutoff are evanescent.

The example measures the group velocity of the dominant propagating frequency
component at the detector and compares it to the analytical TE10 prediction:
    vg = c0 * sqrt(1 - (fc / f)^2)

Grid:   80 x 32 x 32, dx = 1.5 mm (domain: 120 x 48 x 48 mm)
Source: PlaneSource (yz plane) at x=5, component='Ez', GaussianPulse
PEC:    tangential E zeroed at y=0, y=31, z=0, z=31 after each step
Steps:  500
Output: examples/output/3d_03_waveguide.png (4-panel figure)

Run:  python examples/3d/3d_03_waveguide.py
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.grid import YeeGrid, C0
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PlaneSource, SourceCollection
from core.fdtd3d import FDTD3D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NX, NY, NZ = 80, 32, 32
DX = 1.5e-3                     # 1.5 mm isotropic cell spacing
N_STEPS = 500
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Waveguide dimensions (in cells): PEC walls at y=0, y=31, z=0, z=31
# Guide interior spans y in [1, 30], z in [1, 30] — width a = 31*dx
A_CELLS = 31                     # waveguide width in y (cells)
A_METERS = A_CELLS * DX          # physical width (m)

# Cutoff frequency for TE10 mode: fc = c0 / (2*a)
FC = C0 / (2.0 * A_METERS)

# Gaussian pulse centred at ~1.2 * fc so that bandwidth includes sub-cutoff
# and super-cutoff frequencies
F_CENTER = 1.2 * FC
SIGMA = 1.0 / (2.0 * math.pi * FC / 2.5)  # broad bandwidth straddling fc

# Source and detector positions
SRC_X = 5                        # PlaneSource at x=5
DET_X, DET_Y, DET_Z = 60, 16, 16  # detector position

# Snapshot step for spatial field plots
SNAP_STEP = 400


def apply_pec_walls(fields: FieldSet) -> None:
    """Zero tangential E-field components at PEC waveguide walls.

    PEC condition: tangential E = 0 at conducting surfaces.
      - y-walls (y=0, y=31): tangential components are Ex and Ez
      - z-walls (z=0, z=31): tangential components are Ex and Ey
    """
    # y-walls: zero Ex and Ez at y=0 and y=NY-1
    fields.Ex[:, 0, :] = 0.0
    fields.Ex[:, -1, :] = 0.0
    fields.Ez[:, 0, :] = 0.0
    fields.Ez[:, -1, :] = 0.0

    # z-walls: zero Ex and Ey at z=0 and z=NZ-1
    fields.Ex[:, :, 0] = 0.0
    fields.Ex[:, :, -1] = 0.0
    fields.Ey[:, :, 0] = 0.0
    fields.Ey[:, :, -1] = 0.0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Build simulation components ---
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    # Gaussian pulse source — bandwidth straddles cutoff
    pulse = GaussianPulse(amplitude=1.0, sigma=SIGMA)
    src = PlaneSource(
        pulse, plane="yz", position=SRC_X, component="Ez",
        grid=grid, N_steps=N_STEPS,
    )
    sources = SourceCollection([src])

    sim = FDTD3D(grid, fields, boundary, sources, n_check=200)

    # --- Detector time series ---
    det_ez = np.zeros(N_STEPS, dtype=np.float32)

    # --- Print physics ---
    print(f"3D Rectangular Waveguide — TE10 Mode Propagation")
    print(f"  Grid: {NX} x {NY} x {NZ}, dx = {DX*1e3:.1f} mm")
    print(f"  Domain: {NX*DX*1e3:.0f} x {NY*DX*1e3:.0f} x {NZ*DX*1e3:.0f} mm")
    print(f"  Waveguide width (y): a = {A_METERS*1e3:.1f} mm")
    print(f"  TE10 cutoff frequency: fc = {FC/1e9:.3f} GHz")
    print(f"  Pulse centre frequency: f0 = {F_CENTER/1e9:.3f} GHz")
    print(f"  Pulse bandwidth (1/e): {pulse.bandwidth/1e9:.3f} GHz")
    print(f"  Device: {DEVICE}")
    print(f"  Running {N_STEPS} steps...")

    # --- Time-stepping loop with manual PEC enforcement ---
    is_cuda = str(grid.device).startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for n in range(N_STEPS):
            sim.step()
            # Enforce PEC at waveguide walls
            apply_pec_walls(fields)
            # Record detector signal
            det_ez[n] = fields.Ez[DET_X, DET_Y, DET_Z].item()

    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    mcells = N_STEPS * NX * NY * NZ / max(elapsed, 1e-9) / 1e6
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # --- Extract field data for plotting ---
    # Panel 1: Ez at xy mid-plane (z=16), step 400
    ez_xy = fields.Ez[:, :, NZ // 2].detach().cpu().numpy().T  # shape (NY, NX)

    # Panel 2: Ez at xz mid-plane (y=16), step 400
    ez_xz = fields.Ez[:, NY // 2, :].detach().cpu().numpy().T  # shape (NZ, NX)

    # Panel 3: Ez cross-section at x=60 (yz plane) — TE10 mode shape
    ez_yz = fields.Ez[DET_X, :, :].detach().cpu().numpy().T    # shape (NZ, NY)

    # --- Frequency analysis for group velocity ---
    dt = grid.dt
    t_arr = np.arange(N_STEPS) * dt

    # Find dominant frequency via FFT of detector signal
    spectrum = np.fft.rfft(det_ez)
    freqs = np.fft.rfftfreq(N_STEPS, d=dt)
    mag = np.abs(spectrum)
    # Only consider frequencies above cutoff
    above_cutoff_mask = freqs > FC
    if np.any(above_cutoff_mask):
        mag_above = mag.copy()
        mag_above[~above_cutoff_mask] = 0.0
        f_dom_idx = np.argmax(mag_above)
        f_dominant = freqs[f_dom_idx]
    else:
        f_dominant = F_CENTER

    # Analytical group velocity at dominant frequency
    if f_dominant > FC:
        vg_analytical = C0 * math.sqrt(1.0 - (FC / f_dominant) ** 2)
    else:
        vg_analytical = 0.0

    # Measure group velocity from time of arrival at detector
    # Distance from source to detector
    dist = (DET_X - SRC_X) * DX
    # Find peak arrival time (envelope peak at detector)
    env = np.abs(det_ez)
    # Smooth envelope for peak detection
    from scipy.ndimage import uniform_filter1d
    env_smooth = uniform_filter1d(env, size=15)
    peak_idx = np.argmax(env_smooth)
    t_arrival = t_arr[peak_idx]
    # Source peak time
    t_source = pulse.peak_time

    if t_arrival > t_source and t_arrival > 0:
        vg_measured = dist / (t_arrival - t_source)
    else:
        vg_measured = 0.0

    # --- Print group velocity comparison ---
    print(f"\nGroup Velocity Analysis:")
    print(f"  Dominant frequency: f = {f_dominant/1e9:.3f} GHz")
    print(f"  Cutoff frequency:   fc = {FC/1e9:.3f} GHz")
    print(f"  Analytical vg = c0 * sqrt(1 - (fc/f)^2) = {vg_analytical/C0:.4f} c0"
          f" = {vg_analytical:.4e} m/s")
    print(f"  Measured vg (peak arrival) = {vg_measured/C0:.4f} c0"
          f" = {vg_measured:.4e} m/s")
    if vg_analytical > 0:
        error_pct = abs(vg_measured - vg_analytical) / vg_analytical * 100
        print(f"  Relative error: {error_pct:.1f}%")
    print(f"  Mode propagates above cutoff: {f_dominant > FC}")

    # --- 4-panel figure ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Ez at xy mid-plane (z=16)
    ax = axes[0, 0]
    vmax = max(np.abs(ez_xy).max(), 1e-12)
    ext_xy = [0, NX * DX * 1e3, 0, NY * DX * 1e3]
    im1 = ax.imshow(
        ez_xy, origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, extent=ext_xy, aspect="auto"
    )
    ax.axhline(0, color="black", lw=2, label="PEC wall")
    ax.axhline(NY * DX * 1e3, color="black", lw=2)
    ax.scatter(
        [DET_X * DX * 1e3], [DET_Y * DX * 1e3],
        color="yellow", s=50, zorder=5, marker="*", label="detector"
    )
    ax.set(xlabel="x (mm)", ylabel="y (mm)",
           title=f"Ez — xy plane (z={NZ//2}), step {SNAP_STEP}")
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im1, ax=ax, shrink=0.9).set_label("Ez (V/m)")

    # Panel 2: Ez at xz mid-plane (y=16)
    ax = axes[0, 1]
    vmax2 = max(np.abs(ez_xz).max(), 1e-12)
    ext_xz = [0, NX * DX * 1e3, 0, NZ * DX * 1e3]
    im2 = ax.imshow(
        ez_xz, origin="lower", cmap="RdBu_r",
        vmin=-vmax2, vmax=vmax2, extent=ext_xz, aspect="auto"
    )
    ax.axhline(0, color="black", lw=2, label="PEC wall")
    ax.axhline(NZ * DX * 1e3, color="black", lw=2)
    ax.set(xlabel="x (mm)", ylabel="z (mm)",
           title=f"Ez — xz plane (y={NY//2}), step {SNAP_STEP}")
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im2, ax=ax, shrink=0.9).set_label("Ez (V/m)")

    # Panel 3: Ez cross-section at x=60 (yz plane) — TE10 mode shape
    ax = axes[1, 0]
    vmax3 = max(np.abs(ez_yz).max(), 1e-12)
    ext_yz = [0, NY * DX * 1e3, 0, NZ * DX * 1e3]
    im3 = ax.imshow(
        ez_yz, origin="lower", cmap="RdBu_r",
        vmin=-vmax3, vmax=vmax3, extent=ext_yz, aspect="auto"
    )
    ax.set(xlabel="y (mm)", ylabel="z (mm)",
           title=f"Ez cross-section at x={DET_X} — TE10 mode shape")
    fig.colorbar(im3, ax=ax, shrink=0.9).set_label("Ez (V/m)")

    # Panel 4: Time signal at detector with group velocity annotation
    ax = axes[1, 1]
    t_ns = t_arr * 1e9
    ax.plot(t_ns, det_ez, "b-", lw=0.8, label="Ez detector")
    ax.axvline(t_source * 1e9, color="gray", ls="--", lw=0.8, label="source peak")
    if peak_idx > 0:
        ax.axvline(t_arr[peak_idx] * 1e9, color="red", ls="--", lw=0.8,
                   label=f"arrival (vg={vg_measured/C0:.3f}c)")
    ax.set(xlabel="Time (ns)", ylabel="Ez (V/m)",
           title=f"Detector ({DET_X},{DET_Y},{DET_Z}) — group velocity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Annotate group velocity comparison
    ax.text(
        0.98, 0.55,
        f"vg(analytical) = {vg_analytical/C0:.4f} c\n"
        f"vg(measured)   = {vg_measured/C0:.4f} c\n"
        f"fc = {FC/1e9:.2f} GHz\n"
        f"f_dom = {f_dominant/1e9:.2f} GHz",
        transform=ax.transAxes, fontsize=8,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        f"3D Rectangular Waveguide — TE10 Mode\n"
        f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f}mm, "
        f"fc={FC/1e9:.2f} GHz, {N_STEPS} steps",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "3d_03_waveguide.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
