"""
3d_04_waveguide_te10.py — 3D FDTD rectangular waveguide TE10 mode validation.

Validates guided-wave propagation in a rectangular waveguide with perfectly-
conducting walls.  PEC walls are enforced by zeroing tangential E-field
components inside wall cell slabs after each FDTD step — the standard
FDTD PEC technique, as used by the 2D waveguide example.

A broadband Gaussian point source injects Ey at the guide centre.  The TE10
mode dominates above the cutoff frequency.

Validation checks:
  - Wave propagates along z (Ey vs z time snapshots)
  - Transverse profile Ey(x) shows the half-sine TE10 pattern
  - Ey(y) is approximately uniform (TE10 has no y variation)

Grid:   32 (x) × 16 (y) × 96 (z) cells, dx=dy=dz=1 mm
Domain: 32 mm × 16 mm × 96 mm
Guide interior: x=[4,28), y=[2,14), z=[0,96)
PEC walls: zero tangential E inside x∈[0,4), x∈[28,32), y∈[0,2), y∈[14,16)

Run:  python examples/3d/3d_04_waveguide_te10.py
Out:  examples/output/3d_04_waveguide_te10.png
"""

import sys
import time
import math
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
NX, NY, NZ = 32, 16, 96
DX = 1e-3                  # 1 mm isotropic cell spacing
N_STEPS = 500
SNAP_STEPS = [200, 350, 500]
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Waveguide interior cell bounds (inclusive start, exclusive end)
X_INT_START, X_INT_END = 4, 28    # interior x in [4, 28)
Y_INT_START, Y_INT_END = 2, 14    # interior y in [2, 14)

# Wall slabs (everything outside the interior)
# x-min wall: x in [0, X_INT_START)
# x-max wall: x in [X_INT_END, NX)
# y-min wall: y in [0, Y_INT_START)
# y-max wall: y in [Y_INT_END, NY)

# TE10 guide parameters for theoretical profile
A_CELLS = X_INT_END - X_INT_START     # 24 cells = 24 mm guide width

# Source at centre of interior cross-section, near z=8
SRC_I = (X_INT_START + X_INT_END) // 2   # = 16
SRC_J = (Y_INT_START + Y_INT_END) // 2   # = 8
SRC_K = 8                                 # near input end


def apply_pec_walls(fields: FieldSet) -> None:
    """Zero tangential E-field components in PEC wall regions.

    For a rectangular waveguide with PEC walls, the tangential E-field
    components at the conductor surface and inside the conductor must be zero.
    This implements the FDTD perfect-electric-conductor condition by forcing
    the E-field to zero inside all four wall slabs after every field update.

    Both Ex and Ey are zeroed everywhere in the wall volumes.  Ez is also
    zeroed in the wall regions to prevent spurious fields.
    """
    # x-min wall: i in [0, X_INT_START)
    fields.Ey[:X_INT_START, :, :] = 0.0
    fields.Ex[:X_INT_START, :, :] = 0.0
    fields.Ez[:X_INT_START, :, :] = 0.0

    # x-max wall: i in [X_INT_END, NX)
    fields.Ey[X_INT_END:, :, :] = 0.0
    fields.Ex[X_INT_END:, :, :] = 0.0
    fields.Ez[X_INT_END:, :, :] = 0.0

    # y-min wall: j in [0, Y_INT_START)
    fields.Ey[:, :Y_INT_START, :] = 0.0
    fields.Ex[:, :Y_INT_START, :] = 0.0
    fields.Ez[:, :Y_INT_START, :] = 0.0

    # y-max wall: j in [Y_INT_END, NY)
    fields.Ey[:, Y_INT_END:, :] = 0.0
    fields.Ex[:, Y_INT_END:, :] = 0.0
    fields.Ez[:, Y_INT_END:, :] = 0.0


def te10_profile(x_indices: np.ndarray, a_start: int, a_cells: int) -> np.ndarray:
    """Return sin(pi*(x - a_start) / a_cells) inside guide, zero in walls."""
    phase = np.pi * (x_indices - a_start) / a_cells
    return np.where(
        (x_indices >= a_start) & (x_indices < a_start + a_cells),
        np.sin(phase),
        0.0,
    )


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_start_wall = time.perf_counter()

    c0 = 299_792_458.0
    fc = c0 / (2.0 * A_CELLS * DX)     # TE10 cutoff frequency

    print("=" * 64)
    print("WaveForge 3D — Rectangular Waveguide TE10 Mode (32×16×96)")
    print(f"Grid: {NX}x{NY}x{NZ}, dx={DX*1e3:.1f} mm, device={DEVICE}")
    print(f"Domain: {NX*DX*1e3:.0f}x{NY*DX*1e3:.0f}x{NZ*DX*1e3:.0f} mm")
    print(f"Guide interior x=[{X_INT_START},{X_INT_END}), y=[{Y_INT_START},{Y_INT_END})")
    print(f"TE10 cutoff: {fc/1e9:.3f} GHz  (a={A_CELLS*DX*1e3:.0f} mm)")
    print("=" * 64)

    # ── Grid ─────────────────────────────────────────────────────────────────
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    print(f"dt = {grid.dt:.4e} s")

    # ── Fields + boundary (free-space, PEC via post-step zeroing) ─────────────
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    # ── Source ────────────────────────────────────────────────────────────────
    pulse = GaussianPulse(amplitude=1.0, sigma=20 * grid.dt)
    src = PointSource(
        pulse, SRC_I, SRC_J, component="Ey", k=SRC_K, grid=grid, N_steps=N_STEPS
    )
    sources = SourceCollection([src])

    # ── Simulator (free-space, no Ca/Cb) ─────────────────────────────────────
    sim = FDTD3D(grid, fields, boundary, sources, n_check=100)

    # ── Time-stepping loop with inline PEC enforcement ────────────────────────
    snaps: dict[int, np.ndarray] = {}

    print(f"\nRunning {N_STEPS} steps...")
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(N_STEPS):
        sim.step()
        apply_pec_walls(fields)          # enforce PEC walls every step

        if step + 1 in SNAP_STEPS:
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            snaps[step + 1] = fields.Ey.detach().cpu().numpy().copy()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mcells = N_STEPS * NX * NY * NZ / elapsed / 1e6

    print(f"Done: {elapsed:.2f}s | {mcells:.1f} Mcells/s | field_max={sim.last_field_max:.3e}")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # ── Extract final-step data (step 500) ───────────────────────────────────
    Ey_final = snaps[500]                              # shape (32, 16, 96)

    # 2D xz slice at y=SRC_J (guide centre height)
    Ey_xz = Ey_final[:, SRC_J, :]                     # shape (32, 96)

    # Ey along z at guide centre — all three time steps
    Ey_z_snaps = {s: snaps[s][SRC_I, SRC_J, :] for s in SNAP_STEPS}

    # Transverse profile at z=48 (midpoint of guide)
    Z_PROBE = NZ // 2                                  # = 48
    Ey_x_profile = Ey_final[:, SRC_J, Z_PROBE]        # shape (32,)
    Ey_y_profile = Ey_final[SRC_I, :, Z_PROBE]        # shape (16,)

    # ── Theoretical TE10 overlay ──────────────────────────────────────────────
    x_idx = np.arange(NX)
    theo_raw = te10_profile(x_idx, X_INT_START, A_CELLS)

    ey_peak_interior = np.max(np.abs(Ey_x_profile[X_INT_START:X_INT_END]))
    theo_norm = theo_raw * (ey_peak_interior if ey_peak_interior > 0 else 1.0)

    # ── Metrics ──────────────────────────────────────────────────────────────
    peak_ey = float(
        np.abs(Ey_final[X_INT_START:X_INT_END, Y_INT_START:Y_INT_END, :]).max()
    )
    print(f"\nPeak |Ey| in guide interior: {peak_ey:.4e} V/m")
    print(f"TE10 cutoff frequency: {fc/1e9:.4f} GHz")
    print(f"Throughput: {mcells:.1f} Mcells/s")

    # ── Plot ──────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "WaveForge 3D — Rectangular Waveguide TE10 Mode (32x16x96)",
        fontsize=13, fontweight="bold",
    )

    # Panel [0,0]: Ey(x,z) 2D slice at y=SRC_J, step 500
    ax = axes[0, 0]
    vmax = float(np.abs(Ey_xz).max()) or 1e-12
    im = ax.imshow(
        Ey_xz.T, origin="lower", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        extent=[0, NX * DX * 1e3, 0, NZ * DX * 1e3],
        aspect="auto",
    )
    ax.axvline(x=X_INT_START * DX * 1e3, color="k", lw=0.8, ls="--", label="walls")
    ax.axvline(x=X_INT_END * DX * 1e3, color="k", lw=0.8, ls="--")
    ax.set(
        title=f"Ey(x,z) at y={SRC_J} (guide centre), step 500",
        xlabel="x (mm)", ylabel="z (mm)",
    )
    plt.colorbar(im, ax=ax, label="Ey (V/m)")
    ax.legend(fontsize=8)

    # Panel [0,1]: Ey along z at guide centre, three time snapshots
    ax = axes[0, 1]
    z_mm = np.arange(NZ) * DX * 1e3
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for (s, ey_z), col in zip(Ey_z_snaps.items(), colors):
        ax.plot(z_mm, ey_z, color=col, lw=1.2, label=f"step {s}")
    ax.set(
        title=f"Ey along z (x={SRC_I}, y={SRC_J}) — propagation snapshots",
        xlabel="z (mm)", ylabel="Ey (V/m)",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel [1,0]: Ey(x) at z=48, y=SRC_J — TE10 half-sine validation
    ax = axes[1, 0]
    x_mm = x_idx * DX * 1e3
    ax.plot(x_mm, Ey_x_profile, "b-", lw=1.5, label="FDTD Ey")
    ax.plot(x_mm, theo_norm, "r--", lw=1.5, label=r"Theory: $A\sin(\pi(x-x_0)/a)$")
    ax.axvspan(0, X_INT_START * DX * 1e3, alpha=0.15, color="gray", label="PEC wall")
    ax.axvspan(X_INT_END * DX * 1e3, NX * DX * 1e3, alpha=0.15, color="gray")
    ax.set(
        title=f"TE10 transverse profile Ey(x) at z={Z_PROBE}mm, y={SRC_J}, step 500",
        xlabel="x (mm)", ylabel="Ey (V/m)",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel [1,1]: Ey(y) at x=SRC_I, z=48 — should be approximately uniform
    ax = axes[1, 1]
    y_mm = np.arange(NY) * DX * 1e3
    ax.plot(y_mm, Ey_y_profile, "b-", lw=1.5)
    ax.axvspan(0, Y_INT_START * DX * 1e3, alpha=0.15, color="gray", label="PEC wall")
    ax.axvspan(Y_INT_END * DX * 1e3, NY * DX * 1e3, alpha=0.15, color="gray")
    ax.set(
        title=f"Ey(y) at x={SRC_I}, z={Z_PROBE}mm, step 500 (TE10: uniform in y)",
        xlabel="y (mm)", ylabel="Ey (V/m)",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "3d_04_waveguide_te10.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    total_elapsed = time.perf_counter() - t_start_wall
    print(f"\nSaved: {out_path}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Peak |Ey| in guide interior: {peak_ey:.4e} V/m")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")
    print("=" * 64)


if __name__ == "__main__":
    main()
