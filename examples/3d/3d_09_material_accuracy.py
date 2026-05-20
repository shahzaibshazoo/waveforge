"""
3d_09_material_accuracy.py — 3D FDTD multi-layer tissue material accuracy benchmark.

Physics accuracy validation: EM wave attenuation through a skin-fat-muscle tissue
slab is compared against the analytical complex-propagation-constant prediction.

Simulation: PlaneSource (Ez) illuminates a layered tissue slab along z.
  - Air gap:  z=[0,10]   — free space
  - Skin:     z=[10,14]  — eps_r=38, sigma=1.4 S/m  (4 mm)
  - Fat:      z=[14,35]  — eps_r=5,  sigma=0.05 S/m (21 mm)
  - Muscle:   z=[35,65]  — eps_r=52, sigma=1.7 S/m  (30 mm)
  - Air gap:  z=[65,80]  — free space

Grid: 16×16×80, dx=dy=dz=1 mm, 600 steps.

Output: examples/output/3d_09_material_accuracy.png

Run:  python examples/3d/3d_09_material_accuracy.py
"""

import sys
import math
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import GaussianPulse, PlaneSource, SourceCollection
from core.fdtd3d import FDTD3D
from core.materials import Material, MaterialMap3D, TISSUE_LIBRARY

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NX, NY, NZ = 16, 16, 80
DX = 1e-3                          # 1 mm isotropic cell spacing
N_STEPS = 600
RECORD_EVERY = 50                  # record Ez along z every N steps
OUTPUT_DIR = Path(__file__).parent.parent / "output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Layer boundaries in cell indices (along z)
Z_SKIN_START   = 10
Z_SKIN_END     = 14
Z_FAT_START    = 14
Z_FAT_END      = 35
Z_MUSCLE_START = 35
Z_MUSCLE_END   = 65

# Source position
SRC_K = 5   # PlaneSource injected at z=5 (in the air gap)

# Probe position (centre of transverse plane)
IX_CENTRE = NX // 2   # = 8
IY_CENTRE = NY // 2   # = 8

# Tissue monitors at specific z-positions (for time-series panel)
Z_PROBES = [8, 20, 45, 70]

# ---------------------------------------------------------------------------
# Tissue material definitions (at ~GHz frequencies typical for microwave imaging)
# ---------------------------------------------------------------------------

MAT_SKIN   = Material("skin",   eps_r=38.0, sigma=1.4,  mu_r=1.0)
MAT_FAT    = Material("fat",    eps_r=5.0,  sigma=0.05, mu_r=1.0)
MAT_MUSCLE = Material("muscle", eps_r=52.0, sigma=1.7,  mu_r=1.0)


# ---------------------------------------------------------------------------
# Analytical attenuation helper
# ---------------------------------------------------------------------------

def attenuation_db_per_m(eps_r: float, sigma_cond: float, freq: float) -> float:
    """Alpha in dB/m for a lossy dielectric at frequency *freq* (Hz)."""
    omega = 2.0 * math.pi * freq
    eps0 = 8.854e-12
    mu0 = 4.0 * math.pi * 1e-7
    eps_complex = eps_r - 1j * sigma_cond / (omega * eps0)
    k = omega * np.sqrt(mu0 * eps0 * eps_complex + 0j)
    alpha = abs(k.imag)          # Np/m
    return alpha * 20.0 / math.log(10)   # dB/m


def measure_attenuation_db(ez_envelope: np.ndarray, z1: int, z2: int) -> float:
    """Measure attenuation in dB between two z-indices from the field envelope.

    Uses the peak |Ez| values at *z1* and *z2*.  Returns nan when either
    peak is too small to be meaningful (<1e-20).
    """
    amp_in  = float(ez_envelope[z1])
    amp_out = float(ez_envelope[z2])
    if amp_in < 1e-20 or amp_out < 1e-20:
        return float("nan")
    return 20.0 * math.log10(amp_in / amp_out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_wall_start = time.perf_counter()

    # ── Grid ─────────────────────────────────────────────────────────────────
    grid = YeeGrid(NX, NY, dx=DX, dy=DX, Nz=NZ, dz=DX, device=DEVICE)
    dt = grid.dt

    # Pulse sigma and derived centre frequency
    sigma_time = 20.0 * dt
    f0 = 1.0 / (2.0 * math.pi * sigma_time)   # Hz — Gaussian bandwidth centre

    print("=" * 45)
    print("WaveForge 3D — Multi-Layer Tissue Accuracy")
    print(f"Grid: {NX}x{NY}x{NZ}, dx=1mm, {N_STEPS} steps")
    print(f"Layers: Air | Skin(4mm) | Fat(21mm) | Muscle(30mm) | Air")
    print(f"Source: PlaneSource Ez at z=5mm, f0~{f0/1e9:.1f} GHz")
    print(f"Device: {DEVICE}")
    print("-" * 45)

    # ── Materials ────────────────────────────────────────────────────────────
    mm = MaterialMap3D(grid)
    # Painter's algorithm — add layers in z order; last written wins.
    mm.add_box(
        corner_min=(0, 0, Z_SKIN_START),
        corner_max=(NX - 1, NY - 1, Z_SKIN_END - 1),
        material=MAT_SKIN,
    )
    mm.add_box(
        corner_min=(0, 0, Z_FAT_START),
        corner_max=(NX - 1, NY - 1, Z_FAT_END - 1),
        material=MAT_FAT,
    )
    mm.add_box(
        corner_min=(0, 0, Z_MUSCLE_START),
        corner_max=(NX - 1, NY - 1, Z_MUSCLE_END - 1),
        material=MAT_MUSCLE,
    )
    Ca, Cb = mm.build3d()

    # ── Fields + boundary ────────────────────────────────────────────────────
    fields = FieldSet(grid)
    boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

    # ── Source ───────────────────────────────────────────────────────────────
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma_time)
    plane_src = PlaneSource(
        pulse, plane="xy", position=SRC_K, component="Ez",
        grid=grid, N_steps=N_STEPS,
    )
    sources = SourceCollection([plane_src])

    # ── Simulator ────────────────────────────────────────────────────────────
    sim = FDTD3D(grid, fields, boundary, sources, Ca=Ca, Cb=Cb, n_check=100)

    # ── Time-stepping with recording ─────────────────────────────────────────
    # ez_snapshots: list of 1-D arrays (shape Nz) along z at centre (x=8, y=8)
    ez_snapshots: list[tuple[int, np.ndarray]] = []

    # Time series at four z-probe positions
    ez_timeseries: dict[int, list[float]] = {z: [] for z in Z_PROBES}

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for step in range(N_STEPS):
            sim.step()

            # Record Ez along z-axis centre every RECORD_EVERY steps
            if (step + 1) % RECORD_EVERY == 0:
                snap = fields.Ez[IX_CENTRE, IY_CENTRE, :].detach().cpu().numpy().copy()
                ez_snapshots.append((step + 1, snap))

            # Record time series at each probe
            for z_p in Z_PROBES:
                val = float(fields.Ez[IX_CENTRE, IY_CENTRE, z_p].item())
                ez_timeseries[z_p].append(val)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mcells = N_STEPS * NX * NY * NZ / elapsed / 1e6

    # ── Build envelope: max |Ez| across all recorded snapshots ───────────────
    # Peak amplitude profile: for each z-cell take the maximum over all snapshots
    all_snaps = np.array([s for _, s in ez_snapshots])   # (n_snaps, Nz)
    ez_envelope = np.max(np.abs(all_snaps), axis=0)       # (Nz,)

    # Snapshot at step ~400 (8th snapshot, index 7)
    step_400_idx = next(
        (i for i, (s, _) in enumerate(ez_snapshots) if s >= 400), -1
    )
    ez_step400 = ez_snapshots[step_400_idx][1] if step_400_idx >= 0 else all_snaps[-1]

    # ── Analytical reference ─────────────────────────────────────────────────
    alpha_skin_db_m   = attenuation_db_per_m(MAT_SKIN.eps_r,   MAT_SKIN.sigma,   f0)
    alpha_fat_db_m    = attenuation_db_per_m(MAT_FAT.eps_r,    MAT_FAT.sigma,    f0)
    alpha_muscle_db_m = attenuation_db_per_m(MAT_MUSCLE.eps_r, MAT_MUSCLE.sigma, f0)

    # Analytical total dB loss through each layer (alpha * thickness)
    skin_thickness_m   = (Z_SKIN_END   - Z_SKIN_START)   * DX   # 4 mm
    fat_thickness_m    = (Z_FAT_END    - Z_FAT_START)    * DX   # 21 mm
    muscle_thickness_m = (Z_MUSCLE_END - Z_MUSCLE_START) * DX   # 30 mm

    skin_theory_db   = alpha_skin_db_m   * skin_thickness_m
    fat_theory_db    = alpha_fat_db_m    * fat_thickness_m
    muscle_theory_db = alpha_muscle_db_m * muscle_thickness_m

    # ── Numerical attenuation measurement ───────────────────────────────────
    # Use one cell inside each boundary to avoid interface reflections
    skin_meas_db = measure_attenuation_db(
        ez_envelope, Z_SKIN_START + 1, Z_SKIN_END - 1
    )
    fat_meas_db = measure_attenuation_db(
        ez_envelope, Z_FAT_START + 1, Z_FAT_END - 1
    )
    muscle_meas_db = measure_attenuation_db(
        ez_envelope, Z_MUSCLE_START + 1, Z_MUSCLE_END - 1
    )

    # Analytical attenuation over the same slightly-shorter spans used above
    skin_theory_meas   = alpha_skin_db_m   * (Z_SKIN_END   - Z_SKIN_START   - 2) * DX
    fat_theory_meas    = alpha_fat_db_m    * (Z_FAT_END    - Z_FAT_START    - 2) * DX
    muscle_theory_meas = alpha_muscle_db_m * (Z_MUSCLE_END - Z_MUSCLE_START - 2) * DX

    def pct_error(meas: float, theory: float) -> float:
        if math.isnan(meas) or theory == 0.0:
            return float("nan")
        return abs(meas - theory) / abs(theory) * 100.0

    err_skin   = pct_error(skin_meas_db,   skin_theory_meas)
    err_fat    = pct_error(fat_meas_db,    fat_theory_meas)
    err_muscle = pct_error(muscle_meas_db, muscle_theory_meas)

    valid_errors = [e for e in [err_skin, err_fat, err_muscle] if not math.isnan(e)]
    mean_error = sum(valid_errors) / len(valid_errors) if valid_errors else float("nan")
    PASS_THRESHOLD = 25.0
    result_str = "PASS" if (not math.isnan(mean_error) and mean_error < PASS_THRESHOLD) else "FAIL"

    # ── Derived alpha_measured (dB/m) from layer measurements ────────────────
    def meas_db_per_m(meas_db: float, span_cells: int) -> float:
        if math.isnan(meas_db) or span_cells <= 0:
            return float("nan")
        return meas_db / (span_cells * DX)

    alpha_skin_meas   = meas_db_per_m(skin_meas_db,   Z_SKIN_END   - Z_SKIN_START   - 2)
    alpha_fat_meas    = meas_db_per_m(fat_meas_db,    Z_FAT_END    - Z_FAT_START    - 2)
    alpha_muscle_meas = meas_db_per_m(muscle_meas_db, Z_MUSCLE_END - Z_MUSCLE_START - 2)

    # ── Print report ─────────────────────────────────────────────────────────
    print("=" * 45)
    print("WaveForge 3D — Multi-Layer Tissue Accuracy")
    print(f"Grid: {NX}x{NY}x{NZ}, dx=1mm, {N_STEPS} steps")
    print(f"Layers: Air | Skin(4mm) | Fat(21mm) | Muscle(30mm) | Air")
    print(f"Source: PlaneSource Ez at z=5mm, f0~{f0/1e9:.1f} GHz")
    print("-" * 45)
    print(
        f"{'Layer':<12}{'alpha_theory(dB/m)':>20}"
        f"{'alpha_measured(dB/m)':>22}{'Error%':>8}"
    )

    def fmt(v: float) -> str:
        return f"{v:8.1f}" if not math.isnan(v) else "     N/A"

    print(
        f"{'Skin':<12}{alpha_skin_db_m:>20.1f}"
        f"{fmt(alpha_skin_meas):>22}{fmt(err_skin):>8}"
    )
    print(
        f"{'Fat':<12}{alpha_fat_db_m:>20.1f}"
        f"{fmt(alpha_fat_meas):>22}{fmt(err_fat):>8}"
    )
    print(
        f"{'Muscle':<12}{alpha_muscle_db_m:>20.1f}"
        f"{fmt(alpha_muscle_meas):>22}{fmt(err_muscle):>8}"
    )
    print("-" * 45)
    print(f"Mean accuracy error: {mean_error:.1f}%")
    print(f"Result: {result_str} (threshold: {PASS_THRESHOLD:.0f}%)")
    print("=" * 45)
    print(f"Throughput: {mcells:.1f} Mcells/s  |  elapsed: {elapsed:.2f}s")
    print(f"WAVEFORGE_BENCH: {mcells:.1f} Mcells/s")

    # ── Plotting ─────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)

    z_mm = np.arange(NZ) * DX * 1e3    # z positions in mm

    # Colour-coded layer bands
    layer_shading = [
        (Z_SKIN_START   * DX * 1e3, Z_SKIN_END   * DX * 1e3, "#FF9999", "Skin"),
        (Z_FAT_START    * DX * 1e3, Z_FAT_END    * DX * 1e3, "#FFEE99", "Fat"),
        (Z_MUSCLE_START * DX * 1e3, Z_MUSCLE_END * DX * 1e3, "#99BBFF", "Muscle"),
    ]

    def add_layer_shading(ax: plt.Axes, add_legend: bool = True) -> None:
        for z0, z1, color, label in layer_shading:
            ax.axvspan(z0, z1, alpha=0.25, color=color,
                       label=label if add_legend else None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "WaveForge 3D — Multi-Layer Tissue Attenuation Accuracy",
        fontsize=13, fontweight="bold",
    )

    # ── Panel [0,0]: Ez along z at step ~400 ─────────────────────────────────
    ax = axes[0, 0]
    add_layer_shading(ax)
    ax.plot(z_mm, ez_step400, color="tab:blue", lw=1.2, label=f"Ez (step ~400)")
    ax.set(
        title=f"Ez amplitude along z-axis (step ~400, x={IX_CENTRE}, y={IY_CENTRE})",
        xlabel="z (mm)", ylabel="Ez (V/m)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel [0,1]: Envelope |Ez| vs z with analytical decay overlay ─────────
    ax = axes[0, 1]
    add_layer_shading(ax)
    ax.plot(z_mm, ez_envelope, color="tab:blue", lw=1.5, label="|Ez| envelope (FDTD)")

    # Build analytical decay curve piecewise
    # Normalise at z just before the skin layer
    ref_z = Z_SKIN_START - 1
    ref_amp = float(ez_envelope[ref_z]) if float(ez_envelope[ref_z]) > 1e-20 else 1.0
    z_idx = np.arange(NZ, dtype=float)
    analytical = np.zeros(NZ)

    for k in range(NZ):
        if k < Z_SKIN_START:
            # Air: no attenuation
            analytical[k] = ref_amp
        elif k < Z_SKIN_END:
            depth = (k - Z_SKIN_START) * DX
            analytical[k] = ref_amp * np.exp(
                -alpha_skin_db_m / (20.0 / math.log(10)) * depth
            )
        elif k < Z_FAT_END:
            skin_exit_amp = ref_amp * np.exp(
                -alpha_skin_db_m / (20.0 / math.log(10)) * skin_thickness_m
            )
            depth = (k - Z_FAT_START) * DX
            analytical[k] = skin_exit_amp * np.exp(
                -alpha_fat_db_m / (20.0 / math.log(10)) * depth
            )
        elif k < Z_MUSCLE_END:
            fat_exit_amp = ref_amp * np.exp(
                -(alpha_skin_db_m / (20.0 / math.log(10)) * skin_thickness_m
                  + alpha_fat_db_m / (20.0 / math.log(10)) * fat_thickness_m)
            )
            depth = (k - Z_MUSCLE_START) * DX
            analytical[k] = fat_exit_amp * np.exp(
                -alpha_muscle_db_m / (20.0 / math.log(10)) * depth
            )
        else:
            analytical[k] = analytical[Z_MUSCLE_END - 1]

    ax.plot(z_mm, analytical, "r--", lw=1.5, label="Analytical exp decay")
    ax.set(
        title="Peak |Ez| envelope vs z — FDTD vs analytical",
        xlabel="z (mm)", ylabel="|Ez| peak (V/m)",
    )
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel [1,0]: Ez time series at 4 probe positions ─────────────────────
    ax = axes[1, 0]
    t_ns = np.arange(N_STEPS) * dt * 1e9    # time in nanoseconds
    colors_probe = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for z_p, col in zip(Z_PROBES, colors_probe):
        ts = np.array(ez_timeseries[z_p])
        ax.plot(t_ns, ts, color=col, lw=1.0, label=f"z={z_p}mm")
    ax.set(
        title=f"Ez time series at probe depths (x={IX_CENTRE}, y={IY_CENTRE})",
        xlabel="time (ns)", ylabel="Ez (V/m)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel [1,1]: Attenuation bar chart ───────────────────────────────────
    ax = axes[1, 1]
    layers_bar = ["Skin", "Fat", "Muscle"]
    theory_vals = [skin_theory_meas, fat_theory_meas, muscle_theory_meas]
    meas_vals   = [
        skin_meas_db if not math.isnan(skin_meas_db) else 0.0,
        fat_meas_db  if not math.isnan(fat_meas_db)  else 0.0,
        muscle_meas_db if not math.isnan(muscle_meas_db) else 0.0,
    ]

    x_bar = np.arange(len(layers_bar))
    w = 0.35
    bars_theory = ax.bar(x_bar - w / 2, theory_vals, w,
                         color="tab:blue", alpha=0.8, label="Theory (alpha*d)")
    bars_meas   = ax.bar(x_bar + w / 2, meas_vals, w,
                         color="tab:orange", alpha=0.8, label="FDTD measured")

    for i, (th, ms) in enumerate(zip(theory_vals, meas_vals)):
        if ms > 0:
            err_i = abs(ms - th) / abs(th) * 100.0 if th != 0 else float("nan")
            ax.text(
                x_bar[i], max(th, ms) + 0.02 * max(theory_vals),
                f"{err_i:.1f}%", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x_bar)
    ax.set_xticklabels(layers_bar)
    ax.set(
        title=(
            f"Attenuation per layer: FDTD vs Analytical\n"
            f"Mean error: {mean_error:.1f}%  |  {result_str}"
        ),
        ylabel="Attenuation (dB)",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "3d_09_material_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    total_elapsed = time.perf_counter() - t_wall_start
    print(f"\nSaved: {out_path}")
    print(f"Total wall-clock time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
