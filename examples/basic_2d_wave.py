"""
basic_2d_wave.py — End-to-end WaveForge smoke test and usage tutorial.

Demonstrates:
  - 2D TM FDTD simulation of a Gaussian pulse propagating in free space
  - Grid: 128x128 cells, dx=dy=1 mm, first-order Mur ABC on all four edges
  - Source: GaussianPulse at grid centre, soft-injected into Hz
  - Runs N_STEPS=500 time steps, saves a Hz snapshot PNG, and prints throughput

Run from the project root::

    /home/zuu/miniconda3/bin/python examples/basic_2d_wave.py
"""

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt

# Add src/ to path so the example works from any directory.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC
from core.sources import GaussianPulse, PointSource, SourceCollection
from core.fdtd2d import FDTD2D

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------

NX, NY = 128, 128
DX = DY = 1e-3          # 1 mm cell size
N_STEPS = 500
OUTPUT_DIR = Path(__file__).parent / "output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full 2D TM FDTD pipeline and save a Hz field snapshot."""

    import matplotlib
    matplotlib.use("Agg")

    # Step A — Grid
    grid = YeeGrid(NX, NY, dx=DX, dy=DY, device=DEVICE)
    print(f"Grid shape: {grid.shape}  dt: {grid.dt:.6e} s  device: {grid.device}")

    # Step B — Fields
    fields = FieldSet(grid)

    # Step C — Source
    sigma = 30 * grid.dt                      # pulse width ~30 timesteps
    pulse = GaussianPulse(amplitude=1.0, sigma=sigma)
    src = PointSource(pulse, NX // 2, NY // 2, "Hz", grid=grid, N_steps=N_STEPS)
    sources = SourceCollection([src])

    # Step D — Boundary
    boundary = MurABC(grid, fields.Hz)

    # Step E — Solver
    sim = FDTD2D(grid, fields, boundary, sources, n_check=50)

    # Step F — Run with progress
    print("Running simulation...")
    sim.run(N_STEPS, verbose=True)
    print(
        f"Done. {sim.steps_completed} steps, "
        f"{sim.mcells_per_second:.1f} Mcells/s, "
        f"field_max={sim.last_field_max:.3e}"
    )

    # Step G — Save Hz snapshot
    OUTPUT_DIR.mkdir(exist_ok=True)
    from visualization.plot2d import plot_field
    fig, ax = plot_field(fields, component="Hz", grid=grid, sim=sim)
    out_path = OUTPUT_DIR / "hz_final.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Step H — Energy report
    print(f"Total EM energy: {fields.total_energy():.6e} J")
    print(f"Max |Hz|: {fields.Hz.abs().max().item():.6e} A/m")
    print(f"Max |Ex|: {fields.Ex.abs().max().item():.6e} V/m")


if __name__ == "__main__":
    main()
