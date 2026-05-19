"""
cpu_benchmark.py — Local CPU benchmark: CUDA-MEEP vs Meep

Run with:
    /home/zuu/miniconda3/bin/conda run -n pymeep python benchmarks/cpu_benchmark.py

Saves results to: benchmarks/cpu_results.json
Share this file alongside the Colab GPU results for the full comparison.
"""

import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from core import YeeGrid, FieldSet, MurABC, GaussianPulse, PointSource, SourceCollection, FDTD2D

# ── Config ────────────────────────────────────────────────────────────────────
GRID_SIZES = [64, 128, 256, 512]
N_WARMUP   = 20
N_STEPS    = 200
DX         = 1e-3   # 1 mm

# ── CUDA-MEEP CPU benchmark ───────────────────────────────────────────────────

def run_cuda_meep_cpu(N):
    grid     = YeeGrid(N, N, dx=DX, dy=DX, device='cpu')
    fields   = FieldSet(grid)
    boundary = MurABC(grid, fields.Hz)
    pulse    = GaussianPulse(amplitude=1.0, sigma=30 * grid.dt)
    src      = PointSource(pulse, N//2, N//2, 'Hz', grid=grid, N_steps=N_WARMUP + N_STEPS)
    sim      = FDTD2D(grid, fields, boundary, SourceCollection([src]), n_check=500)

    sim.run(N_WARMUP)                      # warmup — JIT, cache warm
    t0 = time.perf_counter()
    sim.run(N_STEPS)
    elapsed = time.perf_counter() - t0

    mcells_s = N_STEPS * N * N / elapsed / 1e6
    ms_step  = elapsed / N_STEPS * 1000
    return mcells_s, ms_step


# ── Meep CPU benchmark ────────────────────────────────────────────────────────

def run_meep_cpu(N):
    import meep as mp, os
    os.environ['MEEP_VERBOSITY'] = '0'

    courant  = 0.5
    dt_meep  = courant / N          # Meep dt in Meep units (1 unit = 1 m)
    t_target = N_STEPS * dt_meep

    def make_sim():
        return mp.Simulation(
            cell_size=mp.Vector3(1, 1),
            resolution=N,
            sources=[mp.Source(
                mp.GaussianSource(frequency=1.0, fwidth=0.5),
                component=mp.Hz,
                center=mp.Vector3(0, 0)
            )],
            boundary_layers=[mp.Absorber(thickness=0.1)]
        )

    # Warmup
    s = make_sim()
    s.run(until=5.0 / N)
    s.reset_meep()

    # Timed run
    s = make_sim()
    t0 = time.perf_counter()
    s.run(until=t_target)
    elapsed = time.perf_counter() - t0
    s.reset_meep()

    actual_steps = max(1, int(round(t_target / dt_meep)))
    mcells_s = actual_steps * N * N / elapsed / 1e6
    ms_step  = elapsed / actual_steps * 1000
    return mcells_s, ms_step


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import platform, datetime

    print('=' * 60)
    print('  LOCAL CPU BENCHMARK: CUDA-MEEP vs Meep')
    print(f'  Python {sys.version.split()[0]}  |  PyTorch {torch.__version__}')
    print(f'  Platform: {platform.processor() or platform.machine()}')
    print(f'  Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 60)

    cuda_meep_rows = []
    meep_rows      = []

    # ── CUDA-MEEP (CPU) ───────────────────────────────────────────────────────
    print('\n--- CUDA-MEEP (CPU / PyTorch) ---')
    for N in GRID_SIZES:
        m, ms = run_cuda_meep_cpu(N)
        cuda_meep_rows.append({'N': N, 'mcells_s': round(m, 2), 'ms_step': round(ms, 4)})
        print(f'  {N:4d}²   {m:8.1f} Mcells/s   {ms:8.3f} ms/step')

    # ── Meep (CPU) ────────────────────────────────────────────────────────────
    print('\n--- Meep (CPU) ---')
    meep_ok = False
    try:
        import meep  # noqa: F401
        meep_ok = True
    except ImportError:
        print('  Meep not importable in this environment. Run with:')
        print('  /home/zuu/miniconda3/bin/conda run -n pymeep python benchmarks/cpu_benchmark.py')

    if meep_ok:
        for N in GRID_SIZES:
            m, ms = run_meep_cpu(N)
            meep_rows.append({'N': N, 'mcells_s': round(m, 2), 'ms_step': round(ms, 4)})
            print(f'  {N:4d}²   {m:8.1f} Mcells/s   {ms:8.3f} ms/step')

    # ── Comparison table ──────────────────────────────────────────────────────
    print()
    print('=' * 62)
    print(f'  {"Grid":5s}  {"CUDA-MEEP CPU":>14s}  {"Meep CPU":>12s}  {"CUDA-MEEP/Meep":>14s}')
    print('=' * 62)
    for i, N in enumerate(GRID_SIZES):
        cm = cuda_meep_rows[i]['mcells_s']
        mp_row = meep_rows[i] if meep_rows else None
        mp_val = mp_row['mcells_s'] if mp_row else None
        ratio  = f'{cm/mp_val:.2f}x' if mp_val else 'N/A'
        mp_str = f'{mp_val:10.1f}' if mp_val else f'{"N/A":>10s}'
        print(f'  {N}²    {cm:12.1f}  {mp_str}  {ratio:>14s}')
    print('=' * 62)

    # ── Save to JSON ──────────────────────────────────────────────────────────
    results = {
        'meta': {
            'date':      datetime.datetime.now().isoformat(),
            'platform':  platform.processor() or platform.machine(),
            'python':    sys.version.split()[0],
            'torch':     torch.__version__,
            'device':    'cpu',
            'n_warmup':  N_WARMUP,
            'n_steps':   N_STEPS,
            'dx_mm':     DX * 1e3,
        },
        'cuda_meep_cpu': cuda_meep_rows,
        'meep_cpu':      meep_rows,
    }

    out_path = Path(__file__).parent / 'cpu_results.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f'\nResults saved to: {out_path}')
    print('Share this file with the Colab GPU results for full comparison.')


if __name__ == '__main__':
    main()
