#!/usr/bin/env python3
"""
generate_brain_dataset.py — CLI for WaveForge brain haemorrhage dataset generation.

Generates a medically realistic FDTD simulation dataset with all 5
publication-quality rules enforced:
  1. Cole-Cole frequency-dependent tissue properties (Gabriel 1996)
  2. Anatomically constrained bleed placement
  3. Blood aging physics (acute/subacute/chronic)
  4a. Balanced class distribution (25% each: healthy/EDH/SDH/ICH)
  4b. Independent test phantom (PHANTOM_B)

Usage:
    # Generate 2000 training samples (PHANTOM_A) at 1 GHz, 64³ grid
    python datasets/generate_brain_dataset.py \
        --n_samples 2000 \
        --output_dir /path/to/output \
        --phantom A

    # Generate 400 test samples (PHANTOM_B)
    python datasets/generate_brain_dataset.py \
        --n_samples 400 \
        --output_dir /path/to/output/test \
        --phantom B \
        --base_seed 999999

    # Quick validation run (10 samples, fast 48³ grid)
    python datasets/generate_brain_dataset.py \
        --n_samples 10 \
        --output_dir /tmp/brain_test \
        --grid_size 48 \
        --n_steps 100

    # High-resolution run for publication
    python datasets/generate_brain_dataset.py \
        --n_samples 5000 \
        --output_dir /kaggle/working/brain_dataset \
        --freq_ghz 1.0 \
        --grid_size 64 \
        --dx_mm 3.0 \
        --n_tx 8 \
        --n_steps 300 \
        --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src to path (works whether run from project root or datasets/)
_here = Path(__file__).parent
sys.path.insert(0, str(_here.parent / 'src'))

import torch

from datasets.generator import BrainDatasetGenerator


def parse_args():
    p = argparse.ArgumentParser(
        description='WaveForge brain haemorrhage FDTD dataset generator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument('--n_samples', type=int, required=True,
                   help='Total number of samples to generate')
    p.add_argument('--output_dir', type=str, required=True,
                   help='Directory where .npz samples will be saved')

    # Phantom
    p.add_argument('--phantom', choices=['A', 'B'], default='A',
                   help='A=training phantom, B=test phantom (independent geometry)')
    p.add_argument('--base_seed', type=int, default=0,
                   help='Base random seed for reproducibility')

    # Physics — narrow-band or UWB
    p.add_argument('--freq_ghz', type=float, default=1.0,
                   help='Centre frequency in GHz (used in narrow-band mode)')
    p.add_argument('--freq_low_ghz', type=float, default=None,
                   help='UWB lower frequency in GHz (enables UWB mode when set with --freq_high_ghz)')
    p.add_argument('--freq_high_ghz', type=float, default=None,
                   help='UWB upper frequency in GHz (e.g. --freq_low_ghz 3 --freq_high_ghz 10)')

    # Grid
    p.add_argument('--grid_size', type=int, default=64,
                   help='Cubic grid dimension N (NxNxN, domain = N*dx_mm)')
    p.add_argument('--dx_mm', type=float, default=3.0,
                   help='Cell size in mm (3mm = lambda/10 at 1.5GHz in brain)')

    # Antennas
    p.add_argument('--n_tx', type=int, default=8,
                   help='Number of TX/RX antenna elements in the ring')
    p.add_argument('--ring_radius', type=int, default=30,
                   help='Antenna ring radius in cells')

    # Simulation
    p.add_argument('--n_steps', type=int, default=300,
                   help='Time steps per TX simulation')
    p.add_argument('--device', type=str, default='',
                   help='PyTorch device (cuda/cpu, auto-detected if empty)')

    # Output
    p.add_argument('--quiet', action='store_true',
                   help='Suppress per-sample progress output')
    p.add_argument('--dry_run', action='store_true',
                   help='Print configuration and exit without simulating')

    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Determine frequency mode
    uwb_mode = args.freq_low_ghz is not None and args.freq_high_ghz is not None
    if uwb_mode:
        freq_low  = args.freq_low_ghz  * 1e9
        freq_high = args.freq_high_ghz * 1e9
        freq_centre = (freq_low + freq_high) / 2.0
    else:
        freq_low = freq_high = None
        freq_centre = args.freq_ghz * 1e9

    # Print configuration
    print('=' * 60)
    print('WaveForge Brain Haemorrhage Dataset Generator')
    print('=' * 60)
    print(f'  Samples:    {args.n_samples}')
    print(f'  Phantom:    {args.phantom} ({"training" if args.phantom == "A" else "test/independent"})')
    if uwb_mode:
        print(f'  Mode:       UWB  {args.freq_low_ghz:.2f}–{args.freq_high_ghz:.2f} GHz')
        print(f'  Tissue fc:  {freq_centre/1e9:.2f} GHz (Cole-Cole evaluated at centre freq)')
    else:
        print(f'  Frequency:  {args.freq_ghz:.2f} GHz')
    print(f'  Grid:       {args.grid_size}³ cells, dx={args.dx_mm}mm')
    print(f'  Domain:     {args.grid_size * args.dx_mm:.0f}mm cube')
    print(f'  Antennas:   {args.n_tx} elements, r={args.ring_radius} cells')
    print(f'  Steps:      {args.n_steps} per TX')
    print(f'  Device:     {device}')
    print(f'  Output:     {args.output_dir}')
    print(f'  Base seed:  {args.base_seed}')

    # Estimate runtime
    if device == 'cuda':
        est_s_per_sample = args.n_tx * args.n_steps * args.grid_size**3 / 567e6 * 2
    else:
        est_s_per_sample = args.n_tx * args.n_steps * args.grid_size**3 / 6e6 * 2
    est_total_h = est_s_per_sample * args.n_samples / 3600
    print(f'  Est. time:  ~{est_s_per_sample:.0f}s/sample → ~{est_total_h:.1f}h total')
    print('=' * 60)

    if args.dry_run:
        print('DRY RUN — exiting without simulation.')
        return 0

    if device == 'cpu' and args.n_samples > 50:
        print(f'WARNING: Running {args.n_samples} samples on CPU will take '
              f'~{est_total_h:.0f} hours. Use --device cuda for GPU acceleration.')
        response = input('Continue anyway? [y/N] ').strip().lower()
        if response != 'y':
            print('Aborted.')
            return 1

    # Build generator
    gen = BrainDatasetGenerator(
        output_dir=args.output_dir,
        freq_hz=freq_centre,
        freq_low_hz=freq_low,
        freq_high_hz=freq_high,
        grid_size=args.grid_size,
        dx_mm=args.dx_mm,
        n_tx=args.n_tx,
        ring_radius_cells=args.ring_radius,
        n_steps=args.n_steps,
        device=device,
        seed=args.base_seed,
    )

    # Generate
    t0 = time.time()
    manifest = gen.generate_balanced_dataset(
        n_samples=args.n_samples,
        phantom_id='train' if args.phantom == 'A' else 'test',
        base_seed=args.base_seed,
        show_progress=not args.quiet,
    )
    elapsed = time.time() - t0

    # Final report
    print()
    print('=' * 60)
    print('GENERATION COMPLETE')
    print('=' * 60)
    print(f'  Completed:  {manifest["n_completed"]} samples')
    print(f'  Failed:     {manifest["n_failed"]} (bleed placement)')
    print(f'  Class dist: {manifest["class_counts"]}')
    print(f'  Total time: {elapsed/3600:.2f}h ({elapsed:.0f}s)')
    print(f'  Avg speed:  {elapsed/max(manifest["n_completed"],1):.1f}s/sample')
    print(f'  Output:     {args.output_dir}')
    print(f'  Manifest:   {args.output_dir}/manifest.json')
    print('=' * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
