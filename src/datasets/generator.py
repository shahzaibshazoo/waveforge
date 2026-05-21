"""
generator.py — Brain haemorrhage FDTD dataset generator.

Generates individual simulation samples and batches, enforcing all 5
dataset quality rules:

  Rule 1: Cole-Cole frequency-dependent tissue properties
  Rule 2: Anatomically constrained bleed placement
  Rule 3: Blood aging physics (4 stages)
  Rule 4a: Balanced class distribution
  Rule 4b: Independent test phantom (PHANTOM_B)

Usage:
    from datasets.generator import BrainDatasetGenerator

    gen = BrainDatasetGenerator(
        output_dir='./brain_dataset',
        freq_hz=1e9,
        grid_size=64,
        dx_mm=3.0,
        n_tx=8,
        n_steps=300,
        device='cuda',
    )
    gen.generate_balanced_dataset(
        n_samples=2000,
        phantom_id='A',
        show_progress=True,
    )
"""

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.grid import YeeGrid
from core.fields import FieldSet
from core.boundaries import MurABC3D
from core.sources import ModulatedGaussian, GaussianPulse, UWBPulse
from core.fdtd3d import FDTD3D

from .brain.phantom import BrainPhantom3D, PHANTOM_A, PHANTOM_B, sample_random_geometry
from .brain.antenna import AntennaRing


# Bleed type distribution for a balanced dataset
_BLEED_TYPES = ['epidural', 'subdural', 'intracerebral']
_BLEED_AGES  = ['acute', 'subacute', 'chronic']
# Weight distribution: 40% acute, 35% subacute, 25% chronic
_AGE_WEIGHTS = [0.40, 0.35, 0.25]
# Weight distribution: 30% small, 50% medium, 20% large
_SIZE_CATS   = ['small', 'medium', 'large']
_SIZE_WEIGHTS = [0.30, 0.50, 0.20]


class BrainDatasetGenerator:
    """Generate a balanced intracranial haemorrhage FDTD simulation dataset.

    Parameters
    ----------
    output_dir : str or Path
        Directory where .npz sample files will be saved.
    freq_hz : float
        Centre frequency in Hz (default 1 GHz). Cole-Cole tissue properties
        are evaluated at this frequency.
    grid_size : int
        Cubic grid dimension N (creates N×N×N grid). Default 64.
    dx_mm : float
        Cell size in mm. Default 3.0 mm (lambda/10 at 1.5 GHz in brain).
    n_tx : int
        Number of transmit/receive antenna elements. Default 8.
    ring_radius_cells : int
        Antenna ring radius in cells. Default 30.
    n_steps : int
        Time steps per simulation. Default 300.
    device : str
        PyTorch device ('cuda' or 'cpu'). Default 'cuda'.
    seed : int, optional
        Master random seed for reproducibility.
    """

    def __init__(
        self,
        output_dir: str,
        freq_hz: float = 1e9,
        freq_low_hz: Optional[float] = None,
        freq_high_hz: Optional[float] = None,
        grid_size: int = 64,
        dx_mm: float = 3.0,
        n_tx: int = 8,
        ring_radius_cells: int = 30,
        n_steps: int = 300,
        device: str = 'cuda',
        seed: Optional[int] = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Frequency config: UWB mode if freq_low_hz/freq_high_hz given
        if freq_low_hz is not None and freq_high_hz is not None:
            if freq_high_hz <= freq_low_hz:
                raise ValueError(f"freq_high_hz must be > freq_low_hz")
            self._freq_low = float(freq_low_hz)
            self._freq_high = float(freq_high_hz)
            self._freq_hz = (freq_low_hz + freq_high_hz) / 2.0  # centre for tissue props
            self._uwb_mode = True
        else:
            self._freq_low = None
            self._freq_high = None
            self._freq_hz = float(freq_hz)
            self._uwb_mode = False

        self._N = grid_size
        self._dx = dx_mm * 1e-3
        self._n_tx = n_tx
        self._ring_r = ring_radius_cells
        self._n_steps = n_steps
        self._device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        self._seed = seed
        self._rng = random.Random(seed)

        # Build a reference grid (reused across samples)
        self._grid = YeeGrid(
            grid_size, grid_size,
            dx=self._dx, dy=self._dx,
            Nz=grid_size, dz=self._dx,
            device=self._device
        )

        # Build reference antenna ring
        self._ring = AntennaRing(
            n_elements=n_tx,
            ring_radius_cells=ring_radius_cells,
            z_plane=grid_size // 2,
            grid=self._grid,
        )

        if self._uwb_mode:
            bw = (self._freq_high - self._freq_low) / 1e9
            print(f"BrainDatasetGenerator ready (UWB mode):")
            print(f"  Band: {self._freq_low/1e9:.2f}–{self._freq_high/1e9:.2f} GHz  BW={bw:.2f} GHz")
            print(f"  Tissue props at fc={self._freq_hz/1e9:.2f} GHz (centre frequency)")
        else:
            print(f"BrainDatasetGenerator ready:")
            print(f"  Frequency: {self._freq_hz/1e9:.2f} GHz")
        print(f"  Grid: {grid_size}³, dx={dx_mm}mm, domain={grid_size*dx_mm:.0f}mm")
        print(f"  Antennas: {n_tx} elements, r={ring_radius_cells} cells")
        print(f"  Steps: {n_steps}, device: {self._device}")
        print(f"  Output: {self._output_dir}")

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def _run_simulation(
        self,
        phantom: BrainPhantom3D,
        tx_idx: int,
    ) -> tuple[np.ndarray, float]:
        """Run a single TX simulation and return recorded signals.

        Returns
        -------
        tuple[np.ndarray, float]
            (signals_row, mcells_per_second)
            signals_row shape: (n_rx, n_steps)
        """
        Ca, Cb = phantom.build()

        grid = self._grid
        fields = FieldSet(grid)
        boundary = MurABC3D(grid, fields.Hx, fields.Hy, fields.Hz)

        if self._uwb_mode:
            # UWB Gaussian monocycle: energy concentrated in [f_low, f_high]
            waveform = UWBPulse(
                amplitude=1.0,
                f_low=self._freq_low,
                f_high=self._freq_high,
            )
        else:
            # Narrow-band: modulated Gaussian at fc, bandwidth ~fc/2
            sigma_t = 1.0 / (2.0 * math.pi * (self._freq_hz * 0.5))
            waveform = ModulatedGaussian(
                amplitude=1.0,
                fc=self._freq_hz,
                sigma=sigma_t,
            )

        sources = self._ring.build_sources(waveform, tx_idx, self._n_steps)
        sim = FDTD3D(grid, fields, boundary, sources, Ca=Ca, Cb=Cb, n_check=99999)

        # Map component name to field tensor
        comp = self._ring._component
        field_map = {
            'Ex': fields.Ex, 'Ey': fields.Ey, 'Ez': fields.Ez,
            'Hx': fields.Hx, 'Hy': fields.Hy, 'Hz': fields.Hz,
        }
        field_tensor = field_map[comp]

        if self._device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for step in range(self._n_steps):
                sim.step()
                self._ring.record(field_tensor, step, tx_idx)

        if self._device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        mcells = self._n_steps * self._N ** 3 / elapsed / 1e6

        # Extract this TX row from the signal buffer
        all_signals = self._ring.get_signals()
        return all_signals[tx_idx].copy(), mcells

    def _run_full_mimo(
        self,
        phantom: BrainPhantom3D,
    ) -> tuple[np.ndarray, float]:
        """Run all TX simulations and return the full (N_tx, N_rx, N_steps) matrix.

        Returns
        -------
        tuple[np.ndarray, float]
            (signals, mean_mcells_per_second)
        """
        signals = np.zeros(
            (self._n_tx, self._n_tx, self._n_steps), dtype=np.float32
        )
        self._ring.reset()

        mcells_list = []
        for tx in range(self._n_tx):
            row, mc = self._run_simulation(phantom, tx)
            signals[tx] = row
            mcells_list.append(mc)

        return signals, float(np.mean(mcells_list))

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def generate_sample(
        self,
        label: int,
        seed: int,
        phantom_id: str = 'train',
        bleed_age: Optional[str] = None,
        bleed_size: Optional[str] = None,
    ) -> dict:
        """Generate a single dataset sample with a unique randomised phantom.

        Each sample gets its own head geometry drawn from the published
        population distribution (skull thickness, brain radius, head scale).
        This ensures the model must generalise across anatomy rather than
        memorising a fixed phantom.

        Parameters
        ----------
        label : int
            0=healthy, 1=epidural, 2=subdural, 3=intracerebral.
        seed : int
            Per-sample seed. Fully determines phantom geometry, bleed
            placement, and blood aging stage. Same seed = same sample.
        phantom_id : str
            'train' or 'test'. Uses separate seed spaces so train/test
            phantoms never overlap.
        bleed_age : str, optional
            'acute', 'subacute', or 'chronic'. Sampled if None.
        bleed_size : str, optional
            'small', 'medium', or 'large'. Sampled if None.

        Returns
        -------
        dict or None
            Complete sample dict, or None if bleed placement failed.
        """
        rng = random.Random(seed)

        # Each sample gets a UNIQUE phantom geometry from the population
        # distribution. Test samples use a separate seed space (seed + 10^7)
        # so they are structurally different from any training phantom.
        geom_seed = seed if phantom_id == 'train' else seed + 10_000_000
        geom = sample_random_geometry(
            seed=geom_seed,
            grid_size=self._N,
            dx_mm=self._dx * 1e3,
        )

        phantom_target = BrainPhantom3D(
            self._grid, self._freq_hz,
            geometry=geom, seed=seed
        )

        bleed_config = None
        if label > 0:
            type_map = {1: 'epidural', 2: 'subdural', 3: 'intracerebral'}
            btype = type_map[label]
            age = bleed_age or rng.choices(_BLEED_AGES, weights=_AGE_WEIGHTS)[0]
            size = bleed_size or rng.choices(_SIZE_CATS, weights=_SIZE_WEIGHTS)[0]

            # Try up to 50 (seed, size) combinations before giving up.
            # First try the requested size; fall back to smaller sizes if the
            # randomised geometry has a thin zone that doesn't fit.
            size_fallback = {
                'large': ['large', 'medium', 'small'],
                'medium': ['medium', 'small'],
                'small': ['small'],
            }
            for attempt in range(50):
                # Vary both the geometry seed and the size on repeated failures
                trial_geom_seed = geom_seed + attempt
                trial_geom = sample_random_geometry(
                    trial_geom_seed, self._N, self._dx * 1e3
                )
                phantom_target = BrainPhantom3D(
                    self._grid, self._freq_hz,
                    geometry=trial_geom, seed=seed + attempt
                )
                # Try each size from largest to smallest
                for trial_size in size_fallback.get(size, [size]):
                    bleed_config = phantom_target.add_random_bleed(
                        btype, age, trial_size
                    )
                    if bleed_config is not None:
                        geom = trial_geom  # use the geometry that worked
                        break
                if bleed_config is not None:
                    break

            if bleed_config is None:
                return None  # Genuinely impossible for this type — caller retries

        # Rebuild reference phantom with the same winning geometry
        phantom_ref = BrainPhantom3D(
            self._grid, self._freq_hz,
            geometry=geom, seed=seed + 100000
        )

        # Run simulations
        signals_total, mc_total = self._run_full_mimo(phantom_target)
        signals_ref, mc_ref   = self._run_full_mimo(phantom_ref)
        signals_scattered = signals_total - signals_ref

        # DAS image from scattered signals — pass phantom geometry for layered-medium delays
        from .brain.tissue_library import TISSUES
        head_geom = {
            'cx_cells': geom.center[0],
            'cy_cells': geom.center[1],
            'layers': [
                (geom.scalp_outer_r, 40.0),   # scalp
                (geom.skull_outer_r, 13.1),   # skull cortical
                (geom.skull_inner_r, 44.0),   # dura
                (geom.csf_inner_r,   68.0),   # CSF
                (geom.gray_matter_r, 52.7),   # gray matter
                (geom.white_matter_r, 38.1),  # white matter
            ],
        }
        das_image = self._ring.compute_das_image(signals_scattered, head_geometry=head_geom)

        # Bleed metadata
        bleed_center = bleed_config.center if bleed_config else (0, 0, 0)
        bleed_r = bleed_config.radius if bleed_config else 0
        bleed_r_mm = bleed_r * self._dx * 1e3
        bleed_volume_ml = (4 / 3) * math.pi * bleed_r_mm ** 3 / 1e3 if bleed_r > 0 else 0.0

        meta = phantom_target.get_metadata()

        return {
            # Signals
            'signals_total':      signals_total,
            'signals_reference':  signals_ref,
            'signals_scattered':  signals_scattered,
            # Image
            'das_image':          das_image,
            # Labels
            'label':              np.int32(label),
            'bleed_type':         (bleed_config.bleed_type if bleed_config else 'none'),
            'bleed_age':          (bleed_config.age if bleed_config else 'none'),
            # Geometry
            'bleed_center_cells': np.array(bleed_center, dtype=np.int32),
            'bleed_center_mm':    np.array(bleed_center, dtype=np.float32) * float(self._dx * 1e3),
            'bleed_radius_cells': np.int32(bleed_r),
            'bleed_radius_mm':    np.float32(bleed_r_mm),
            'bleed_volume_ml':    np.float32(bleed_volume_ml),
            # Simulation params
            'freq_hz':            np.float32(self._freq_hz),
            'freq_low_hz':        np.float32(self._freq_low or self._freq_hz),
            'freq_high_hz':       np.float32(self._freq_high or self._freq_hz),
            'uwb_mode':           np.bool_(self._uwb_mode),
            'dx_mm':              np.float32(self._dx * 1e3),
            'grid_shape':         np.array([self._N, self._N, self._N], dtype=np.int32),
            'n_tx':               np.int32(self._n_tx),
            'n_rx':               np.int32(self._n_tx),
            'n_steps':            np.int32(self._n_steps),
            'dt_s':               np.float64(self._grid.dt),
            # Phantom — unique geometry per sample
            'phantom_id':            phantom_id,
            'phantom_seed':          np.int32(seed),
            'phantom_skull_inner_r': np.int32(geom.skull_inner_r),
            'phantom_gray_r':        np.int32(geom.gray_matter_r),
            'phantom_scalp_outer_r': np.int32(geom.scalp_outer_r),
            # Performance
            '_mcells_per_second': np.float32((mc_total + mc_ref) / 2),
        }

    def save_sample(self, sample: dict, sample_idx: int) -> Path:
        """Save a sample dict to a numbered .npz file."""
        path = self._output_dir / f'sample_{sample_idx:06d}.npz'
        # Separate string fields from arrays
        arrays = {k: v for k, v in sample.items()
                  if isinstance(v, (np.ndarray, np.generic))}
        # Store string fields as 0-d object arrays
        for key in ('bleed_type', 'bleed_age', 'phantom_id'):
            arrays[key] = np.array(sample[key])
        np.savez_compressed(path, **arrays)
        return path

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_balanced_dataset(
        self,
        n_samples: int,
        phantom_id: str = 'train',
        base_seed: int = 0,
        show_progress: bool = True,
    ) -> dict:
        """Generate a class-balanced dataset.

        Each sample gets its own unique randomised head geometry drawn from
        the population distribution — n_samples unique phantoms total.
        This prevents the model from memorising a fixed skull shape.

        Splits n_samples equally across 4 classes (healthy, EDH, SDH, ICH).

        Parameters
        ----------
        n_samples : int
            Total number of samples to generate.
        phantom_id : str
            'train' or 'test'. Test uses a separate seed space to guarantee
            no phantom overlap with training data.
        base_seed : int
            Starting seed (incremented per sample).
        show_progress : bool
            Print progress updates.

        Returns
        -------
        dict
            Manifest dict with indices, paths, and statistics.
        """
        n_per_class = n_samples // 4
        labels = ([0] * n_per_class + [1] * n_per_class +
                  [2] * n_per_class + [3] * n_per_class)
        # Handle remainder
        labels += [i % 4 for i in range(n_samples - len(labels))]
        random.Random(base_seed).shuffle(labels)

        manifest = {
            'n_requested': n_samples,
            'n_completed': 0,
            'n_failed': 0,
            'phantom_id': phantom_id,
            'class_counts': {0: 0, 1: 0, 2: 0, 3: 0},
            'sample_paths': [],
            'sample_indices': [],
            'labels': [],
            'bleed_types': [],
            'bleed_ages': [],
            'freq_hz': self._freq_hz,
            'grid_shape': [self._N, self._N, self._N],
            'dx_mm': self._dx * 1e3,
        }

        t_start = time.time()
        sample_idx = 0

        for i, label in enumerate(labels):
            seed = base_seed + i * 7 + label * 1000  # ensure diverse seeds

            sample = self.generate_sample(
                label=label,
                seed=seed,
                phantom_id=phantom_id,
            )

            if sample is None:
                manifest['n_failed'] += 1
                if show_progress:
                    print(f"  [{i+1}/{n_samples}] SKIP (bleed placement failed, label={label})")
                continue

            path = self.save_sample(sample, sample_idx)
            manifest['n_completed'] += 1
            manifest['class_counts'][int(label)] += 1
            manifest['sample_paths'].append(str(path))
            manifest['sample_indices'].append(sample_idx)
            manifest['labels'].append(int(label))
            manifest['bleed_types'].append(str(sample['bleed_type']))
            manifest['bleed_ages'].append(str(sample['bleed_age']))

            if show_progress:
                elapsed = time.time() - t_start
                mc = float(sample['_mcells_per_second'])
                eta = (elapsed / (i + 1)) * (n_samples - i - 1)
                print(
                    f"  [{i+1:4d}/{n_samples}] label={label} "
                    f"type={sample['bleed_type']:15} "
                    f"age={sample['bleed_age']:9} "
                    f"{mc:.1f}Mc/s  ETA:{eta/60:.0f}min"
                )

            sample_idx += 1

        # Save manifest
        manifest_path = self._output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        if show_progress:
            print(f"\nDone: {manifest['n_completed']}/{n_samples} samples saved")
            print(f"Class counts: {manifest['class_counts']}")
            print(f"Manifest: {manifest_path}")

        return manifest
