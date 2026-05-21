"""
datasets — Brain haemorrhage FDTD dataset generation for WaveForge.

This module uses the WaveForge core library to generate medically realistic
simulation datasets. It does NOT modify the core library.

Top-level API:
    from datasets.generator import BrainDatasetGenerator
    from datasets.brain import BrainPhantom3D, tissue_at_freq, AntennaRing
"""
from .brain import BrainPhantom3D, PHANTOM_A, PHANTOM_B
from .brain import tissue_at_freq, all_tissues_at_freq
from .brain.antenna import AntennaRing
from .generator import BrainDatasetGenerator
