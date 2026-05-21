"""
datasets.brain — Medically realistic brain phantom and tissue library.

Provides:
    BrainPhantom3D   — anatomical head phantom with frequency-dependent tissues
    PHANTOM_A        — standard adult male proportions (training)
    PHANTOM_B        — alternative proportions (test split)
    TISSUES          — Gabriel 1996 Cole-Cole parameters for all tissues
    BLOOD_AGING      — blood Cole-Cole parameters by age stage
    tissue_at_freq   — get (eps_r, sigma) for any tissue at any frequency
    all_tissues_at_freq — get all tissues at a given frequency

Usage:
    from datasets.brain import BrainPhantom3D
    phantom = BrainPhantom3D(grid, freq_hz=1e9)
    phantom.add_random_bleed(bleed_type='subdural', age='acute', size='medium')
    Ca, Cb = phantom.build()
"""

from .tissue_library import (
    ColeColeParams,
    TISSUES,
    BLOOD_AGING,
    tissue_at_freq,
    all_tissues_at_freq,
    cole_cole_eps,
    cole_cole_to_fdtd,
    cole_cole_spectrum,
)

from .phantom import (
    BrainPhantom3D,
    BleedConfig,
    PhantomGeometry,
    PHANTOM_A,
    PHANTOM_B,
    BLEED_SIZES,
)

__all__ = [
    "BrainPhantom3D",
    "BleedConfig",
    "PhantomGeometry",
    "PHANTOM_A",
    "PHANTOM_B",
    "BLEED_SIZES",
    "ColeColeParams",
    "TISSUES",
    "BLOOD_AGING",
    "tissue_at_freq",
    "all_tissues_at_freq",
    "cole_cole_eps",
    "cole_cole_to_fdtd",
    "cole_cole_spectrum",
]
