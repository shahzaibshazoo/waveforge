"""
tissue_library.py — Gabriel 1996 4-pole Cole-Cole parameters for biological tissues.

Provides frequency-dependent dielectric properties for all tissues relevant to
brain haemorrhage imaging. Values from:

    Gabriel, S., Lau, R.W. and Gabriel, C. (1996)
    "The dielectric properties of biological tissues: II. Measurements in the
    frequency range 10 Hz to 20 GHz"
    Phys. Med. Biol. 41:2251-2269

Each tissue is characterised by the 4-pole Cole-Cole model:

    eps*(omega) = eps_inf
                  + sum_{n=1}^{4} [ delta_eps_n / (1 + (j*omega*tau_n)^(1-alpha_n)) ]
                  + sigma_s / (j * omega * eps_0)

Usage:
    from datasets.brain.tissue_library import TISSUES, cole_cole_eps

    # Get complex permittivity at 1 GHz
    eps_complex = cole_cole_eps(TISSUES['gray_matter'], 1e9)
    eps_r = eps_complex.real
    sigma = -eps_complex.imag * 2 * pi * 1e9 * EPS0
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

EPS0 = 8.8541878128e-12
MU0 = 1.2566370614e-6


@dataclass(frozen=True)
class ColeColeParams:
    """4-pole Cole-Cole parameters for a single tissue type."""

    name: str
    eps_inf: float
    sigma_s: float  # static ionic conductivity (S/m)

    # Pole 1 (alpha dispersion — low frequency)
    delta_eps_1: float
    tau_1: float  # relaxation time (s)
    alpha_1: float  # distribution parameter [0, 1)

    # Pole 2 (beta dispersion — RF)
    delta_eps_2: float
    tau_2: float
    alpha_2: float

    # Pole 3 (delta dispersion — microwave)
    delta_eps_3: float
    tau_3: float
    alpha_3: float

    # Pole 4 (gamma dispersion — water relaxation)
    delta_eps_4: float
    tau_4: float
    alpha_4: float


def cole_cole_eps(tissue: ColeColeParams, freq_hz: float) -> complex:
    """Compute complex relative permittivity at a given frequency.

    Parameters
    ----------
    tissue : ColeColeParams
        Tissue Cole-Cole parameters.
    freq_hz : float
        Frequency in Hz.

    Returns
    -------
    complex
        Complex relative permittivity eps_r' - j*eps_r''
        where eps_r' is the real part (permittivity) and
        eps_r'' relates to total loss (conductive + dielectric).
    """
    omega = 2.0 * math.pi * freq_hz
    eps = complex(tissue.eps_inf, 0.0)

    poles = [
        (tissue.delta_eps_1, tissue.tau_1, tissue.alpha_1),
        (tissue.delta_eps_2, tissue.tau_2, tissue.alpha_2),
        (tissue.delta_eps_3, tissue.tau_3, tissue.alpha_3),
        (tissue.delta_eps_4, tissue.tau_4, tissue.alpha_4),
    ]

    for delta_eps, tau, alpha in poles:
        if delta_eps == 0.0:
            continue
        denom = 1.0 + (1j * omega * tau) ** (1.0 - alpha)
        eps += delta_eps / denom

    # Static conductivity term
    if tissue.sigma_s > 0 and omega > 0:
        eps += tissue.sigma_s / (1j * omega * EPS0)

    return eps


def cole_cole_to_fdtd(tissue: ColeColeParams, freq_hz: float) -> tuple[float, float]:
    """Convert Cole-Cole tissue model to FDTD-compatible (eps_r, sigma) at a frequency.

    Extracts the real permittivity and effective conductivity from the
    complex permittivity. This is the standard approach for narrowband
    FDTD simulations where the source bandwidth is centred at freq_hz.

    Parameters
    ----------
    tissue : ColeColeParams
        Tissue Cole-Cole parameters.
    freq_hz : float
        Centre frequency in Hz.

    Returns
    -------
    tuple[float, float]
        (eps_r, sigma_eff) — real permittivity and effective conductivity (S/m).
    """
    eps_complex = cole_cole_eps(tissue, freq_hz)
    eps_r = eps_complex.real
    omega = 2.0 * math.pi * freq_hz
    sigma_eff = -eps_complex.imag * omega * EPS0
    return max(eps_r, 1.0), max(sigma_eff, 0.0)


def cole_cole_spectrum(
    tissue: ColeColeParams,
    freq_min: float = 1e8,
    freq_max: float = 10e9,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute eps_r and sigma over a frequency range.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        (frequencies, eps_r_array, sigma_array) — all shape (n_points,)
    """
    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)
    eps_r = np.zeros(n_points)
    sigma = np.zeros(n_points)
    for i, f in enumerate(freqs):
        eps_r[i], sigma[i] = cole_cole_to_fdtd(tissue, f)
    return freqs, eps_r, sigma


# ============================================================
# Gabriel 1996 4-pole Cole-Cole parameters
# Source: Table 1 & Table 2 from Gabriel et al. (1996)
# Validated against: IT'IS Foundation database (itis.swiss/virtual-population)
# ============================================================

TISSUES: dict[str, ColeColeParams] = {
    'scalp': ColeColeParams(
        name='scalp (wet skin)',
        eps_inf=4.0,
        sigma_s=0.0002,
        delta_eps_1=32.0, tau_1=7.23e-12, alpha_1=0.0,
        delta_eps_2=1100.0, tau_2=32.48e-9, alpha_2=0.2,
        delta_eps_3=0.0, tau_3=159.15e-6, alpha_3=0.2,
        delta_eps_4=0.0, tau_4=15.92e-3, alpha_4=0.2,
    ),
    'skull_cortical': ColeColeParams(
        name='skull (cortical bone)',
        eps_inf=2.5,
        sigma_s=0.02,
        delta_eps_1=10.0, tau_1=13.26e-12, alpha_1=0.2,
        delta_eps_2=180.0, tau_2=79.58e-9, alpha_2=0.2,
        delta_eps_3=5e4, tau_3=159.15e-6, alpha_3=0.2,
        delta_eps_4=1e5, tau_4=15.92e-3, alpha_4=0.0,
    ),
    'csf': ColeColeParams(
        name='cerebrospinal fluid',
        eps_inf=4.0,
        sigma_s=2.0,
        delta_eps_1=65.0, tau_1=7.96e-12, alpha_1=0.1,
        delta_eps_2=40.0, tau_2=1.59e-9, alpha_2=0.0,
        delta_eps_3=0.0, tau_3=159.15e-6, alpha_3=0.2,
        delta_eps_4=0.0, tau_4=15.92e-3, alpha_4=0.0,
    ),
    'gray_matter': ColeColeParams(
        name='brain gray matter',
        eps_inf=4.0,
        sigma_s=0.02,
        delta_eps_1=45.0, tau_1=7.96e-12, alpha_1=0.1,
        delta_eps_2=400.0, tau_2=15.92e-9, alpha_2=0.15,
        delta_eps_3=2e5, tau_3=106.1e-6, alpha_3=0.22,
        delta_eps_4=4.5e7, tau_4=5.31e-3, alpha_4=0.0,
    ),
    'white_matter': ColeColeParams(
        name='brain white matter',
        eps_inf=4.0,
        sigma_s=0.02,
        delta_eps_1=32.0, tau_1=7.96e-12, alpha_1=0.1,
        delta_eps_2=100.0, tau_2=7.96e-9, alpha_2=0.1,
        delta_eps_3=4e4, tau_3=53.05e-6, alpha_3=0.3,
        delta_eps_4=3.5e7, tau_4=7.96e-3, alpha_4=0.02,
    ),
    'blood': ColeColeParams(
        name='blood (fresh, oxygenated)',
        eps_inf=4.0,
        sigma_s=0.7,
        delta_eps_1=56.0, tau_1=8.38e-12, alpha_1=0.1,
        delta_eps_2=5200.0, tau_2=132.63e-9, alpha_2=0.1,
        delta_eps_3=0.0, tau_3=159.15e-6, alpha_3=0.2,
        delta_eps_4=0.0, tau_4=15.92e-3, alpha_4=0.0,
    ),
    'dura': ColeColeParams(
        name='dura mater',
        eps_inf=4.0,
        sigma_s=0.5,
        delta_eps_1=40.0, tau_1=7.96e-12, alpha_1=0.1,
        delta_eps_2=120.0, tau_2=15.92e-9, alpha_2=0.15,
        delta_eps_3=3e4, tau_3=159.15e-6, alpha_3=0.2,
        delta_eps_4=3e7, tau_4=15.92e-3, alpha_4=0.0,
    ),
    'muscle': ColeColeParams(
        name='muscle',
        eps_inf=4.0,
        sigma_s=0.2,
        delta_eps_1=50.0, tau_1=7.23e-12, alpha_1=0.1,
        delta_eps_2=7000.0, tau_2=353.68e-9, alpha_2=0.1,
        delta_eps_3=1.2e6, tau_3=318.31e-6, alpha_3=0.1,
        delta_eps_4=2.5e7, tau_4=2.27e-3, alpha_4=0.0,
    ),
    'fat': ColeColeParams(
        name='fat (adipose)',
        eps_inf=2.5,
        sigma_s=0.01,
        delta_eps_1=3.0, tau_1=23.0e-12, alpha_1=0.2,
        delta_eps_2=15.0, tau_2=159.15e-9, alpha_2=0.1,
        delta_eps_3=3.3e4, tau_3=159.15e-6, alpha_3=0.05,
        delta_eps_4=1e7, tau_4=7.96e-3, alpha_4=0.01,
    ),
}

# ============================================================
# Blood aging variants — modified Cole-Cole for haemorrhage stages
# Based on: Mustafa et al. (2014), Semenov et al. (2008)
# ============================================================

BLOOD_AGING: dict[str, ColeColeParams] = {
    'acute': TISSUES['blood'],  # fresh blood, 0-6 hours

    'subacute': ColeColeParams(
        name='blood (subacute, 6h-3d)',
        eps_inf=4.0,
        sigma_s=0.5,
        delta_eps_1=46.0, tau_1=8.38e-12, alpha_1=0.1,
        delta_eps_2=4000.0, tau_2=132.63e-9, alpha_2=0.1,
        delta_eps_3=0.0, tau_3=159.15e-6, alpha_3=0.2,
        delta_eps_4=0.0, tau_4=15.92e-3, alpha_4=0.0,
    ),

    'chronic': ColeColeParams(
        name='blood (chronic clot, >3d)',
        eps_inf=4.0,
        sigma_s=0.35,
        delta_eps_1=38.0, tau_1=8.38e-12, alpha_1=0.1,
        delta_eps_2=3000.0, tau_2=132.63e-9, alpha_2=0.15,
        delta_eps_3=0.0, tau_3=159.15e-6, alpha_3=0.2,
        delta_eps_4=0.0, tau_4=15.92e-3, alpha_4=0.0,
    ),
}


# ============================================================
# Quick-access: pre-evaluated at common frequencies
# ============================================================

def tissue_at_freq(name: str, freq_hz: float) -> tuple[float, float]:
    """Get (eps_r, sigma) for a named tissue at a given frequency.

    Parameters
    ----------
    name : str
        Tissue name (key from TISSUES or BLOOD_AGING dicts).
    freq_hz : float
        Frequency in Hz.

    Returns
    -------
    tuple[float, float]
        (eps_r, sigma_eff)

    Raises
    ------
    KeyError
        If tissue name is not found.

    Examples
    --------
    >>> eps_r, sigma = tissue_at_freq('gray_matter', 1e9)
    >>> print(f"Gray matter at 1 GHz: eps_r={eps_r:.1f}, sigma={sigma:.2f} S/m")
    """
    if name in TISSUES:
        return cole_cole_to_fdtd(TISSUES[name], freq_hz)
    elif name in BLOOD_AGING:
        return cole_cole_to_fdtd(BLOOD_AGING[name], freq_hz)
    else:
        all_keys = list(TISSUES.keys()) + list(BLOOD_AGING.keys())
        raise KeyError(f"Unknown tissue '{name}'. Available: {all_keys}")


def all_tissues_at_freq(freq_hz: float) -> dict[str, tuple[float, float]]:
    """Get (eps_r, sigma) for ALL tissues at a given frequency.

    Returns
    -------
    dict[str, tuple[float, float]]
        {tissue_name: (eps_r, sigma)} for all tissues + blood aging stages.
    """
    result = {}
    for name, params in TISSUES.items():
        result[name] = cole_cole_to_fdtd(params, freq_hz)
    for name, params in BLOOD_AGING.items():
        result[f'blood_{name}'] = cole_cole_to_fdtd(params, freq_hz)
    return result
