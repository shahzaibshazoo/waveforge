"""
3d_head_phantom_visualisation.py — Publication-quality brain phantom visualisation.

Produces a 4-panel figure showing the 3D head phantom geometry and antenna ring
configuration. This is a pure visualisation — no FDTD simulation is run.

Panels:
  1. 3D volumetric cross-section (voxels, half-head cut, bleed, antenna ring)
  2. Axial (XY) slice at z=centre with antenna ring overlay
  3. Sagittal (XZ) slice at y=centre with tissue layers
  4. 3D antenna ring schematic — head wireframe + TX elements + wave arcs

Run: python examples/3d/3d_head_phantom_visualisation.py
Out: examples/output/3d_head_phantom_visualisation.png
     docs/assets/3d_head_phantom_visualisation.png
"""

import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add src to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Import project modules — fall back to inline definitions when torch is
# absent (e.g. a visualisation-only environment), since this script only
# needs the geometry constants and antenna position arithmetic.
try:
    from datasets.brain.phantom import sample_random_geometry, BrainPhantom3D, PHANTOM_A
    from datasets.brain.antenna import AntennaRing
    from core.grid import YeeGrid
    _HAVE_PROJECT_MODULES = True
except (ImportError, ModuleNotFoundError):
    _HAVE_PROJECT_MODULES = False


# ---------------------------------------------------------------------------
# Inline fallback geometry — mirrors phantom.py and antenna.py exactly
# ---------------------------------------------------------------------------

if not _HAVE_PROJECT_MODULES:
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class PhantomGeometry:
        center: tuple
        scalp_outer_r: int
        skull_outer_r: int
        skull_inner_r: int
        dura_inner_r: int
        csf_inner_r: int
        gray_matter_r: int
        white_matter_r: int

    # Standard adult male proportions — identical to phantom.PHANTOM_A
    PHANTOM_A = PhantomGeometry(
        center=(32, 32, 32),
        scalp_outer_r=28,
        skull_outer_r=26,
        skull_inner_r=23,
        dura_inner_r=20,
        csf_inner_r=17,
        gray_matter_r=17,
        white_matter_r=10,
    )

    class AntennaRing:
        """Minimal antenna-ring that only computes element positions."""

        def __init__(self, n_elements, ring_radius_cells, z_plane, grid=None,
                     centre=None):
            self._n = n_elements
            self._r = ring_radius_cells
            self._z = z_plane
            cx = centre[0] if centre else (64 // 2)
            cy = centre[1] if centre else (64 // 2)
            self._positions = []
            for k in range(n_elements):
                angle = 2.0 * math.pi * k / n_elements
                i = int(round(cx + ring_radius_cells * math.cos(angle)))
                j = int(round(cy + ring_radius_cells * math.sin(angle)))
                i = max(1, min(62, i))
                j = max(1, min(62, j))
                self._positions.append((i, j, z_plane))

        @property
        def positions(self):
            return list(self._positions)

    def sample_random_geometry(seed, grid_size=64, dx_mm=3.0):
        return PHANTOM_A

    class YeeGrid:
        """Stub — not needed for visualisation."""
        pass


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

GRID_SIZE = 64
DX_MM = 3.0
N_TX = 8
RING_RADIUS = 30      # cells
Z_PLANE = 32
SEED = 42
FREQ_GHZ = 1.0

BLEED_CENTER = (38, 36, 32)
BLEED_RADIUS = 5

# Tissue label codes
AIR = 0
SCALP_L = 1
SKULL_L = 2
CSF_L = 3
GRAY_L = 4
WHITE_L = 5
BLEED_L = 6

# Tissue colour palette — publication-quality RGBA
TISSUE_COLORS = {
    AIR:     (0.96, 0.96, 0.98, 0.00),  # transparent
    SCALP_L: (0.957, 0.761, 0.631, 1.0),  # skin tone   #F4C2A1
    SKULL_L: (0.961, 0.961, 0.863, 1.0),  # bone white  #F5F5DC
    CSF_L:   (0.659, 0.847, 0.918, 1.0),  # pale blue   #A8D8EA
    GRAY_L:  (0.545, 0.580, 0.278, 1.0),  # gray-green  #8B9467
    WHITE_L: (0.941, 0.922, 0.847, 1.0),  # light cream #F0EBD8
    BLEED_L: (0.780, 0.082, 0.082, 1.0),  # deep red
}

# Hex colours for matplotlib patches
TISSUE_HEX = {
    SCALP_L: '#F4C2A1',
    SKULL_L: '#F5F5DC',
    CSF_L:   '#A8D8EA',
    GRAY_L:  '#8B9467',
    WHITE_L: '#F0EBD8',
    BLEED_L: '#C71515',
}

TISSUE_NAMES = {
    SCALP_L: 'Scalp',
    SKULL_L: 'Skull',
    CSF_L:   'CSF / Epidural / Subdural',
    GRAY_L:  'Gray Matter',
    WHITE_L: 'White Matter',
    BLEED_L: 'Intracerebral Bleed',
}


# ---------------------------------------------------------------------------
# Build tissue label map
# ---------------------------------------------------------------------------

def build_label_map(grid_size, geom, bleed_center, bleed_radius):
    """Build integer label array (grid_size^3) from PHANTOM_A geometry.

    Labels applied outside-in (painter's algorithm):
        0=air, 1=scalp, 2=skull, 3=CSF/epidural/subdural,
        4=gray matter, 5=white matter, 6=bleed
    """
    label = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

    cx, cy, cz = geom.center

    ii = np.arange(grid_size)
    ix, iy, iz = np.meshgrid(ii, ii, ii, indexing='ij')
    dist = np.sqrt(
        (ix - cx).astype(np.float32)**2 +
        (iy - cy).astype(np.float32)**2 +
        (iz - cz).astype(np.float32)**2
    )

    label[dist <= geom.scalp_outer_r] = SCALP_L
    label[dist <= geom.skull_outer_r] = SKULL_L
    # Between skull_inner_r and skull_outer_r is skull bone.
    # skull_inner_r inward → CSF / epidural / subdural
    label[dist <= geom.skull_inner_r] = CSF_L
    label[dist <= geom.csf_inner_r]   = GRAY_L
    label[dist <= geom.white_matter_r] = WHITE_L

    # Place intracerebral bleed
    bx, by, bz = bleed_center
    bdist = np.sqrt(
        (ix - bx).astype(np.float32)**2 +
        (iy - by).astype(np.float32)**2 +
        (iz - bz).astype(np.float32)**2
    )
    label[bdist <= bleed_radius] = BLEED_L

    return label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def label_to_rgba(label_map):
    """Convert integer label array to RGBA float32 array."""
    rgba = np.zeros((*label_map.shape, 4), dtype=np.float32)
    for code, color in TISSUE_COLORS.items():
        rgba[label_map == code] = color
    return rgba


def make_sphere_surface(cx, cy, cz, r, n=40):
    """Return (X, Y, Z) surface arrays for a sphere wireframe/surface."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    X = cx + r * np.outer(np.cos(u), np.sin(v))
    Y = cy + r * np.outer(np.sin(u), np.sin(v))
    Z = cz + r * np.outer(np.ones_like(u), np.cos(v))
    return X, Y, Z


def draw_cylinder(ax, pos, direction, length=4.0, radius=1.2, color='#2B7BB9',
                  resolution=12):
    """Draw a solid cylinder in a 3D axes using Poly3DCollection quad strips."""
    px, py, pz = pos
    dx_, dy_, dz_ = direction

    norm = math.sqrt(dx_**2 + dy_**2 + dz_**2 + 1e-12)
    dx_, dy_, dz_ = dx_ / norm, dy_ / norm, dz_ / norm

    if abs(dx_) < 0.9:
        perp1 = np.cross([dx_, dy_, dz_], [1, 0, 0])
    else:
        perp1 = np.cross([dx_, dy_, dz_], [0, 1, 0])
    perp1 = perp1 / (np.linalg.norm(perp1) + 1e-12)
    perp2 = np.cross([dx_, dy_, dz_], perp1)

    theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    cap_bottom = np.array([
        [px + radius * (math.cos(t) * perp1[0] + math.sin(t) * perp2[0])
         for t in theta],
        [py + radius * (math.cos(t) * perp1[1] + math.sin(t) * perp2[1])
         for t in theta],
        [pz + radius * (math.cos(t) * perp1[2] + math.sin(t) * perp2[2])
         for t in theta],
    ])
    offset = np.array([[dx_ * length], [dy_ * length], [dz_ * length]])
    cap_top = cap_bottom + offset

    verts = []
    for i in range(resolution):
        i2 = (i + 1) % resolution
        v0 = cap_bottom[:, i]
        v1 = cap_bottom[:, i2]
        v2 = cap_top[:, i2]
        v3 = cap_top[:, i]
        verts.append([list(v0), list(v1), list(v2), list(v3)])

    poly = Poly3DCollection(verts, alpha=0.88, facecolor=color, edgecolor='none')
    ax.add_collection3d(poly)


# ---------------------------------------------------------------------------
# Panel 1: 3D volumetric cross-section
# ---------------------------------------------------------------------------

def draw_panel1(ax, label_map, ant_positions, geom, bleed_center, bleed_radius,
                grid_size):
    """Half-head voxel cut-away with bleed sphere and antenna cylinders."""
    half = grid_size // 2

    # Only render the back half (j >= half) and non-air voxels
    show = (label_map > 0)
    show[:, :half, :] = False

    rgba = label_to_rgba(label_map)

    ax.voxels(show, facecolors=rgba, edgecolor='none', alpha=0.9)

    # Bleed sphere — slightly inflated for visibility
    bx, by, bz = bleed_center
    sX, sY, sZ = make_sphere_surface(bx, by, bz, bleed_radius + 0.6, n=22)
    ax.plot_surface(sX, sY, sZ, color='#C71515', alpha=0.75, linewidth=0)

    # Antenna cylinders pointing inward
    cx, cy, cz = geom.center
    for ai, aj, ak in ant_positions:
        ddx = float(cx - ai)
        ddy = float(cy - aj)
        norm = math.sqrt(ddx**2 + ddy**2 + 1e-12)
        ddx_n, ddy_n = ddx / norm, ddy / norm

        # Place cylinder tip just outside the scalp, pointing inward
        start_x = ai - ddx_n * 2.5
        start_y = aj - ddy_n * 2.5
        draw_cylinder(ax,
                      pos=(start_x, start_y, float(ak)),
                      direction=(ddx_n, ddy_n, 0.0),
                      length=6.5, radius=1.1,
                      color='#1A6EBD')
        ax.scatter([ai], [aj], [ak], color='#0D3E6E', s=30, zorder=5)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, grid_size)
    ax.set_xlabel('X (cells)', fontsize=8, labelpad=2)
    ax.set_ylabel('Y (cells)', fontsize=8, labelpad=2)
    ax.set_zlabel('Z (cells)', fontsize=8, labelpad=2)
    ax.set_title('Brain Head Phantom — Cross-Section View',
                 fontsize=10, fontweight='bold', pad=6)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=-52)

    legend_handles = [
        mpatches.Patch(color=TISSUE_HEX[t], label=TISSUE_NAMES[t])
        for t in (SCALP_L, SKULL_L, CSF_L, GRAY_L, WHITE_L, BLEED_L)
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize=6.2, framealpha=0.88,
              bbox_to_anchor=(-0.04, 1.02))


# ---------------------------------------------------------------------------
# Panel 2: Axial XY slice (z = Z_PLANE)
# ---------------------------------------------------------------------------

def draw_panel2(ax, label_map, ant_positions, bleed_center, grid_size, dx_mm):
    """Top-down XY slice with antenna ring overlay."""
    tissue_slice = label_map[:, :, Z_PLANE].T  # (Ny, Nx)

    H, W = tissue_slice.shape
    img = np.zeros((H, W, 4), dtype=np.float32)
    for code, color in TISSUE_COLORS.items():
        img[tissue_slice == code] = color
    img[tissue_slice == AIR] = (0.96, 0.96, 0.98, 1.0)

    extent = [0, grid_size * dx_mm, 0, grid_size * dx_mm]
    ax.imshow(img, origin='lower', extent=extent, aspect='equal',
              interpolation='nearest')

    cx_mm = (grid_size // 2) * dx_mm
    cy_mm = (grid_size // 2) * dx_mm

    # Antenna dots + labels
    for idx, (ai, aj, ak) in enumerate(ant_positions):
        xmm = ai * dx_mm
        ymm = aj * dx_mm
        ax.plot(xmm, ymm, 'o', color='#1A6EBD', markersize=9,
                markeredgecolor='white', markeredgewidth=0.8, zorder=10)
        ox = xmm - cx_mm
        oy = ymm - cy_mm
        onorm = math.sqrt(ox**2 + oy**2 + 1e-12)
        lx = xmm + ox / onorm * 9.0
        ly = ymm + oy / onorm * 9.0
        ax.text(lx, ly, f'Ant {idx + 1}', fontsize=6.5, ha='center',
                va='center', color='#1A3A6E', fontweight='bold')

    # Bleed marker
    bx, by, bz = bleed_center
    ax.plot(bx * dx_mm, by * dx_mm, 'X', color='#C71515', markersize=12,
            markeredgecolor='white', markeredgewidth=0.8, zorder=11)
    ax.text(bx * dx_mm + 5, by * dx_mm + 4, 'Bleed',
            fontsize=7.5, color='#C71515', fontweight='bold')

    # Ring radius annotation
    ring_mm = RING_RADIUS * dx_mm
    circle = plt.Circle((cx_mm, cy_mm), ring_mm, fill=False,
                         color='#1A6EBD', linestyle='--',
                         linewidth=1.0, alpha=0.55)
    ax.add_patch(circle)
    ax.annotate('', xy=(cx_mm + ring_mm, cy_mm), xytext=(cx_mm, cy_mm),
                arrowprops=dict(arrowstyle='->', color='#1A6EBD', lw=1.0))
    ax.text(cx_mm + ring_mm / 2, cy_mm + 5.5,
            f'r = {ring_mm:.0f} mm', fontsize=7, color='#1A6EBD', ha='center')

    ax.set_xlabel('X (mm)', fontsize=9)
    ax.set_ylabel('Y (mm)', fontsize=9)
    ax.set_title('Axial Plane — Antenna Ring Configuration',
                 fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)

    legend_handles = [
        mpatches.Patch(color=TISSUE_HEX[t], label=TISSUE_NAMES[t])
        for t in (SCALP_L, SKULL_L, CSF_L, GRAY_L, WHITE_L, BLEED_L)
    ]
    ax.legend(handles=legend_handles, loc='lower left',
              fontsize=6.2, framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel 3: Sagittal XZ slice (y = centre)
# ---------------------------------------------------------------------------

def draw_panel3(ax, label_map, bleed_center, grid_size, dx_mm, geom):
    """Side XZ slice at y=centre with tissue layer annotations."""
    y = grid_size // 2
    tissue_slice = label_map[:, y, :].T  # (Nz, Nx)

    H, W = tissue_slice.shape
    img = np.zeros((H, W, 4), dtype=np.float32)
    for code, color in TISSUE_COLORS.items():
        img[tissue_slice == code] = color
    img[tissue_slice == AIR] = (0.96, 0.96, 0.98, 1.0)

    extent = [0, grid_size * dx_mm, 0, grid_size * dx_mm]
    ax.imshow(img, origin='lower', extent=extent, aspect='equal',
              interpolation='nearest')

    # Bleed marker
    bx, by, bz = bleed_center
    ax.plot(bx * dx_mm, bz * dx_mm, 'X', color='#C71515', markersize=12,
            markeredgecolor='white', markeredgewidth=0.8, zorder=11)
    ax.text(bx * dx_mm + 5, bz * dx_mm + 4, 'Bleed',
            fontsize=7.5, color='#C71515', fontweight='bold')

    # Layer annotations with arrows — placed at 45-degree diagonal
    cx_mm = (grid_size // 2) * dx_mm
    cz_mm = (grid_size // 2) * dx_mm
    layers_annot = [
        (geom.scalp_outer_r, 'Scalp'),
        (geom.skull_outer_r, 'Skull'),
        (geom.skull_inner_r, 'CSF'),
        (geom.csf_inner_r,   'Gray M.'),
        (geom.white_matter_r, 'White M.'),
    ]
    angle_deg = 42.0
    rad = math.radians(angle_deg)
    for r_cells, name in layers_annot:
        r_mm = r_cells * dx_mm
        xpt = cx_mm + r_mm * math.cos(rad)
        zpt = cz_mm + r_mm * math.sin(rad)
        ax.annotate(
            name,
            xy=(xpt, zpt),
            xytext=(xpt + 7, zpt + 7),
            fontsize=6.2,
            color='#1A1A1A',
            arrowprops=dict(arrowstyle='->', color='#444', lw=0.7),
            bbox=dict(boxstyle='round,pad=0.15', fc='white',
                      alpha=0.75, edgecolor='none'),
        )

    ax.set_xlabel('X (mm)', fontsize=9)
    ax.set_ylabel('Z (mm)', fontsize=9)
    ax.set_title('Sagittal Plane — Tissue Cross-Section',
                 fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)

    legend_handles = [
        mpatches.Patch(color=TISSUE_HEX[t], label=TISSUE_NAMES[t])
        for t in (SCALP_L, SKULL_L, CSF_L, GRAY_L, WHITE_L, BLEED_L)
    ]
    ax.legend(handles=legend_handles, loc='lower left',
              fontsize=6.2, framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel 4: 3D antenna ring schematic
# ---------------------------------------------------------------------------

def _antenna_box_verts(pos, direction, size=2.5, depth=1.8):
    """Return face vertex lists for a small rectangular antenna element."""
    px, py, pz = pos
    dx_, dy_, dz_ = direction
    norm = math.sqrt(dx_**2 + dy_**2 + dz_**2 + 1e-12)
    dx_, dy_, dz_ = dx_ / norm, dy_ / norm, dz_ / norm

    if abs(dx_) < 0.9:
        p1 = np.cross([dx_, dy_, dz_], [1, 0, 0])
    else:
        p1 = np.cross([dx_, dy_, dz_], [0, 1, 0])
    p1 = p1 / (np.linalg.norm(p1) + 1e-12) * size
    p2 = np.cross([dx_, dy_, dz_], p1 / np.linalg.norm(p1)) * (size / 2.0)
    dn = np.array([dx_, dy_, dz_]) * depth

    o = np.array([px, py, pz])
    c0 = o - p1 / 2 - p2 / 2
    c1 = c0 + p1
    c2 = c0 + p1 + p2
    c3 = c0 + p2
    c4, c5, c6, c7 = c0 + dn, c1 + dn, c2 + dn, c3 + dn

    faces = [
        [c0, c1, c2, c3],   # front
        [c4, c5, c6, c7],   # back
        [c0, c1, c5, c4],   # bottom
        [c2, c3, c7, c6],   # top
        [c1, c2, c6, c5],   # right
        [c0, c3, c7, c4],   # left
    ]
    return [[list(v) for v in face] for face in faces]


def draw_panel4(ax, ant_positions, geom, grid_size):
    """Clean 3D polar schematic: head wireframe + TX elements + wave arcs."""
    cx, cy, cz = geom.center
    r_head = geom.scalp_outer_r

    # Head — translucent wireframe sphere
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 32)
    sX = cx + r_head * np.outer(np.cos(u), np.sin(v))
    sY = cy + r_head * np.outer(np.sin(u), np.sin(v))
    sZ = cz + r_head * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_wireframe(sX, sY, sZ, color='#BBBBBB', linewidth=0.3, alpha=0.3,
                      rstride=3, cstride=3)
    ax.plot_surface(sX, sY, sZ, color='#E0E0E0', alpha=0.10, linewidth=0)

    # Antenna elements and connection lines
    for idx, (ai, aj, ak) in enumerate(ant_positions):
        ddx = float(cx - ai)
        ddy = float(cy - aj)
        norm = math.sqrt(ddx**2 + ddy**2 + 1e-12)
        ddx_n, ddy_n = ddx / norm, ddy / norm

        verts = _antenna_box_verts(
            pos=(float(ai), float(aj), float(ak)),
            direction=(ddx_n, ddy_n, 0.0),
            size=2.6, depth=1.9,
        )
        poly = Poly3DCollection(verts, alpha=0.92,
                                facecolor='#1A6EBD',
                                edgecolor='#0D3E6E',
                                linewidths=0.4)
        ax.add_collection3d(poly)

        # Dashed line from antenna to head surface
        surf_x = cx - ddx_n * r_head
        surf_y = cy - ddy_n * r_head
        ax.plot([float(ai), surf_x], [float(aj), surf_y], [float(ak), float(ak)],
                color='#2196F3', linewidth=0.9, alpha=0.55, linestyle='--')

        # Label offset outward
        lx = cx - ddx_n * (r_head + 8)
        ly = cy - ddy_n * (r_head + 8)
        ax.text(lx, ly, float(ak) + 1.5, f'TX{idx + 1}',
                fontsize=7, color='#1A3A6E', fontweight='bold',
                ha='center', va='center')

    # Microwave wave arcs radiating from TX1 inward
    tx1_i, tx1_j, tx1_k = ant_positions[0]
    angle0 = math.atan2(float(tx1_j) - cy, float(tx1_i) - cx)

    arc_params = [(5, 0.92, 1.5), (9, 0.72, 1.25), (13, 0.52, 1.0),
                  (17, 0.32, 0.8)]
    for wave_r, wave_alpha, wave_lw in arc_params:
        arc_t = np.linspace(angle0 - math.pi / 3, angle0 + math.pi / 3, 50)
        ax_arc = cx + wave_r * np.cos(arc_t)
        ay_arc = cy + wave_r * np.sin(arc_t)
        az_arc = np.full(50, float(cz))
        ax.plot(ax_arc, ay_arc, az_arc,
                color='#FF6B35', linewidth=wave_lw,
                alpha=wave_alpha, linestyle='-')

    pad = 12
    ax.set_xlim(cx - r_head - pad, cx + r_head + pad)
    ax.set_ylim(cy - r_head - pad, cy + r_head + pad)
    ax.set_zlim(cz - r_head - 8, cz + r_head + 8)
    ax.set_xlabel('X (cells)', fontsize=8, labelpad=2)
    ax.set_ylabel('Y (cells)', fontsize=8, labelpad=2)
    ax.set_zlabel('Z (cells)', fontsize=8, labelpad=2)
    ax.set_title('8-Element Antenna Ring — Microwave Illumination',
                 fontsize=10, fontweight='bold', pad=6)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=28, azim=38)

    ant_patch = mpatches.Patch(color='#1A6EBD', label='TX/RX Element')
    wave_patch = mpatches.Patch(color='#FF6B35', label='Microwave Signal')
    head_patch = mpatches.Patch(color='#BBBBBB', label='Head (wireframe)',
                                alpha=0.5)
    ax.legend(handles=[ant_patch, wave_patch, head_patch],
              loc='upper left', fontsize=7, framealpha=0.9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("WaveForge — 3D Brain Phantom Visualisation")
    print(f"Grid: {GRID_SIZE}^3, dx={DX_MM} mm, "
          f"{N_TX} antennas, ring_r={RING_RADIUS} cells")
    if _HAVE_PROJECT_MODULES:
        print("Using project modules (phantom + antenna).")
    else:
        print("Using inline fallback geometry (torch not available).")
    print("=" * 60)

    # Build grid (or use stub) and antenna ring
    if _HAVE_PROJECT_MODULES:
        grid = YeeGrid(GRID_SIZE, GRID_SIZE, dx=3e-3, dy=3e-3,
                       Nz=GRID_SIZE, dz=3e-3, device='cpu')
        ring = AntennaRing(N_TX, RING_RADIUS, Z_PLANE, grid)
    else:
        ring = AntennaRing(N_TX, RING_RADIUS, Z_PLANE)
    ant_positions = ring.positions

    # Use PHANTOM_A geometry
    geom = PHANTOM_A
    label_map = build_label_map(GRID_SIZE, geom, BLEED_CENTER, BLEED_RADIUS)

    print(f"Label map shape : {label_map.shape}")
    print(f"Unique labels   : {np.unique(label_map).tolist()}")
    print(f"Antenna positions: {ant_positions}")
    print(f"Bleed at {BLEED_CENTER}, radius={BLEED_RADIUS} cells")

    # -----------------------------------------------------------------------
    # Figure — 2x2 layout
    # -----------------------------------------------------------------------
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 14), facecolor='#F8F9FA')
    fig.suptitle(
        'WaveForge — Brain Head Phantom & 8-Element Microwave Antenna Ring',
        fontsize=14, fontweight='bold', color='#1A1A2E', y=0.98,
    )

    gs = GridSpec(2, 2, figure=fig,
                  left=0.04, right=0.97,
                  top=0.93, bottom=0.04,
                  wspace=0.24, hspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')

    # Style 2D axes
    for ax_ in (ax2, ax3):
        ax_.set_facecolor('#F0F2F5')
        for spine in ax_.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.8)

    # Style 3D axes
    for ax_ in (ax1, ax4):
        ax_.set_facecolor('#F0F2F5')
        ax_.xaxis.pane.fill = False
        ax_.yaxis.pane.fill = False
        ax_.zaxis.pane.fill = False
        ax_.xaxis.pane.set_edgecolor('#DDDDDD')
        ax_.yaxis.pane.set_edgecolor('#DDDDDD')
        ax_.zaxis.pane.set_edgecolor('#DDDDDD')
        ax_.grid(True, color='#E0E0E0', linewidth=0.35, alpha=0.6)

    print("\nRendering Panel 1: 3D volumetric cross-section...")
    draw_panel1(ax1, label_map, ant_positions, geom,
                BLEED_CENTER, BLEED_RADIUS, GRID_SIZE)

    print("Rendering Panel 2: Axial (XY) slice...")
    draw_panel2(ax2, label_map, ant_positions, BLEED_CENTER, GRID_SIZE, DX_MM)

    print("Rendering Panel 3: Sagittal (XZ) slice...")
    draw_panel3(ax3, label_map, BLEED_CENTER, GRID_SIZE, DX_MM, geom)

    print("Rendering Panel 4: 3D antenna ring schematic...")
    draw_panel4(ax4, ant_positions, geom, GRID_SIZE)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    out_dir = Path(__file__).parent.parent / 'output'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / '3d_head_phantom_visualisation.png'
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nSaved: {out_path}")

    docs_assets = Path(__file__).parent.parent.parent / 'docs' / 'assets'
    docs_assets.mkdir(parents=True, exist_ok=True)
    docs_path = docs_assets / '3d_head_phantom_visualisation.png'
    fig.savefig(str(docs_path), dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved: {docs_path}")

    plt.close(fig)
    print("\nWAVEFORGE_BENCH: N/A (visualisation only)")
    print("=" * 60)


if __name__ == '__main__':
    main()
