"""
WaveForge Brain Phantom — UWB Microwave Imaging Array
Publication-quality 3D visualization of a human head model with antenna ring array.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ---------------------------------------------------------------------------
# Utility: half-sphere mesh (cutaway on x>0 side)
# ---------------------------------------------------------------------------

def make_cutaway_sphere(radius, n_lat=40, n_lon=40, cutaway_fraction=0.45):
    """
    Return (X, Y, Z) arrays for a sphere surface with a wedge removed on the
    +x side so inner layers are visible.  cutaway_fraction is the fraction of
    the full 2*pi longitude that is cut away (centred on phi=0).
    """
    phi_cut = cutaway_fraction * np.pi          # half-angle of the cut
    phi_start = phi_cut
    phi_end   = 2 * np.pi - phi_cut

    lat  = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon  = np.linspace(phi_start, phi_end, n_lon)
    LON, LAT = np.meshgrid(lon, lat)

    X = radius * np.cos(LAT) * np.cos(LON)
    Y = radius * np.cos(LAT) * np.sin(LON)
    Z = radius * np.sin(LAT)
    return X, Y, Z


def make_full_sphere(radius, n_lat=30, n_lon=30):
    """Full sphere surface arrays."""
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon)
    LON, LAT = np.meshgrid(lon, lat)
    X = radius * np.cos(LAT) * np.cos(LON)
    Y = radius * np.cos(LAT) * np.sin(LON)
    Z = radius * np.sin(LAT)
    return X, Y, Z


# ---------------------------------------------------------------------------
# Utility: antenna cylinder pointing inward
# ---------------------------------------------------------------------------

def antenna_cylinder(cx, cy, length=15, radius=3, n=16):
    """
    Return a list of Poly3DCollection faces for a cylinder whose axis
    points from (cx, cy, 0) toward the origin.  The cylinder tip is
    at the inner end (toward origin).
    """
    # Unit vector from antenna position toward origin
    dist = np.sqrt(cx**2 + cy**2)
    ux, uy = -cx / dist, -cy / dist          # inward unit vector in XY plane

    # Cylinder centre: antenna outer base at (cx,cy,0), tip at (cx+ux*length, ...)
    # We build the cylinder in local coords then rotate.
    t  = np.linspace(0, 2 * np.pi, n + 1)
    # Local frame: axis = (ux, uy, 0), perp1 = (-uy, ux, 0), perp2 = (0, 0, 1)
    p1x, p1y, p1z = -uy, ux, 0.0
    p2x, p2y, p2z =  0.0, 0.0, 1.0

    base_pts  = []
    tip_pts   = []
    for ti in t:
        dx = radius * (np.cos(ti) * p1x + np.sin(ti) * p2x)
        dy = radius * (np.cos(ti) * p1y + np.sin(ti) * p2y)
        dz = radius * (np.cos(ti) * p1z + np.sin(ti) * p2z)
        base_pts.append((cx + dx,         cy + dy,         dz))
        tip_pts.append( (cx + ux*length + dx, cy + uy*length + dy, dz))

    faces = []
    for i in range(n):
        quad = [base_pts[i], base_pts[i+1], tip_pts[i+1], tip_pts[i]]
        faces.append(quad)

    # Cap at base
    cap_base = [base_pts[i] for i in range(n)]
    faces.append(cap_base)
    # Cap at tip
    cap_tip  = [tip_pts[i]  for i in range(n)]
    faces.append(cap_tip)

    return faces


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')

gs = GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.05)

ax3d  = fig.add_subplot(gs[0], projection='3d')
ax_ins = fig.add_subplot(gs[1])

ax3d.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')

# ---------------------------------------------------------------------------
# Layer definitions  (radii in mm, displayed as-is)
# ---------------------------------------------------------------------------

layers = [
    # (radius_mm, color,          alpha,  label,             zorder)
    (82,  '#D4A574',  0.20,  'Scalp (r=82 mm)',          1),
    (72,  '#C8C8C8',  0.28,  'Skull (r=72 mm)',           2),
    (54,  '#C09898',  0.40,  'Brain tissue (r=54 mm)',    3),
    (30,  '#E8C4C4',  0.60,  'White matter (r=30 mm)',    4),
]

bleed_radius = 12  # mm, offset inside brain
bleed_center = np.array([18.0, 12.0, 8.0])   # mm offset from origin

# ---- draw layers from outside in (so outer surfaces render first) ----------

for (r, color, alpha, label, zo) in layers:
    X, Y, Z = make_cutaway_sphere(r, n_lat=50, n_lon=50, cutaway_fraction=0.40)
    surf = ax3d.plot_surface(
        X, Y, Z,
        color=color, alpha=alpha,
        linewidth=0, antialiased=True,
        zorder=zo,
    )

# ---- haemorrhage (full sphere, fully opaque red) ---------------------------

Xb, Yb, Zb = make_full_sphere(bleed_radius, n_lat=24, n_lon=24)
ax3d.plot_surface(
    Xb + bleed_center[0],
    Yb + bleed_center[1],
    Zb + bleed_center[2],
    color='#CC1111', alpha=1.0,
    linewidth=0, antialiased=True,
    zorder=5,
)

# ---------------------------------------------------------------------------
# Antenna ring (8 elements at z=0, r=90 mm)
# ---------------------------------------------------------------------------

n_ant      = 8
ant_radius = 90.0   # mm
ant_angles = np.linspace(0, 2 * np.pi, n_ant, endpoint=False)

ant_positions = []
for angle in ant_angles:
    cx = ant_radius * np.cos(angle)
    cy = ant_radius * np.sin(angle)
    ant_positions.append((cx, cy, 0.0))

antenna_color = '#FFA040'

for (cx, cy, _) in ant_positions:
    faces = antenna_cylinder(cx, cy, length=18, radius=2.5, n=14)
    col = Poly3DCollection(faces, alpha=0.92, linewidth=0.3,
                           edgecolor='#FFCC80', zorder=6)
    col.set_facecolor(antenna_color)
    ax3d.add_collection3d(col)

    # Small marker dot at base of each antenna
    ax3d.scatter([cx], [cy], [0], color='#FFCC00', s=18, zorder=7, depthshade=False)

# ---------------------------------------------------------------------------
# TX→RX signal arcs between selected antenna pairs
# ---------------------------------------------------------------------------

# Pick 4 TX→RX pairs (across the ring)
arc_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
arc_color = '#40E0FF'

for (ti, ri) in arc_pairs:
    tx = np.array(ant_positions[ti])
    rx = np.array(ant_positions[ri])
    # Parametric arc with slight z bow
    t_vals = np.linspace(0, 1, 60)
    pts = np.outer(1 - t_vals, tx) + np.outer(t_vals, rx)
    # Add a gentle upward bow peaking at midpoint
    bow_z = 18 * np.sin(np.pi * t_vals)
    ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2] + bow_z,
              '--', color=arc_color, linewidth=0.8, alpha=0.70, zorder=8)

    # Arrow head at RX end
    ax3d.quiver(pts[-2, 0], pts[-2, 1], pts[-2, 2] + bow_z[-2],
                rx[0] - pts[-2, 0], rx[1] - pts[-2, 1], 0,
                color=arc_color, alpha=0.75, length=4, arrow_length_ratio=0.8,
                linewidth=0.5)

# ---------------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------------

lim = 110
ax3d.set_xlim(-lim, lim)
ax3d.set_ylim(-lim, lim)
ax3d.set_zlim(-lim, lim)
ax3d.set_box_aspect([1, 1, 1])

ax3d.set_xlabel('X (mm)', color='#AAAAAA', fontsize=8, labelpad=4)
ax3d.set_ylabel('Y (mm)', color='#AAAAAA', fontsize=8, labelpad=4)
ax3d.set_zlabel('Z (mm)', color='#AAAAAA', fontsize=8, labelpad=4)

ax3d.tick_params(colors='#666666', labelsize=7)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('#222222')
ax3d.yaxis.pane.set_edgecolor('#222222')
ax3d.zaxis.pane.set_edgecolor('#222222')
ax3d.grid(False)

ax3d.view_init(elev=22, azim=-55)

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

ax3d.set_title(
    'WaveForge Brain Phantom — UWB Microwave Imaging Array',
    color='white', fontsize=12, fontweight='bold', pad=10,
)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

legend_elements = [
    mpatches.Patch(facecolor='#D4A574', alpha=0.6, edgecolor='#AAAAAA', label='Scalp (r=82 mm)'),
    mpatches.Patch(facecolor='#C8C8C8', alpha=0.6, edgecolor='#AAAAAA', label='Skull (r=72 mm)'),
    mpatches.Patch(facecolor='#C09898', alpha=0.7, edgecolor='#AAAAAA', label='Brain tissue (r=54 mm)'),
    mpatches.Patch(facecolor='#E8C4C4', alpha=0.8, edgecolor='#AAAAAA', label='White matter (r=30 mm)'),
    mpatches.Patch(facecolor='#CC1111', alpha=1.0, edgecolor='#FF4444', label='Haemorrhage (r=12 mm)'),
    mpatches.Patch(facecolor='#FFA040', alpha=0.9, edgecolor='#FFCC80', label='UWB Antenna (×8, r=90 mm)'),
    plt.Line2D([0], [0], linestyle='--', color='#40E0FF', linewidth=1.2, label='TX→RX signal arc'),
]

leg = ax3d.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(-0.02, 1.02),
    fontsize=7.5,
    framealpha=0.25,
    facecolor='#1a1f2e',
    edgecolor='#444455',
    labelcolor='white',
    ncol=1,
)

# ---------------------------------------------------------------------------
# Inset: Gaussian monocycle pulse waveform (0.5–1.5 GHz UWB)
# ---------------------------------------------------------------------------

ax_ins.set_facecolor('#0d1117')

# Gaussian monocycle: derivative of Gaussian envelope
# Centre freq ~1 GHz -> sigma chosen so bandwidth ~0.5-1.5 GHz
t_ns   = np.linspace(-1.5, 1.5, 500)          # nanoseconds
fc     = 1.0e9                                  # centre 1 GHz
bw     = 0.7e9                                  # 70% fractional BW
sigma  = 1.0 / (2 * np.pi * bw / np.sqrt(2 * np.log(2)))   # ~0.187 ns
sigma_ns = sigma * 1e9

pulse  = -(t_ns / sigma_ns) * np.exp(-0.5 * (t_ns / sigma_ns)**2)
pulse /= np.max(np.abs(pulse))

ax_ins.plot(t_ns, pulse, color='#40E0FF', linewidth=1.5, label='Gaussian monocycle')
ax_ins.fill_between(t_ns, pulse, 0, alpha=0.15, color='#40E0FF')
ax_ins.axhline(0, color='#444455', linewidth=0.6, linestyle='--')

# Frequency annotation
ax_ins.text(0.97, 0.93, '0.5–1.5 GHz\nUWB band', transform=ax_ins.transAxes,
            ha='right', va='top', fontsize=7, color='#AAAACC',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1f2e',
                      edgecolor='#444455', alpha=0.8))

ax_ins.set_xlabel('Time (ns)', color='#AAAAAA', fontsize=7.5)
ax_ins.set_ylabel('Amplitude (norm.)', color='#AAAAAA', fontsize=7.5)
ax_ins.set_title('UWB Transmit Pulse', color='#CCCCCC', fontsize=8.5, pad=6)

ax_ins.tick_params(colors='#666666', labelsize=7)
ax_ins.spines['bottom'].set_color('#333344')
ax_ins.spines['left'].set_color('#333344')
ax_ins.spines['top'].set_color('#333344')
ax_ins.spines['right'].set_color('#333344')

ax_ins.set_xlim(-1.5, 1.5)
ax_ins.set_ylim(-1.2, 1.2)
ax_ins.yaxis.set_tick_params(labelsize=7, labelcolor='#888888')
ax_ins.xaxis.set_tick_params(labelsize=7, labelcolor='#888888')

# Add a second x-axis showing approximate frequency range label
ax_ins2 = ax_ins.inset_axes([0.0, -0.28, 1.0, 0.18])
ax_ins2.set_facecolor('#0d1117')
ax_ins2.set_xlim(0, 3)
ax_ins2.set_ylim(0, 1)
ax_ins2.axvline(0.5, color='#FFA040', linewidth=1.0, alpha=0.7)
ax_ins2.axvline(1.5, color='#FFA040', linewidth=1.0, alpha=0.7)
ax_ins2.fill_betweenx([0, 1], 0.5, 1.5, alpha=0.18, color='#FFA040')
ax_ins2.set_xlabel('Frequency (GHz)', color='#AAAAAA', fontsize=7)
ax_ins2.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
ax_ins2.set_xticklabels(['0.5', '1', '1.5', '2', '2.5'],
                         fontsize=6.5, color='#888888')
ax_ins2.set_yticks([])
for sp in ax_ins2.spines.values():
    sp.set_color('#333344')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_path = '/home/zuu/GPU-MEEP/internal_bleeding/figures/head_antenna_array.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f'Saved: {out_path}')
