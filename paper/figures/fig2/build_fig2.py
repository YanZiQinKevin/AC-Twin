"""Build the AC-Twin Fig. 2 method overview (editable SVG + 600 dpi PNG).

Draft v4 — the draft-v3 layout with the figure-set style rules applied:
  * no in-figure title and no footer sentence; both belong to the caption, so
    every figure in the set carries only the drawing;
  * the canvas is cropped to the drawn content (180 x 88 mm) instead of leaving
    the two freed bands as dead space, at an unchanged physical scale;
  * in-figure type is raised by 8% for legibility at 100% column width.
Draft v3 presentation (kept):
  * two tinted macro regions separate the inference path from the sealed
    training/evaluation truth region;
  * key modules are drawn as extruded slabs, everything else as flat
    tinted panels;
  * the figure body carries no method formulas or hyperparameters; the
    action count and fixed retention coverage are shown explicitly.
    The illustrative scene in Stage I reuses the supplied figure asset.

The final composition omits the training-label tag and the redundant prefix
label. Only the supplied figure assets are loaded by the build entry point;
no experiment data or statistical analysis is accessed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SCENE_NPZ = (ROOT / "data/object_centric_stage/stage2_merge_1200"
             / "object_twinworld_stage2_merge_1200.npz")

MM = 1 / 25.4

# The canvas keeps draft v3's physical scale exactly (180.0 mm per x-unit and
# 100.0 mm per y-unit, the two constants the icon and scene helpers below rely
# on), but the horizontal bands that carried the old in-figure title and footer
# are cropped away: the axes now show only the drawn content with a 2.5 mm
# margin, so no physical size in this file changes.
AX_Y0, AX_Y1 = 0.0491, 0.9310        # draft-v3 y band kept in the canvas
FIG_W = 180 * MM
FIG_H = (AX_Y1 - AX_Y0) * 100.0 * MM  # 88.2 mm

FROZEN = "#2F5D7C"
def _mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def lighten(c: str, t: float) -> str:
    return _mix(c, "#FFFFFF", t)


def darken(c: str, t: float) -> str:
    return _mix(c, "#000000", t)


# Colour system derived from one published palette (Advanced Materials case on
# color.amfe.space, doi 10.1002/adma.202405009). Each hue keeps a fixed meaning;
# lighter and darker steps are generated from the same source hues.
# Colour system: the six supplied hues. Each hue keeps one meaning; lighter and
# darker steps are generated from the same hues.
_PALETTE_BLUE = "#4D93D6"        # AC-Twin audit / Direct audit head
_PALETTE_BLUE_LIGHT = "#A4C8EA"  # method tints and the inference region
_PALETTE_RED = "#D57873"         # counterfactual branches, truth-only family
_PALETTE_PEACH = "#F1C2AD"       # truth-region tint
_PALETTE_GREY = "#A8A8A9"        # frozen predictor
_PALETTE_GREY_LIGHT = "#D9D9D9"  # frames and neutral bars

FROZEN = _PALETTE_GREY
FROZEN_TOP = lighten(_PALETTE_GREY, 0.42)
FROZEN_SIDE = darken(_PALETTE_GREY, 0.22)
FROZEN_SLAB = darken(_PALETTE_GREY, 0.18)   # white labels need more contrast
METHOD = _PALETTE_BLUE
METHOD_TOP = lighten(_PALETTE_BLUE, 0.38)
METHOD_SIDE = darken(_PALETTE_BLUE, 0.24)
BRANCH = _PALETTE_RED
BRANCH_SOFT = lighten(_PALETTE_PEACH, 0.45)
TRUTH = darken(_PALETTE_RED, 0.32)
TRUTH_TOP = darken(_PALETTE_RED, 0.12)
TRUTH_SIDE = darken(_PALETTE_RED, 0.50)
FRAME = _PALETTE_GREY_LIGHT
INK = "#2B2B2B"
MUTED = "#8C8C8E"
LIGHT = lighten(_PALETTE_GREY_LIGHT, 0.55)
WHITE = "#FFFFFF"
INFER_TINT = lighten(_PALETTE_BLUE_LIGHT, 0.55)
TRUTH_TINT = lighten(_PALETTE_PEACH, 0.55)
FEATURE_FRONT = lighten(_PALETTE_BLUE_LIGHT, 0.25)
FEATURE_TOP = lighten(_PALETTE_BLUE_LIGHT, 0.62)
FEATURE_SIDE = darken(_PALETTE_BLUE_LIGHT, 0.12)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    }
)

# Fig. 1/3 set the smallest in-figure type at 6.6 pt; draft v3 went down to
# 6.6 pt as well, which is too small at 100% column width. One scale factor
# keeps every label's relative hierarchy intact.
FONT_SCALE = 1.08

DASH = (0, (3.0, 2.2))
DASH_FINE = (0, (2.2, 1.7))

# the illustrative prefix is shown over the last few observed steps only, so that
# the equal-metric-scale scene stays legible inside a narrow stage column
PREFIX_STEPS = 4

# illustrative vehicle footprint for the thumbnail, in metres
CAR_LEN, CAR_WID = 4.6, 1.9

# frame geometry (axes fraction)
STAGE_Y, STAGE_H = 0.372, 0.476
STAGE_TOP = STAGE_Y + STAGE_H          # 0.848
INFER_BOX = (0.012, 0.352, 0.725, 0.550)
TRUTH_BOX = (0.012, 0.078, 0.725, 0.214)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def rounded(ax, x, y, w, h, *, fc=WHITE, ec=FRAME, lw=0.9, radius=0.010,
            linestyle="-", zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.002,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=linestyle,
        transform=ax.transAxes, clip_on=False, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def slab(ax, x, y, w, h, *, front, top, side, depth=0.010, ec="none",
         lw=0.0, zorder=4):
    """Extruded module block: front face plus top and right faces."""
    ax.add_patch(Polygon(
        [(x, y + h), (x + depth, y + h + depth),
         (x + w + depth, y + h + depth), (x + w, y + h)],
        closed=True, facecolor=top, edgecolor=ec, linewidth=lw,
        transform=ax.transAxes, clip_on=False, zorder=zorder - 1))
    ax.add_patch(Polygon(
        [(x + w, y), (x + w + depth, y + depth),
         (x + w + depth, y + h + depth), (x + w, y + h)],
        closed=True, facecolor=side, edgecolor=ec, linewidth=lw,
        transform=ax.transAxes, clip_on=False, zorder=zorder - 1))
    rounded(ax, x, y, w, h, fc=front, ec=ec, lw=lw, radius=0.006, zorder=zorder)


def arrow(ax, x0, y0, x1, y1, *, color=MUTED, lw=1.0, ms=6.5, zorder=6,
          linestyle="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, transform=ax.transAxes, linestyle=linestyle,
        shrinkA=0, shrinkB=0, clip_on=False, zorder=zorder))


def text(ax, x, y, s, *, size=7.8, color=INK, weight="normal", ha="center",
         va="center", zorder=9):
    return ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va,
                   fontsize=size * FONT_SCALE, color=color, fontweight=weight,
                   linespacing=1.10, zorder=zorder, fontfamily="DejaVu Sans")


def spaced(s: str) -> str:
    return "\u2009".join(s)


# --------------------------------------------------------------------------
# iconfont icons: parsed to filled vector paths and recoloured to the palette
# --------------------------------------------------------------------------
ICON_DIR = OUT / "icon"
SCENE_IMG = ICON_DIR / "Observed prefix.png"
_ICON_CACHE: dict[str, tuple] = {}


def _arc_to_cubics(x0, y0, rx, ry, phi, large, sweep, x1, y1):
    """Endpoint-parameterised SVG arc -> list of cubic Bezier control tuples."""
    import math
    if rx == 0 or ry == 0:
        return [(x1, y1, x1, y1, x1, y1)]
    phi = math.radians(phi)
    cosp, sinp = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x0 - x1) / 2.0, (y0 - y1) / 2.0
    x1p = cosp * dx2 + sinp * dy2
    y1p = -sinp * dx2 + cosp * dy2
    rx, ry = abs(rx), abs(ry)
    lam = x1p ** 2 / rx ** 2 + y1p ** 2 / ry ** 2
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1p ** 2 - ry ** 2 * x1p ** 2
    den = rx ** 2 * y1p ** 2 + ry ** 2 * x1p ** 2
    coef = math.sqrt(max(num / den, 0.0)) if den else 0.0
    if large == sweep:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cosp * cxp - sinp * cyp + (x0 + x1) / 2.0
    cy = sinp * cxp + cosp * cyp + (y0 + y1) / 2.0

    def _angle(ux, uy, vx, vy):
        n = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, (ux * vx + uy * vy) / n))) if n else 0.0
        return -a if (ux * vy - uy * vx) < 0 else a

    theta0 = _angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = _angle((x1p - cxp) / rx, (y1p - cyp) / ry,
                    (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if sweep == 0 and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep == 1 and dtheta < 0:
        dtheta += 2 * math.pi
    steps = max(1, int(math.ceil(abs(dtheta) / (math.pi / 2.0))))
    out = []
    for i in range(steps):
        a0 = theta0 + dtheta * i / steps
        a1 = theta0 + dtheta * (i + 1) / steps
        alpha = 4.0 / 3.0 * math.tan((a1 - a0) / 4.0)

        def pt(a):
            return (cx + rx * math.cos(a) * cosp - ry * math.sin(a) * sinp,
                    cy + rx * math.cos(a) * sinp + ry * math.sin(a) * cosp)

        def dv(a):
            return (-rx * math.sin(a) * cosp - ry * math.cos(a) * sinp,
                    -rx * math.sin(a) * sinp + ry * math.cos(a) * cosp)

        p0, p1, d0, d1 = pt(a0), pt(a1), dv(a0), dv(a1)
        out.append((p0[0] + alpha * d0[0], p0[1] + alpha * d0[1],
                    p1[0] - alpha * d1[0], p1[1] - alpha * d1[1], p1[0], p1[1]))
    return out


def _emit_subpaths(d, verts, codes):
    """Append one SVG path definition to matplotlib vertex/code lists."""
    import re
    from matplotlib.path import Path

    tokens = re.findall(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)", d)
    cx = cy = sx = sy = 0.0
    prev_c = prev_q = None
    for raw_cmd, arg in tokens:
        nums = [float(v) for v in re.findall(
            r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", arg)]
        cmd = raw_cmd.upper()
        rel = raw_cmd.islower()
        if cmd != "C" and cmd != "S":
            prev_c = None
        if cmd != "Q" and cmd != "T":
            prev_q = None
        if cmd == "M":
            for i in range(0, len(nums), 2):
                x = nums[i] + (cx if rel else 0.0)
                y = nums[i + 1] + (cy if rel else 0.0)
                if i == 0:
                    verts.append((x, y))
                    codes.append(Path.MOVETO)
                    sx, sy = x, y
                else:
                    verts.append((x, y))
                    codes.append(Path.LINETO)
                cx, cy = x, y
        elif cmd == "L":
            for i in range(0, len(nums), 2):
                x = nums[i] + (cx if rel else 0.0)
                y = nums[i + 1] + (cy if rel else 0.0)
                verts.append((x, y))
                codes.append(Path.LINETO)
                cx, cy = x, y
        elif cmd == "H":
            for v in nums:
                cx = v + (cx if rel else 0.0)
                verts.append((cx, cy))
                codes.append(Path.LINETO)
        elif cmd == "V":
            for v in nums:
                cy = v + (cy if rel else 0.0)
                verts.append((cx, cy))
                codes.append(Path.LINETO)
        elif cmd == "C":
            for i in range(0, len(nums), 6):
                p = [(nums[i + k] + (cx if rel else 0.0),
                      nums[i + k + 1] + (cy if rel else 0.0))
                     for k in (0, 2, 4)]
                verts.extend(p)
                codes.extend([Path.CURVE4] * 3)
                prev_c = p[1]
                cx, cy = p[2]
        elif cmd == "S":
            for i in range(0, len(nums), 4):
                p0 = (2 * cx - prev_c[0], 2 * cy - prev_c[1]) if prev_c else (cx, cy)
                p1 = (nums[i] + (cx if rel else 0.0), nums[i + 1] + (cy if rel else 0.0))
                p2 = (nums[i + 2] + (cx if rel else 0.0), nums[i + 3] + (cy if rel else 0.0))
                verts.extend([p0, p1, p2])
                codes.extend([Path.CURVE4] * 3)
                prev_c = p1
                cx, cy = p2
        elif cmd == "Q":
            for i in range(0, len(nums), 4):
                p0 = (nums[i] + (cx if rel else 0.0), nums[i + 1] + (cy if rel else 0.0))
                p1 = (nums[i + 2] + (cx if rel else 0.0), nums[i + 3] + (cy if rel else 0.0))
                verts.extend([p0, p1])
                codes.extend([Path.CURVE3] * 2)
                prev_q = p0
                cx, cy = p1
        elif cmd == "T":
            for i in range(0, len(nums), 2):
                p0 = (2 * cx - prev_q[0], 2 * cy - prev_q[1]) if prev_q else (cx, cy)
                p1 = (nums[i] + (cx if rel else 0.0), nums[i + 1] + (cy if rel else 0.0))
                verts.extend([p0, p1])
                codes.extend([Path.CURVE3] * 2)
                prev_q = p0
                cx, cy = p1
        elif cmd == "A":
            for i in range(0, len(nums), 7):
                rx, ry, rot, large, sweep = nums[i:i + 5]
                x = nums[i + 5] + (cx if rel else 0.0)
                y = nums[i + 6] + (cy if rel else 0.0)
                for seg in _arc_to_cubics(cx, cy, rx, ry, rot, int(large),
                                          int(sweep), x, y):
                    verts.extend([(seg[0], seg[1]), (seg[2], seg[3]),
                                  (seg[4], seg[5])])
                    codes.extend([Path.CURVE4] * 3)
                cx, cy = x, y
        elif cmd == "Z":
            verts.append((sx, sy))
            codes.append(Path.CLOSEPOLY)
            cx, cy = sx, sy


def _icon_geometry(fname):
    import re
    from matplotlib.path import Path
    if fname not in _ICON_CACHE:
        src = (ICON_DIR / fname).read_text(encoding="utf-8")
        found = re.search(r'viewBox="([-\d.\s]+)"', src)
        box = [float(v) for v in found.group(1).split()]
        verts, codes = [], []
        for d in re.findall(r'<path[^>]*?\sd="([^"]+)"', src):
            _emit_subpaths(d, verts, codes)
        _ICON_CACHE[fname] = (verts, codes, box)
    return _ICON_CACHE[fname]


def draw_icon(ax, fname, cx, cy, size_mm, color, *, zorder=9):
    """Draw one iconfont glyph as a filled path at a physical size in mm."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    if not (ICON_DIR / fname).exists():
        return False
    verts, codes, (vx, vy, vw, vh) = _icon_geometry(fname)
    if not verts:
        return False
    unit = size_mm / max(vw, vh)                 # mm per user unit
    ox, oy = vx + vw / 2.0, vy + vh / 2.0
    pts = [((x - ox) * unit / 180.0 + cx, -(y - oy) * unit / 100.0 + cy)
           for x, y in verts]
    ax.add_patch(PathPatch(Path(pts, codes), facecolor=color, edgecolor="none",
                           transform=ax.transAxes, clip_on=False, zorder=zorder))
    return True


def draw_icon_image(ax, fname, cx, cy, size_mm, *, zorder=8):
    """Place a raster icon at a physical size in mm, aspect preserved."""
    path = ICON_DIR / fname
    if not path.exists():
        return False
    img = plt.imread(path)
    half_w = size_mm / 2.0 / 180.0
    half_h = size_mm / 2.0 / 100.0
    ax.imshow(img, extent=(cx - half_w, cx + half_w, cy - half_h, cy + half_h),
              aspect="auto", interpolation="lanczos", transform=ax.transAxes,
              zorder=zorder)
    return True


def lock(ax, x, y, color=TRUTH, scale=1.0, zorder=9):
    """Lock glyph; (x, y) is the lower-left corner of the requested box."""
    size = 3.0 * scale                                  # mm
    return draw_icon(ax, "锁,密码,锁定.svg", x + size / 2 / 180.0,
                     y + size / 2 / 100.0, size, color, zorder=zorder)


def gauge(ax, cx, cy, r=0.018, color=METHOD):
    """Dial glyph used for the audit-risk output; r is a half-height."""
    return draw_icon(ax, "异常检测 仪表盘.svg", cx, cy, r * 200.0 * 0.55, color,
                     zorder=8)


# --------------------------------------------------------------------------
# illustrative prefix (frozen Stage-2 Merge task data, ego-local frame)
# --------------------------------------------------------------------------
def load_prefix_example():
    """First prefix of the frozen task data with three or more objects present
    and more than 1.5 m of ego lateral displacement, in ego-local coordinates."""
    if not SCENE_NPZ.exists():
        return None
    with np.load(SCENE_NPZ, allow_pickle=False) as data:
        history = np.asarray(data["object_history"], dtype=np.float64)
    for episode in range(history.shape[0]):
        scene = history[episode]                       # (steps, objects, feats)
        present = scene[:, :, 0] > 0.5
        ego = int(np.argmax(scene[0, :, 1]))
        if not present[:, ego].all():
            continue
        if int(present[-1].sum()) < 3:
            continue
        ego_xy = scene[:, ego, 2:4]
        if abs(ego_xy[-1, 1] - ego_xy[0, 1]) < 1.5:
            continue

        cos_h, sin_h = scene[-1, ego, 6], scene[-1, ego, 7]
        angle = np.arctan2(sin_h, cos_h)
        rot = np.array([[np.cos(-angle), -np.sin(-angle)],
                        [np.sin(-angle), np.cos(-angle)]])
        tracks = []
        for obj in range(scene.shape[1]):
            keep = present[:, obj]
            if keep.sum() < 2:
                continue
            pts = scene[keep, obj, 2:4][-PREFIX_STEPS:]
            last = np.flatnonzero(keep)[-1]
            rel = (np.arctan2(scene[last, obj, 7], scene[last, obj, 6])
                   - angle)
            tracks.append(((pts - ego_xy[-1]) @ rot.T, obj == ego, rel))
        return tracks
    return None


def draw_prefix_scene(ax, x, y, w, h, tracks):
    """Draw the real prefix tracks inside the given axes-fraction rectangle."""
    if not tracks:
        return
    all_pts = np.vstack([t[0] for t in tracks])
    span_x = max(all_pts[:, 0].max() - all_pts[:, 0].min(), 1.0)
    span_y = max(all_pts[:, 1].max() - all_pts[:, 1].min(), 1.0)
    # the physical canvas is 180 x 100 mm, so equal metric scale needs this weight
    scale = min((w * 180.0) / span_x, (h * 100.0) / span_y)     # mm per metre
    cx = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2.0
    cy = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2.0
    ox, oy = x + w / 2.0, y + h / 2.0

    def to_axes(pts):
        px = ox + (pts[:, 0] - cx) * scale / 180.0
        py = oy + (pts[:, 1] - cy) * scale / 100.0
        return px, py

    for pts, is_ego, rel_heading in tracks:
        px, py = to_axes(pts)
        ax.plot(px, py, color=FROZEN if is_ego else lighten(_PALETTE_GREY, 0.35),
                lw=1.30 if is_ego else 0.75, solid_capstyle="round",
                transform=ax.transAxes, zorder=4)
        # vehicle box at the last observed step, real footprint and heading
        fwd = np.array([np.cos(rel_heading), np.sin(rel_heading)])
        lft = np.array([-np.sin(rel_heading), np.cos(rel_heading)])
        tail = pts[-1]
        ring = np.array([
            tail + fwd * CAR_LEN / 2 + lft * CAR_WID / 2,
            tail + fwd * CAR_LEN / 2 - lft * CAR_WID / 2,
            tail - fwd * CAR_LEN / 2 - lft * CAR_WID / 2,
            tail - fwd * CAR_LEN / 2 + lft * CAR_WID / 2,
        ])
        bx, by = to_axes(ring)
        ax.add_patch(Polygon(np.column_stack([bx, by]), closed=True,
                             facecolor=FROZEN if is_ego else lighten(_PALETTE_GREY, 0.25),
                             edgecolor=WHITE if is_ego else lighten(_PALETTE_GREY, 0.35),
                             linewidth=0.35, transform=ax.transAxes, zorder=6))


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------
def macro_region(ax, box, *, tint, edge):
    x, y, w, h = box
    rounded(ax, x, y, w, h, fc=tint, ec=edge, lw=0.95, radius=0.014,
            linestyle=DASH, zorder=0)


def stage_box(ax, x, w, label):
    rounded(ax, x, STAGE_Y, w, STAGE_H, fc=WHITE, ec=FRAME, lw=0.85,
            radius=0.010, linestyle=DASH, zorder=1)
    text(ax, x + 0.010, STAGE_TOP - 0.028, label, size=8.0, color=TRUTH,
         weight="bold", ha="left")


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def draw_scene_card(ax, x, y, w, tracks):
    """Real scene panel: the supplied map/trajectory image, else frozen tracks."""
    if SCENE_IMG.exists():
        ratio = 3.058                                    # supplied image w/h
        h = (w * 180.0) / ratio / 100.0
        rounded(ax, x - 0.004, y - 0.004, w + 0.008, h + 0.008, fc=WHITE,
                ec=FRAME, lw=0.5, radius=0.004, zorder=2)
        ax.imshow(plt.imread(SCENE_IMG), extent=(x, x + w, y, y + h),
                  aspect="auto", interpolation="lanczos",
                  transform=ax.transAxes, zorder=3)
        return
    draw_prefix_scene(ax, x, y, w, 0.114, tracks)


def draw_stage_i(ax, x, w, tracks):
    cx = x + w / 2
    draw_scene_card(ax, x + 0.014, 0.650, w - 0.028, tracks)
    rounded(ax, x + 0.023, 0.606, w - 0.046, 0.028,
            fc=INFER_TINT, ec=FROZEN, lw=0.7, radius=0.004)
    text(ax, cx, 0.620, "Observed factual suffix", size=7.0)
    arrow(ax, cx, 0.600, cx, 0.588, color=BRANCH, lw=0.95)

    # the ten branches of the frozen action contract, as five action groups
    origin_y, tip_y = 0.588, 0.492
    step = (w - 0.024) / 5.0
    for group in range(5):
        for offset in (0.011, 0.028):
            px = x + 0.012 + group * step + offset
            ax.plot([cx, px], [origin_y, tip_y], color=BRANCH, lw=0.62,
                    transform=ax.transAxes, zorder=3)
            ax.add_patch(Circle((px, tip_y), 0.0026, facecolor=BRANCH_SOFT,
                                edgecolor=BRANCH, linewidth=0.7,
                                transform=ax.transAxes, zorder=5))
    text(ax, cx, 0.455, "Counterfactual action branches", size=7.6)
    text(ax, cx, 0.430, "5 actions x 2 durations", size=7.0, color=MUTED)


def draw_stage_ii(ax, x, w):
    cx = x + w / 2
    bw = w * 0.88

    slab(ax, cx - bw / 2, 0.685, bw, 0.078,
         front=FROZEN_SLAB, top=FROZEN_TOP, side=FROZEN_SIDE)
    text(ax, cx, 0.724, "Frozen prediction\nstack", size=7.2, color=WHITE,
         weight="bold", zorder=9)

    arrow(ax, cx, 0.678, cx, 0.652, color=FROZEN_SIDE, lw=0.95)
    text(ax, cx, 0.632, "Predicted cost", size=7.9, color=MUTED)

    bar_y = 0.552
    for i, hh in enumerate((0.030, 0.045, 0.024, 0.039, 0.028)):
        ax.add_patch(Rectangle((cx - 0.042 + i * 0.021, bar_y), 0.014, hh,
                               facecolor=FROZEN, edgecolor="none",
                               transform=ax.transAxes, zorder=4))

    arrow(ax, cx, 0.542, cx, 0.516, color=FROZEN, lw=0.95)
    dot_y = 0.488
    for i in range(10):
        px = cx - 0.076 + i * 0.017
        chosen = i == 6
        ax.add_patch(Circle(
            (px, dot_y), 0.0050 if chosen else 0.0029,
            facecolor=METHOD if chosen else WHITE,
            edgecolor=METHOD if chosen else FRAME,
            linewidth=0.9 if chosen else 0.6,
            transform=ax.transAxes, zorder=6))
    text(ax, cx, 0.446, "Selected action", size=7.9, color=METHOD)
    text(ax, cx, 0.420, "minimum predicted cost", size=6.6, color=MUTED)


def draw_stage_iii(ax, x, w):
    cx = x + w / 2
    bw = w * 0.82

    slab(ax, cx - bw / 2, 0.662, bw, 0.058,
         front=FEATURE_FRONT, top=FEATURE_TOP, side=FEATURE_SIDE,
         ec=METHOD, lw=0.6)
    text(ax, cx, 0.691, "Inference-legal\nfeatures", size=7.0,
         color=darken(_PALETTE_BLUE, 0.35), weight="bold", zorder=9)

    arrow(ax, cx, 0.656, cx, 0.638, color=METHOD, lw=0.95)
    slab(ax, cx - bw / 2, 0.580, bw, 0.058,
         front=METHOD, top=METHOD_TOP, side=METHOD_SIDE)
    text(ax, cx, 0.609, "Direct audit head", size=7.6, color=WHITE,
         weight="bold", zorder=9)

    arrow(ax, cx, 0.574, cx, 0.550, color=METHOD, lw=0.95)
    draw_icon_image(ax, "audit.png", cx, 0.505, 8.0, zorder=8)
    text(ax, cx, 0.448, "Regret audit score", size=7.4, color=METHOD)


def draw_stage_iv(ax, x, w):
    cx = x + w / 2
    bw = w * 0.86

    rounded(ax, cx - bw / 2, 0.656, bw, 0.058, fc=WHITE, ec=METHOD,
            lw=0.95, radius=0.008, zorder=3)
    draw_icon(ax, "authcode.svg", cx - bw / 2 + 0.014, 0.685, 4.4, METHOD,
              zorder=8)
    text(ax, cx + 0.013, 0.685, "Rank by audit score", size=7.0, color=METHOD,
         weight="bold")

    arrow(ax, cx, 0.649, cx, 0.619, color=METHOD, lw=0.95)
    slab(ax, cx - bw / 2, 0.552, bw, 0.058,
         front=METHOD, top=METHOD_TOP, side=METHOD_SIDE)
    draw_icon(ax, "筛选.svg", cx - bw / 2 + 0.014, 0.581, 4.2, WHITE,
              zorder=10)
    text(ax, cx + 0.013, 0.581, "Retain lowest-risk", size=7.6, color=WHITE,
         weight="bold", zorder=9)

    arrow(ax, cx, 0.545, cx, 0.515, color=METHOD, lw=0.95)
    bar_w, bar_h, bar_y = bw, 0.028, 0.472
    bar_x = cx - bar_w / 2
    rounded(ax, bar_x, bar_y, bar_w * 0.78, bar_h, fc=METHOD, ec=METHOD,
            lw=0.6, radius=0.004, zorder=3)
    rounded(ax, bar_x + bar_w * 0.80, bar_y, bar_w * 0.20, bar_h,
            fc=LIGHT, ec=FRAME, lw=0.6, radius=0.004, zorder=3)
    tick_x = bar_x + bar_w * 0.79
    ax.plot([tick_x, tick_x], [bar_y - 0.013, bar_y + bar_h + 0.013],
            color=MUTED, lw=0.8, transform=ax.transAxes, zorder=5)
    text(ax, tick_x, bar_y + bar_h + 0.017, "80%", size=7.0,
         color=METHOD, weight="bold")
    text(ax, bar_x + bar_w * 0.30, bar_y - 0.030, "retained", size=7.0,
         color=METHOD, weight="bold")
    text(ax, bar_x + bar_w, bar_y - 0.030, "not retained", size=7.0,
         color=MUTED, ha="right")


def draw_evaluation_chip(ax):
    """The truth chain ends in an evaluation of the audit, outside the inference path."""
    w, h = 0.120, 0.056
    x, y = 0.870 - w / 2, 0.297
    rounded(ax, x, y, w, h, fc=WHITE, ec=TRUTH, lw=0.7, radius=0.007,
            linestyle=DASH_FINE, zorder=1)
    lock(ax, x + 0.007, y + h / 2 - 0.007, scale=0.8)
    text(ax, x + 0.027, y + h / 2, "evaluates\nthe audit", size=6.6,
         color=darken(_PALETTE_RED, 0.22), ha="left")
    arrow(ax, 0.739, 0.301, x - 0.006, y + h / 2, color=TRUTH, lw=0.8,
          ms=5.5, linestyle=DASH_FINE)


def draw_truth_region(ax):
    x, y, w, h = TRUTH_BOX
    macro_region(ax, TRUTH_BOX, tint=TRUTH_TINT, edge=lighten(_PALETTE_RED, 0.58))
    lock(ax, x + 0.014, y + h - 0.031, scale=0.9)
    text(ax, x + 0.034, y + h - 0.023, spaced("TRAINING & EVALUATION ONLY"),
         size=6.6, color=TRUTH, ha="left", weight="bold")

    item_w, gap = 0.142, 0.036
    start_x = x + 0.038
    cy = y + h - 0.112
    for i, label in enumerate(("True outcomes", "True cost", "Regret labels",
                               "Offline evaluation")):
        ix = start_x + i * (item_w + gap)
        if i == 0:
            slab(ax, ix, cy - 0.027, item_w, 0.054,
                 front=TRUTH, top=TRUTH_TOP, side=TRUTH_SIDE, zorder=5)
            text(ax, ix + item_w / 2, cy, label, size=7.2, color=WHITE,
                 weight="bold", zorder=9)
        else:
            rounded(ax, ix, cy - 0.027, item_w, 0.054, fc=WHITE, ec=TRUTH,
                    lw=0.85, radius=0.006, zorder=3)
            text(ax, ix + item_w / 2, cy, label, size=7.2, color=TRUTH,
                 zorder=9)
        if i < 3:
            arrow(ax, ix + item_w + 0.008, cy, ix + item_w + gap - 0.008, cy,
                  color=TRUTH, lw=0.85, ms=5.5)

    text(ax, x + 0.038, y + 0.026,
         "Sealed: used only to build training labels and for offline evaluation.",
         size=6.8, color=MUTED, ha="left")


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def build(stem: str):
    # Reuse the supplied scene asset without accessing experiment data.
    if not SCENE_IMG.exists():
        raise FileNotFoundError(f"Required figure asset is missing: {SCENE_IMG}")
    tracks = []

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(AX_Y0, AX_Y1)
    ax.axis("off")

    macro_region(ax, INFER_BOX, tint=INFER_TINT, edge=lighten(_PALETTE_BLUE_LIGHT, 0.40))
    text(ax, INFER_BOX[0] + 0.014, INFER_BOX[1] + INFER_BOX[3] - 0.024,
         spaced("INPUTS & AUDIT SCORING"), size=6.6, color=MUTED, ha="left")
    draw_truth_region(ax)
    draw_evaluation_chip(ax)

    inner_left, gap = 0.026, 0.011
    avail = 0.723 - inner_left - 2 * gap
    weights = (1.34, 1.0, 1.0)
    widths = [avail * v / sum(weights) for v in weights]
    xs, cursor = [], inner_left
    for wd in widths:
        xs.append(cursor)
        cursor += wd + gap

    for xx, wd, label in zip(xs, widths,
                             ("Twin event", "Prediction", "Risk audit")):
        stage_box(ax, xx, wd, label)

    draw_stage_i(ax, xs[0], widths[0], tracks)
    draw_stage_ii(ax, xs[1], widths[1])
    draw_stage_iii(ax, xs[2], widths[2])

    # Training-only supervision bypasses the inference feature path.
    train_x = xs[2] + widths[2] * 0.10
    arrow(ax, 0.477, 0.207, train_x, 0.340, color=TRUTH, lw=0.85,
          linestyle=DASH_FINE)
    arrow(ax, train_x, 0.340, train_x, 0.579, color=TRUTH, lw=0.85,
          linestyle=DASH_FINE)
    text(ax, train_x + 0.011, 0.371, "training only", size=6.4,
         color=TRUTH, ha="left")

    col_x, col_w = 0.752, 0.236
    stage_box(ax, col_x, col_w, "Retention")
    draw_stage_iv(ax, col_x, col_w)

    for i in range(2):
        arrow(ax, xs[i] + widths[i] + 0.0015, 0.560,
              xs[i + 1] - 0.0015, 0.560, color=MUTED, lw=0.95, ms=6)
    arrow(ax, xs[2] + widths[2] + 0.0015, 0.560, col_x - 0.0015, 0.560,
          color=MUTED, lw=0.95, ms=6)

    fig.savefig(OUT / f"{stem}.svg", facecolor=WHITE)
    fig.savefig(OUT / f"{stem}.png", dpi=600, facecolor=WHITE)
    fig.savefig(OUT / f"{stem}.pdf", dpi=600, facecolor=WHITE,
                metadata={"Title": "AC-Twin decision-risk auditing"})
    plt.close(fig)
    print(f"wrote {stem}.svg/.png/.pdf  ({FIG_W/MM:.0f} x {FIG_H/MM:.0f} mm); "
          "prefix example: supplied figure asset")


if __name__ == "__main__":
    build("Fig2_AC_Twin_method_overview_v4")
