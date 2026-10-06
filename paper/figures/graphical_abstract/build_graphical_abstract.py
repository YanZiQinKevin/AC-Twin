"""Build the unnumbered AC-Twin graphical abstract (SVG + 600 dpi PNG + PDF).

This is the submission graphical abstract retained outside the numbered main
figures in visual-narrative blueprint v1.0. It states that, inside one retrospective
counterfactual task, the average prediction error misses two things a decision
needs -- the response-conditioned value of the factual evidence and whether the
selected action carries high regret -- and that AC-Twin measures and audits the
two separately.

Layout (180 x 84 mm, i.e. 100% of a two-column page width, asymmetric):

    (a) Twin event           28%   observed prefix -> two worlds
    (b) Evidence-value lens  34%   three prediction variants + frozen V_evt
    (c) Decision audit       38%   ten branches -> risk head -> retention
    bottom band                     the three measured evidence scopes

Deterministic and read-only with respect to every frozen experiment: the only
measured quantity in the figure is the height of the four V_evt dots in panel
(b); it is read from the frozen Stage-2 confirmatory report and asserted against
the per-bin `V_evt` field before anything is written. The risk-coverage glyph in
panel (c) is illustrative only and is labelled as such in the figure legend.

Run from the repository root:
    python paper/figures/graphical_abstract/build_graphical_abstract.py
Outputs (inside paper/figures/graphical_abstract/):
    GA_AC_Twin_overview.svg / .png / .pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
REPORT = ROOT / "experiments/stage2_confirmatory/merge/report.json"

MM = 1 / 25.4

# --------------------------------------------------------------------------
# canvas: every coordinate in this file is millimetres, so the drawing is
# isotropic (1 data unit = 1 mm in x and y at the delivered size).
# --------------------------------------------------------------------------
FIG_W_MM, FIG_H_MM = 180.0, 84.0
FIG_W, FIG_H = FIG_W_MM * MM, FIG_H_MM * MM

MARGIN = 3.0
ROW_Y0, ROW_Y1 = 20.0, 81.0          # the A/B/C main row
BAND_Y0, BAND_Y1 = 4.5, 13.5         # the scope band

# width split 28 / 34 / 38 over the drawable width, with a narrow A-B gap and a
# wider B-C gap that carries the rotated "motivates" phrase.
A_X0, A_X1 = 3.00, 48.36
B_X0, B_X1 = 52.36, 107.44
C_X0, C_X1 = 115.44, 177.00


# --------------------------------------------------------------------------
# palette: the same six hues as Fig. 1 / Fig. 2 / Fig. 3; one hue, one meaning
# --------------------------------------------------------------------------
def _mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def lighten(c: str, t: float) -> str:
    return _mix(c, "#FFFFFF", t)


def darken(c: str, t: float) -> str:
    return _mix(c, "#000000", t)


BLUE = "#4D93D6"          # AC-Twin method / audit
BLUE_LIGHT = "#A4C8EA"    # method tints
RED = "#D57873"           # counterfactual branches and the truth family
PEACH = "#F1C2AD"         # warm tints
GREY = "#A8A8A9"          # frozen predictor / neutral ink
GREY_LIGHT = "#D9D9D9"    # frames and neutral bars
INK = "#2B2B2B"
MUTED = "#8C8C8E"
WHITE = "#FFFFFF"

FRAME = GREY_LIGHT
FAINT = lighten(GREY_LIGHT, 0.62)
INFER_TINT = lighten(BLUE_LIGHT, 0.66)     # panel (c) hero backing
NEUTRAL_TINT = lighten(GREY_LIGHT, 0.66)   # panel (b) backing
BAND_TINT = lighten(BLUE_LIGHT, 0.74)      # scope band chips (one tint only)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def rounded(ax, x, y, w, h, *, fc=WHITE, ec=FRAME, lw=0.8, radius=1.2,
            linestyle="-", zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=linestyle,
        clip_on=False, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def slab(ax, x, y, w, h, *, front, top, side, depth=2.0, zorder=4):
    """Extruded module block: front face plus top and right faces."""
    ax.add_patch(Polygon(
        [(x, y + h), (x + depth, y + h + depth),
         (x + w + depth, y + h + depth), (x + w, y + h)],
        closed=True, facecolor=top, edgecolor="none", clip_on=False,
        zorder=zorder - 1))
    ax.add_patch(Polygon(
        [(x + w, y), (x + w + depth, y + depth),
         (x + w + depth, y + h + depth), (x + w, y + h)],
        closed=True, facecolor=side, edgecolor="none", clip_on=False,
        zorder=zorder - 1))
    rounded(ax, x, y, w, h, fc=front, ec="none", lw=0.0, radius=0.9,
            zorder=zorder)


def arrow(ax, x0, y0, x1, y1, *, color=MUTED, lw=0.9, ms=5.5, zorder=6,
          linestyle="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, linestyle=linestyle, shrinkA=0, shrinkB=0,
        clip_on=False, zorder=zorder))


def text(ax, x, y, s, *, size=8.0, color=INK, weight="normal", ha="center",
         va="center", zorder=9, rotation=0.0):
    return ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
                   fontweight=weight, linespacing=1.18, rotation=rotation,
                   rotation_mode="anchor", clip_on=False, zorder=zorder,
                   fontfamily="DejaVu Sans")


def dot(ax, cx, cy, d, *, fc, ec="none", lw=0.0, zorder=7):
    ax.add_patch(Ellipse((cx, cy), d, d, facecolor=fc, edgecolor=ec,
                         linewidth=lw, clip_on=False, zorder=zorder))


def bezier(p0, p1, p2, n=60):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p = ((1 - t) ** 2) * np.array(p0) + 2 * (1 - t) * t * np.array(p1) \
        + (t ** 2) * np.array(p2)
    return p[:, 0], p[:, 1]


def header(ax, x, letter, title):
    text(ax, x, 79.0, f"({letter})", size=9.6, weight="bold", ha="left")
    text(ax, x + 6.6, 79.0, title, size=9.2, weight="bold", ha="left")


# --------------------------------------------------------------------------
# frozen input: the only measured quantity in this figure
# --------------------------------------------------------------------------
BIN_ORDER = (("zero", "zero"), ("low", "positive_low"),
             ("mid", "positive_mid"), ("high", "positive_high"))


def load_v_evt():
    """Four frozen V_evt point estimates, asserted against the report twice."""
    stats = json.loads(REPORT.read_text(encoding="utf-8"))["statistics"]
    out = []
    for label, key in BIN_ORDER:
        point = float(stats[f"v_{label}"]["point"])
        assert abs(float(stats["bins"][key]["V_evt"]) - point) < 1e-12, label
        out.append(point)
    assert stats["n_episodes"] == 1200
    return out


# --------------------------------------------------------------------------
# panel (a) Twin event
# --------------------------------------------------------------------------
def draw_panel_a(ax):
    header(ax, A_X0, "a", "Twin event")

    sx0, sx1 = A_X0 + 1.5, A_X0 + 23.5
    sy0, sy1 = 40.0, 62.0
    rounded(ax, sx0, sy0, sx1 - sx0, sy1 - sy0, fc=FAINT, ec=GREY_LIGHT,
            lw=0.7, radius=1.4, zorder=1)

    hist_x = [sx0 + 2.2, sx0 + 6.4, sx0 + 10.6, sx0 + 15.0, sx0 + 20.4]
    hist_y = [46.6, 49.4, 47.6, 51.2, 50.2]
    ax.plot(hist_x, hist_y, color=INK, lw=1.1, solid_capstyle="round",
            zorder=4)
    for i, (px, py) in enumerate(zip(hist_x, hist_y)):
        if i in (1, 2, 3):
            dot(ax, px, py, 1.7, fc=WHITE, ec=INK, lw=0.85, zorder=5)
    dot(ax, hist_x[-1], hist_y[-1], 2.3, fc=INK, zorder=6)

    text(ax, 0.5 * (sx0 + sx1), 36.4, "Observed prefix", size=8.0)

    fork = (sx1, hist_y[-1])
    dot(ax, fork[0], fork[1], 2.6, fc=WHITE, ec=INK, lw=1.0, zorder=6)

    up_x, up_y = bezier(fork, (36.0, 52.5), (44.0, 67.0))
    dn_x, dn_y = bezier(fork, (36.0, 48.0), (44.0, 35.0))
    ax.plot(up_x, up_y, color=INK, lw=1.2, solid_capstyle="round", zorder=4)
    ax.plot(dn_x, dn_y, color=RED, lw=1.2, solid_capstyle="round", zorder=4)
    dot(ax, up_x[-1], up_y[-1], 1.9, fc=INK, zorder=5)
    dot(ax, dn_x[-1], dn_y[-1], 1.9, fc=RED, zorder=5)

    text(ax, 31.5, 70.4, "Factual continuation", size=7.5)
    text(ax, 31.5, 30.8, "Alternative ego action", size=7.5, color=RED)


# --------------------------------------------------------------------------
# panel (b) Evidence-value lens
# --------------------------------------------------------------------------
def draw_panel_b(ax, v_evt):
    rounded(ax, B_X0 - 1.5, ROW_Y0 - 1.5, (B_X1 - B_X0) + 3.0,
            (ROW_Y1 - ROW_Y0) + 3.0, fc=NEUTRAL_TINT, ec="none", radius=2.0,
            zorder=0)
    header(ax, B_X0 + 1.5, "b", "Evidence-value lens")

    cards = (("history-\nonly", GREY, False),
             ("correct\nsuffix", BLUE, True),
             ("shuffled\nsuffix", GREY, False))
    inner0, inner1 = B_X0 + 2.0, B_X1 - 2.0
    gap = 1.6
    cw = ((inner1 - inner0) - 2 * gap) / 3.0
    cy0, cy1 = 63.0, 74.5
    for i, (label, accent, hero) in enumerate(cards):
        cx0 = inner0 + i * (cw + gap)
        cx = cx0 + 0.5 * cw
        rounded(ax, cx0, cy0, cw, cy1 - cy0,
                fc=INFER_TINT if hero else WHITE,
                ec=accent if hero else GREY_LIGHT,
                lw=1.1 if hero else 0.8, radius=1.2, zorder=2)
        gy = cy0 + 8.6
        rel_x = (-5.0, -1.6, 1.8, 5.0)
        rel_y = (-1.2, 0.6, 0.2, 2.0)
        ax.plot([cx + dx for dx in rel_x], [gy + dy for dy in rel_y],
                color=accent, lw=1.3 if hero else 1.0, solid_capstyle="round",
                zorder=4)
        dot(ax, cx + rel_x[-1], gy + rel_y[-1], 1.6, fc=accent, zorder=5)
        text(ax, cx, cy0 + 3.4, label, size=7.5,
             color=INK if hero else MUTED, weight="bold" if hero else "normal")

    # frozen V_evt dots: height only, no numerals in the figure body
    vmin, vmax = -0.12, 0.40
    y_lo, y_hi = 26.0, 56.0

    def vy(v):
        return y_lo + (v - vmin) / (vmax - vmin) * (y_hi - y_lo)

    xs = [63.0, 75.0, 87.0, 99.0]
    ys = [vy(v) for v in v_evt]
    ax.plot([60.5, 105.5], [vy(0.0)] * 2, color=GREY, lw=0.8,
            linestyle=(0, (2.6, 2.0)), zorder=2)
    ax.plot(xs, ys, color=BLUE_LIGHT, lw=1.0, zorder=3)
    for x, y in zip(xs, ys):
        dot(ax, x, y, 2.7, fc=BLUE, ec=WHITE, lw=0.8, zorder=6)
    for x, label in zip(xs, ("zero", "low", "mid", "high")):
        text(ax, x, 23.2, label, size=7.5, color=MUTED)
    text(ax, 57.4, 41.0, "$V_{evt}$", size=8.0, rotation=90)


# --------------------------------------------------------------------------
# panel (c) Decision audit
# --------------------------------------------------------------------------
def draw_panel_c(ax):
    rounded(ax, C_X0 - 1.5, ROW_Y0 - 1.5, (C_X1 - C_X0) + 1.5,
            (ROW_Y1 - ROW_Y0) + 3.0, fc=INFER_TINT, ec="none", radius=2.0,
            zorder=0)
    header(ax, C_X0 + 1.5, "c", "Decision audit")
    text(ax, C_X1, 76.4, "ten candidate actions", size=7.5, color=MUTED,
         ha="right")

    node = (135.0, 67.5)
    ticks_x = np.linspace(120.0, 150.0, 10)
    for i, tx in enumerate(ticks_x):
        sel = (i == 5)
        ax.plot([tx, tx], [73.0, 75.2],
                color=BLUE if sel else GREY, lw=1.7 if sel else 0.9,
                solid_capstyle="round", zorder=4)
        ax.plot([tx, node[0]], [73.0, node[1]],
                color=BLUE if sel else GREY_LIGHT,
                lw=1.3 if sel else 0.7, solid_capstyle="round", zorder=3)
    dot(ax, node[0], node[1], 2.4, fc=BLUE, zorder=6)
    text(ax, node[0], 64.6, "selected action", size=8.0)

    arrow(ax, node[0], 62.6, node[0], 59.4)
    slab(ax, 122.0, 53.6, 26.0, 5.0, front=BLUE, top=lighten(BLUE, 0.38),
         side=darken(BLUE, 0.24), depth=1.6)
    text(ax, 135.0, 56.1, "Direct risk head", size=8.0, color=WHITE,
         weight="bold", zorder=10)

    arrow(ax, node[0], 51.8, node[0], 48.6)
    ax.add_patch(Arc((135.0, 44.2), 6.6, 6.6, theta1=0.0, theta2=180.0,
                     color=BLUE, lw=1.5, zorder=5))
    ax.plot([135.0, 136.9], [44.2, 46.6], color=INK, lw=1.2,
            solid_capstyle="round", zorder=6)
    dot(ax, 135.0, 44.2, 1.5, fc=INK, zorder=7)
    text(ax, 135.0, 40.4, "audit risk score", size=8.0)

    arrow(ax, node[0], 38.4, node[0], 35.6)
    bar_x0, bar_x1 = 120.0, 150.0
    split = bar_x0 + 0.8 * (bar_x1 - bar_x0)
    rounded(ax, bar_x0, 28.8, bar_x1 - bar_x0, 5.6, fc=GREY_LIGHT, ec="none",
            radius=0.8, zorder=3)
    rounded(ax, bar_x0, 28.8, split - bar_x0, 5.6, fc=BLUE, ec="none",
            radius=0.8, zorder=4)
    ax.plot([split, split], [28.4, 34.8], color=WHITE, lw=1.0,
            linestyle=(0, (2.0, 1.6)), zorder=5)
    text(ax, 0.5 * (bar_x0 + split), 31.6, "retained", size=7.5, color=WHITE,
         weight="bold", zorder=6)
    text(ax, 0.5 * (split + bar_x1), 26.2, "flagged", size=7.5, color=MUTED)

    # illustrative risk-coverage glyph (schematic; the legend says so)
    gx0, gx1, gy0 = 155.0, 176.0, 30.0
    ax.plot([gx0, gx1], [gy0, gy0], color=GREY_LIGHT, lw=0.8, zorder=2)
    ax.plot([gx0, gx0], [gy0, 64.0], color=GREY_LIGHT, lw=0.8, zorder=2)
    t = np.linspace(0.0, 1.0, 60)
    curve_y = gy0 + 30.0 * (1.0 - (1.0 - t) ** 2.0)
    ax.plot(gx0 + t * (gx1 - gx0), curve_y, color=BLUE, lw=1.4,
            solid_capstyle="round", zorder=4)
    k = int(0.8 * (len(t) - 1))
    ax.plot([gx0 + t[k] * (gx1 - gx0)] * 2, [gy0, curve_y[k]], color=GREY,
            lw=0.8, linestyle=(0, (2.4, 1.8)), zorder=3)
    dot(ax, gx0 + t[k] * (gx1 - gx0), curve_y[k], 2.0, fc=BLUE, ec=WHITE,
        lw=0.7, zorder=5)
    text(ax, 0.5 * (gx0 + gx1), 24.6, "risk\u2013coverage\nevaluation",
         size=7.5, color=MUTED)


# --------------------------------------------------------------------------
# bottom band: the three measured scopes (names only, one tint)
# --------------------------------------------------------------------------
def draw_scope_band(ax):
    text(ax, MARGIN, 16.6, "Measured evidence scopes", size=7.5, color=MUTED,
         ha="left")
    names = ("Merge\nin distribution",
             "Preregistered\ngenerator mismatch",
             "MetaDrive / Waymo\nplatform-internal retraining")
    gap = 8.0
    cw = (177.0 - MARGIN - 2 * gap) / 3.0
    for i, name in enumerate(names):
        x = MARGIN + i * (cw + gap)
        rounded(ax, x, BAND_Y0, cw, BAND_Y1 - BAND_Y0, fc=BAND_TINT,
                ec=GREY_LIGHT, lw=0.8, radius=1.4, zorder=2)
        text(ax, x + 0.5 * cw, 0.5 * (BAND_Y0 + BAND_Y1), name, size=8.0)
        if i < 2:
            arrow(ax, x + cw + 2.0, 0.5 * (BAND_Y0 + BAND_Y1),
                  x + cw + gap - 2.0, 0.5 * (BAND_Y0 + BAND_Y1), ms=5.0)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def main():
    v_evt = load_v_evt()

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, FIG_W_MM)
    ax.set_ylim(0.0, FIG_H_MM)
    ax.axis("off")

    draw_panel_a(ax)
    draw_panel_b(ax, v_evt)
    draw_panel_c(ax)
    draw_scope_band(ax)

    mid_y = 0.5 * (ROW_Y0 + ROW_Y1)
    arrow(ax, A_X1 + 0.6, mid_y, B_X0 - 0.6, mid_y, ms=5.5)
    text(ax, 0.5 * (B_X1 + C_X0), 56.0,
         "Motivates a decision-level audit", size=7.5, color=MUTED,
         rotation=90.0)

    stem = "GA_AC_Twin_overview"
    written, blocked = [], []
    for ext in ("svg", "png", "pdf"):
        target = OUT / f"{stem}.{ext}"
        try:
            if ext == "png":
                fig.savefig(target, dpi=600, facecolor=WHITE)
            elif ext == "pdf":
                fig.savefig(target, dpi=600, facecolor=WHITE,
                            metadata={"Title": "AC-Twin overview"})
            else:
                fig.savefig(target, facecolor=WHITE)
            written.append(ext)
        except PermissionError:
            blocked.append(ext)
    plt.close(fig)

    print("wrote %s/%s  (%.0f x %.0f mm)"
          % (stem, "/".join(written), FIG_W_MM, FIG_H_MM))
    print("  V_evt dots (frozen): " + "  ".join("%.4f" % v for v in v_evt))
    if blocked:
        print("WARNING: %s held open by another program; close it and re-run"
              % ", ".join(blocked))


if __name__ == "__main__":
    main()
