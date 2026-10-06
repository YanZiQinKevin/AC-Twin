"""Build Fig. 1: the response-conditioned structure of factual-evidence value.

Read the existing figure-folder CSV exports without changing their values.
Rounded thresholds label schematic bins; bin widths do not encode response scale.

Run from the repository root:
    python paper/figures/fig1/build_fig1.py
Outputs (inside paper/figures/fig1/):
    Fig1_evidence_value_structure.svg / .png / .pdf

Layout: 180 x 96 mm, i.e. 100% of a two-column page width. Panel (a) defines the
four response bins, (b) gives the raw per-bin losses, (c) is the hero panel of
the event-specific value with its frozen intervals, and (d) the prospectively specified
contrasts.
"""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
MM = 1 / 25.4
FIG_W, FIG_H = 180 * MM, 104 * MM

# Same six hues as Fig. 2 (paper/figures/fig2/build_fig2.py); each hue keeps one
# meaning across the figure set.
BLUE = "#4D93D6"
BLUE_LIGHT = "#A4C8EA"
RED = "#D57873"
PEACH = "#F1C2AD"
GREY = "#A8A8A9"
GREY_LIGHT = "#D9D9D9"
INK = "#2B2B2B"
MUTED = "#8C8C8E"

BIN_KEYS = ("zero", "positive_low", "positive_mid", "positive_high")
BIN_LABELS = ("zero", "low", "mid", "high")
CONTRASTS = (
    ("c_presence", "zero vs all\npositive response", 0.0),
    ("c_high_low", "high vs low", 0.0),
    ("c_high_mid", "high vs mid", 0.0),
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.6,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def lighten(c: str, t: float) -> str:
    a = [int(c[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (255 - x) * t):02X}" for x in a)


def darken(c: str, t: float) -> str:
    a = [int(c[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x * (1 - t)):02X}" for x in a)


def load():
    with (OUT / "fig1_bins_plot_data.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = {row['bin']: row for row in csv.DictReader(fh)}
    bins, value = [], {}
    for key, label in zip(BIN_KEYS, BIN_LABELS):
        row = rows[label]
        bins.append({name: int(row[name]) if name.startswith('n_') else float(row[name])
                     for name in row if name != 'bin'})
        value[key] = dict(point=float(row['v_evt']), ci_lower=float(row['v_evt_ci_lower']),
                          ci_upper=float(row['v_evt_ci_upper']))
    with (OUT / "fig1_contrasts_plot_data.csv").open(encoding="utf-8-sig", newline="") as fh:
        contrasts = {row['contrast']: {k: float(row[k]) for k in ('point','ci_lower','ci_upper')}
                     for row in csv.DictReader(fh)}
    stats = {'response_bins': {'t1': 0.2484, 't2': 1.8990}, 'point_screen': 0.05}
    return stats, bins, value, contrasts


def write_csv(bins, value, contrasts):
    with (OUT / "fig1_bins_plot_data.csv").open("w", newline="",
                                                encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["bin", "n_agent_time", "n_episodes", "mean_l_no",
                    "mean_l_suffix", "mean_l_shuffled", "v_evt",
                    "v_evt_ci_lower", "v_evt_ci_upper", "rho_no"])
        for key, label, b in zip(BIN_KEYS, BIN_LABELS, bins):
            v = value[key]
            w.writerow([label, b["n_agent_time"], b["n_episodes"],
                        b["mean_l_no"], b["mean_l_suffix"], b["mean_l_shuffled"],
                        v["point"], v["ci_lower"], v["ci_upper"], b["rho_no"]])
    with (OUT / "fig1_contrasts_plot_data.csv").open("w", newline="",
                                                     encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["contrast", "point", "ci_lower", "ci_upper", "n_episodes"])
        for c, _, _ in CONTRASTS:
            d = contrasts[c]
            w.writerow([c, d["point"], d["ci_lower"], d["ci_upper"],
                        d.get("n_episodes", 1200)])


def header(ax, letter, title):
    ax.text(0.0, 1.10, letter, transform=ax.transAxes, fontsize=9.6,
            fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0.56, 1.10, title, transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", color=INK, ha="center", va="bottom")


def draw_bins(ax, stats):
    """Panel (a): the frozen response bins, zero-inflated bin drawn separately."""
    thr = stats["response_bins"]
    t1, t2 = thr["t1"], thr["t2"]
    edge = np.array([0.0, 1.0, 2.0, 3.0])
    zero_w = 1.0
    tints = (lighten(BLUE_LIGHT, 0.78), lighten(BLUE_LIGHT, 0.48),
             lighten(BLUE_LIGHT, 0.10))
    ax.axvspan(-zero_w, edge[0], color=lighten(PEACH, 0.40), zorder=0)
    for i in range(3):
        ax.axvspan(edge[i], edge[i + 1], color=tints[i], zorder=0)
    ax.set_xlim(-zero_w, edge[-1])
    ax.set_ylim(0.0, 1.0)
    ax.axvline(edge[0], color=RED, lw=1.8, zorder=3)
    for x, label in ((edge[1], f"t1 = {t1:.4f}"), (edge[2], f"t2 = {t2:.4f}")):
        ax.axvline(x, color=darken(GREY, 0.15), lw=0.9, ls=(0, (2.4, 1.8)),
                   zorder=2)
        ax.text(x + 0.004, 0.88, label, fontsize=6.8, color=INK,
                ha="left", va="center")

    centres = [-zero_w / 2.0] + [0.5 * (edge[i] + edge[i + 1]) for i in range(3)]
    for centre, label, count, colour in zip(
            centres, BIN_LABELS, (1200, 788, 751, 513),
            (RED, INK, INK, INK)):
        ax.text(centre, 0.62, label, fontsize=8.2, color=colour,
                ha="center", va="center")
        ax.text(centre, 0.26, f"episodes: {count:,}", fontsize=6.8, color=colour,
                ha="center", va="center")
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def draw_losses(ax, bins):
    x = np.arange(4)
    series = (("mean_l_no", "no suffix encoding", lighten(GREY, 0.25), 1.1),
              ("mean_l_shuffled", "shuffled encoding", GREY, 1.1),
              ("mean_l_suffix", "same-event encoding", BLUE, 1.9))
    for key, label, colour, lw in series:
        y = [b[key] for b in bins]
        ax.plot(x, y, color=colour, lw=lw, marker="o", ms=4.4,
                markeredgecolor="white", markeredgewidth=0.5, label=label,
                zorder=5 if key == "mean_l_suffix" else 3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=7.4)
    ax.set_xlabel("response bin", fontsize=8.0)
    ax.set_ylabel("Mean position error (m)", fontsize=8.0)
    ax.tick_params(labelsize=7.4, length=2.4, pad=1.8)
    ax.grid(axis="y", color=lighten(GREY_LIGHT, 0.35), lw=0.55, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.legend(loc="upper left", fontsize=7.0, frameon=False, handlelength=1.6,
              borderpad=0.1, labelspacing=0.25)


def draw_value(ax, value):
    x = np.arange(4)
    point = np.array([value[k]["point"] for k in BIN_KEYS])
    lo = np.array([value[k]["ci_lower"] for k in BIN_KEYS])
    hi = np.array([value[k]["ci_upper"] for k in BIN_KEYS])
    ax.axhline(0.0, color=darken(GREY, 0.2), lw=0.9, zorder=2)
    ax.errorbar(x, point, yerr=[point - lo, hi - point], fmt="o", ms=6.0,
                color=BLUE, ecolor=darken(BLUE, 0.12), elinewidth=1.4,
                capsize=2.6, capthick=1.1, markeredgecolor="white",
                markeredgewidth=0.7, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=7.4)
    ax.set_xlim(-0.45, 3.45)
    ax.set_xlabel("response bin", fontsize=8.0)
    ax.set_ylabel("event-specific value  $V_{evt}$", fontsize=8.0)
    ax.tick_params(labelsize=7.4, length=2.4, pad=1.8)
    ax.grid(axis="y", color=lighten(GREY_LIGHT, 0.35), lw=0.55, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)


def draw_contrasts(ax, contrasts, screen):
    rows = list(reversed(CONTRASTS))
    for idx, (key, label, _) in enumerate(rows):
        d = contrasts[key]
        ax.plot([d["ci_lower"], d["ci_upper"]], [idx, idx],
                color=darken(BLUE, 0.05), lw=1.7, solid_capstyle="butt", zorder=3)
        for v in (d["ci_lower"], d["ci_upper"]):
            ax.plot([v, v], [idx - 0.12, idx + 0.12], color=darken(BLUE, 0.05),
                    lw=0.9, zorder=3)
        ax.plot([d["point"]], [idx], marker="o", ms=5.0, color=BLUE,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax.axvline(0.0, color=darken(GREY, 0.2), lw=0.9, zorder=2)
    ax.axvline(screen, color=RED, lw=0.9, ls=(0, (2.4, 1.8)), zorder=2)
    ax.text(screen, len(rows) - 0.45, f"  screen {screen:g}", fontsize=6.8,
            color=RED, ha="left", va="center")
    for idx, (_, label, _) in enumerate(rows):
        ax.text(0.006, idx + 0.20, label.replace("\n", " "), fontsize=7.2,
                color=INK, ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.25))
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0.0, 0.38)
    ax.set_xlabel("contrast  (95% CI)", fontsize=7.8)
    ax.tick_params(axis="x", labelsize=7.2, length=2.2, pad=1.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=lighten(GREY_LIGHT, 0.35), lw=0.55, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)


def main():
    stats, bins, value, contrasts = load()

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    ax_a = fig.add_axes([0.070, 0.665, 0.915, 0.185])
    draw_bins(ax_a, stats)
    header(ax_a, "(a)", "frozen response bins")
    # Panel (a) has no axis furniture, so its scale note goes into the white
    # clearance under the strip instead of on top of the tinted bins.
    fig.text(0.070, 0.626,
             "Schematic bins; zero <= 0.0001 m; episode counts overlap.",
             fontsize=6.6, color=MUTED, ha="left", va="center")

    panel_y, panel_h, panel_w, gap = 0.110, 0.395, 0.272, 0.048
    ax_b = fig.add_axes([0.072, panel_y, panel_w, panel_h])
    ax_c = fig.add_axes([0.072 + panel_w + gap, panel_y, panel_w, panel_h])
    ax_d = fig.add_axes([0.072 + 2 * (panel_w + gap), panel_y, panel_w, panel_h])
    draw_losses(ax_b, bins)
    draw_value(ax_c, value)
    draw_contrasts(ax_d, contrasts, stats.get("point_screen", 0.05))
    header(ax_b, "(b)", "raw per-bin losses")
    header(ax_c, "(c)", "evidence value")
    header(ax_d, "(d)", "specified contrasts")

    written, blocked = [], []
    for ext in ("svg", "png", "pdf"):
        target = OUT / f"Fig1_evidence_value_structure.{ext}"
        try:
            fig.savefig(target, dpi=600, facecolor="white")
            written.append(ext)
        except PermissionError:
            blocked.append(ext)
    plt.close(fig)
    print("wrote %s  (180 x 104 mm)" % "/".join(written))
    if blocked:
        print("WARNING: %s held open by another program; close it and re-run"
              % ", ".join(blocked))
    for key, b, v in zip(BIN_KEYS, bins, value.values()):
        print("  %-14s V_evt=%.6f [%.6f, %.6f]  n_ep=%d"
              % (key, v["point"], v["ci_lower"], v["ci_upper"], b["n_episodes"]))


if __name__ == "__main__":
    main()
