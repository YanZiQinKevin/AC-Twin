"""Build Fig. 3: the three measured scopes of decision-risk auditing.

Read-only with respect to every frozen experiment: the script loads the frozen
per-group quantities and the frozen confirmation reports, re-draws the
risk-coverage curves descriptively, and asserts that the plotted 80% coverage
value of the primary baseline equals the adjudicated report value before
writing anything.

Both frozen plans require that every standard baseline be reported: the
validation-frozen primary baseline is emphasised, and the other two standard
baselines are drawn alongside it.

Run from the repository root:
    python paper/figures/fig3/build_fig3.py
Outputs (inside paper/figures/fig3/):
    Fig3_three_scope_decision_risk.svg / .png / .pdf
    A_merge_plot_data.csv, B_joint_ood_plot_data.csv, C_metadrive_plot_data.csv

Layout: 180 x 118 mm, i.e. 100% of a two-column page width. Panels (a)-(c) are
the three risk-coverage panels; panel (d) is the two-facet effect strip.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MM = 1 / 25.4
FIG_W, FIG_H = 180 * MM, 118 * MM

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

# Baseline styling by role: all baselines are solid lines, so the curves stay
# readable when they overlap. The scope's frozen primary baseline is the darkest
# and heaviest; the other two standard baselines step down in lightness.
PRIMARY_STYLE = dict(color="#5C5C5F", ls="-", lw=1.9)
SECONDARY_STYLES = [
    dict(color="#96969B", ls="-", lw=1.15),
    dict(color="#C4C4C8", ls="-", lw=1.15),
]

STANDARD_BASELINES = ("MARGIN_ONLY", "SEED_MARGIN_STD", "SEED_VOTE_ENTROPY")


def _mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def lighten(c: str, t: float) -> str:
    return _mix(c, "#FFFFFF", t)


def darken(c: str, t: float) -> str:
    return _mix(c, "#000000", t)


# stem, canonical scope name, model relation, npz, report, metric path, primary
SPECS = [
    ("A_merge", "Merge ID", "in-distribution\nsame frozen Merge weights",
     "stage4cr2_confirm/reports/confirm_group_quantities.npz",
     "stage4cr2_confirm/reports/stage4cr2_confirmatory_report.json",
     ("decision", "metrics"), "SEED_VOTE_ENTROPY"),
    ("B_joint_ood", "JOINT_OOD", "prospectively specified mismatch\nsame frozen Merge weights",
     "stage4d_ood/reports/d1_group_quantities.npz",
     "stage4d_ood/reports/stage4d_ood_confirmatory_report.json",
     ("decision", "metrics"), "SEED_VOTE_ENTROPY"),
    ("C_metadrive", "MetaDrive", "new platform\nplatform-internal retraining",
     "stage4d_d2f/confirm/confirm_group_quantities.npz",
     "stage4d_d2f/confirm/confirm_report.json",
     ("metrics",), "SEED_MARGIN_STD"),
]

REPORT_REL = {
    "A_merge": "stage4cr2_confirm/reports/stage4cr2_confirmatory_report.json",
    "B_joint_ood": "stage4d_ood/reports/stage4d_ood_confirmatory_report.json",
    "C_metadrive": "stage4d_d2f/confirm/confirm_report.json",
}
METRIC_PATH = {
    "A_merge": ("decision", "metrics"),
    "B_joint_ood": ("decision", "metrics"),
    "C_metadrive": ("metrics",),
}
PANEL_LETTER = {"A_merge": "(a)", "B_joint_ood": "(b)", "C_metadrive": "(c)"}

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


def dig(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def curve(score: np.ndarray, regret: np.ndarray) -> np.ndarray:
    order = np.argsort(score, kind="stable")
    return np.cumsum(regret[order], dtype=np.float64) / np.arange(1, len(regret) + 1)


def load_curves():
    """Per-scope curves for the audit and every standard baseline, with checks."""
    loaded = {}
    for stem, name, relation, npz_rel, report_rel, mpath, primary in SPECS:
        base = ROOT / "experiments/stage4_reliability"
        with np.load(base / npz_rel, allow_pickle=False) as data:
            regret = np.asarray(data["regret"], dtype=np.float64)
            direct = curve(np.asarray(data["ensemble_risk"], dtype=np.float64), regret)
            others = {b: curve(np.asarray(data[f"baseline_risk__{b}"], dtype=np.float64),
                               regret)
                      for b in STANDARD_BASELINES}
        report = json.loads((base / report_rel).read_text(encoding="utf-8"))
        metrics = dig(report, mpath)
        k80 = int(0.8 * len(regret))
        assert len(regret) == 2000, stem
        assert np.isclose(direct[k80 - 1], metrics["point"]["regret80_model"], atol=1e-12)
        assert np.isclose(others[primary][k80 - 1],
                          metrics["point"]["regret80_baseline"], atol=1e-12)
        columns = ["coverage", "direct_mean_regret"] + [f"baseline__{b}"
                                                        for b in STANDARD_BASELINES]
        with (OUT / f"{stem}_plot_data.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerows(
                [k / len(regret), float(d)] + [float(others[b][k - 1])
                                               for b in STANDARD_BASELINES]
                for k, d in enumerate(direct, 1))
        loaded[stem] = dict(name=name, relation=relation, primary=primary,
                            x=np.arange(1, len(regret) + 1) / len(regret),
                            direct=direct, others=others, metrics=metrics,
                            n=len(regret))
        print("%-13s n=%d  regret@80 direct=%.12f primary(%s)=%.12f  report match"
              % (stem, len(regret), direct[k80 - 1], primary, others[primary][k80 - 1]))
    return loaded


def load_effects():
    """Relative improvements and 95% CIs, read from the frozen reports."""
    out = {}
    for stem in ("A_merge", "B_joint_ood", "C_metadrive"):
        report = json.loads(
            (ROOT / "experiments/stage4_reliability" / REPORT_REL[stem])
            .read_text(encoding="utf-8"))
        m = dig(report, METRIC_PATH[stem])
        out[stem] = {
            "aurc": (m["point"]["i_aurc_regret"],
                     m["ci_lower"]["i_aurc_regret"], m["ci_upper"]["i_aurc_regret"]),
            "r80": (m["point"]["i_regret80"],
                    m["ci_lower"]["i_regret80"], m["ci_upper"]["i_regret80"]),
        }
    return out


def draw_curve_panel(ax, item, first: bool):
    secondary = [b for b in STANDARD_BASELINES if b != item["primary"]]
    handles = {}
    for style, baseline in zip(SECONDARY_STYLES, secondary):
        handles[baseline], = ax.plot(item["x"], item["others"][baseline],
                                     zorder=3, label=baseline, **style)
    handles["primary"], = ax.plot(item["x"], item["others"][item["primary"]],
                                  zorder=4,
                                  label=f"{item['primary']}\n(frozen primary)",
                                  **PRIMARY_STYLE)
    handles["direct"], = ax.plot(item["x"], item["direct"], color=BLUE, lw=2.2,
                                 zorder=5, label="AC-Twin audit")

    ax.axvline(0.8, color=darken(GREY, 0.15), ls=(0, (2.4, 1.8)), lw=0.9, zorder=2)
    ax.set_xlim(0.0, 1.0)
    top = float(max(item["direct"].max(),
                    *(c.max() for c in item["others"].values()))) * 1.20
    ax.set_ylim(0.0, top)
    ax.set_xlabel("Coverage", fontsize=8.0)
    if first:
        ax.set_ylabel("Mean retained regret", fontsize=8.0)
    ax.tick_params(labelsize=7.4, length=2.4, pad=1.8)
    ax.grid(axis="y", color=lighten(GREY_LIGHT, 0.35), lw=0.55, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.text(0.8, top * 0.985, " 80%", fontsize=7.0, color=MUTED,
            ha="left", va="top")
    order = [handles["direct"], handles["primary"]] + \
            [handles[b] for b in secondary]
    ax.legend(handles=order, loc="upper left", fontsize=6.6, frameon=False,
              handlelength=1.6, borderpad=0.1, labelspacing=0.30)


def draw_effect_panel(ax, effects, metric: str, title: str, order,
                      row_labels: bool = True):
    rows = list(reversed(order))
    for idx, stem in enumerate(rows):
        point, lo, hi = effects[stem][metric]
        ax.plot([lo, hi], [idx, idx], color=darken(BLUE, 0.05), lw=1.7,
                solid_capstyle="butt", zorder=3)
        for x in (lo, hi):
            ax.plot([x, x], [idx - 0.12, idx + 0.12], color=darken(BLUE, 0.05),
                    lw=0.9, zorder=3)
        ax.plot([point], [idx], marker="o", ms=4.8, color=BLUE,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax.axvline(0.0, color=GREY, lw=0.9, zorder=2)
    for idx in range(1, len(rows)):
        ax.axhline(idx - 0.5, color=lighten(GREY_LIGHT, 0.45), lw=0.55, zorder=0)
    ax.set_yticks(range(len(rows)))
    if row_labels:
        ax.set_yticklabels([effects[s]["name"] for s in rows], fontsize=7.6)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0.0, 0.56)
    ax.set_xlabel("relative improvement (95% CI)", fontsize=7.4)
    ax.tick_params(axis="x", labelsize=7.2, length=2.2, pad=1.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=lighten(GREY_LIGHT, 0.35), lw=0.55, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.set_title(title, fontsize=8.6, color=INK, pad=6.0)


def main():
    data = load_curves()
    effects = load_effects()
    for stem, name, *_ in SPECS:
        effects[stem]["name"] = name

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    panel_w, gap, panel_y, panel_h = 0.270, 0.050, 0.545, 0.320
    for i, stem in enumerate(("A_merge", "B_joint_ood", "C_metadrive")):
        ax = fig.add_axes([0.075 + i * (panel_w + gap), panel_y, panel_w, panel_h])
        item = data[stem]
        draw_curve_panel(ax, item, first=(i == 0))
        ax.text(0.0, 1.245, PANEL_LETTER[stem], transform=ax.transAxes,
                fontsize=9.6, fontweight="bold", color=INK,
                ha="left", va="center")
        ax.text(0.5, 1.245, item["name"], transform=ax.transAxes,
                fontsize=9.6, fontweight="bold", color=INK,
                ha="center", va="center")
        ax.text(0.5, 1.075, item["relation"], transform=ax.transAxes,
                fontsize=7.6, color=MUTED, ha="center", va="center",
                linespacing=1.25)

    strip_y, strip_h = 0.125, 0.290
    order = ("A_merge", "B_joint_ood", "C_metadrive")
    ax_a = fig.add_axes([0.115, strip_y, 0.375, strip_h])
    ax_b = fig.add_axes([0.555, strip_y, 0.375, strip_h])
    draw_effect_panel(ax_a, effects, "aurc", "AURC-regret", order)
    draw_effect_panel(ax_b, effects, "r80", "Regret@80", order,
                      row_labels=False)
    fig.text(0.078, strip_y + strip_h + 0.008, "(d)", fontsize=9.6,
             fontweight="bold", color=INK, ha="left", va="bottom")

    written, blocked = [], []
    for ext in ("svg", "png", "pdf"):
        target = OUT / f"Fig3_three_scope_decision_risk.{ext}"
        try:
            fig.savefig(target, dpi=600, facecolor="white")
            written.append(ext)
        except PermissionError:
            blocked.append(ext)
    plt.close(fig)
    print("wrote %s  (180 x 118 mm)" % "/".join(written))
    if blocked:
        print("WARNING: %s still held open by another program; close it and "
              "re-run to refresh that format" % ", ".join(blocked))


if __name__ == "__main__":
    main()
