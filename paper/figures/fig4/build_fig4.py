"""Build Fig. 4: the credibility closure of the confirmed Merge audit effect.

Read-only with respect to every frozen experiment. The script loads the frozen
Stage 4C-R2 confirmation tables and the Stage 4D D1 profile-selection record,
re-draws four robustness panels descriptively, and asserts before writing that
the relative improvements it draws equal the adjudicated report values.

Panels (blueprint v1.0 section 4, Fig. 4):
    (a) Baseline ladder -- the audit against every evaluated standard baseline,
        with the two non-adjudicated reference scores faded to point estimates.
    (b) Five-seed paired effects of the audit head.
    (c) Strata and trim -- direction stability only, not independent
        confirmation.
    (d) Prospectively specified mismatch qualification -- which generator-mismatch
        profile was eligible, and which one was selected by frozen priority.
        The confirmed mismatch effect itself is carried by Fig. 3b and Table 6,
        not by this panel.

Run from the repository root:
    python paper/figures/fig4/build_fig4.py
Outputs (inside paper/figures/fig4/):
    Fig4_credibility_closure.svg / .png / .pdf
    fig4_panel_a_data.csv ... fig4_panel_d_data.csv

Layout: 180 x 126 mm, i.e. 100% of a two-column page width; 2 x 2 panels.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MM = 1 / 25.4
FIG_W, FIG_H = 180 * MM, 126 * MM

R2_REPORTS = ROOT / "experiments/stage4_reliability/stage4cr2_confirm/reports"
OOD_DIR = ROOT / "experiments/stage4_reliability/stage4d_ood"
BASELINE_TABLE = R2_REPORTS / "Table_C2_baselines.csv"
SEED_TABLE = R2_REPORTS / "Table_C3_seeds_strata_trim.csv"
R2_REPORT = R2_REPORTS / "stage4cr2_confirmatory_report.json"
OOD_SELECTION = OOD_DIR / "pilot/ood_profile_selection.json"
OOD_REPORT = OOD_DIR / "reports/stage4d_ood_confirmatory_report.json"

# Same six hues as Fig. 1 / Fig. 2 / Fig. 3; each hue keeps one meaning.
BLUE = "#4D93D6"
BLUE_LIGHT = "#A4C8EA"
RED = "#D57873"
PEACH = "#F1C2AD"
GREY = "#A8A8A9"
GREY_LIGHT = "#D9D9D9"
INK = "#2B2B2B"
MUTED = "#8C8C8E"


def _mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def lighten(c: str, t: float) -> str:
    return _mix(c, "#FFFFFF", t)


def darken(c: str, t: float) -> str:
    return _mix(c, "#000000", t)


FAINT = lighten(GREY_LIGHT, 0.55)
BAR = darken(BLUE, 0.05)
BAR_SOFT = lighten(BLUE, 0.30)

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
        "axes.unicode_minus": False,
    }
)

# Panel geometry, in figure coordinates. Margins: the left gutter of (a)/(c) and
# the right edge of the (d) header both keep clear canvas, and the two (b)
# sub-axes are exactly the same width so one data unit has one length in both.
AX_A = [0.178, 0.618, 0.397, 0.290]
AX_B1 = [0.700, 0.618, 0.128, 0.290]
AX_B2 = [0.855, 0.618, 0.128, 0.290]
AX_C = [0.178, 0.100, 0.397, 0.300]
AX_D1 = [0.615, 0.310, 0.370, 0.090]
AX_D2 = [0.800, 0.100, 0.185, 0.190]
HEAD_Y_TOP, SUB_Y_TOP = 0.948, 0.921
HEAD_Y_BOT, SUB_Y_BOT1, SUB_Y_BOT2 = 0.478, 0.450, 0.426

# Panel (a) row order: the three standard baselines the audit was adjudicated
# against, then the two reference scores that carry no adjudicated interval.
A_MAIN = ("MARGIN_ONLY", "SEED_VOTE_ENTROPY", "SEED_MARGIN_STD")
A_REFERENCE = ("COMBINED_TOP2_MAX", "CONSTANT_RANDOM")
METRIC_LABEL = {"aurc": "AURC-regret", "regret80": "Regret@80"}
# The two non-adjudicated rows are named by the role the frozen table gives
# them, so a reader can see why they carry no interval.
REFERENCE_LABEL = {"historical_reference": "historical reference",
                   "curve_reference": "curve reference"}
# Frozen Table_C2 names the two intervals inconsistently -- the AURC columns use
# `ci95_*` while the Regret@80 upper column uses `ci_*` -- so the columns are
# mapped explicitly rather than built from a format string.
CI_COLS = {
    "aurc": ("i_aurc_point", "i_aurc_ci95_lower", "i_aurc_ci95_upper"),
    "regret80": ("i_regret80_point", "i_regret80_ci95_lower",
                 "i_regret80_ci_upper"),
}
XTICKS = (0.0, 0.2, 0.4)


# --------------------------------------------------------------------------
# frozen inputs
# --------------------------------------------------------------------------
def _read_csv(path):
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_baseline_ladder():
    """Panel (a): every evaluated score, with the audit's relative improvement.

    Every drawn interval is asserted against the adjudicated record. The two
    reference scores have no adjudicated interval, so only their point estimate
    is drawn, using the same construction the assertions validate.
    """
    rows = {r["name"]: r for r in _read_csv(BASELINE_TABLE)}
    direct = rows["DIRECT_ENSEMBLE"]
    model = {"aurc": float(direct["aurc_regret"]),
             "regret80": float(direct["regret_at_80"])}

    out = []
    for name in A_MAIN + A_REFERENCE:
        r = rows[name]
        absolute = {"aurc": float(r["aurc_regret"]),
                    "regret80": float(r["regret_at_80"])}
        adjudicated = name in A_MAIN
        item = {"name": name, "adjudicated": adjudicated, "role": r["role"],
                "primary": r["role"] == "primary_standard_baseline"}
        for metric in ("aurc", "regret80"):
            derived = 1.0 - model[metric] / absolute[metric]
            if adjudicated:
                pcol, lcol, hcol = CI_COLS[metric]
                point, lo, hi = float(r[pcol]), float(r[lcol]), float(r[hcol])
                assert abs(derived - point) < 1e-12, (name, metric)
                item[metric] = (point, lo, hi)
            else:
                item[metric] = (derived, None, None)
        out.append(item)
    return out


def load_seeds():
    seeds = {}
    for r in _read_csv(SEED_TABLE):
        if r["kind"] in ("aurc", "regret80") and r["name"].startswith("seed_"):
            seeds.setdefault(r["name"], {})[r["kind"]] = (
                float(r["point"]), float(r["ci95_lower"]), float(r["ci95_upper"]))
    assert len(seeds) == 5, len(seeds)
    return seeds


def load_strata_trim():
    out = []
    for r in _read_csv(SEED_TABLE):
        if r["kind"] in ("stratum", "trim"):
            out.append((r["name"], r["kind"], float(r["point"]),
                        float(r["ci95_lower"]), float(r["ci95_upper"])))
    assert len(out) == 6, len(out)
    return out


def load_ood():
    selection = json.loads(OOD_SELECTION.read_text(encoding="utf-8"))
    report = json.loads(OOD_REPORT.read_text(encoding="utf-8"))
    shift = report["decision"]["domain_shift"]
    profiles = [{"name": name,
                 "qualified": bool(selection["qualified"][name]),
                 "priority": idx + 1,
                 "selected": name == selection["selected_profile"]}
                for idx, name in enumerate(selection["priority"])]
    assert sum(p["selected"] for p in profiles) == 1
    smd = sorted(((k, abs(v)) for k, v in shift["abs_smd"].items()),
                 key=lambda kv: kv[1])
    return {
        "profiles": profiles,
        "smd": smd,
        "smd_min": float(shift["smd_min"]),
        "n_shifted": int(shift["n_shifted"]),
        "auc": (float(shift["domain_classifier"]["auc_point"]),
                float(shift["domain_classifier"]["auc_ci_lower"]),
                float(shift["domain_classifier"]["auc_ci_upper"])),
        "n_id": int(shift["domain_classifier"]["n_id"]),
        "n_ood": int(shift["domain_classifier"]["n_ood"]),
    }


def load_ensemble_effect():
    m = json.loads(R2_REPORT.read_text(encoding="utf-8"))["decision"]["metrics"]
    return {"aurc": (m["point"]["i_aurc_regret"], m["ci_lower"]["i_aurc_regret"],
                     m["ci_upper"]["i_aurc_regret"]),
            "regret80": (m["point"]["i_regret80"], m["ci_lower"]["i_regret80"],
                         m["ci_upper"]["i_regret80"])}


# --------------------------------------------------------------------------
# drawing primitives
# --------------------------------------------------------------------------
def ci_row(ax, y, point, lo, hi, *, color, marker="o", ms=4.4, lw=1.6,
           filled=True, zorder=4):
    if lo is not None:
        ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt",
                zorder=zorder - 1)
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.13, y + 0.13], color=color, lw=lw * 0.55,
                    zorder=zorder - 1)
    ax.plot([point], [y], marker=marker, ms=ms,
            color=color if filled else "white", markeredgecolor=color,
            markeredgewidth=0.9, zorder=zorder)


def forest_frame(ax, n_rows, xmax, *, xlabel=None, labels=None, tick_size=7.4):
    ax.axvline(0.0, color=GREY, lw=0.9, zorder=2)
    for i in range(1, n_rows):
        ax.axhline(i - 0.5, color=FAINT, lw=0.55, zorder=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels if labels else [""] * n_rows, fontsize=tick_size)
    ax.set_ylim(-0.62, n_rows - 0.38)
    ax.set_xlim(0.0, xmax)
    ax.set_xticks(XTICKS)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.2, labelpad=2.0)
    ax.tick_params(axis="x", labelsize=7.0, length=2.2, pad=1.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=FAINT, lw=0.55, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)


def panel_text(fig, x, y, s, *, size=8.8, weight="normal", color=INK,
               ha="center"):
    fig.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha,
             va="center")


def mid_x(box):
    """Centre of an [x, y, w, h] axes box -- the width is box[2], not the edge."""
    return box[0] + 0.5 * box[2]


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
def draw_baseline_ladder(ax, ladder):
    n = len(ladder)
    labels = [""] * n
    for idx, item in enumerate(ladder):
        y = n - 1 - idx                     # rows are drawn top-down
        name = item["name"]
        if item["primary"]:
            name += "\n(frozen primary)"
        elif not item["adjudicated"]:
            name += "\n(%s)" % REFERENCE_LABEL[item["role"]]
        labels[y] = name                    # ... so labels are indexed by row
        for metric, dy, marker, filled in (("aurc", +0.17, "o", True),
                                           ("regret80", -0.17, "s", False)):
            point, lo, hi = item[metric]
            if item["adjudicated"]:
                ci_row(ax, y + dy, point, lo, hi,
                       color=BAR if metric == "aurc" else BAR_SOFT,
                       marker=marker, filled=filled, ms=4.4, lw=1.6)
            else:
                ci_row(ax, y + dy, point, None, None, color=GREY, marker="D",
                       filled=False, ms=3.4)
    ax.axhline(1.5, color=GREY_LIGHT, lw=0.8, ls=(0, (2.4, 1.8)), zorder=1)
    forest_frame(ax, n, 0.70, labels=labels, tick_size=7.2,
                 xlabel="relative improvement over each baseline "
                 "(95% CI for adjudicated rows)")
    # The reading aid sits in the empty band above the longest row, clear of
    # both the legend (upper left) and the reference diamonds (lower right).
    ax.text(0.99, 1.005, "longer bar = weaker baseline",
            transform=ax.transAxes, fontsize=7.0, color=MUTED, ha="right",
            va="top", clip_on=False)
    # Two entries only: the method name is already in the panel title, and the
    # narrow legend then fits the free strip left of the shortest bar, so it
    # occludes no data row.
    handles = [
        plt.Line2D([], [], color=BAR, marker="o", ms=4.4, lw=1.6,
                   label="AURC-regret"),
        plt.Line2D([], [], color=BAR_SOFT, marker="s", ms=4.4, lw=1.6,
                   markerfacecolor="white", markeredgewidth=0.9,
                   label="Regret@80"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.0, frameon=False,
              handlelength=1.3, handletextpad=0.6, borderpad=0.1,
              labelspacing=0.26)


def draw_seeds(ax_l, ax_r, seeds, ensemble):
    names = list(seeds)
    n = len(names)
    labels = [""] * n
    for idx, name in enumerate(names):
        labels[n - 1 - idx] = name.replace("seed_", "seed ")
    for ax, metric, first in ((ax_l, "aurc", True), (ax_r, "regret80", False)):
        for idx, name in enumerate(names):
            point, lo, hi = seeds[name][metric]
            assert lo > 0.0, (name, metric)   # every seed keeps the sign
            ci_row(ax, n - 1 - idx, point, lo, hi, color=BAR, lw=1.5)
        ep = ensemble[metric][0]
        ax.axvline(ep, color=RED, lw=0.9, ls=(0, (2.4, 1.8)), zorder=3)
        forest_frame(ax, n, 0.58, labels=labels if first else None,
                     tick_size=7.0)
        if not first:
            ax.set_yticklabels([])
        ax.text(ep - 0.006, -0.50, "ensemble", fontsize=7.0, color=RED,
                ha="right", va="bottom", zorder=6)


def draw_strata_trim(ax, rows):
    n = len(rows)
    labels = [""] * n
    for idx, (name, kind, point, lo, hi) in enumerate(rows):
        y = n - 1 - idx
        labels[y] = (name.replace("_i_aurc", "").replace("tstar", "t*")
                     .replace("interaction", "min distance")
                     .replace("trim1", "trim 1%").replace("trim5", "trim 5%")
                     .replace("_", " "))
        ci_row(ax, y, point, lo, hi, color=BAR, lw=1.5)
    ax.axhline(1.5, color=GREY_LIGHT, lw=0.8, ls=(0, (2.4, 1.8)), zorder=1)
    forest_frame(ax, n, 0.62, labels=labels, tick_size=7.4,
                 xlabel="relative improvement in AURC-regret (95% CI)")


def draw_ood(ax_matrix, ax_smd, ood):
    ax_matrix.set_xlim(0.0, 1.0)
    ax_matrix.set_ylim(0.0, 1.0)
    ax_matrix.axis("off")
    cols = (0.46, 0.68, 0.88)
    for cx, head in zip(cols, ("eligible", "priority", "selected")):
        ax_matrix.text(cx, 0.90, head, fontsize=7.0, color=MUTED,
                       ha="center", va="center")
    for idx, prof in enumerate(ood["profiles"]):
        y = 0.63 - idx * 0.26
        ax_matrix.text(0.16, y, prof["name"].replace("_OOD", ""), fontsize=7.4,
                       color=INK, ha="center", va="center",
                       fontweight="bold" if prof["selected"] else "normal")
        # "no" is muted italics, not red: red is reserved for the two reference
        # lines in (b)/(d) and must not also mean "this profile failed"
        ax_matrix.text(cols[0], y, "yes" if prof["qualified"] else "no",
                       fontsize=7.4, color=INK if prof["qualified"] else MUTED,
                       fontstyle="normal" if prof["qualified"] else "italic",
                       ha="center", va="center")
        ax_matrix.text(cols[1], y, "%d" % prof["priority"], fontsize=7.4,
                       color=MUTED, ha="center", va="center")
        if prof["selected"]:
            # a square, not a circle: the SMD dots below are circles, and the
            # two must not be read as the same quantity
            ax_matrix.plot([cols[2]], [y], marker="s", ms=5.0, color=BLUE,
                           markeredgecolor="white", markeredgewidth=0.7)

    feats = [f for f, _ in ood["smd"]]
    vals = [v for _, v in ood["smd"]]
    ys = list(range(len(feats)))[::-1]
    ax_smd.axvline(ood["smd_min"], color=RED, lw=0.9, ls=(0, (2.4, 1.8)),
                   zorder=2)
    for y, v in zip(ys, vals):
        shifted = v >= ood["smd_min"]
        ax_smd.plot([0.0, v], [y, y], color=FAINT, lw=0.9, zorder=1)
        ax_smd.plot([v], [y], marker="o", ms=4.0,
                    color=BAR if shifted else GREY, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=4)
    ax_smd.set_yticks(ys)
    ax_smd.set_yticklabels(feats, fontsize=7.0)
    ax_smd.set_ylim(-0.62, len(feats) - 0.38)
    ax_smd.set_xlim(0.0, 2.5)
    ax_smd.set_xlabel("absolute SMD", fontsize=7.2, labelpad=2.0)
    ax_smd.tick_params(axis="x", labelsize=7.0, length=2.2, pad=1.6)
    ax_smd.tick_params(axis="y", length=0)
    ax_smd.grid(axis="x", color=FAINT, lw=0.55, zorder=0)
    for side in ("top", "right", "left"):
        ax_smd.spines[side].set_visible(False)
    ax_smd.spines["bottom"].set_color(GREY)
    ax_smd.text(ood["smd_min"], len(feats) - 0.42,
                "  threshold %.1f" % ood["smd_min"], fontsize=7.0, color=RED,
                ha="left", va="center")


# --------------------------------------------------------------------------
# exports
# --------------------------------------------------------------------------
def write_csvs(ladder, seeds, strata, ood):
    with (OUT / "fig4_panel_a_data.csv").open("w", newline="",
                                              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["baseline", "role", "metric", "point", "ci_lower", "ci_upper"])
        for item in ladder:
            role = ("frozen_primary" if item["primary"] else
                    "standard_baseline" if item["adjudicated"] else "reference")
            for m in ("aurc", "regret80"):
                p, lo, hi = item[m]
                w.writerow([item["name"], role, METRIC_LABEL[m], p,
                            "" if lo is None else lo, "" if hi is None else hi])
    with (OUT / "fig4_panel_b_data.csv").open("w", newline="",
                                              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "metric", "point", "ci_lower", "ci_upper"])
        for name, vals in seeds.items():
            for m in ("aurc", "regret80"):
                w.writerow([name, METRIC_LABEL[m], *vals[m]])
    with (OUT / "fig4_panel_c_data.csv").open("w", newline="",
                                              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "kind", "point", "ci_lower", "ci_upper"])
        for name, kind, p, lo, hi in strata:
            w.writerow([name, kind, p, lo, hi])
    with (OUT / "fig4_panel_d_data.csv").open("w", newline="",
                                              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["profile", "eligible", "priority", "selected"])
        for prof in ood["profiles"]:
            w.writerow([prof["name"], prof["qualified"], prof["priority"],
                        prof["selected"]])
        w.writerow([])
        w.writerow(["feature", "absolute_smd", "threshold", "shifted"])
        for feat, val in ood["smd"]:
            w.writerow([feat, val, ood["smd_min"], val >= ood["smd_min"]])


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def main():
    ladder = load_baseline_ladder()
    seeds = load_seeds()
    strata = load_strata_trim()
    ood = load_ood()
    ensemble = load_ensemble_effect()
    write_csvs(ladder, seeds, strata, ood)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    ax_a = fig.add_axes(AX_A)
    ax_b1 = fig.add_axes(AX_B1)
    ax_b2 = fig.add_axes(AX_B2)
    ax_c = fig.add_axes(AX_C)
    ax_d1 = fig.add_axes(AX_D1)
    ax_d2 = fig.add_axes(AX_D2)

    draw_baseline_ladder(ax_a, ladder)
    draw_seeds(ax_b1, ax_b2, seeds, ensemble)
    draw_strata_trim(ax_c, strata)
    draw_ood(ax_d1, ax_d2, ood)

    panel_text(fig, 0.062, HEAD_Y_TOP, "(a)", size=9.6, weight="bold", ha="left")
    panel_text(fig, mid_x(AX_A), HEAD_Y_TOP, "AC-Twin audit: baseline ladder")
    panel_text(fig, AX_B1[0] - 0.082, HEAD_Y_TOP, "(b)", size=9.6,
               weight="bold", ha="left")
    panel_text(fig, 0.5 * (AX_B1[0] + AX_B2[0] + AX_B2[2]), HEAD_Y_TOP,
               "Five-seed paired effects")
    panel_text(fig, mid_x(AX_B1), SUB_Y_TOP, METRIC_LABEL["aurc"], size=7.4)
    panel_text(fig, mid_x(AX_B2), SUB_Y_TOP, METRIC_LABEL["regret80"], size=7.4)

    panel_text(fig, 0.062, HEAD_Y_BOT, "(c)", size=9.6, weight="bold", ha="left")
    panel_text(fig, mid_x(AX_C), HEAD_Y_BOT, "Strata and trim")
    panel_text(fig, mid_x(AX_C), SUB_Y_BOT1,
               "direction stability only, not independent confirmation",
               size=7.0, color=MUTED)
    panel_text(fig, AX_D1[0] - 0.082, HEAD_Y_BOT, "(d)", size=9.6,
               weight="bold", ha="left")
    panel_text(fig, mid_x(AX_D1), HEAD_Y_BOT,
               "Specified mismatch qualification")
    auc, lo, hi = ood["auc"]
    panel_text(fig, mid_x(AX_D1), SUB_Y_BOT1,
               "%d of %d SMDs shifted" % (ood["n_shifted"], len(ood["smd"])),
               size=7.0, color=MUTED)
    panel_text(fig, mid_x(AX_D1), SUB_Y_BOT2,
               "domain AUC %.3f [%.3f, %.3f]" % (auc, lo, hi),
               size=7.0, color=MUTED)
    # (b) shares one x-label, centred across both sub-axes
    panel_text(fig, 0.5 * (AX_B1[0] + AX_B2[0] + AX_B2[2]), 0.576,
               "relative improvement (95% CI)", size=7.2)

    stem = "Fig4_credibility_closure"
    written, blocked = [], []
    for ext in ("svg", "png", "pdf"):
        target = OUT / f"{stem}.{ext}"
        try:
            if ext == "png":
                fig.savefig(target, dpi=600, facecolor="white")
            elif ext == "pdf":
                fig.savefig(target, dpi=600, facecolor="white",
                            metadata={"Title": "AC-Twin credibility closure"})
            else:
                fig.savefig(target, facecolor="white")
            written.append(ext)
        except PermissionError:
            blocked.append(ext)
    plt.close(fig)

    print("wrote %s/%s  (180 x 126 mm)" % (stem, "/".join(written)))
    print("  panel a rows=%d | panel b seeds=%d | panel c rows=%d | panel d "
          "profiles=%d" % (len(ladder), len(seeds), len(strata),
                           len(ood["profiles"])))
    if blocked:
        print("WARNING: %s held open by another program; close it and re-run"
              % ", ".join(blocked))


if __name__ == "__main__":
    main()
