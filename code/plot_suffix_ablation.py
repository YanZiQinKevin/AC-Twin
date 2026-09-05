"""Create the publication figure for the factual-suffix ablation study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments" / "suffix_evidence_ablation"
OUTPUT_DIR = ROOT / "paper" / "figures" / "fig_suffix_evidence"
REPORT_PATH = EXPERIMENT_DIR / "report.json"
PREDICTION_PATH = EXPERIMENT_DIR / "test_predictions_and_errors.npz"
OUTPUT_STEM = OUTPUT_DIR / "Fig_suffix_evidence"

CONDITIONS = [
    "history_only",
    "shuffled_full",
    "last_frame",
    "prefix_1",
    "prefix_5",
    "prefix_10",
    "full_20",
    "residual_innovation_20",
]

DISPLAY_LABELS = {
    "history_only": "History only",
    "shuffled_full": "Shuffled suffix",
    "last_frame": "Last frame",
    "prefix_1": "1-step suffix",
    "prefix_5": "5-step suffix",
    "prefix_10": "10-step suffix",
    "full_20": "20-step suffix",
    "residual_innovation_20": "Residual proxy",
}

COLORS = {
    "history_only": "#767676",
    "shuffled_full": "#9A4D8E",
    "last_frame": "#A9A9A9",
    "prefix_1": "#8FB7D9",
    "prefix_5": "#3775BA",
    "prefix_10": "#0F4D92",
    "full_20": "#42949E",
    "residual_innovation_20": "#B64342",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def export_source_data(report: dict, predictions: np.lib.npyio.NpzFile) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "condition_seed_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "display_label",
                "training_seed",
                "effect_ADE_at_20_m",
                "effect_FDE_at_20_m",
            ],
        )
        writer.writeheader()
        for condition in CONDITIONS:
            for row in report["training_runs"][condition]:
                writer.writerow(
                    {
                        "condition": condition,
                        "display_label": DISPLAY_LABELS[condition],
                        "training_seed": row["seed"],
                        "effect_ADE_at_20_m": row["effect_ADE_at_20"],
                        "effect_FDE_at_20_m": row["effect_FDE_at_20"],
                    }
                )

    shuffled = predictions["episode_fde_shuffled_full"].astype(float)
    factual = predictions["episode_fde_full_20"].astype(float)
    with (OUTPUT_DIR / "paired_episode_fde.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "test_episode_index",
                "shuffled_suffix_FDE_m",
                "same_episode_suffix_FDE_m",
                "paired_difference_m",
            ],
        )
        writer.writeheader()
        for episode_index, reference, candidate in zip(
            predictions["test_indices"], shuffled, factual
        ):
            writer.writerow(
                {
                    "test_episode_index": int(episode_index),
                    "shuffled_suffix_FDE_m": reference,
                    "same_episode_suffix_FDE_m": candidate,
                    "paired_difference_m": reference - candidate,
                }
            )


def make_figure(report: dict, predictions: np.lib.npyio.NpzFile) -> plt.Figure:
    apply_style()
    fig = plt.figure(figsize=(7.20, 3.55))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.18, 0.92, 1.0],
        left=0.145,
        right=0.985,
        bottom=0.19,
        top=0.91,
        wspace=0.50,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    # a | Across-condition comparison. Dots are training seeds; bars are mean ± s.d.
    y_positions = np.arange(len(CONDITIONS))[::-1]
    fixed_jitter = np.linspace(-0.07, 0.07, 5)
    for y, condition in zip(y_positions, CONDITIONS):
        values = np.asarray(
            [run["effect_FDE_at_20"] for run in report["training_runs"][condition]],
            dtype=float,
        )
        jitter = fixed_jitter[: values.size]
        color = COLORS[condition]
        ax_a.scatter(
            values,
            y + jitter,
            s=12,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        ax_a.errorbar(
            values.mean(),
            y,
            xerr=values.std(ddof=0),
            fmt="D",
            ms=3.2,
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=2.4,
            capthick=0.9,
            zorder=4,
        )
    ax_a.axvline(
        report["condition_metrics"]["history_only"]["effect_FDE_at_20"],
        color="#767676",
        linestyle="--",
        linewidth=0.8,
        alpha=0.65,
    )
    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([DISPLAY_LABELS[c] for c in CONDITIONS])
    ax_a.set_xlim(1.9, 4.65)
    ax_a.set_xlabel("Effect-FDE at horizon 20 (m)\n← lower is better")
    ax_a.grid(axis="x", color="#E2E2E2", linewidth=0.55, zorder=0)
    ax_a.set_axisbelow(True)
    add_panel_label(ax_a, "a")

    # b | Evidence-length response.
    length_conditions = ["history_only", "prefix_1", "prefix_5", "prefix_10", "full_20"]
    lengths = np.asarray([0, 1, 5, 10, 20])
    length_positions = np.arange(lengths.size)
    seed_values = np.asarray(
        [
            [run["effect_FDE_at_20"] for run in report["training_runs"][condition]]
            for condition in length_conditions
        ],
        dtype=float,
    )
    means = seed_values.mean(axis=1)
    sds = seed_values.std(axis=1, ddof=0)
    ax_b.fill_between(
        length_positions,
        means - sds,
        means + sds,
        color="#0F4D92",
        alpha=0.14,
        linewidth=0,
    )
    ax_b.plot(
        length_positions,
        means,
        color="#0F4D92",
        marker="o",
        markersize=4.0,
        linewidth=1.6,
        zorder=3,
    )
    for x, values in zip(length_positions, seed_values):
        ax_b.scatter(
            np.full(values.size, x),
            values,
            s=9,
            facecolor="white",
            edgecolor="#3775BA",
            linewidth=0.55,
            alpha=0.85,
            zorder=2,
        )
    ax_b.axvspan(2.75, 4.25, color="#DDEAF4", alpha=0.45, linewidth=0)
    ax_b.text(3.5, 3.43, "plateau", color="#0F4D92", ha="center", va="center", fontsize=6.5)
    ax_b.set_xticks(length_positions)
    ax_b.set_xticklabels(lengths)
    ax_b.set_xlim(-0.3, 4.3)
    ax_b.set_ylim(2.0, 3.6)
    ax_b.set_xlabel("Observed factual-suffix length (steps)")
    ax_b.set_ylabel("Effect-FDE at horizon 20 (m)")
    ax_b.grid(axis="y", color="#E2E2E2", linewidth=0.55)
    ax_b.set_axisbelow(True)
    add_panel_label(ax_b, "b")

    # c | Paired episode-level difference for the confirmatory contrast.
    shuffled = predictions["episode_fde_shuffled_full"].astype(float)
    factual = predictions["episode_fde_full_20"].astype(float)
    paired_difference = shuffled - factual
    primary = report["primary_analysis"]
    bins = np.linspace(paired_difference.min(), paired_difference.max(), 28)
    ax_c.hist(
        paired_difference,
        bins=bins,
        color="#86AEC7",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
    )
    ax_c.axvline(0, color="#4D4D4D", linestyle="--", linewidth=0.9)
    ax_c.axvspan(
        primary["ci_lower"],
        primary["ci_upper"],
        color="#0F4D92",
        alpha=0.16,
        linewidth=0,
    )
    ax_c.axvline(
        primary["mean_paired_difference_reference_minus_candidate"],
        color="#0F4D92",
        linewidth=1.6,
    )
    ax_c.text(
        0.98,
        0.96,
        "mean = 1.13 m\n95% CI 0.86–1.39\nn = 300 episodes",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#272727",
    )
    ax_c.text(
        0.97,
        0.08,
        "same-episode suffix better →",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#0F4D92",
    )
    ax_c.set_xlabel("Paired effect-FDE difference (m)")
    ax_c.set_ylabel("Test episodes")
    ax_c.grid(axis="y", color="#E2E2E2", linewidth=0.55)
    ax_c.set_axisbelow(True)
    add_panel_label(ax_c, "c")

    return fig


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    with np.load(PREDICTION_PATH) as predictions:
        export_source_data(report, predictions)
        fig = make_figure(report, predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"Saved figure and source data to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
