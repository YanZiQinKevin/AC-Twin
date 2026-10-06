"""Build the single manuscript comparison table and the reference ledger.

Only persisted Stage 5 development summaries and the engineering-only timing
receipt are read.  No scientific metric is recomputed and no frozen artifact is
modified.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments" / "stage5_sota_comparison"
HERE = Path(__file__).resolve().parent
REF_DIR = ROOT / "paper" / "references"
TIMING_PATH = HERE / "stage5_inference_timing_receipt.json"

METHOD_LABELS = {
    "CONFIDNET_TCP_ADAPTED": "ConfidNet (adapted)",
    "SELECTIVENET_REGRET_ADAPTED": "SelectiveNet (adapted)",
    "RWR_NN_KNN_REJECT_ADAPTED": "RwR-kNNRej (adapted)",
    "CONFORMAL_REGRESSION_REJECT_ADAPTED": "Conformal regression + reject (adapted)",
    "CCTR_DRIVING_CALIBRATION_ADAPTED": "CCTR (adapted)",
}
INTERNAL_ORDER = ("MARGIN_ONLY", "SEED_MARGIN_STD", "SEED_VOTE_ENTROPY")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_rows() -> list[dict[str, Any]]:
    registry = _json(STAGE / "literature" / "method_registry.json")
    selection = _json(STAGE / "development" / "final_method_selection.json")
    report = _json(STAGE / "development" / "validation_comparison_report.json")
    timing = _json(TIMING_PATH)
    if timing.get("scope") != "ENGINEERING_TIMING_ONLY_NO_SCIENTIFIC_RESULTS":
        raise ValueError("timing receipt has the wrong scope")

    works = {row.get("method_id"): row for row in registry["works"]
             if row.get("method_id")}
    candidates = report["candidate_metrics"]
    rows: list[dict[str, Any]] = []

    published = sorted(
        selection["selected"].items(),
        key=lambda item: (int(works[item[0]]["year"]), METHOD_LABELS[item[0]]),
    )
    for method_id, selected in published:
        metrics = candidates[f"{method_id}#{selected['config_key']}"]
        work = works[method_id]
        latency = timing["benchmark"][method_id]
        rows.append({
            "category": "published",
            "method": METHOD_LABELS[method_id],
            "ref_key": work["work_id"],
            "venue": work["venue"],
            "aurc_regret": float(metrics["aurc_regret"]),
            "regret_at_80": float(metrics["regret_at_80"]),
            "pr_auc_minus_prevalence": float(metrics["pr_auc_minus_prevalence"]),
            "spearman": float(metrics["spearman_risk_regret"]),
            "param_count_per_seed": int(selected["param_count"]),
            "approx_forward_flops_per_group": 2 * int(selected["param_count"]),
            "inference_ms_per_group": float(latency["median_ms_per_group"]),
            "inference_iqr": [float(latency["q1_ms_per_group"]),
                              float(latency["q3_ms_per_group"])],
        })

    baselines = report["internal_baseline_metrics"]
    for name in INTERNAL_ORDER:
        metrics = baselines[name]
        rows.append({
            "category": "internal",
            "method": name,
            "ref_key": "",
            "venue": "—",
            "aurc_regret": float(metrics["aurc_regret"]),
            "regret_at_80": float(metrics["regret_at_80"]),
            "pr_auc_minus_prevalence": float(metrics["pr_auc_minus_prevalence"]),
            "spearman": float(metrics["spearman_risk_regret"]),
            "param_count_per_seed": None,
            "approx_forward_flops_per_group": None,
            "inference_ms_per_group": None,
            "inference_iqr": None,
        })

    metrics = report["actwin_metrics"]
    latency = timing["benchmark"]["AC-Twin Direct"]
    rows.append({
        "category": "ours",
        "method": "AC-Twin audit",
        "ref_key": "",
        "venue": "This work",
        "aurc_regret": float(metrics["aurc_regret"]),
        "regret_at_80": float(metrics["regret_at_80"]),
        "pr_auc_minus_prevalence": float(metrics["pr_auc_minus_prevalence"]),
        "spearman": float(metrics["spearman_risk_regret"]),
        "param_count_per_seed": int(latency["param_count_per_seed"]),
        "approx_forward_flops_per_group": 2 * int(latency["param_count_per_seed"]),
        "inference_ms_per_group": float(latency["median_ms_per_group"]),
        "inference_iqr": [float(latency["q1_ms_per_group"]),
                          float(latency["q3_ms_per_group"])],
    })
    return rows


def build_reference_rows() -> list[dict[str, str]]:
    registry = _json(STAGE / "literature" / "method_registry.json")
    access_date = str(registry["access_date"])
    rows: list[dict[str, str]] = []
    for work in registry["works"]:
        in_experiment = bool(work.get("in_experiment", False))
        status = str(work["status"])
        if in_experiment:
            used_in = "Comparison table; Related Work"
            experimental_role = "Adapted experimental comparator"
        elif status.startswith("SUPPLEMENTAL"):
            used_in = "Related Work; supplemental positioning"
            experimental_role = "Supplemental candidate, not main comparison"
        elif status.startswith("REGISTERED_ALTERNATE"):
            used_in = "Related Work; method audit"
            experimental_role = "Registered alternate, not activated"
        else:
            used_in = "Related Work / scope positioning"
            experimental_role = "Not an experimental comparator"
        rows.append({
            "ref_key": str(work["work_id"]),
            "manuscript_no": "",
            "authors": str(work["authors"]),
            "year": str(work["year"]),
            "title": str(work["title"]),
            "venue": str(work["venue"]),
            "persistent_id": str(work.get("doi_or_arxiv") or ""),
            "code_url": str(work.get("code_url") or ""),
            "used_in": used_in,
            "experimental_role": experimental_role,
            "verification": status,
            "access_date": access_date,
        })
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _write_comparison(rows: list[dict[str, Any]]) -> None:
    csv_path = HERE / "table_stage5_comparison_main.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    best = {
        "aurc_regret": min(row["aurc_regret"] for row in rows),
        "regret_at_80": min(row["regret_at_80"] for row in rows),
        "pr_auc_minus_prevalence": max(row["pr_auc_minus_prevalence"] for row in rows),
        "spearman": max(row["spearman"] for row in rows),
    }

    def display(row: dict[str, Any], key: str, digits: int = 4) -> str:
        text = _fmt(row[key], digits)
        if abs(float(row[key]) - float(best[key])) <= 1e-12:
            return f"**{text}**"
        return text

    lines = [
        "# Table III. Comparison with adapted published methods on the shared development benchmark",
        "",
        "> **Scope:** `ADAPTIVE_DEVELOPMENT_ONLY`; Stage 5 terminal state "
        "`STAGE5_MERGE_COMPARISON_UNDERPOWERED`. The comparison holdout was never "
        "created. The table supports a transparent development benchmark, not a "
        "confirmatory SOTA-superiority claim.",
        "",
        "| Method | Venue | AURC-regret ↓ | Regret@80 ↓ | PR-AUC − prevalence ↑ | Spearman ↑ | Params/seed | Approx. forward FLOPs/group (MAC×2) ↓ | Inference (ms/group) ↓ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    previous = None
    for row in rows:
        if previous is not None and row["category"] != previous:
            lines.append("|  |  |  |  |  |  |  |  |  |")
        label = row["method"]
        if row["ref_key"]:
            label += f" [`{row['ref_key']}`]"
        if row["category"] == "ours":
            label = f"**{label}**"
        lines.append(
            f"| {label} | {row['venue']} | {display(row, 'aurc_regret')} | "
            f"{display(row, 'regret_at_80')} | "
            f"{display(row, 'pr_auc_minus_prevalence')} | "
            f"{display(row, 'spearman')} | "
            f"{row['param_count_per_seed'] if row['param_count_per_seed'] is not None else '—'} | "
            f"{row['approx_forward_flops_per_group'] if row['approx_forward_flops_per_group'] is not None else '—'} | "
            f"{_fmt(row['inference_ms_per_group'], 6)} |"
        )
        previous = row["category"]
    lines.extend([
        "",
        "All learned rows use five model seeds. Parameters are reported per seed. "
        "Approximate forward FLOPs are the per-seed simplified parameterized-forward "
        "estimate MAC×2 per group; they exclude the shared upstream predictor and method-specific "
        "non-parametric search or calibration operations. Inference time is the median "
        "single-group (batch size 1) five-seed ensemble "
        "risk-head time after model/input preload on an NVIDIA GeForce RTX 3070 Ti; "
        "host materialisation is included "
        "and the shared upstream predictor is excluded. Internal baselines are "
        "closed-form scores and are therefore not assigned trainable parameters or "
        "risk-head latency. Full grids, secondary metrics, calibration diagnostics, "
        "pairwise intervals and power results remain in `TABLE_STAGE5_SOTA.md` as "
        "audit material, not additional manuscript comparison tables.",
        "",
        "Reference keys map to `paper/references/REFERENCE_LEDGER.md`; manuscript "
        "numbers will be assigned only after the bibliography order is fixed.",
    ])
    (HERE / "TABLE_STAGE5_COMPARISON_MAIN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _write_references(rows: list[dict[str, str]]) -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (REF_DIR / "reference_ledger.csv").open(
            "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# AC-Twin manuscript reference ledger",
        "",
        "> 初始版本收录 Stage 5 已核验的 16 篇文献。后续正文引用统一追加到本表，"
        "不得从模型记忆补写元数据。`Manuscript No.` 在全文引用顺序冻结后机械填入。",
        "",
        "## 使用规则",
        "",
        "- 稳定主键为 `Ref key`；优先按 DOI/arXiv/PMLR/OpenReview 标识去重。",
        "- `Verification` 记录该文献在当前项目中的资格，而不是论文质量判断。",
        "- 已发表方法在 AC-Twin 数据上使用适配实现；表内不得暗示复现了原论文数据集数值。",
        "- 新增引用时同步更新本 Markdown 与 `reference_ledger.csv`。",
        "",
        "| Ref key | Manuscript No. | Authors | Year | Title | Venue | DOI/arXiv/PMLR | Code | Used in | Experimental role | Verification | Accessed |",
        "| --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        code = f"[link]({row['code_url']})" if row["code_url"] else "—"
        lines.append(
            f"| `{row['ref_key']}` |  | {row['authors']} | {row['year']} | "
            f"{row['title']} | {row['venue']} | {row['persistent_id'] or '—'} | "
            f"{code} | {row['used_in']} | {row['experimental_role']} | "
            f"`{row['verification']}` | {row['access_date']} |"
        )
    (REF_DIR / "REFERENCE_LEDGER.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    comparison = build_comparison_rows()
    references = build_reference_rows()
    _write_comparison(comparison)
    _write_references(references)
    print(f"[stage5-paper] comparison_rows={len(comparison)} "
          f"reference_rows={len(references)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
