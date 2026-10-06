"""Rebuild Stage 5 literature artifacts from method_registry.json (single source of truth).

Outputs (all under the project tree):
- experiments/stage5_sota_comparison/literature/method_registry.md
- paper/tables/table1/table1_source.json   (rebuildable source for the xlsx)
- paper/tables/table1/table1_data.xlsx     (Table I, refreshed: 16 works + this work)

AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3, Task 1.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "experiments/stage5_sota_comparison/literature/method_registry.json"
MD_OUT = ROOT / "experiments/stage5_sota_comparison/literature/method_registry.md"
T1_DIR = ROOT / "paper/tables/table1"

T1_COLUMNS = [
    "Work", "Target and task", "Factual suffix / paired CF truth", "Action-regret audit",
    "Reject / defer meaning", "Guarantee", "Original paper",
]

# Table I positioning text per work (registry is the citation source; these columns
# are the paper-facing positioning, consistent with the frozen evidence boundary).
T1_POSITIONING = {
    "zhang2026driving": (
        "Retrospective counterfactual prediction for driving world models",
        "Factual suffix as transported evidence; paired CF for evaluation",
        "No multi-action regret audit",
        "None",
        "No selective-risk guarantee",
        "https://arxiv.org/abs/2608.11601"),
    "khandelwal2020wimp": (
        "What-if counterfactual motion forecasting",
        "Counterfactual conditioning at inference; no paired CF truth audit",
        "No",
        "None",
        "None",
        "https://arxiv.org/abs/2008.10587"),
    "tan2024prosim": (
        "Promptable closed-loop traffic simulation",
        "Prompt-conditioned rollouts; no factual-suffix evidence contract",
        "No",
        "None",
        "None",
        "https://arxiv.org/abs/2409.05863"),
    "fujii2024tgvcrn": (
        "Counterfactual treatment outcomes over time in multiagent systems",
        "Factual observational trajectories; no driving factual suffix",
        "No",
        "None",
        "None",
        "https://arxiv.org/abs/2206.01900"),
    "jiang2023cfgode": (
        "Continuous-time counterfactual outcome estimation (multi-agent)",
        "Factual outcome trajectories; no driving factual suffix",
        "No",
        "None",
        "None",
        "https://arxiv.org/abs/2306.11216"),
    "zou2022oosr": (
        "Outcome-oriented counterfactual treatment prediction",
        "Factual outcomes; no driving factual suffix",
        "Treatment-choice oriented, not decision-regret audit",
        "None",
        "None",
        "https://proceedings.mlr.press/v162/zou22a.html"),
    "zou2023fohl": (
        "Counterfactual prediction from factual-observation heterogeneity",
        "Factual outcome as input evidence; no driving factual suffix",
        "No",
        "None",
        "None",
        "https://proceedings.mlr.press/v213/zou23a.html"),
    "corbiere2019confidnet": (
        "Learned model confidence for failure prediction",
        "No driving factual suffix or paired driving CF",
        "No",
        "Classification failure-risk ranking",
        "None (ranking only)",
        "https://arxiv.org/abs/1910.04851"),
    "geifman2017selective": (
        "Selective classification with guaranteed risk",
        "No driving factual suffix or paired driving CF",
        "No",
        "Classification rejection",
        "High-probability selective-risk bound under its setting",
        "https://arxiv.org/abs/1705.08500"),
    "geifman2019selectivenet": (
        "End-to-end selective prediction with integrated reject option",
        "No driving factual suffix or paired driving CF",
        "No",
        "Learned selection/rejection",
        "Target-coverage training objective, no post-hoc guarantee",
        "https://arxiv.org/abs/1901.09192"),
    "liu2019deepgamblers": (
        "Abstention via portfolio theory (classification)",
        "No driving factual suffix or paired driving CF",
        "No",
        "Learned abstention",
        "None",
        "https://arxiv.org/abs/1907.00208"),
    "li2024rwr": (
        "Consistent no-rejection learning for regression with rejection",
        "No driving factual suffix or paired driving CF",
        "No",
        "Regression deferral at cost",
        "Consistency under weak realizability",
        "https://arxiv.org/abs/2307.02932"),
    "angelopoulos2024crc": (
        "Conformal risk control for monotone losses",
        "General framework; no paired driving CF requirement",
        "No study of this multi-action regret task",
        "Prediction-set/output adjustment, not offline action retention",
        "Expected monotone-loss control under exchangeability",
        "https://arxiv.org/abs/2208.02814"),
    "johansson2024conformal": (
        "Conformal regression with reject option",
        "No driving factual suffix or paired driving CF",
        "No",
        "Conformal rejection for regression",
        "Valid intervals on retained instances",
        "https://proceedings.mlr.press/v230/johansson24a.html"),
    "cao2024cctr": (
        "Calibrating trajectory-prediction uncertainty for planning",
        "No driving factual suffix or paired driving CF",
        "No",
        "None (calibration for planner consumption)",
        "Calibration error / NLL improvement",
        "https://doi.org/10.1609/aaai.v38i19.30085"),
    "zaoui2020regression": (
        "Regression with reject option (kNN)",
        "No driving factual suffix or paired driving CF",
        "No",
        "Regression rejection at fixed rate",
        "Optimal-rate characterization",
        "https://papers.nips.cc/paper/2020/hash/"),
    "this_work": (
        "Response-conditioned factual-evidence value and decision-regret audit",
        "Factual suffix in frozen predictor; paired CF truth for labels/evaluation only",
        "Yes",
        "Offline selective action retention under a decision-regret endpoint",
        "Confirmed selective-regret improvement vs frozen baselines (Stage 4 chain); SOTA comparison reported in this stage's table",
        "This paper"),
}

WORK_LABELS = {
    "zhang2026driving": "Zhang et al. (2026)",
    "khandelwal2020wimp": "Khandelwal et al. (2020)",
    "tan2024prosim": "Tan et al. (2024)",
    "fujii2024tgvcrn": "Fujii et al. (2024)",
    "jiang2023cfgode": "Jiang et al. (2023)",
    "zou2022oosr": "Zou et al. (2022)",
    "zou2023fohl": "Zou et al. (2023)",
    "corbiere2019confidnet": "Corbiere et al. (2019)",
    "geifman2017selective": "Geifman & El-Yaniv (2017)",
    "geifman2019selectivenet": "Geifman & El-Yaniv (2019)",
    "liu2019deepgamblers": "Liu et al. (2019)",
    "li2024rwr": "Li et al. (2024)",
    "angelopoulos2024crc": "Angelopoulos et al. (2024)",
    "johansson2024conformal": "Johansson et al. (2024)",
    "cao2024cctr": "Cao et al. (2024)",
    "zaoui2020regression": "Zaoui et al. (2020)",
    "this_work": "This work",
}

T1_ORDER = [
    "zhang2026driving", "khandelwal2020wimp", "tan2024prosim",
    "fujii2024tgvcrn", "jiang2023cfgode", "zou2022oosr", "zou2023fohl",
    "corbiere2019confidnet", "geifman2017selective", "geifman2019selectivenet",
    "liu2019deepgamblers", "li2024rwr", "angelopoulos2024crc",
    "johansson2024conformal", "cao2024cctr", "zaoui2020regression",
    "this_work",
]


def build_markdown(registry: dict) -> str:
    lines = [
        "# Stage 5 method registry (machine source: `method_registry.json`)",
        "",
        f"- Plan: `{registry['plan_id']}`",
        f"- Frozen: {registry['frozen_date']}; access date for all citations: {registry['access_date']}",
        f"- Works: {len(registry['works'])} (blocks: counterfactual driving / counterfactual outcome / "
        "failure & selective prediction / regression rejection & risk control)",
        "",
        "## Experimental candidates (P1-P5) and supplemental",
        "",
        "| Method ID | Source | Venue | Code / license | Status | Adaptation (one line) |",
        "|---|---|---|---|---|---|",
    ]
    for w in registry["works"]:
        if w["registry_role"] not in ("main_candidate", "supplemental_candidate", "technical_alternate_pool"):
            continue
        code = w.get("code_url") or "NONE"
        lic = w.get("license") or "NONE"
        adapt = (w.get("adaptation") or {}).get("summary", w.get("reason", "")).replace("|", "\\|")
        lines.append(
            f"| `{w.get('method_id', '-')}` | {w['authors'].split(',')[0]} et al. ({w['year']}) | "
            f"{w['venue']} | {code} / {lic} | {w['status']} | {adapt} |")
    lines += [
        "",
        "## Literature-only positioning works (enter Table I, not the experiment)",
        "",
        "| Work | Venue | arXiv/DOI | Status | Reason |",
        "|---|---|---|---|---|",
    ]
    for w in registry["works"]:
        if w["registry_role"] != "literature_only":
            continue
        reason = w.get("reason", "").replace("|", "\\|")
        lines.append(
            f"| {w['authors'].split(',')[0]} et al. ({w['year']}), *{w['title']}* | {w['venue']} | "
            f"{w['doi_or_arxiv']} | {w['status']} | {reason} |")
    lines += [
        "",
        "## Verification notes",
        "",
    ]
    for note in registry["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(f"Sanity evidence: `../phase_l/sanity/sanity_report.json` "
                 f"(5/5 PASS, ENGINEERING_SANITY_ONLY). Compatibility decisions: `compatibility_decision.json`.")
    return "\n".join(lines) + "\n"


def build_table1(registry: dict) -> None:
    import openpyxl
    rows = []
    for wid in T1_ORDER:
        pos = T1_POSITIONING[wid]
        rows.append([WORK_LABELS[wid], *pos])
    source = {
        "source_type": "STAGE5_TABLE1_SOURCE",
        "plan_id": registry["plan_id"],
        "rebuilt_from": "experiments/stage5_sota_comparison/literature/method_registry.json",
        "columns": T1_COLUMNS,
        "rows": rows,
    }
    (T1_DIR / "table1_source.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Table I"
    ws.append(T1_COLUMNS)
    for row in rows:
        ws.append(row)
    for col, width in zip("ABCDEFG", (26, 46, 46, 40, 42, 46, 44)):
        ws.column_dimensions[col].width = width
    ws2 = wb.create_sheet("Sources")
    ws2.append(["Source file / section", "Use"])
    ws2.append(["experiments/stage5_sota_comparison/literature/method_registry.json",
                "Single citation/adaptation source (frozen 2026-09-27)"])
    ws2.append(["experiments/stage5_sota_comparison/literature/compatibility_decision.json",
                "Endpoint/sanity verification per method"])
    ws2.append(["paper/tables/2026-09-25_AC-Twin主表草案_v0.1.md", "Frozen source; read-only"])
    ws2.column_dimensions["A"].width = 70
    ws2.column_dimensions["B"].width = 60
    wb.save(T1_DIR / "table1_data.xlsx")


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    MD_OUT.write_text(build_markdown(registry), encoding="utf-8")
    print(f"wrote {MD_OUT}")
    build_table1(registry)
    print(f"wrote {T1_DIR / 'table1_source.json'} and table1_data.xlsx "
          f"({len(T1_ORDER) - 1} literature rows + this work)")


if __name__ == "__main__":
    main()
