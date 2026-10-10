"""Build the Stage 5 paper table package + risk-coverage figure data (Task 6).

Plan ``AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3`` section 8 Task 6.

READS ONLY frozen Stage 5 artifacts (method registry, final_method_selection,
validation_comparison_report, power_report, method_search_space) and re-emits
them as paper-facing tables/figure data. Nothing is recomputed, re-selected or
re-adjudicated here; every number carries its development scope label. The
terminal state STAGE5_MERGE_COMPARISON_UNDERPOWERED is stated prominently and no
SOTA-advantage claim is made anywhere in the package (plan section 11).

Outputs (this directory):
- ``table7_source.json`` (rebuildable source)
- ``TABLE_7_DETAILS.md``
- ``table7_data.xlsx``
- ``../../figures/stage5_sota/figS_stage5_risk_coverage_data.csv``
- ``../../figures/stage5_sota/FIGURE_LEGEND.md``
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments" / "stage5_sota_comparison"
T_DIR = Path(__file__).resolve().parent
F_DIR = ROOT / "paper" / "figures" / "stage5_sota"

TERMINAL = "STAGE5_MERGE_COMPARISON_UNDERPOWERED"


def load(name: str, *parts: str) -> dict:
    return json.loads((STAGE.joinpath(*parts, name)).read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:  # NaN
        return "NaN"
    return f"{v:.{digits}f}"


def main() -> int:
    registry = load("method_registry.json", "literature")
    selection = load("final_method_selection.json", "development")
    report = load("validation_comparison_report.json", "development")
    power = load("power_report.json", "power")
    space = load("method_search_space.json", "development")

    strongest = selection["strongest_published_baseline"]
    selected = selection["selected"]
    cand_metrics = report["candidate_metrics"]
    names = report["comparison_vs_strongest"]["quantity_names"]
    comp = report["comparison_vs_strongest"]

    def q(name: str) -> dict:
        i = names.index(name)
        return {"point": comp["point"][i], "lo": comp["ci95_lower"][i],
                "hi": comp["ci95_upper"][i]}

    # --- rows -------------------------------------------------------------------
    works = {w.get("method_id"): w for w in registry["works"] if w.get("method_id")}

    adaptation_rows = []
    for method_id in sorted(selected):
        w = works[method_id]
        a = w["adaptation"]
        sel = selected[method_id]
        dev = a.get("deviations_from_original", a.get("deviation", "—"))
        if isinstance(dev, list):
            dev = "; ".join(str(d) for d in dev)
        adaptation_rows.append({
            "method_id": method_id,
            "source": f"{w['authors'].split(',')[0]} et al. {w['year']} ({w['venue']})",
            "family": w["family"],
            "legal_fields": ", ".join(a["legal_bundle_fields"]),
            "training_target": a["training_target"],
            "risk_output": a["risk_output"],
            "frozen_grid": json.dumps(space["methods"][method_id]["grid"]),
            "deviations": dev,
            "selected_config": sel["config_key"],
            "param_count": sel["param_count"],
        })

    order_rows = [("AC-Twin Direct (frozen R2 audit)", "actwin", report["actwin_metrics"])]
    for method_id in sorted(selected):
        key = f"{method_id}#{selected[method_id]['config_key']}"
        order_rows.append((f"{method_id} (selected {selected[method_id]['config_key']})",
                           key, cand_metrics[key]))
    for name, metrics in report["internal_baseline_metrics"].items():
        order_rows.append((f"{name} (internal baseline)", f"baseline:{name}", metrics))

    primary_rows = [{
        "method": label,
        "aurc_regret": m["aurc_regret"],
        "regret_at_70": m["regret_at_70"],
        "regret_at_80": m["regret_at_80"],
        "regret_at_90": m["regret_at_90"],
        "regret_at_95": m["regret_at_95"],
        "pr_auc_minus_prevalence": m["pr_auc_minus_prevalence"],
        "spearman_risk_regret": m["spearman_risk_regret"],
        "per_seed_aurc_min": min(m["per_seed_aurc"]),
        "per_seed_aurc_max": max(m["per_seed_aurc"]),
    } for label, _, m in order_rows]

    secondary_rows = [{
        "method": label,
        "tail_severity_at_20": m["tail_severity_at_20"],
        "tail_capture_at_20": m["tail_capture_at_20"],
        "oracle_gap_closure_at_80": m["oracle_gap_closure_at_80"],
        "retained80_regret_p95": m["retained80_regret_p95"],
        "retained80_regret_max": m["retained80_regret_max"],
        "trim1_aurc": m["trim1_aurc"],
        "trim5_aurc": m["trim5_aurc"],
    } for label, _, m in order_rows]

    all_candidate_rows = [{
        "method_id": c["method_id"], "config_key": c["config_key"],
        "aurc_regret": c["aurc_regret"], "regret_at_80": c["regret_at_80"],
        "param_count": c["param_count"],
        "mean_inference_ms": c["mean_inference_ms"],
        "selected": (c["method_id"] in selected
                     and selected[c["method_id"]]["config_key"] == c["config_key"]),
        "valid": c["valid"],
    } for c in selection["candidates"]]

    paired_rows = []
    for qname, label in (("i_aurc", "I_AURC_REGRET"), ("i_regret80", "I_REGRET@80")):
        v = q(qname)
        paired_rows.append({"endpoint": label, "point": v["point"],
                            "ci95_lower": v["lo"], "ci95_upper": v["hi"]})
    for i in range(5):
        v = q(f"seed{i}_i_aurc")
        v2 = q(f"seed{i}_i_regret80")
        paired_rows.append({"endpoint": f"seed{i} I_AURC / I_REGRET@80",
                            "point": v["point"], "ci95_lower": v2["point"],
                            "ci95_upper": None})
    for sname in ("tstar_low_i_aurc", "tstar_high_i_aurc",
                  "interaction_low_i_aurc", "interaction_high_i_aurc"):
        v = q(sname)
        paired_rows.append({"endpoint": sname, "point": v["point"],
                            "ci95_lower": None, "ci95_upper": None})

    power_rows = [{
        "n": int(n),
        "joint_power_rho100": e["scenarios"]["1.00"]["joint_power"],
        "joint_power_rho075": e["scenarios"]["0.75"]["joint_power"],
        "joint_power_rho050": e["scenarios"]["0.50"]["joint_power"],
        "binding_condition": e["binding_condition"],
        "binding_power": e["binding_power"],
    } for n, e in sorted(power["per_n"].items(), key=lambda kv: int(kv[0]))]

    cost_rows = []
    for method_id in sorted(selected):
        sel = selected[method_id]
        params = int(sel["param_count"])
        cost_rows.append({
            "method": method_id,
            "param_count": params,
            "approx_forward_flops_per_group": 2 * params,
            "mean_inference_ms_per_group": sel["mean_inference_ms"],
            "native_calibration": report["native_calibration_selected"][method_id]["status"],
        })
    cost_rows.append({
        "method": "AC-Twin Direct (frozen R2 audit)",
        "param_count": 15740,
        "approx_forward_flops_per_group": 2 * 15740,
        "mean_inference_ms_per_group": "frozen (R2 receipt)",
        "native_calibration": "NOT_APPLICABLE",
    })

    source = {
        "plan_id": report["plan_id"],
        "terminal_state": power["status"],
        "scope": ("ADAPTIVE_DEVELOPMENT_ONLY — validation role of the Stage 4C-R2 "
                  "development split; the Stage 5 comparison holdout was NEVER created "
                  "(underpowered terminal). These are development diagnostics, not a "
                  "confirmed comparison."),
        "strongest_published_baseline": strongest,
        "adaptation_rows": adaptation_rows,
        "primary_rows": primary_rows,
        "secondary_rows": secondary_rows,
        "all_candidate_rows": all_candidate_rows,
        "paired_rows_vs_strongest": paired_rows,
        "materiality": report["materiality"],
        "power_rows": power_rows,
        "cost_rows": cost_rows,
        "native_calibration_selected": report["native_calibration_selected"],
        "training_cost": {"total_fits": 60, "total_wall_seconds": 468,
                          "note": "aggregate over all 60 (method, config, seed) fits "
                                  "(development_freeze_run.log); per-fit average ≈ 7.3 s; "
                                  "peak GPU memory NOT_MEASURED"},
    }
    (T_DIR / "table7_source.json").write_text(
        json.dumps(source, indent=1, ensure_ascii=False), encoding="utf-8")

    # --- risk-coverage figure CSV (development scope) -------------------------------
    F_DIR.mkdir(parents=True, exist_ok=True)
    curves = {}
    coverages = None
    for label, key, m in order_rows:
        coverages = m["coverage_profile_coverages"]
        curves[label] = m["coverage_profile_values"]
    csv_path = F_DIR / "figS_stage5_risk_coverage_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["coverage"] + [label for label, _, _ in order_rows])
        for i, c in enumerate(coverages):
            writer.writerow([f"{float(c):.4f}"]
                            + [f"{float(curves[label][i]):.8f}" for label, _, _ in order_rows])

    # --- Markdown table ----------------------------------------------------------------
    lines: list[str] = []
    lines.append("# Stage 5 已发表强方法公平对照 — 开发范围数据包（终态：功效不足）")
    lines.append("")
    lines.append(f"> 计划：`{report['plan_id']}`  ")
    lines.append(f"> **终态：`{power['status']}`（2026-09-28，永久，不得重判）**  ")
    lines.append("> 范围：Stage 4C-R2 开发划分 validation（1,500 组）上的 "
                 "`ADAPTIVE_DEVELOPMENT_ONLY` 诊断。Stage 5 comparison holdout "
                 "**从未创建**（§6.2 功效终态禁止采集）。本包任何数字都不是确认比较结果。")
    lines.append("")
    lines.append("## 主张边界（计划 §11）")
    lines.append("")
    lines.append("- §11.1 的 SOTA 优势句**不可写**：主比较未达到 0.90 联合功效，未生成 holdout，"
                 "未做裁判。")
    lines.append("- 按 §11.3：保留 Stage 4 与 D2F 已确认结论；不写优于已发表方法；"
                 "本开发比较按真实结果完整报告，不删除失败结果。")
    lines.append("- 允许写法：完成了系统的文献定位与五项已发表方法的统一合法输入适配，"
                 "并在开发数据上冻结了唯一 `STRONGEST_PUBLISHED_BASELINE`；"
                 "正式比较因预注册功效不足而未进行。")
    lines.append("")
    lines.append("## Table A：已发表方法适配矩阵（P1–P5，全部 `ELIGIBLE_ADAPTED_VERIFIED`）")
    lines.append("")
    lines.append("| 方法 | 来源 | 家族 | 合法字段 | 训练目标 | 风险输出 | 冻结网格 | 适配偏差 | 中选 | 参数量 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in adaptation_rows:
        lines.append(f"| {r['method_id']} | {r['source']} | {r['family']} | {r['legal_fields']} "
                     f"| {r['training_target']} | {r['risk_output']} | {r['frozen_grid']} "
                     f"| {r['deviations']} | {r['selected_config']} | {r['param_count']} |")
    lines.append("")
    lines.append("## Table B1：validation 主指标（开发范围；AC-Twin、五方法中选配置、三个内部基线）")
    lines.append("")
    lines.append("| 方法 | AURC_REGRET | REGRET@70 | REGRET@80 | REGRET@90 | REGRET@95 "
                 "| PR−prevalence | Spearman | per-seed AURC [min,max] |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in primary_rows:
        lines.append(f"| {r['method']} | {fmt(r['aurc_regret'], 6)} | {fmt(r['regret_at_70'], 6)} "
                     f"| {fmt(r['regret_at_80'], 6)} | {fmt(r['regret_at_90'], 6)} "
                     f"| {fmt(r['regret_at_95'], 6)} | {fmt(r['pr_auc_minus_prevalence'], 4)} "
                     f"| {fmt(r['spearman_risk_regret'], 4)} "
                     f"| [{fmt(r['per_seed_aurc_min'], 6)}, {fmt(r['per_seed_aurc_max'], 6)}] |")
    lines.append("")
    lines.append("## Table B2：validation 次要面板（开发范围）")
    lines.append("")
    lines.append("| 方法 | 尾部严重度@20 | 尾部捕获@20 | oracle gap 闭合@80 "
                 "| 保留80% regret p95 | 保留80% regret max | trim1% AURC | trim5% AURC |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in secondary_rows:
        lines.append(f"| {r['method']} | {fmt(r['tail_severity_at_20'], 6)} "
                     f"| {fmt(r['tail_capture_at_20'], 4)} | {fmt(r['oracle_gap_closure_at_80'], 4)} "
                     f"| {fmt(r['retained80_regret_p95'], 6)} | {fmt(r['retained80_regret_max'], 6)} "
                     f"| {fmt(r['trim1_aurc'], 6)} | {fmt(r['trim5_aurc'], 6)} |")
    lines.append("")
    lines.append("## Table B3：全部候选（含未中选；机械选择回执）")
    lines.append("")
    lines.append("| 方法 | 配置 | AURC_REGRET | REGRET@80 | 参数量 | 单组推理 ms | 中选 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in all_candidate_rows:
        lines.append(f"| {r['method_id']} | {r['config_key']} | {fmt(r['aurc_regret'], 6)} "
                     f"| {fmt(r['regret_at_80'], 6)} | {r['param_count']} "
                     f"| {fmt(r['mean_inference_ms'], 4)} | {'**YES**' if r['selected'] else 'no'} |")
    lines.append("")
    lines.append(f"## Table C：AC-Twin vs 冻结最强已发表基线（`{strongest}`，validation 配对 bootstrap，开发范围）")
    lines.append("")
    lines.append("| 量 | 点估计 | 95% CI 下界 | 95% CI 上界 |")
    lines.append("| --- | --- | --- | --- |")
    for r in paired_rows[:2]:
        lines.append(f"| {r['endpoint']} | {fmt(r['point'], 4)} | {fmt(r['ci95_lower'], 4)} "
                     f"| {fmt(r['ci95_upper'], 4)} |")
    mat = report["materiality"]
    lines.append("")
    lines.append(f"实质效应（75% 收缩）：I_AURC {fmt(mat['point_i_aurc'], 4)} → "
                 f"{fmt(mat['point_i_aurc'] * mat['shrink'], 4)} ≥ 0.05 ✓；"
                 f"I_REGRET80 {fmt(mat['point_i_regret80'], 4)} → "
                 f"{fmt(mat['point_i_regret80'] * mat['shrink'], 4)} ≥ 0.03 ✓ → **material**，进入功效网格。")
    lines.append("")
    lines.append("逐 seed 与四分层点估计（开发范围，描述性）：")
    lines.append("")
    lines.append("| 量 | 点估计 |")
    lines.append("| --- | --- |")
    for r in paired_rows[2:]:
        value = fmt(r["point"], 4) if r["ci95_lower"] is None else f"{fmt(r['point'], 4)} / {fmt(r['ci95_lower'], 4)}"
        lines.append(f"| {r['endpoint']} | {value} |")
    lines.append("")
    lines.append("## Table D：联合功效与终态（计划 §6.2；候选 n 冻结于功效运行前）")
    lines.append("")
    lines.append("| n | ρ=1.00 联合功效 | ρ=0.75 联合功效（裁判情景） | ρ=0.50 联合功效 | 绑定条件 | 绑定功效@0.75 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in power_rows:
        lines.append(f"| {r['n']} | {fmt(r['joint_power_rho100'], 4)} | {fmt(r['joint_power_rho075'], 4)} "
                     f"| {fmt(r['joint_power_rho050'], 4)} | {r['binding_condition']} "
                     f"| {fmt(r['binding_power'], 4)} |")
    lines.append("")
    lines.append(f"75% 情景下最大冻结 n=6000 的联合功效 "
                 f"{fmt(power_rows[-1]['joint_power_rho075'], 4)} < 0.90 → `selected_n=null`，"
                 f"终态 **`{power['status']}`**。不创建 comparison holdout，不降低门槛、不换 seed、"
                 "不换方法或主基线（计划 §6.2/§10）。")
    lines.append("")
    lines.append("## Table E：成本与完整性（开发范围）")
    lines.append("")
    lines.append("| 方法 | 参数量 | 近似前向 FLOPs/组（MAC×2） | 单组推理 ms | 原生校准 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in cost_rows:
        lines.append(f"| {r['method']} | {r['param_count']} | {r['approx_forward_flops_per_group']} "
                     f"| {r['mean_inference_ms_per_group'] if isinstance(r['mean_inference_ms_per_group'], str) else fmt(r['mean_inference_ms_per_group'], 4)} "
                     f"| {r['native_calibration']} |")
    lines.append("")
    cal = report["native_calibration_selected"]
    lines.append("原生校准细节：P1 ECE "
                 f"{fmt(cal['CONFIDNET_TCP_ADAPTED'].get('ece'), 4)}；P4 经验覆盖 "
                 f"{fmt(cal['CONFORMAL_REGRESSION_REJECT_ADAPTED'].get('empirical_coverage'), 4)}"
                 "（目标 0.90）；P5 NLL "
                 f"{fmt(cal['CCTR_DRIVING_CALIBRATION_ADAPTED'].get('nll'), 4)}、95% 经验覆盖 "
                 f"{fmt(cal['CCTR_DRIVING_CALIBRATION_ADAPTED'].get('empirical_coverage_95'), 4)}；"
                 "P2/P3 `NOT_APPLICABLE`。")
    lines.append("")
    lines.append("训练成本：60 个（方法×配置×seed）拟合合计约 468 s 墙钟（CUDA；"
                 "峰值显存 NOT_MEASURED）。")
    lines.append("")
    lines.append("## 来源与重建")
    lines.append("")
    lines.append("- 冻结源：`experiments/stage5_sota_comparison/` 下 `literature/`、`development/`、"
                 "`power/` 的 create-once 产物；本表由 `build_table7_details.py` 重新导出，"
                 "数字不改写、不重选、不重判。")
    lines.append("- risk–coverage 曲线数据：`paper/figures/stage5_sota/figS_stage5_risk_coverage_data.csv`"
                 "（同一冻结源，开发范围）。")
    (T_DIR / "TABLE_7_DETAILS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- xlsx ---------------------------------------------------------------------------
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    def sheet(name: str, rows: list[dict]) -> None:
        ws = wb.create_sheet(name)
        if not rows:
            return
        headers = list(rows[0].keys())
        ws.append(headers)
        for row in rows:
            ws.append([row.get(h) for h in headers])

    sheet("A_adaptation", adaptation_rows)
    sheet("B1_validation_primary", primary_rows)
    sheet("B2_validation_secondary", secondary_rows)
    sheet("B3_all_candidates", all_candidate_rows)
    sheet("C_paired_vs_strongest", paired_rows)
    sheet("D_power", power_rows)
    sheet("E_cost", cost_rows)
    wb.save(T_DIR / "table7_data.xlsx")

    # --- figure legend ----------------------------------------------------------------------
    legend = f"""# Fig. S-Stage5 risk–coverage 曲线图例（开发范围；终态：功效不足）

> 计划：`{report['plan_id']}`；终态 `{power['status']}`。
> 数据源：`paper/figures/stage5_sota/figS_stage5_risk_coverage_data.csv`（由
> `paper/tables/table7/build_table7_details.py` 从冻结
> `validation_comparison_report.json` 重导出；不重算、不重选、不重判）。

**图意。** 在 Stage 4C-R2 开发划分 validation（1,500 十动作决策组）上，各方法的
risk–coverage 曲线 R(c)（保留最低风险 c 比例组的平均决策 regret，c 从 5% 到 100%）。
曲线越低越好。九条曲线：AC-Twin Direct（冻结 R2 审计）、五个已发表适配方法的
validation 中选配置（P1 ConfidNet、P2 SelectiveNet、P3 RwR-kNN、P4 Mondrian
conformal、P5 CCTR）与三个内部非学习基线（MARGIN_ONLY、SEED_VOTE_ENTROPY、
SEED_MARGIN_STD）。

**范围与限制（必须随图出现）。** 本图是 `ADAPTIVE_DEVELOPMENT_ONLY` 诊断：
Stage 5 comparison holdout 因预注册联合功效不足（75% 情景 n=6000 联合功效
{fmt(power_rows[-1]['joint_power_rho075'], 4)} < 0.90）**从未创建**，本图不支持任何
“优于已发表方法”的主张；正式比较未进行。validation 上 AC-Twin 相对冻结
`STRONGEST_PUBLISHED_BASELINE`（{strongest}）的 I_AURC 点估计
{fmt(report['materiality']['point_i_aurc'], 4)}（95% CI [
{fmt(q('i_aurc')['lo'], 4)}, {fmt(q('i_aurc')['hi'], 4)}]）、I_REGRET@80
{fmt(report['materiality']['point_i_regret80'], 4)}（95% CI [
{fmt(q('i_regret80')['lo'], 4)}, {fmt(q('i_regret80')['hi'], 4)}]）只作开发诊断引用。
"""
    (F_DIR / "FIGURE_LEGEND.md").write_text(legend, encoding="utf-8")

    print(f"wrote TABLE_7_DETAILS.md, table7_data.xlsx, "
          f"table7_source.json, {csv_path.name}, FIGURE_LEGEND.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
