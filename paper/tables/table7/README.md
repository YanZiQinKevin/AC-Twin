# Table 7 — Stage 5 已发表方法开发基准（终态：功效不足）

计划：`AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3`（§8 Task 6）
终态：**`STAGE5_MERGE_COMPARISON_UNDERPOWERED`（2026-09-28，永久，不得重判）**

## 内容

| 文件 | 说明 |
| --- | --- |
| `TABLE_7.md` | **论文唯一 Comparison 表入口**：五个已发表适配方法、三个内部基线与 AC-Twin 的紧凑九行比较；不再把 A–E 拆成论文表 |
| `table7_comparison_main.csv` | 上述唯一论文表的机读副本 |
| `stage5_inference_timing_receipt.json` | 独立工程计时收据；不含真值与科研指标，补齐 AC-Twin 及五个学习型对照的同机、batch=1、五-seed risk-head 单组延迟 |
| `TABLE_7_DETAILS.md` | **内部审计材料，不作为另一张论文 Comparison 表**：适配矩阵 A、validation 主/次指标 B1/B2、全部候选回执 B3、配对比较 C、功效与终态 D、原始成本记录 E |
| `table7_data.xlsx` | 同一数据的机读版（7 个 sheet，与 MD 一一对应） |
| `table7_source.json` | 可重建源：两张表的全部数字 + 范围标签 + 终态 |
| `build_table7_details.py` | 构建脚本：只读冻结产物，重导出本包 |
| `build_stage5_manuscript_assets.py` | 只读冻结结果、工程计时收据和文献注册表，重建唯一论文表与 reference ledger |

配套图数据：`../../figures/stage5_sota/figS_stage5_risk_coverage_data.csv` + `FIGURE_LEGEND.md`（同一脚本生成）。

## 重建

```bash
python paper/tables/table7/build_table7_details.py
python paper/tables/table7/build_stage5_manuscript_assets.py
```

只读 `experiments/stage5_sota_comparison/` 下 `literature/`、`development/`、`power/` 的
create-once 冻结产物；不改写、不重选、不重判任何实验数字。

## 范围与主张边界（必须随表出现）

- 所有数字是 Stage 4C-R2 开发划分 validation（1,500 组）上的 `ADAPTIVE_DEVELOPMENT_ONLY`
  诊断；Stage 5 comparison holdout **从未创建**（§6.2 功效终态禁止采集）。
- §11.1 的 SOTA 优势句**不可写**；按 §11.3 保留 Stage 4 与 D2F 已确认结论，不写优于已发表方法。
- 冻结 `STRONGEST_PUBLISHED_BASELINE = CCTR_DRIVING_CALIBRATION_ADAPTED`（永不替换）。
- 计时仅为工程成本测量：复用旧 validation 的合法输入，不读取真值，不保存预测，不改变任何 Gate 或终态。
