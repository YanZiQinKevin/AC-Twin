# Fig. S-Stage5 risk–coverage 曲线图例（开发范围；终态：功效不足）

> 计划：`AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3`；终态 `STAGE5_MERGE_COMPARISON_UNDERPOWERED`。
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
0.8354 < 0.90）**从未创建**，本图不支持任何
“优于已发表方法”的主张；正式比较未进行。validation 上 AC-Twin 相对冻结
`STRONGEST_PUBLISHED_BASELINE`（CCTR_DRIVING_CALIBRATION_ADAPTED）的 I_AURC 点估计
0.1404（95% CI [
-0.0097, 0.2710]）、I_REGRET@80
0.1778（95% CI [
0.0807, 0.2789]）只作开发诊断引用。
