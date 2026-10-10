# Table 8 — Supervised controls and feature ablations on development data

对应正文§3.5、§4.4及补充方法SM11。五项对照均来自同一1,500组开发验证数据。

## 中文表体（与正文一致）

**Table 8 | 开发范围的监督对照与特征消融。**

| 评价方法 / 参照 | AURC-regret 降低比例 [99% CI]（%） | Regret@80 降低比例 [99% CI]（%） |
| --- | ---: | ---: |
| 完整输入 / 仅基线信号 | 43.4 [28.2, 55.4] | 26.5 [14.5, 38.1] |
| 完整输入 / 移除显式风险通道 | 11.6 [-3.1, 22.9] | -2.0 [-11.8, 7.0] |
| 仅基线信号 / SEED_VOTE_ENTROPY | -2.2 [-19.3, 12.5] | 2.1 [-4.2, 5.9] |
| 仅基线信号 / MARGIN_ONLY | 7.0 [-8.9, 20.6] | 5.2 [-3.0, 14.8] |
| 仅基线信号 / SEED_MARGIN_STD | 11.3 [-13.0, 30.6] | 10.1 [-4.4, 23.8] |

表注：评价来自同一1,500个完整十动作开发组，属于开发性证据。正值表示评价方法相对参照降低regret指标。五项AURC-regret比较和五项Regret@80比较分别组成一个家族，每项99%区间对应其家族名义95%，由10,000次整组配对bootstrap得到；不将两家族合并为整体95%覆盖。

## English table and caption

**Table 8 | Supervised controls and feature ablations on development data.**

| Comparison (method / reference) | AURC-regret reduction [99% CI] (%) | Regret@80 reduction [99% CI] (%) |
| --- | ---: | ---: |
| Complete input / Baseline signals only | 43.4 [28.2, 55.4] | 26.5 [14.5, 38.1] |
| Complete input / Explicit risk channels removed | 11.6 [-3.1, 22.9] | -2.0 [-11.8, 7.0] |
| Baseline signals only / SEED_VOTE_ENTROPY | -2.2 [-19.3, 12.5] | 2.1 [-4.2, 5.9] |
| Baseline signals only / MARGIN_ONLY | 7.0 [-8.9, 20.6] | 5.2 [-3.0, 14.8] |
| Baseline signals only / SEED_MARGIN_STD | 11.3 [-13.0, 30.6] | 10.1 [-4.4, 23.8] |

Note. Evaluation uses the same 1,500 complete ten-action development groups. Positive values indicate lower regret metrics for the evaluated method than for its reference. Five AURC-regret contrasts and five Regret@80 contrasts form separate families. Each 99% percentile interval corresponds to nominal 95% Bonferroni family coverage, using 10,000 whole-group paired bootstrap resamples; joint 95% coverage across both families is not claimed. All results are conditional development evidence.

## 比较口径

相对改善为1−方法指标/参照指标，正值表示指标降低；机读数据保留原始无量纲比例，正文以百分数显示并四舍五入至一位小数。五项主要AURC-regret与五项次Regret@80分别组成家族，每项99%百分位区间对应其五比较家族Bonferroni名义95%覆盖，不合并声称两个家族整体95%覆盖。开发验证组曾参与历史开发决策，结果不构成新确认。

完整输入相对三标量监督组合有条件追加收益；三标量监督组合相对三项固定启发式的校正区间均跨零。移除显式风险通道的对比两项区间均跨零，不建立通道的独立增量，也不证明等效或风险信息无用。名义参数量相同不等于有效容量相同。

## 数字来源

- `experiments/paper_revision/m1_supervised_ablation_v02/evaluation/contrasts.csv`：C1–C5的improvement行，分别提取aurc_regret和regret_at_80的point/ci99_lower/ci99_upper。
- `evaluation/bootstrap_receipt.json`：重采样次数、区间分位与家族口径。
- `config.json`及`models/model_inventory.json`：输入方案、训练与参数合同。
- `paper/manuscript/AC-Twin_论文初稿_中文.md`§4.4：中文表体与表注。

本包只整理冻结结果，不训练、推理、重采样或重判。
