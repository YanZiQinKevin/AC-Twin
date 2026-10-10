# Fig. 3｜事实证据价值

> 当前正文编号（2026-10-08）：**Fig. 3**。目录 `fig1` 与资产文件名沿用既有命名；下方日期记录中的旧图号为当时快照。

> 2026-10-04 叙事同步：c 面板改为独立点与区间，不连接成连续关系；d 首项直接标明 zero vs all positive response。数值和区间不变。

> 同日测量含义修订：b 图例明确为 no suffix encoding / shuffled encoding / same-event encoding；三组同结构模型分别训练，共享本事件事实轨迹重建参照。负值描述受评模型的相对误差，不解释为事实信息本身的负价值。

当前版本为 **v1（完成）**：四面板，180 × 104 mm，按双栏通栏 100% 比例设计，配色与 Fig. 2 / Fig. 3 相同。

## 当前交付

- `Fig1_evidence_value_structure.svg` / `.png`（600 dpi）/ `.pdf`：主版本；
- `fig1_bins_plot_data.csv`：四箱的 `n_agent_time`、`n_episodes`、三个平均损失、`V_evt` 及其 CI、`rho_no`；
- `fig1_contrasts_plot_data.csv`：三个前瞻指定对比的点估计与 95% CI；
- `fig1_data.xlsx`：原有数据表（与本目录 CSV 同源）；
- `build_fig1.py`：确定性重绘脚本；
- `FIGURE_LEGEND.md`：正文图注草稿。

## 运行

```powershell
python paper/figures/fig1/build_fig1.py
```

脚本只读 Stage 2 确认性裁判报告 `experiments/stage2_confirmatory/merge/report.json`，
并在绘图前断言四箱 `bins.*.V_evt` 与 `statistics.v_zero / v_low / v_mid / v_high` 的点估计逐位一致（atol 1e-12）。

## 面板与取数字段

| 面板 | 内容 | 字段 |
| --- | --- | --- |
| (a) | 四个响应箱的定义与冻结阈值 | `statistics.response_bins`（`zero_tolerance` / `t1` / `t2`）；箱计数取 `statistics.bins.*.n_episodes` |
| (b) | 每箱三个平均损失（对数纵轴） | `statistics.bins.*.mean_l_no` / `mean_l_suffix` / `mean_l_shuffled` |
| (c) | 四箱 `V_evt` 与其冻结 95% CI（hero 面板） | `statistics.v_zero / v_low / v_mid / v_high` 的 `point / ci_lower / ci_upper` |
| (d) | 三个前瞻指定对比的森林图 | `statistics.c_presence` / `c_high_low` / `c_high_mid`，筛选线取 `statistics.point_screen` |

## 冻结阈值的出处

箱边界沿用 Stage 1 的冻结分箱，未在确认数据上重新拟合：

- 数值：`experiments/stage2_confirmatory/merge/report.json#statistics.response_bins`；
- 冻结工件：`experiments/stage1_evidence_value/merge/response_bins.json`，其 sha256 `8f6aeb078b12007b84da55d3c682cfce861f74a199640e2e1cad4bf45e5cc15f`
  被登记在 `experiments/stage2_confirmatory/merge/protocol_lock.json` 的 `paths/response_bins_json`。

## 版面与表述规则

- 四箱 episode 数可重叠（一个 episode 可贡献多个箱），**不得相加**；
- 中箱 CI 整体为负（`[-0.1129, -0.0199]`）、低箱 CI 跨零（`[-0.0110, 0.1697]`），二者如实显示；
- 不使用"双刃剑"、严格单调、对称 U 形或因果机制表述；
- `C_rebound` 与 `C_high_low` 是同一个数（前者为无裁判 CI 的摘要点估计），本图不另列；
- 稳健性（leave-one-model-seed / leave-one-action-type / leave-one-episode / high-error-deletion）不在本图内，见补充材料。
