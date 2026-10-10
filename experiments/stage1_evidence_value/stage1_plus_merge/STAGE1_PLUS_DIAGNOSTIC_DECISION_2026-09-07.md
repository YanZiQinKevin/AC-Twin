# Stage 1 Plus 轻量形状诊断与功效分析 — 决策文档

> 文档状态：**DEVELOPMENT_DIAGNOSTIC_ONLY**  
> 计划编号：`AC-TWIN-RFER-STAGE1-PLUS-2026-09-07-V1`  
> 上位研究计划：`AC-TWIN-RFER-2026-09-07-V1`  
> 日期：2026-09-07  
> claim_boundary：`development_diagnostic_only; does_not_reopen_or_rejudge_stage1_gate_a; does_not_confirm_monotonic_decay_or_u_shape; no_mechanism_or_causal_claim; stage2_and_router_remain_blocked_pending_user_approval`

---

## 1. 性质与边界

- 本诊断仅使用已多次查看的 Merge development-test 开发数据，**不构成确认性结论**。
- 不重新判定 Stage 1 Gate S/A/B；原书面状态（merge=`DEVELOPMENT_GO_TO_NEW_HOLDOUT`）保持不变。
- Stage 2 与 router 仍处阻塞，等待用户审批；本诊断不授权采集、训练或 router。
- 运行标记：`training_performed=False`、`inference_performed=False`、`stage1_gate_a_unchanged=True`。

## 2. 输入完整性

- 场景：`merge`；种子顺序 `[11, 22, 33, 44, 55]`；development-test episode 数 = 300。
- split：train=1400 / validation=300 / development_test=300。
- 源数据、Stage 1 配置、response bins、test manifest 与 15 个 test prediction 的 SHA256 已在运行前核验通过，明细见 `input_hashes.json`；运行前后 Stage 1 merge 目录全部文件 hash 守恒。

## 3. 原 Stage 1 复现（点估计，容差 1e-10）

| 量 | 本次计算 | report.json | abs diff | 通过 |
|---|---:|---:|---:|:---:|
| `v_evt_low` | 0.280775424031 | 0.280775424031 | 0.00e+00 | ✓ |
| `v_evt_low_minus_high` | 0.067795793923 | 0.067795793923 | 0.00e+00 | ✓ |
| `spearman_r_eta` | 0.287322123668 | 0.287322123668 | 0.00e+00 | ✓ |
| `trend_log_r` | 0.283038029154 | 0.283038029154 | 1.61e-15 | ✓ |

| 分箱 V_evt | 本次计算 | report.json | abs diff | 通过 |
|---|---:|---:|---:|:---:|
| zero | 0.338865307590 | 0.338865307590 | 0.00e+00 | ✓ |
| positive_low | 0.031953321589 | 0.031953321589 | 0.00e+00 | ✓ |
| positive_mid | -0.024744016359 | -0.024744016359 | 0.00e+00 | ✓ |
| positive_high | 0.212979630108 | 0.212979630108 | 0.00e+00 | ✓ |

原报告 CI 端点按协议原样引用，不要求以新诊断 bootstrap 种子重现。

## 4. D1 zero/positive 存在性

- zero agent-time 占比 = 0.8434
- zero 占 original_low 打乱后缀误差和比例 = 0.8107
- zero 占 original_low 正确后缀误差和比例 = 0.7452
- `V_evt_zero` = 0.3389 [0.2875, 0.3878]
- `V_evt_positive_all` = 0.1413 [0.0984, 0.1806]
- `C_presence` = 0.1976 [0.1313, 0.2617]
- leave-one-seed-out `C_presence` = [0.2096, 0.1911, 0.2065, 0.1797, 0.2014]
- **PRESENCE_DIFFERENCE_SUPPORTED = True**

## 5. D2 仅正响应幅度趋势

- positive-only `Spearman(R, eta_evt)` = 0.0059 [-0.0676, 0.0774]
- positive-only 趋势 `log1p(R)` 系数 = -0.0814 [-0.1669, 0.0032]
- `V_evt_positive_low - V_evt_positive_high` = -0.1810 [-0.3491, -0.0283]
- leave-one-seed-out spearman = [0.0142, 0.0046, -0.0126, 0.0421, -0.0193]
- leave-one-seed-out trend = [-0.0797, -0.0961, -0.0979, -0.0304, -0.0872]
- leave-one-seed-out low-high = [-0.1811, -0.1848, -0.2008, -0.1328, -0.2084]
- **POSITIVE_MAGNITUDE_DECAY_SUPPORTED = False**

## 6. D3 高响应反弹

- `C_high_low` = 0.1810 [0.0268, 0.3499]
- `C_high_mid` = 0.2377 [0.1437, 0.3436]
- `C_rebound` = min 对比 = 0.1810 [0.0270, 0.2930]
- leave-one-seed-out `C_rebound` = [0.1811, 0.1848, 0.2008, 0.1328, 0.2084]
- leave-one-action-type-out `C_rebound`：
  - action 0: C_rebound=0.1558（保留 episode 240，移除 60）
  - action 1: C_rebound=0.1717（保留 episode 240，移除 60）
  - action 2: C_rebound=0.1034（保留 episode 240，移除 60）
  - action 3: C_rebound=0.2707（保留 episode 240，移除 60）
  - action 4: C_rebound=0.1628（保留 episode 240，移除 60）
- leave-one-episode-out：min = 0.1487，反转数 = 0
- 高误差删除 drop_top_1pct：C_rebound = 0.1869（保留 297 episode）
- 高误差删除 drop_top_5pct：C_rebound = 0.1897（保留 285 episode）
- 各正响应箱独立 episode 数 = {'positive_low': 188, 'positive_mid': 169, 'positive_high': 111}
- **ROBUST_HIGH_REBOUND_SUPPORTED = True**

## 7. 完整分箱表

| 响应箱 | V_evt | rho_no | meanL_no | meanL_suffix | meanL_shuffled | agent-time | episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 0.338865 | 0.370779 | 0.116 | 0.073 | 0.111 | 20241 | 300 |
| positive_low | 0.031953 | 0.128836 | 0.432 | 0.377 | 0.389 | 1343 | 188 |
| positive_mid | -0.024744 | -0.004185 | 0.933 | 0.937 | 0.915 | 1249 | 169 |
| positive_high | 0.212980 | 0.236242 | 3.039 | 2.321 | 2.949 | 1167 | 111 |

## 8. 功效分析（单项 + 联合）

- 候选 episode 数：[300, 450, 600, 750, 900, 1200, 1500, 1800, 2400]；Monte Carlo 复制 = 10000；bootstrap 池 = 300；z = 1.959963984540054
- 注意：本功效基于开发数据经验分布的近似，不保证确认性结果，仅用于避免沿用 n=300 的无功效设计。

### 8.1 原 Gate A 四个数值条件单项功效（empirical）

| n | v_evt_low | v_evt_low_minus_high | spearman_r_eta | trend_log_r | joint |
|---|---|---|---|---|---|
| 300 | 1.000 | 0.550 | 1.000 | 1.000 | 0.550 |
| 450 | 1.000 | 0.739 | 1.000 | 1.000 | 0.739 |
| 600 | 1.000 | 0.844 | 1.000 | 1.000 | 0.844 |
| 750 | 1.000 | 0.910 | 1.000 | 1.000 | 0.910 |
| 900 | 1.000 | 0.948 | 1.000 | 1.000 | 0.948 |
| 1200 | 1.000 | 0.986 | 1.000 | 1.000 | 0.986 |
| 1500 | 1.000 | 0.996 | 1.000 | 1.000 | 0.996 |
| 1800 | 1.000 | 0.999 | 1.000 | 1.000 | 0.999 |
| 2400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### 8.2 联合功效（三效应情景）

| n | empirical | 75% | 50% |
|---|---:|---:|---:|
| 300 | 0.550 | 0.297 | 0.032 |
| 450 | 0.739 | 0.492 | 0.093 |
| 600 | 0.844 | 0.641 | 0.167 |
| 750 | 0.910 | 0.757 | 0.270 |
| 900 | 0.948 | 0.845 | 0.384 |
| 1200 | 0.986 | 0.940 | 0.605 |
| 1500 | 0.996 | 0.974 | 0.760 |
| 1800 | 0.999 | 0.994 | 0.867 |
| 2400 | 1.000 | 1.000 | 0.974 |

### 8.3 最弱条件（empirical）

- n=300：`v_evt_low_minus_high` = 0.550
- n=450：`v_evt_low_minus_high` = 0.739
- n=600：`v_evt_low_minus_high` = 0.844
- n=750：`v_evt_low_minus_high` = 0.910
- n=900：`v_evt_low_minus_high` = 0.948
- n=1200：`v_evt_low_minus_high` = 0.986
- n=1500：`v_evt_low_minus_high` = 0.996
- n=1800：`v_evt_low_minus_high` = 0.999
- n=2400：`v_evt_low` = 1.000

### 8.4 样本量建议（按情景）

| 情景 | smallest_n 单项均≥0.90 | smallest_n 联合≥0.80 | smallest_n 联合≥0.90 |
|---|---:|---:|---:|
| empirical | 750 | 600 | 750 |
| 75pct | 1200 | 900 | 1200 |
| 50pct | 2400 | 1800 | 2400 |

## 9. 机械状态

- `PRESENCE_DIFFERENCE_SUPPORTED` = True
- `POSITIVE_MAGNITUDE_DECAY_SUPPORTED` = False
- `ROBUST_HIGH_REBOUND_SUPPORTED` = True
- **唯一 shape_recommendation = `REVISE_H2_TO_NONMONOTONIC_BEFORE_STAGE2`**

## 10. 对 Stage 2 的建议

开发数据显示高响应反弹稳健、正响应内部无单调幅度衰减。建议在新确认集**采集前**正式修订 H2 为**非单调响应依赖**并预注册非单调形状；这不表示 U 形已被确认。是否修订上位计划与 Stage 2 由用户审批决定。

## 11. 禁止主张

- 不确认 U 形，也不确认逐箱单调衰减；
- 不提出碰撞、换道或“事件身份”机制结论；
- 不把 R 与证据价值的关系表述为因果作用；
- 不输出 `GATE_A_CONFIRMED`，不授权 Stage 2 采集或 router；
- 不使用“双刃剑 / dual-edged evidence”表述（Gate B 仍为 NOT_SUPPORTED）。

## 12. 验证

- 完整测试：见本目录 QA / 完整回归；
- 运行前后 Stage 1 封存产物 hash 守恒：运行前后一致（before_after_identical=True）
- `training_performed=False`、`inference_performed=False`。
- 运行耗时 254.6s；Python 3.12.7。
