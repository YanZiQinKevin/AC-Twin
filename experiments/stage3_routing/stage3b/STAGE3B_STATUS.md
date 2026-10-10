# AC-Twin Stage 3B 实施状态

> 计划：`AC-TWIN-RFER-STAGE3B-2026-09-10-V0.1`  
> 当前：**`STAGE3_DEV_NOT_READY_FOR_CONFIRMATION`**（Gate T_dev 已单次裁判，未通过）  
> 前置：`RESPONSE_DEV_QUALIFIED_STAGE3A_PLUS`  
> 终态含义：不进入 crossfit、不准备 Gate C Protocol Lock；回用户评审。

## 恢复锚点

- 冻结决定：沿用 Stage 3 v0.2 的非单调 Dual、Same-information Direct、五组 seed 配对和 Gate T_dev。
- 已完成：Stage 3A Plus 独立复核；历史资格集永久封存。
- 已完成（Task 1）：`stage3b_inputs.py`、5 项专项测试、`input_binding_receipt.json`（receipt=`97634c6d…f04006`）。
- 已完成（Task 2/4 软件）：`code/ac_twin/stage3b_bridge.py`、`code/run_stage3b.py` 驱动（8 子命令）。Stage 3B 专项 25 项测试、全量回归 742 passed（`-W error`）。
- 已执行（用户手动）：screen-gate-hidden → 选 `gate_hidden_dim=32`（val effect-ADE 0.27623<0.27966，receipt 单次不可变）；fit-tier1-five → 四条件×五 pair 共 20 run，签名互异，容量审计 max_rel_diff=0。
- **已执行（用户手动，单次裁判）**：`evaluate-gate-t-dev`（n_resamples=2000, seed 20260910）→ **未通过**，状态 `STAGE3_DEV_NOT_READY_FOR_CONFIRMATION`，report=`gate_t_dev/gate_t_dev_report.json`。
- **下一步：停止。** 按合同不进入 crossfit（report 非 `AWAITING_CROSSFIT`，crossfit 会拒绝）、不准备 Gate C Protocol Lock、不得在失败后扩展搜索空间（禁加 attention/图网络/新损失）。回用户评审。
- 阻断项：Gate C 正式采集/解封未授权且本轮不进入。

## Gate T_dev 裁判结果（开发诊断，非确认结论）

技术不变量全过（reconstruction/ablation/no-oracle-leakage/容量/审计=True）。**失败在实质效应与稳健性**：

| 集 | n | ADE_uniform | ADE_direct | ADE_dual | ADE_oracle | rel_uniform | rel_direct | oracle_gap |
|---|---|---|---|---|---|---|---|---|
| A (merge split2) | 300 | 0.2491 | 0.2523 | 0.2451 | 0.1708 | 1.61% | 2.84% | 0.078 |
| B (replacement) | 400 | 0.2730 | 0.2792 | 0.2721 | 0.1959 | 0.32% | 2.56% | 0.077 |

- 两套均失败于：`dual_vs_direct_point_ge5`、`dual_vs_uniform_point_ge5`（相对改善 <5% 最小实质线）、`ci80_lower_gt0`（rel_uniform 的 80% CI 下界 <0）。
- 通过项：`oracle_beats_uniform`（oracle_gap>0，A/B 均 ~0.08），证明**响应路由存在可利用头间**；leave-one-pair-out 方向不稳定（lopo_stable=False）。
- 解读：oracle 头间明确存在（oracle ADE 远低于 uniform），但**学到的非单调 Dual 路由器只闭合了约 5% 的 oracle gap**（closure≈0.05），且 Direct（同信息、仅少一次显式乘法）与 Uniform（不用响应信息）与 Dual 几乎持平。相对改善低于 5% 实质线且跨 seed-pair 不稳健。
- 性质：A/B 为历史开发诊断集，本结果是**开发阶段不可确认**信号，非论文级否证。

## 设计解释点（供用户复核，可覆盖）

- **oracle-reference 门映射**：v0.2 §4.4 只规定"用真值响应状态作输入、映射在 train/validation 学习"，未冻结函数形式。实现采用确定性单调映射 `w = R/(R+t1)`（`t1`=冻结低/中阈值 0.2484），仅用冻结常量、无学习参数；decoder 仍在 train 学习、validation 选择。定位保持"可达性参考、非公平主张、非理论上界"。
- **router backbone hidden_dim=96**：Stage 3B 有界搜索只变 gate hidden `{32,64}`；router 主干固定为 96（与冻结 response 最终配置及 widest benchmark 一致），`response_feature_dim=32` 由绑定 checkpoint meta 读取。
- **Gate T_dev 执行序**：为省算力，`evaluate-gate-t-dev` 先算 A/B 六项非 cross-fit 条件；六项均过才写 `AWAITING_CROSSFIT` 并提示运行 `crossfit` 补第七项（2-fold C_direct 不反转），任一 `C_direct<=0` 机械停止。七项 AND 最终裁判只执行一次。A=`merge_2000 split 2`、B=`stage3a_replacement_dev` 分别计算、分别裁判，episode 不合并。

## 用户手动长任务命令（PowerShell，于仓库根目录）

```powershell
# 0. 只读预检（可重复，自动通过）
python code/run_stage3b.py preflight

# 1. pair-1 gate hidden 机械筛选 {32,64}（LONG；写单次 receipt）
python code/run_stage3b.py screen-gate-hidden
# 中断/失败后断点续跑：
python code/run_stage3b.py screen-gate-hidden --resume

# 2. Tier 1 四条件 × 五 seed-pair 主矩阵（LONG）
python code/run_stage3b.py fit-tier1-five
python code/run_stage3b.py fit-tier1-five --resume   # 续跑

# 3. Gate T_dev A/B 六项主评估（LONG；单次执行）
python code/run_stage3b.py evaluate-gate-t-dev --n-resamples 2000 --bootstrap-seed 20260910

# 4. 固定 2-fold cross-fit（Direct/Dual），补第七项并给出唯一 Gate T_dev 裁判（LONG；单次）
python code/run_stage3b.py crossfit
python code/run_stage3b.py crossfit --resume          # 续跑
```

执行边界：以上多 seed 路由训练、cross-fit、Gate T_dev 均由用户手动启动；实施模型不代为启动。Gate T_dev/cross-fit 通过后状态为 `ROUTING_DEV_QUALIFIED`；否则 `STAGE3_DEV_NOT_READY_FOR_CONFIRMATION`，不得准备 Gate C Protocol Lock。
