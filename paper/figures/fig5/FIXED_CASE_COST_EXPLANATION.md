# Fig. 5 固定案例的成本解释

> 2026-10-04：只解释作者固定 PNG 中已选定的两个案例；未重选案例、重画 PNG、重新评分或裁判。以下为冻结轨迹及固定成本函数的描述性分解，不是新增总体统计结论。

| 面板 / 组 | 分支 | 固定动作 | 纵向进度（m） | C_int | C_eff | J_true |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| c / 550 | 所选 | FASTER × 3 | 2.0131 | 0.657523 | 0.999994 | 0.726017 |
| c / 550 | oracle | LEFT × 1 | 3.9419 | 0.346960 | 0 | 0.277568 |
| d / 1821 | 所选 | FASTER × 3 | 4.8850 | 0.000487 | 1.000000（舍入） | 0.200389 |
| d / 1821 | oracle | RIGHT × 3 | 22.1717 | 0.000007 | 0 | 0.000005 |

案例 c 的所选分支同时具有较高接近度成本和较低纵向进度，二者共同构成 regret 0.448449。所选分支的接近度最大值出现在预测序列第16帧、周车槽位4，对应纵向/横向间隔约4.5755/0.0671 m；oracle 分支在第18帧、槽位4达到其最大值，对应约3.9271/2.4497 m。该成本按各向异性尺度计算，并非仅看欧氏最近距离。

案例 d 的 regret 0.200384 主要来自效率成本：所选分支的纵向进度为4.8850 m，oracle 为22.1717 m，接近度项在两个分支均很小。因此这个 false-negative 是所定义决策成本的漏检，不能直接称为危险或碰撞的漏检。

图中星号仍表示两自车分支的最大轨迹分离时刻（两案例均第20帧），不是接近度成本最大时刻。本说明不向已确认图稿添加标记。

来源：`Fig5_render_receipt.json#frozen_cases` 固定 group_index；`experiments/stage4_reliability/stage4d_d2f/roles/confirm/groups.npz` 的已冻结轨迹；同阶段 `confirm/confirm_group_quantities.npz` 的动作索引与 J_true；固定成本函数 `code/ac_twin/stage4c_objective.py`。只对550与1821两组调用纯成本分解，复现 J_true 的十动作最大误差均为0.0，无 bootstrap、模型推理或新裁判。复核入口：`paper/manuscript/revisions/2026-10-04_two_level_restructure/inspect_fixed_cases.py`。
