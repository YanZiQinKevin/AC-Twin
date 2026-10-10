# Stage 2B `intersection-v0` v0.4 — HANDOFF / 恢复状态

> 计划：`AC-TWIN-RFER-STAGE2B-INTERSECTION-2026-09-15-V0.4`  
> 当前状态：`STAGE2B_CONFIRMATION_UNDERPOWERED / TERMINAL`  
> v0.3终态：保留 `INCONCLUSIVE_INTERSECTION_RESPONSE_SUPPORT`，不可改写

## 冻结决定

- 固定 profile B：`initial_vehicle_count=15`。
- v0.3 pilot B只作profile选择证据，不进入训练、validation或confirm。
- 删除旧的`positive_episodes>=300`资格项；其余支持与技术条件保持。
- development使用6,000个全新episode和collection seed `2026091609`。
- 不新增profile、不启用动态入流、不修改Gate A2B。
- 用户已授权本计划内长任务自动执行。

## 当前产物

- v0.3 pilot A/B、selection与终态记录完整保留。
- v0.4实施计划已批准。
- v0.4恢复回执已冻结：`profile_selection_v04.json`，profile B。
- v0.4 development manifest与60,000候选seed schedule已冻结，历史与v0.3角色零重叠验证通过。
- Stage 2B专项测试：68 passed。
- 首次长任务前全量回归：970 passed；5项既有Stage 2旧锁测试因其冻结schedule与后来活数据出现1个随机seed重叠而失败，与v0.4变更无关。
- 6,000 episode development采集完成：8,458 attempts；六层各1,000，逐层800 train/200 validation。
- 数据总计4,800 train/1,200 validation；prefix/identity最大绝对差均为0；最大道路/初始身份数均为12，未超过容量20。
- manifest与v0.4计划编号一致，NPZ SHA256与collection receipt一致（`ae34d0f6...b8136d270`）。
- C0三条件×五seed训练完成；Gate S=`GATE_S_PASS`，因此冻结C0，不触发C1。
- Gate S关键结果：Spearman=`0.3847`，相对action-duration基线的MAE改善=`+0.0494`，相对action-duration+geometry基线的MAE改善=`+0.0512`；五seed与三个合格响应箱的方向要求均通过。
- 功效分析已按预注册候选`{1200,1800,2400,3600}`和10,000次Monte Carlo完成，create-once报告SHA256=`847851cf6ee13347c75d06a50cf508fd43ab4d693ff9bb7b02b6388324da73e8`。
- 三个positive箱在全部候选n下的支持概率均为1.0；停止原因不是箱稀疏。
- validation效应为：`C_presence=-0.2568`、`V_zero=-0.1817`、`C_high_low=+0.0557`、`C_high_mid=+0.0199`、`V_high=+0.0807`。由于前两项方向与Gate A2B相反，75%收缩场景在全部候选n下联合功效均为0。
- 独立重算确认train-only阈值、四箱统计与上述对比一致；未发现预测绑定、分箱或统计接线错误。
- confirm lock未创建，confirm数据未采集，冻结推理与解封均未发生。

## 下一动作

v0.4无后续执行动作。任何改变Gate A2B、场景分布、模型或样本上限的尝试均须另立计划；不得在v0.4内挽救。

## 终态与边界

终态为`STAGE2B_CONFIRMATION_UNDERPOWERED`。这是预注册状态机对“最大候选n仍不能达到联合功效”的名称；本次实际绑定原因是development效应方向不支持，而不是样本或响应箱数量不足。该结果是开发阶段的跨场景不迁移边界，不是intersection上的确认性否证，也不改变Stage 2 Merge已经确认的结论。
