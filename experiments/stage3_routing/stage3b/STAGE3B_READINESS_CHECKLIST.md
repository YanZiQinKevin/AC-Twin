# AC-Twin Stage 3B 启动前 Readiness Checklist

> 计划：`AC-TWIN-RFER-STAGE3B-2026-09-10-V0.1`。全部勾选后，用户才手动启动长任务。

- [x] Stage 3B 计划已批准（软件准备范围），状态 `APPROVED_FOR_SOFTWARE_PREPARATION`
- [x] Stage 3B 输入绑定 receipt 已冻结并可只读复验（5 checkpoint + 校准来源 + 数据 SHA256）
- [x] feature-only 正式推理 loader 就绪；注入 CF future/R/label 即失败（5 项测试）
- [x] 路由器只用 split 0/1 训练；历史 A/B（merge split 2 / replacement dev）仅作非确认 Gate T_dev 诊断
- [x] 五组 predictor-router 一一配对，禁 best-seed / 5×5（驱动 `_pair_seeds` 强制）
- [x] Uniform/Direct/Dual 三口径参数差 ≤1%（容量审计 max_rel_diff=0）
- [x] gate hidden dim 仅 {32,64}，pair-1 validation 机械选择，确定性并列取 32
- [x] Tier 1 只训练 Uniform/Direct/Dual/Oracle-reference 四条件
- [x] Direct/Dual 唯一结构差异为 w 是否乘 z_dyn（w=1 等价已由 `_technical_checks` 验证）
- [x] 12 ep/1 epoch smoke 通过，状态仅 `ENGINEERING_SMOKE_ONLY`
- [x] router 预算复核落在已批准 15 h / [7.5,30] h ±25% 内（12.41 h → REUSE_APPROVED_BUDGET）
- [x] 全量回归 `pytest tests/ -W error`：**742 passed / 0 failed**
- [ ] 用户已确认本 checklist，准备手动启动 screen-gate-hidden

> 说明：`evaluate-gate-t-dev` 与 `crossfit` 为单次执行（已有 report 即拒绝）。`crossfit` 任一 A/B `C_direct<=0` 机械停止于 `STAGE3_DEV_NOT_READY_FOR_CONFIRMATION`，不得扩 fold/架构。
