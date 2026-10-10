# Stage 4C-R2 READINESS_CHECKLIST

> 计划：`AC-TWIN-RFER-STAGE4C-R2-DECISION-RISK-SCALING-2026-09-16-V0.2`
> 更新：2026-09-16（Task 1 完成时）
> 规则：readiness 未全过时，runner 不得进入 lock-development。

- [x] 计划获批（v0.2，2026-09-16 用户明确批准第 16 节全部 9 项 + v0.2 种子修订）
- [x] v0.1 development entropy `2026092701` 触发第 5.2 节硬停止（4 个种子与全局
  accepted+reserved 清单重叠），按计划形成 v0.2，未原地替换冲突 seed
- [x] 新 development entropy `2026092880` / 60,000 候选 schedule 经全局
  accepted+reserved 清单（含 Stage 2 冻结域、Stage 4C 未用 confirm、Stage 4C Plus
  预留 dev/confirm、全部现存 `.npy` seed schedule）审计**零重叠**（测试断言，
  清单规模 >200,000）
- [x] 协议模块 `code/ac_twin/stage4cr2_protocol.py` 冻结：十动作/后果函数/regret
  与 Stage 4C bitwise 同构（引用复用，未重定义）；特征合同引用
  `stage4cp_features.py`（9 token×34 维+9 全局）；固定
  `SAME_INFORMATION_DIRECT(width=43, lambda_cls=0.25, dropout=0.10)`，
  参数量逐层复算精确等于 15,740
- [x] 角色 segment 固定：train `[0,24000)` / validation `[24000,32000)` /
  screen `[32000,60000)`，无重分配 API，confirm 角色不存在
- [x] 禁读守卫三层（规范化路径 / 逻辑角色 / SHA256）：旧 Stage 4C screen 四产物、
  Stage 4C Plus phase0 三产物、旧 500-group selection、Stage 4B/Stage 2 confirm
  真值全部拒绝（77 项协议测试证明）
- [x] config `stage4cr2_config.json` 通过 `validate_config`；任何冻结值漂移被拒绝
- [x] development lock 接口为 create-once（二次调用拒绝）且
  `verify_development_lock` 可检测锁体与冻结来源篡改（尚未烧正式锁——
  正式锁在 lock-development 阶段由 runner 触发）
- [x] 阻断测试 `python -m pytest -q -W error tests/test_stage4cr2_protocol.py`
  全部通过，零 warning
- [ ] Task 2 collector + 冻结推理（`collect_stage4cr2_merge_multiaction.py`、
  `stage4cr2_inference.py`）——未开始
- [ ] 正式 development lock（需全部 R2 代码就绪后由 runner 烧录）——未开始
