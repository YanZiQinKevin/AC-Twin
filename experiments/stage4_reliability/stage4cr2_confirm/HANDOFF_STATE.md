# Stage 4C-R2 独立确认 HANDOFF_STATE

> 计划：`AC-TWIN-RFER-STAGE4CR2-CONFIRM-2026-09-17-V0.1`
> 本文件是恢复锚点，由 `code/run_stage4cr2_confirm.py` 的 handoff 阶段重写；骨架由 Task 4 创建。
> 更新时间：2026-09-20T07:28:29

## 当前状态（终态）

- 终态：**STAGE4CR2_DECISION_RISK_CONFIRMED**（§8 合法终态）
- 含义与论文边界：见 `reports/STAGE4CR2_CONFIRMATORY_DECISION.md`（裁判后由统计层写入；成功句式仅当终态为 STAGE4CR2_DECISION_RISK_CONFIRMED 时合法）。

## 一次性/create-once 产物状态

| 产物 | 状态 | 校验 |
|---|---|---|
| protocol lock | 存在 | sha256 `d355653b56d67d19…` |
| collection receipt | 存在 | sha256 `a562c841b6650129…` |
| legal inference inputs | 存在 | 3.7 MB |
| sealed confirmation archive (score side never reads it) | 存在 | 10.0 MB |
| predictions | 存在 | 312.5 MB |
| risk scores | 存在 | 0.2 MB |
| inference receipt | 存在 | sha256 `d8cbdad637bb5177…` |
| determinism receipt | 存在 | sha256 `7e2ddb9fef1cf88e…` |
| score freeze receipt | 存在 | sha256 `f82aa43eaf703600…` |
| confirmation_used.lock | 存在 | sha256 `f6f4f23cc9c70869…` |
| confirm group quantities | 存在 | 0.5 MB |
| quantities receipt | 存在 | sha256 `0b24bf33a671d088…` |
| confirmatory report | 存在 | sha256 `46798b2d3dac5090…` |
| confirmatory decision | 存在 | 0.0 MB |

## 文档同步清单（终态确定后由 controller/人工更新；runner 只打印，不自动改这些文件）

- [x] AGENTS.md
- [x] 00_READ_FIRST_AC_TWIN.md
- [x] 01_AC_TWIN_DOCUMENT_STATUS_INDEX.md
- [x] plans/2026-09-06_AC-Twin反应式事实证据价值与路由研究计划_v1.0.md (master plan v1.13)
- [x] logs/ 当日科研日志 (logs/2026-09-20_AC-Twin科研日志.md)

## 终态复核（2026-09-20 整链最终评审，CLEAN）

- 整链 21 个文件 hash 独立复算全部匹配；主终点数值独立重算逐一吻合；§7 报告集
  清单无缺失无多余。
- Erratum（已登记于 logs/2026-09-20_AC-Twin科研日志.md）：`reports/Table_C1_gates.csv`
  列对齐缺陷（无数值点估计/CI 列内容）；数值完整存在于 hash 绑定的 report JSON 与
  Table_C2/C3，终态不受影响；按 §8 不改码、不手改 CSV。
- `READINESS_CHECKLIST.md` §1 测试计数已更正为 210（文档性更正）。

## 下一动作

- 无 runner 内动作；成功也不自动启动下一阶段。文档同步与 erratum 登记完成，授权结束。

## Blocker

- 无。
