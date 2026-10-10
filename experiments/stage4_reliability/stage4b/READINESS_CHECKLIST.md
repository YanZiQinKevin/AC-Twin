# Stage 4B READINESS_CHECKLIST

> 计划：`AC-TWIN-RFER-STAGE4B-RISK-CONFIRM-2026-09-15-V0.1`
> 更新：2026-09-15T17:38:45

- [x] 功效报告 create-once 已写；75% 收缩为唯一阻断裁判
- [x] 机械选定 `selected_n = 2000`（每层 200）
- [x] 候选 schedule 零重叠（历史 seed 数 27224）
- [x] Stage 4A 冻结链（recovery lock / selection / finalist / geometry / scaler / risk_dataset）hash 全部匹配
- [x] 冻结阈值 `top_error_threshold = 0.41601440122246675`（不重算）

## n=5000 三收缩场景联合功效（并列）

- 0.50 收缩：0.7539
- 0.75 收缩：0.9958
- 1.00 收缩：1.0000

50% 收缩仅作选择性乐观风险提示，不设人工暂停或二次批准点。
