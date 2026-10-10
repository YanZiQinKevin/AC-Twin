# 当前最终源码独立 CPU Smoke QA

日期：2026-09-06  
工程状态：`FINAL_SMOKE_PASS`

本状态只表示当前最终源码的短时工程链路、签名保护和恢复路径通过，不是科研性能判定；本次无科研结论。报告内的 `decision=SMOKE_ONLY` 同样只是边界标记，不是 `PASS_IDENTITY_CONTINUITY`。

## 冻结范围

- 独立配置：`config/identity_continuity_final_smoke.json`
- 独立输出：`experiments/identity_continuity_stage/final_smoke`
- 旧开发测试：merge 对象中心数据，SHA256 `EB6122AC62AF20219FC0B27D346740A935587FF32D82E4000127A50A41466E2B`
- 四条件：`stable_temporal`、`distance_temporal`、`permuted_temporal`、`set_pool`
- 65k、训练种子 11、每条件 2 epoch、训练/验证/开发测试限制 64/16/16、CPU
- 配置签名：`1b3a1fa405e3c17ec6797483e6d3736c998c6bf688e66c4f78bfa4c60d7b0145`

## QA 结果

| 检查 | 证据 | 结果 |
|---|---|---|
| 当前源码全套测试 | `python -m pytest tests -q -W error`：181 passed in 40.64s，退出码 0 | PASS |
| 首轮真实训练 | 四条件均显示 epoch 1/2、2/2 与 complete，退出码 0 | PASS |
| 第二轮恢复 | 完全相同命令；四条件均显示 `resumed=complete`，无 `epoch=` 行，退出码 0 | PASS |
| 逐运行产物 | 16 个 `.pt`/训练日志/预测/metadata 的 SHA256 与 UTC mtime 两轮间全部不变 | PASS |
| 聚合产物 | `normalization.npz`、`predictions.npz`、`report.json` 第二轮确定性重写，SHA256 分别与首轮一致 | PASS |
| 签名 | 四个 metadata 签名与当前最终源码/配置匹配，状态 complete、epochs_run=2 | PASS |
| 数值与容量 | 预测及 ADE/FDE 均有限；三 temporal 条件 65,770 参数，set 64,266 参数，均在 1.5% 容差内 | PASS |
| 评估边界 | `evaluation_mode=development_test`、`decision=SMOKE_ONLY`；未使用冻结正式测试 | PASS |
| 两层 CI | 单一训练种子不足以进行层次 Bootstrap，三个比较均正确标为 `unavailable` | PASS |

## 关键哈希

- `report.json`：`9596C17391E5D38D02D46286BBFCAD750D4326853764D2CF4E673BC541375CC2`
- `predictions.npz`：`07E2F04B48E264778B17B839100CF3C942BCC2272F9AB812930FB527F5DFEA86`
- `normalization.npz`：`3E898A9A942AADB441F45261BDAD94C710F0AC6883191F20897D50539CE025B0`

## 结论边界

`FINAL_SMOKE_PASS` 仅证明当前最终源码在旧 merge 开发数据上的四条件单 seed、2 epoch CPU 短运行及恢复可用。它不估计训练随机性，不提供可用的层次置信区间，不替代两场景、两容量、五种子和新冻结测试集的正式实验，因此不得形成任何身份连续性科研结论。
