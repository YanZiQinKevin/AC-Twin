# Stage 1 验尸实验 — CPU Smoke QA

> 状态：**STAGE1_SMOKE_PASS**
> 日期：2026-09-07
> 配置：`config/stage1_evidence_value_smoke.json`
> 边界：本 smoke 只验证工程数据流、训练、产物与恢复机制，**不形成任何科研结论**，不判定 Gate S/A/B，也不触发 merge/roundabout 正式五种子长训练。

## 1. 执行的命令

```powershell
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_smoke.json --device cpu
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_smoke.json --device cpu --resume
```

## 2. 运行边界（冻结）

- 三条件 `no_suffix / uniform_suffix / shuffled_suffix` × 单种子 11 × 2 epoch；
- 样本限制 train/validation/test = 64/16/16 episodes；
- 模型结构、参数量与正式配置完全一致，仅缩短样本数与 epoch；
- smoke 报告固定为 `SMOKE_ONLY`，未打印 development-test 指标，未输出 Gate A 结论。

## 3. 检查结果（全部通过）

| 检查项 | 结果 |
|---|---|
| 三条件各 1 seed × 2 epoch 完成 | PASS |
| 预测形状 `B×T×N×2` = (16,20,5,2) | PASS |
| 预测全部有限（无 NaN/Inf） | PASS |
| M0 自训练起使用空后缀（suffix 特征与 mask 全零） | PASS（单元测试覆盖） |
| 逐车输出非全零/非常数/非单车模板 | PASS |
| validation/test derangement manifest QA | PASS（self_match=0, duplicate_donor=0, cross_split=0, same_seed=0） |
| 训练日志、权重、val/test 预测、run metadata、签名齐全 | PASS |
| 归一化 `normalization.npz`、响应分箱 `response_bins.json` 写入 | PASS |
| 首轮 `--resume` 不重训，权重与逐运行预测 hash 逐项不变 | PASS |
| 报告 `decision=SMOKE_ONLY` | PASS |

## 4. 产物清单

每个 condition×seed：`<condition>_seed11.pt`、`_training.json`、`_val_prediction.npy`、`_test_prediction.npy`、`_run.json`。
共享产物：`normalization.npz`、`response_bins.json`、`validation_manifest.json`、`test_manifest.json`、`report.json`。

## 5. 明确边界

- Gate S 尚未正式判定（需正式五种子训练后评估）；
- Gate A/B 无任何结论；
- 未实现 router / response head / 动态图 / attention / SIR / 参数溯因；
- 正式长训练由用户手动启动，命令见实施计划第 14 节与本目录 `../PROTOCOL.md`。
