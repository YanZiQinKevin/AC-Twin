# 稳定身份连续性 CPU Smoke QA

日期：2026-09-05  
工程状态：`SMOKE_PASS`

本 QA 只判定短时工程链路，不构成科研性能结论。正式合流/环岛训练和冻结测试均未运行。

## 执行范围

- 场景：merge 开发数据
- 设备：CPU
- 条件：`stable_temporal`、`distance_temporal`、`permuted_temporal`、`set_pool`
- 容量/训练：65k、种子 11、每条件 2 epoch
- 样本限制：训练 64、验证 16、开发测试 16；共核查 96 个实际参与 episode
- 统一输入形状：history `(96, 10, 5, 8)`，suffix `(96, 20, 5, 8)`

## QA 清单

| 检查项 | 证据 | 结果 |
|---|---|---|
| 数据 SHA256 | `EB6122AC62AF20219FC0B27D346740A935587FF32D82E4000127A50A41466E2B`；与 `report.json` 的 source/test dataset hash 一致 | PASS |
| 四条件对象多重集合 | 对实际参与的 96 个 episode、history 与 suffix 逐 episode/帧比较完整对象行多重集合 | PASS |
| 自车 slot | 四条件所有核查帧的 slot 0 与 stable 输入逐元素一致 | PASS |
| 统一 8 维输入 | `presence, is_ego, x, y, vx, vy, cos_heading, sin_heading`；实际张量末维为 8 | PASS |
| 置换复现 | 使用相同配置重新准备输入，四条件 history/suffix 逐元素相同 | PASS |
| 身份时序排列性质 | 已训练 stable 编码器对整段一致非自车置换不变，对逐帧变化置换输出发生变化 | PASS |
| 集合池化排列性质 | 已训练 set 编码器对逐帧非自车置换在浮点容差内不变 | PASS |
| 参数量容差 | temporal 三条件各 65,770（+1.184615%）；set 64,266（-1.129231%）；均不超过 1.5% | PASS |
| NaN/Inf | 源数值数组、归一化产物、预测、ADE/FDE 数组均为有限值 | PASS |
| 纯文本进度 | 首轮四条件均显示 `epoch=1/2`、`epoch=2/2`、train/val、elapsed、eta 和 complete | PASS |
| 断点复用 | 第二次运行同一命令，四条件均输出 `resumed=complete`，无 epoch 行 | PASS |
| 产物稳定性 | 四组模型/日志/逐条件预测/签名元数据 SHA256 与 UTC mtime 均未改变；聚合 report/predictions 被确定性重写但内容 SHA256 不变 | PASS |
| 完整测试 | `python -m pytest tests -q`：124 passed in 22.24s，退出码 0，无 warning 汇总 | PASS |

## 产物摘要

- `report.json` SHA256：`BC17C80111C43903023C7B07AF67C7F615FAE24689A81278BF457359F03AC437`
- `predictions.npz` SHA256：`07E2F04B48E264778B17B839100CF3C942BCC2272F9AB812930FB527F5DFEA86`
- 配置签名：`0dd85db2ecca6a7940c9d29a5755f2fcfaa7571e2a0451c572a17bed6e9e5522`
- 单种子 smoke 的两层 Bootstrap 正确标记为 `unavailable`；原因是至少需要两个训练种子和两个测试 episode。该状态符合 smoke 协议，不能用于科研判定。

## 工程结论

四条件短训练、统一输入、公平性性质、容量约束、数值有限性、纯文本进度、签名保护恢复与产物生成均通过本次短时核查，因此工程状态为 `SMOKE_PASS`。`PASS_IDENTITY_CONTINUITY` 尚未评估。
