# 新冻结测试集采集说明

状态：`COLLECTED_AND_QA_PASSED`（2026-09-06）。合流、环岛的新测试集均已完成采集并通过质量核查；详见[HOLDOUT_QA_2026-09-06.md](HOLDOUT_QA_2026-09-06.md)。这只表示数据可用于正式实验，不代表任何性能假设已经通过。

## 冻结配置

| 场景 | 采集种子 | 目标 pairs | 输出文件 |
| --- | ---: | ---: | --- |
| 合流 `merge-v0` | `20261011` | 1,000 | `data/identity_continuity_stage/merge_holdout_1000/object_twinworld_holdout_1000.npz` |
| 环岛 `roundabout-v0` | `20261013` | 1,000 | `data/identity_continuity_stage/roundabout_holdout_1000/object_twinworld_holdout_1000.npz` |

两份配置均固定 20 步未来、10 步历史、`t_star=10..15`、干预持续时间 `[1, 3]`、25 辆仿真车/5 辆观测车和对象中心输出。每接纳 100 对写入 checkpoint，每接纳 20 对显示进度。首个 checkpoint 只会在第 100 个已接纳 pair 后产生；因此若进程在此前退出，只能从头开始。后续恢复通常从最近 checkpoint 继续、最多重算 99 个已接纳 pair 及该区间的 attempts；若中断恰发生在 checkpoint 写入期间，可能重算该批的 100 个 accepted pair。

## 人工启动与恢复

在项目根目录手动运行所需场景的命令：

```powershell
python code/collect_pilot.py --config config/identity_holdout_merge_collect.json
python code/collect_pilot.py --config config/identity_holdout_roundabout_collect.json
```

终端每 20 个已接纳 pair 会显示 `accepted/1000`、`attempts`、`prefix_terminated`、两条未来分支的短未来计数、`elapsed` 与粗略 `ETA`；完整跳过原因分类请以最终 `.stats.json` 为准。恢复后在新增 accepted pair 的测时积累前，ETA 可能偏乐观。电脑正常休眠时，原进程通常会在唤醒后继续；只有进程退出或被终止时才需再次运行命令。checkpoint 会保留在目标 `.npz` 同目录；默认再次运行**同一条命令**会自动恢复。显式使用 `--resume` 时，若既没有已验证的完整输出、也没有 partial checkpoint，命令会明确报错，绝不会静默开始一次新采集。

不要同时运行同一场景的两个采集进程，也不要在采集期间改动对应 JSON；两者都会破坏可恢复性或冻结性。

## 输出与 QA 边界

成功完成后，每个输出 `.npz` 旁会有同名 `.stats.json`。最终 `.npz` 与 stats 会先在同目录临时文件中写入并验证可读，再先原子替换 stats、最后原子替换 `.npz`；最终 `.npz` 出现才是完成提交点。启动时会验证既有 `.npz` 的必需字段、样本数、内嵌完整配置与当前完整配置的规范化严格匹配，以及 `.stats.json` 与 NPZ 内嵌统计一致；未通过验证的既有输出会明确报错，绝不被误报为完成或覆盖。统计会单独保存 `prefix_terminated`、`factual_future_short` 与 `counterfactual_future_short`，并保留其他跳过原因。只有事实和反事实分支都具有完整 20 步未来的 pair 才会被接纳；提前终止只用于描述采样规模与质量，不能纳入本阶段的性能结论。

采集完成后才可进行 QA：确认输出 pair 数为 1,000、对象中心字段存在、stats 的终止原因计数已写出，并记录配置与数据哈希。QA 通过也只表示数据可用，不代表身份连续性假设、跨场景泛化或任何性能结论已成立。
