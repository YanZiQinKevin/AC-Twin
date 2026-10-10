# 事实证据来源消融实验结果

运行时间：2026-09-04T14:54:16.101769+08:00

## 确认性结论

判定：**STRONG_PASS**。`prefix_10`相对`shuffled_full`，effect-FDE@20改善 32.31%。

episode级配对均值差（shuffled_full - prefix_10）为 1.0775 m，95% CI [0.8509, 1.3078]，n=300。

## 全部条件

| 条件 | effect-ADE@20 (m) | effect-FDE@20 (m) | seed间FDE SD |
|---|---:|---:|---:|
| history_only | 1.8306 | 3.2210 | 0.1746 |
| shuffled_full | 1.9250 | 3.3350 | 0.1906 |
| prefix_10 | 1.2861 | 2.2575 | 0.1002 |

## 解释

- `prefix_10`优于`shuffled_full`才支持同episode事实证据，而不是单纯增加输入长度。
- 所有推断以episode为独立单位；五个训练seed不增加n。

详细对比置信区间、打乱匹配质量和残差模型信息见 `report.json`。
