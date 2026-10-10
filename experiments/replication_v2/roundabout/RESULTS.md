# 事实证据来源消融实验结果

运行时间：2026-09-04T15:12:40.787762+08:00

## 确认性结论

判定：**STRONG_PASS**。`prefix_10`相对`shuffled_full`，effect-FDE@20改善 39.22%。

episode级配对均值差（shuffled_full - prefix_10）为 2.3080 m，95% CI [1.9155, 2.7293]，n=300。

## 全部条件

| 条件 | effect-ADE@20 (m) | effect-FDE@20 (m) | seed间FDE SD |
|---|---:|---:|---:|
| history_only | 2.5492 | 4.5127 | 0.3796 |
| shuffled_full | 3.3614 | 5.8847 | 0.2512 |
| prefix_10 | 2.0657 | 3.5767 | 0.2065 |

## 解释

- `prefix_10`优于`shuffled_full`才支持同episode事实证据，而不是单纯增加输入长度。
- 所有推断以episode为独立单位；五个训练seed不增加n。

详细对比置信区间、打乱匹配质量和残差模型信息见 `report.json`。
