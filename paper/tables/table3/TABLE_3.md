# Table 3 | Data roles and sample sizes for the reported evidence.

| 证据范围 | 独立计数单位 | Train 总量 | 独立 monitor | Validation | 开发评测 / screen / 资格 pilot | Confirm |
| --- | --- | ---: | --- | --- | --- | ---: |
| Merge 事实证据价值 | episode | 1,400 | 未另设，使用 validation | 300（含 checkpoint 选择） | 300 个既有开发评测 episode | 1,200 |
| Merge 十动作 Direct audit | 完整十动作组 | 5,000 | train 内 500；参数拟合用其余 4,500 | 1,500 | 8,000 个一次性 development-screen 组 | 2,000 |
| `JOINT_OOD` | 完整十动作组 | 不新增，复用 Merge | 不新增，复用 Merge | 不新增，复用 Merge | 三种候选配置各 300 组资格 pilot | 2,000 |
| MetaDrive 十动作 Direct audit | 完整十动作组 | 5,000 | 未另设，使用 validation | 1,500（含 early stopping、配置选择） | 1,000 个最终路线 development-screen 组 | 2,000 |
| 已发表方法适配比较 | 完整十动作组 | 5,000，复用 Merge 开发 train | 适配方法不设独立 monitor；Direct 沿用其冻结划分 | 1,500，复用 Merge 开发 validation | 不新增 screen | 0（独立确认集未创建） |


来源：唯一中文源稿中的对应表；原数值和口径不变，仅统一编号。
