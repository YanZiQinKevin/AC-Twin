# Table 2 | Information roles and permitted access.

| 信息角色 | 允许用途 | 禁止用途 |
| --- | --- | --- |
| Inference-legal signals | 待评组的预测、基础选择、特征与风险排序 | 读入任何该组真实反事实后果或真值派生量 |
| Training-only supervision | 允许的开发训练角色内拟合风险目标及训练状态 | 将确认标签用于拟合、配置选择或归一化 |
| Frozen model outputs | 在允许的开发 train 上拟合、按允许的 monitor/validation 选择或校准后冻结；待评组推理可消费其输出 | 用确认数据训练、选择或更新上游模型，或在预测输出中夹带待评组真值 |
| Evaluation-only truth | 评价侧的 regret、响应分层、资格及冻结评分后的区间计算 | 将事后评价信息回流至待评组的推理流程 |


来源：唯一中文源稿中的对应表；原数值和口径不变，仅统一编号。
