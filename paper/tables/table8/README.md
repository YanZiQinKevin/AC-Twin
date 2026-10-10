# Table 8｜监督对照与特征消融

正文表格编号为Table 8，不使用内部评审问题编号。

| 文件 | 说明 |
| --- | --- |
| [TABLE_8.md](TABLE_8.md) | 中英文表体、caption、表注、统计口径及主张边界 |
| [Table_8_Supervised_Controls.xlsx](Table_8_Supervised_Controls.xlsx) | 可编辑Excel，Table 8为三线表，Data为未四舍五入的十个效应记录 |
| [table8_data.csv](table8_data.csv) | 机读效应与99%区间，数值单位为无量纲比例 |
| [table8_source.json](table8_source.json) | 元数据、冻结来源哈希、输入维度、完整数值和显示值 |
| build_table8.mjs | 只读冻结CSV与正文，重建本包并核对五项对照 |

训练/评价细节见[补充方法SM11](../../manuscript/SUPPLEMENTARY_METHODS.md)。当前正文仍以[唯一中文工作稿](../../manuscript/AC-Twin_论文初稿_中文.md)为权威；本目录为其表格资产包，不维护第二份正文。

本表属于开发性证据。主要AURC和次Regret@80各五对比独立做Bonferroni校正；所有十个99%区间完整保留，包括跨零与负向点估计。未使用未经校正95%区间替代。

重建：从项目根目录运行`node paper/tables/table8/build_table8.mjs`，使用已配置的工作区表格依赖；无需训练或读取确认逐组资源。
