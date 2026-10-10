# Fig. 6 Legend

**Fig. 6 | Robustness checks and generator-mismatch qualification.** **a,** Merge improvements over evaluated scores: filled circles denote AURC-regret and open squares Regret@80. Three standard baselines carry paired-bootstrap 95% intervals; COMBINED_TOP2_MAX and CONSTANT_RANDOM have point estimates only. Longer bars indicate weaker baselines. **b,** Five audit-head initializations evaluated on the same groups; the red dashed line marks the ensemble. **c,** Intervention-time and pre-intervention-distance marginal strata, and deletion of the highest-regret 1% or 5%. These checks describe direction stability rather than independent confirmations. **d,** OOD eligibility and fixed priority, assessed without audit-effect access; JOINT_OOD was selected. Three of six absolute standardized mean differences exceed 0.5. The domain classifier uses 5,000 ID and 2,000 OOD groups and is excluded from scoring. Its AUC interval is 0.980–0.986. Panels a–c use 2,000 Merge confirmation groups. Performance under mismatch is in Fig. 4b; complete results are in Table S3.

---

## 中文备忘（不入正文）

**范围串号警告（已修一次，勿再犯）：** 三个范围的"最窄余量"不同 —— Merge 是 **26.0%**、
JOINT_OOD 是 **18.6%**、MetaDrive 是 **10.1%**。本图 (a)–(c) 全部来自 Merge，**只能写 26.0%**；
Table 6 里那个 10.1% 属于 MetaDrive，自带 (MetaDrive/Waymo, Regret@80 against `SEED_VOTE_ENTROPY`)
的范围限定，不得搬到 Merge 面板上。

- (a) 两条参考评分**没有裁判区间**，这一句不能省，否则等于给它们补了 CI；轴标签同理写明
  "for adjudicated rows"。
- (a) 的横轴是"相对每条基线的提升"，所以**条越长 = 该基线越弱**；图内已放
  `longer bar = weaker baseline`，正文若引用 (a) 不得读成"最重要的胜利在最上面"。
- (a) 的"最强"必须带指标限定：AURC 最低是 `MARGIN_ONLY`，Regret@80 最低是 `SEED_VOTE_ENTROPY`。
- (c) 只能写"方向稳定"，必须指向 Table S3；不得把分层当独立 confirm。
- (d) 域分类器只证明失配，不进风险打分；不得写"对任意 OOD 鲁棒"；`BEHAVIOR_OOD` 未通过资格是
  资格筛选的结果，不是"审计失败"。
- 红虚线在图中有两义：在 (b) 是**正式集成裁判值**、在 (d) 是**冻结失配阈值**；(d) 资格矩阵里的
  "no"已刻意改为中性斜体而非红色，避免与这两条参考线混淆。
- **统计相关性别误用：** R2 确认的 bootstrap 对每次重采样是用**同一个重采样索引向量**重算该
  draw 的全部指标，因此 (a) 各行、(b) 各 seed 的区间彼此高度相关。**不得**用"两个区间不重叠"
  推断两行差异显著；行间比较需要专门的配对差区间。
- 本图不承载 Fig. 2 的范围主结果；跨范围结论只回指 Fig. 2 与 Table 6。
