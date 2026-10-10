# Fig. 6 Legend

**Fig. 6 | Robustness checks and generator-mismatch qualification.** **a,** Merge improvements over evaluated scores: filled circles denote AURC-regret and open squares Regret@80. Three standard baselines carry paired-bootstrap 95% intervals; COMBINED_TOP2_MAX and CONSTANT_RANDOM have point estimates only. Longer bars indicate weaker baselines. **b,** Five audit-head initializations evaluated on the same groups; the red dashed line marks the ensemble. **c,** Intervention-time and pre-intervention-distance marginal strata, and deletion of the highest-regret 1% or 5%. These checks describe direction stability rather than independent confirmations. **d,** OOD eligibility and fixed priority, assessed without audit-effect access; JOINT_OOD was selected. Three of six absolute standardized mean differences exceed 0.5. The domain classifier uses 5,000 ID and 2,000 OOD groups and is excluded from scoring. Its AUC interval is 0.980–0.986. Panels a–c use 2,000 Merge confirmation groups. Performance under mismatch is in Fig. 4b; complete results are in Table S3.

---

## Interpretation notes (outside the manuscript caption)

**Do not mix scopes:** narrowest margins differ: Merge **26.0%**,
JOINT_OOD **18.6%**, MetaDrive **10.1%**. Panels (a)–(c) use Merge and **only 26.0% applies**.
The Table 6 value 10.1% belongs to MetaDrive (MetaDrive/Waymo, Regret@80 against `SEED_VOTE_ENTROPY`),
and must not be assigned to Merge panels.

- (a) The two reference scores **have no adjudication CIs**; retain that qualification and the corresponding axis-label wording.
  "for adjudicated rows"。
- (a) The horizontal axis is improvement relative to each baseline, so **longer bars indicate weaker baselines**. The figure states
  `longer bar = weaker baseline`; do not interpret the top bar as the most important victory.
- Specify the endpoint when calling a baseline strongest: `MARGIN_ONLY` for AURC; `SEED_VOTE_ENTROPY` for Regret@80.
- (c) Supports directional stability only, with full values in Table S3; strata are not independent confirmations.
- (d) Domain classification establishes mismatch only and does not enter risk scoring; do not claim arbitrary-OOD robustness. BEHAVIOR_OOD failed
  qualification, which is not an audit failure.
- Red dashed lines denote the adjudicated ensemble effect in (b) and the frozen mismatch threshold in (d). In the qualification matrix,
  `no` is neutral italic text, avoiding confusion with the reference-line color.
- **Statistical dependence:** each R2 confirmation bootstrap draw uses **the same resampling index vector** for
  all endpoints, so row/seed intervals in (a)/(b) are highly correlated. **Do not** infer a significant between-row difference
  from nonoverlapping intervals; use a dedicated paired difference interval.
- Overall cross-scope effects belong in the three-scope figure (current Fig. 4) and Table 6.
