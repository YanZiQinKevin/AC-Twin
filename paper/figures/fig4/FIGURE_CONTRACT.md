# Fig. 4 Figure Contract

## Figure-level claim

The confirmed Merge audit effect is not an artefact of the baseline it was
adjudicated against, of a single audit-head seed, or of one marginal stratum;
and the additional JOINT_OOD confirmation comes from a prospectively specified
generator-mismatch profile that was selected without reading any audit effect.
Panel (d) shows that qualification, not the mismatch effect itself: the effect
is reported in Fig. 3b and Table 6, and the figure legend points there.

## Layout (v1)

Canvas 180 × 126 mm (100% of a two-column page width), 2 × 2 panels:

| Panel | Content | Data |
| --- | --- | --- |
| (a) Baseline ladder (hero) | relative improvement over every evaluated score, two metrics, five rows; the three adjudicated standard baselines carry intervals, the two reference scores are point-only | frozen `Table_C2_baselines.csv` |
| (b) Five-seed paired effects | two side-by-side forests, five seeds, ensemble reference line | frozen `Table_C3_seeds_strata_trim.csv` |
| (c) Strata and trim | four marginal strata + two regret trims, AURC-regret only | frozen `Table_C3_seeds_strata_trim.csv` |
| (d) Pre-registered mismatch qualification | three-profile eligibility/priority matrix + absolute SMD against the 0.5 threshold + domain AUC | frozen `ood_profile_selection.json`, `decision.domain_shift` |

The two metric series in (a) are offset within each row (AURC-regret filled
circle, Regret@80 open square) so that one row carries one baseline; the two
reference scores are drawn as faded diamonds with **no interval**.

## Visual invariants

- Six supplied hues, one meaning each, shared with Figs. 1–3. Two collisions the
  first draft had are resolved: the (d) eligibility matrix marks the selected
  profile with a **square**, never a circle, because the SMD dots below it are
  circles; and `BEHAVIOR_OOD`'s `no` is **muted italic**, not red, so that red
  keeps its single meaning of "reference line" (ensemble value in (b), frozen
  shift threshold in (d)).
- Smallest in-figure type **7.0 pt** (`ensemble` markers, `threshold 0.5` tag,
  panel subtitles, SMD row labels); row labels 7.0–7.4 pt; axis labels 7.2 pt;
  panel letters 9.6 pt bold; panel titles 8.8 pt bold.
  **Recorded density exception (2026-09-28):** blueprint §9.2 sets a 7.5 pt floor
  for new figures. Four dense panels (5–6 forest rows each, plus a three-row
  qualification matrix and a six-row SMD plot) do not fit 7.5 pt in 126 mm, so
  this figure is registered at a 7.0 pt floor. The earlier draft ran down to
  6.2 pt; those tags were raised rather than exempted.
- All four canvas margins measure ≥ 2.1 mm of clear paper (left 2.16, right 2.50,
  top 4.91, bottom 5.63 mm, measured from the rendered PNG), so a substitute
  font cannot push the outermost labels off the page.
- The two (b) sub-axes are the same width (0.128 each) with the same ticks
  (0.0 / 0.2 / 0.4), so one data unit has one length in both and the two metrics
  can be compared row by row.
- Zero reference line in every forest; the two dashed separators split the
  adjudicated rows from the reference rows in (a) and the strata from the trims
  in (c).
- The horizontal axis of (a) is the improvement over each score, so a longer bar
  means a weaker baseline; the panel carries that note in-figure.
- The red dashed line means "adjudicated ensemble value" in (b) and "frozen
  shift threshold" in (d); the figure legend must say so.
- The two point-only rows of (a) are labelled with their frozen role names
  (`historical reference`, `curve reference`) rather than a generic
  "(reference)", so the absence of an interval is self-explanatory. Both values
  are the same ones Table 6 lists in its supporting-facts block, which exists
  to underwrite the "beats every evaluated baseline" sentence: dropping these
  rows would leave that claim table-only.
- In-figure text in (d) is two single-purpose lines — the SMD threshold count
  and the domain-classifier AUC with its interval. The classifier's sample sizes
  and the cross-reference to the mismatch effect (Fig. 3b / Table 6) live in
  the caption, not in the artwork: cross-references inside a figure break when
  numbering moves, and the sample size belongs to a different quantity than the
  six per-feature SMDs on the same panel.

## Integrity assertions (run before anything is written)

- For each of the three standard baselines and both metrics,
  `1 - model / baseline` recomputed from the frozen absolute values must equal
  the adjudicated `i_*_point` to atol 1e-12. The same construction is what
  produces the two reference point estimates, which the frozen table does not
  adjudicate.
- Every seed's 95% CI lower bound must be `> 0`; a sign flip would be a change
  of conclusion and must stop the script rather than be drawn.
- Exactly one OOD profile is selected.

## Claim boundary

- (c) shows direction stability only; no stratum is an independent confirmation.
  Complete stratum and trim numbers live in Table S3, and any "direction is
  stable" statement must point there.
- (d)'s domain classifier is proof of shift only and never enters method
  scoring; `BEHAVIOR_OOD` did not qualify, and `JOINT_OOD` was chosen by frozen
  priority. The panel supports no "robust to arbitrary OOD" claim.
- The two metrics do not agree on one strongest standard baseline, so any
  "strongest baseline" phrasing must name the metric.

## Source bindings

- `experiments/stage4_reliability/stage4cr2_confirm/reports/Table_C2_baselines.csv`
- `experiments/stage4_reliability/stage4cr2_confirm/reports/Table_C3_seeds_strata_trim.csv`
- `experiments/stage4_reliability/stage4cr2_confirm/reports/stage4cr2_confirmatory_report.json#decision.metrics`
- `experiments/stage4_reliability/stage4d_ood/pilot/ood_profile_selection.json`
- `experiments/stage4_reliability/stage4d_ood/reports/stage4d_ood_confirmatory_report.json#decision.domain_shift`
- `paper/2026-09-26_AC-Twin视觉叙事蓝图_v1.0.md` §4 Fig. 4, §12-2
