# Graphical Abstract Contract

## Figure-level claim

Inside one retrospective counterfactual task, the average prediction error misses two things a
decision needs: the response-conditioned value of the factual evidence, and whether the selected
action carries high regret. AC-Twin measures the first (panel b) and audits the second (panel c).
The figure is an overview: it carries no result numbers, no method internals and no scope effect
sizes.

## Layout (v1)

Canvas 180 × 84 mm (100% of a two-column page width), asymmetric — **not** four equal boxes.
All coordinates are millimetres, so the drawing is isotropic at the delivered size.

| Region | Width | Role | Required content |
| --- | --- | --- | --- |
| (a) Twin event | 28% | setting | pre-intervention history opening a factual continuation (top) and an alternative ego action (bottom); 3–4 vehicles, no real road detail |
| (b) Evidence-value lens | 34% | problem discovery | three prediction cards (history-only / correct suffix / shuffled suffix) + four `V_evt` dots over the frozen response bins |
| (c) Decision audit | 38%, hero | problem solving | ten candidate branches → selected action → direct risk head → audit risk score → retained / flagged, with a schematic risk–coverage glyph |
| bottom band | full | evidence boundary | the three measured scopes, names only, one shared tint |

The hero panel is distinguished by the only strong backing tint; (a) carries none and (b) a faint
neutral one, so the auxiliary panels are not weighted equally with (c).

## Visual invariants

- Six supplied hues, one meaning each, shared with Figs. 2–4: blue = AC-Twin method,
  light blue = method tints, warm red = counterfactual branches and the truth family,
  peach = warm tints, grey = frozen predictor / neutral, light grey = frames.
- No in-figure title and no footer line; the caption carries both.
- Panel letters 9.6 pt bold, panel titles 9.2 pt bold, **smallest in-figure type 7.5 pt**
  (the 7.5 pt floor applies rather than the numbered Fig. 1/2/3 6.6 pt exception).
- The B → C link is the rotated phrase `Motivates a decision-level audit`; there is no solid
  causal arrow anywhere in the figure.
- The three scopes in the bottom band share one tint: scope identity is never encoded by
  colour.

## Prohibited in the figure body

- Numerals of any result (per-bin `V_evt` values, intervals, effect sizes), including `23.91%`.
- Method internals: 315 features, layer counts, parameter counts, losses, network diagrams
  (all belong to Fig. 2).
- Experiment-governance strings: stage names, gate names, terminal statuses, locks, hashes.
- The phrase `external generalization`; MetaDrive is fixed as `platform-internal retraining`.
- Any solid arrow implying that a higher average error causes higher regret.

## Frozen input

The only measured quantity in the figure is the **height** of the four dots in (b). It is read from
`experiments/stage2_confirmatory/merge/report.json#statistics` and asserted twice before anything is
written: `statistics.v_<bin>.point` must equal `statistics.bins.<bin>.V_evt` (atol 1e-12) and
`statistics.n_episodes` must be 1200. The risk–coverage glyph in (c) is illustrative and is labelled
as such in both the drawing prompt and the legend.

## Source bindings

- `plans/2026-09-06_AC-Twin反应式事实证据价值与路由研究计划_v1.0.md` §2, §25
- `plans/2026-09-16_AC-Twin_Stage4CR2决策风险扩样开发验证实施计划_v0.2.md` §3
- `experiments/stage2_confirmatory/merge/report.json#statistics`
- `paper/2026-09-26_AC-Twin视觉叙事蓝图_v1.0.md` graphical abstract 条目与视觉规则
