# Fig. 2 Figure Contract

## Figure-level claim

AC-Twin audits the decision regret of the action selected by a frozen counterfactual prediction stack, then retains decision groups by the resulting risk ranking. The overview shows the method flow only; results and comparison baselines belong to the result figures and正文, while the full information-parity protocol is reported in Table S2.

## Layout (v4)

Draft v4 removes the in-figure title and footer sentence (the caption carries both) and
crops the canvas to the drawn content at an unchanged physical scale; the layout below is
otherwise draft v3's. Two tinted macro regions carry the story:

- `INPUTS & AUDIT SCORING` (upper tinted region): the input and scoring flow, holding stages I–III; training-only supervision is distinguished by a red dashed arrow.
- `TRAINING & EVALUATION ONLY` (lower tinted region, sealed): the paired-counterfactual
  truth chain, separated from scoring inputs; a training-only connector targets the audit head, never its inference feature input.
- Stage IV sits in its own column to the right of the inference region: it is the
  output/use of the audit, not part of the scoring path.

## Four stages

| Stage | Role | Required content |
| --- | --- | --- |
| I · Twin event | Form the decision group | Observed prefix, observed factual suffix and the counterfactual branch bundle (five action semantics by two durations) |
| II · Prediction | Produce the base decision | Frozen prediction stack, predicted cost, the selected action (highlighted branch, chosen by minimum predicted cost) |
| III · Risk audit | Score the selected decision | Inference-legal features, one Direct audit-head block, audit-risk dial; supervision shown by the red dashed connector without a separate label tag |
| IV · Retention | Rank complete groups | Rank by audit score, retain the lowest-scored share at fixed coverage, show retained/not-retained segments without a curve; full-sample regret remains the reference |

## Visual invariants

- The canvas is 180 mm wide and 88 mm high, cropped to the drawn content: the figure carries no in-figure title and no footer line. The smallest in-figure type is 7.1 pt at the delivered 100% width, and block labels are 8.2 pt or larger.
- Colours are the six supplied hues: blue and light blue for AC-Twin and the inference region, grey for the frozen predictor, warm red and light peach for the counterfactual branches and the training/evaluation truth family, light grey for frames and neutral elements. Lighter and darker steps are generated from the same hues, and each hue keeps one meaning across the figure.
- **The figure body carries only contract-level counts: the action grid (five actions by two
  durations) and the retention share (80%).** Dimensions, parameter counts, hyperparameters,
  output equations and every measured result are carried by the caption (`FIGURE_LEGEND.md`)
  and by the text.
- The red dashed connector marks supervision; the connector from the truth
  region targets head training, not inference features or scoring. Evaluation truth
  is used after audit scores are fixed.
- Final selection (2026-10-05): keep only the no-red-tag composition under the canonical `Fig2_AC_Twin_method_overview_v4` stem. The scene inset represents the observed prefix; its redundant overlapping label is removed. Historical alternatives are outside this directory under `paper/_figure_archive/`.
- The figure contains no results, metrics, baseline names, dimensions, parameter counts,
  hyperparameters, network layers, losses or experiment-governance labels.
- Selective retention operates across complete decision groups and does not update the
  frozen predictor or execute a fallback policy.

## Source bindings

- `code/ac_twin/stage4c_objective.py::ACTION_PLANS`
- `code/ac_twin/stage4c_protocol.py::COVERAGE_MAIN`
- `code/ac_twin/stage4cp_features.py`
- `code/ac_twin/stage4cp_models.py`
- `plans/2026-09-16_AC-Twin_Stage4CR2决策风险扩样开发验证实施计划_v0.2.md` §3–4
