# Figure contract — factual-suffix evidence ablation

- Core conclusion: Same-episode factual evidence improves retrospective counterfactual effect prediction beyond the benefit of merely adding an equally long input; the useful evidence saturates at approximately 10 observation steps, while a naive one-step residual proxy is not reusable structural innovation.
- Evidence chain:
  - Panel a: compare all eight ablation conditions with the five training-seed results visible.
  - Panel b: isolate the response to increasing factual-suffix length (0, 1, 5, 10, and 20 steps).
  - Panel c: show the confirmatory paired test-episode contrast between the matched shuffled suffix and same-episode 20-step suffix.
- Archetype: quantitative three-panel grid with one hero comparison, one dose-response ablation, and one paired-error distribution.
- Drawing backend: Python with matplotlib.
- Target size: 183 mm wide, approximately 90 mm high; suitable for a double-column IEEE/Nature-style manuscript figure.
- Statistics:
  - Panels a and b show the mean and standard deviation across five independently initialized training runs; all five values are shown.
  - Panel c uses 300 independent test episodes. The reported 95% confidence interval is the existing 10,000-resample paired episode bootstrap interval.
- Source data: existing `report.json` and `test_predictions_and_errors.npz` from `experiments/suffix_evidence_ablation`; figure-ready CSV files are exported beside the figure.
- Export: editable SVG, vector PDF, 600 dpi LZW-compressed TIFF, and 300 dpi PNG preview.
- Reviewer risks addressed: no training seed is treated as an independent test episode; shuffled input is matched within split and condition strata; no significance stars or unsupported causal labels are used.
- Remaining limitation: this figure reports one collected dataset and one scenario family; independent-seed and cross-scene replication remains the next decision gate.
