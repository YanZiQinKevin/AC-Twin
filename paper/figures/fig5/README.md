# Fig. 5: MetaDrive task and illustrative cases

The author-approved main-text asset is [Fig5_metadrive_external_validation.png](Fig5_metadrive_external_validation.png), 4251×3779 px, corresponding to 180 × 160 mm. `AUTHOR_APPROVAL.json` records its identity. Matching SVG/PDF files are alternate formats, not independently selected figures.

Panels: (a) native MetaDrive/Panda3D view; (b) the same scene's map and factual/counterfactual trajectories; (c) a frozen high-regret detection example; (d) the frozen worst false-negative example. `Fig5_selected_platform_raw.png` is the native screenshot used in (a). `Fig5_render_receipt.json` and `Fig5_layout_manifest.json` record source, selection and layout. Caption: `FIGURE_LEGEND.md`; cost interpretation: `FIXED_CASE_COST_EXPLANATION.md`.

The platform view was selected once from 20 frozen candidates for readability, using decreasing visible-entity count, maximum factual/counterfactual ego separation, legibility score, then increasing candidate hash. The selected view contains 5 visible entities and approximately 3.134 m separation. This replaced a previous view with only 0.009983 m separation; no training, inference, gates, adjudication or paper statistics changed. The native FPS overlay may remain.

Historical platform-only and author-assembly assets remain traceable but are not alternative current main-text versions. `build_fig5.py` is the current rendering entry point. The fixed PNG must not be silently regenerated or replaced during repository maintenance. Reproduction requires the frozen MetaDrive simulation environment and provider scenarios; scripts read frozen products without retraining, collection, model re-inference or re-adjudication.

In (b), factual ego trajectories are green; the fixed `LEFT × 1` counterfactual is red dashed and drawn above them. Case IDs, actions, regret, risk percentiles and peak steps are reused from create-once receipts. The best-of-205 success and best-rendered-of-20 platform view are qualitative illustrations, not population success rates. Selected/oracle differences illustrate regret, not executed fallback control. MetaDrive uses platform-internal retraining and freezing, not zero-shot Merge-weight transfer. Overall quantitative claims require the full statistical figures/tables.

Sources:

- `experiments/stage4_reliability/stage4d_external/platform/viz_platform_pool.json`
- `experiments/stage4_reliability/stage4d_external/roles/pilot/`
- `experiments/stage4_reliability/stage4d_d2f/roles/confirm/groups.npz`
- `experiments/stage4_reliability/stage4d_d2f/confirm/confirm_group_quantities.npz`
- `paper/figures/stage4d_d2f/provenance/selection_receipt.json`
- `code/ac_twin/stage4d_external_features.py` and `code/ac_twin/stage4d_paper_visualization.py`
