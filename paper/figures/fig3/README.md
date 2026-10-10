# Fig. 4: Decision-risk confirmation in three scopes

Current main-text number: Fig. 4; historical directory and asset names are retained. The completed figure is 180 × 118 mm at full two-column width.

Outputs: `Fig3_three_scope_decision_risk.svg`, `.png` (600 dpi), `.pdf`; `A_merge_plot_data.csv`, `B_joint_ood_plot_data.csv`, `C_metadrive_plot_data.csv` (2,000 points each); `fig3_data.xlsx` (curves and frozen effect/CI summary); `FIGURE_LEGEND.md`.

Run `python paper/figures/fig3/build_fig3.py`. It reads frozen per-group quantities and confirmation reports, exports descriptive curves and never reruns bootstrap or adjudication. At 80% coverage, values must match `regret80_model` / `regret80_baseline` within atol 1e-12. If a reader locks one output format, that format is skipped with a message while others are written.

Panels (a)–(c) show separate risk–coverage axes for **Merge ID**, **JOINT_OOD** and **MetaDrive**. All three standard baselines are shown; the scope's frozen primary is the darkest/thickest solid line, labelled `frozen primary`. A thin dashed marker indicates 80% coverage. Panel (d) reports relative AURC-regret and Regret@80 improvements with 95% CIs.

`AC-Twin audit` denotes the frozen `DIRECT_ENSEMBLE` method, using five-seed `ensemble_risk`. `Direct audit head` is a component name; AC-Twin denotes the overall framework. Scope distributions, frozen baselines and weight relationships differ; do not rank or pool effects across scopes. MetaDrive uses platform-internal retraining, not zero-shot Merge-weight transfer.

Sources: Merge `experiments/stage4_reliability/stage4cr2_confirm/reports/confirm_group_quantities.npz`, the corresponding OOD/MetaDrive per-group outputs, and the three confirmation reports (`decision.metrics` for Merge/OOD; `metrics` for MetaDrive).
