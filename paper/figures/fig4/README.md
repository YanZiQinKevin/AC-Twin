# Fig. 6: Robustness and OOD qualification

Current main-text number: Fig. 6; historical directory/asset names remain. The completed 2×2 figure is 180 × 126 mm at full two-column width. Outputs: `Fig4_credibility_closure.svg`, `.png` (600 dpi), `.pdf`; `fig4_panel_a_data.csv` through `fig4_panel_d_data.csv`; `fig4_data.xlsx`; caption in `FIGURE_LEGEND.md`.

Run `python paper/figures/fig4/build_fig4.py`. It reads frozen confirmation tables and selection receipts, without bootstrap reruns, baseline reselection or re-adjudication. Standard-baseline improvements recomputed as `1 - AURC_model / AURC_baseline` must match `Table_C2_baselines.csv` `i_*_point` within atol 1e-12. Seed-panel 95% CI lower bounds must be >0.

| Panel | Evidence | Source |
| --- | --- | --- |
| (a) | Three standard baselines with adjudicated AURC/Regret@80 intervals; two reference scores with point estimates only | `Table_C2_baselines.csv`; `stage4cr2_confirmatory_report.json#decision.metrics` |
| (b) | Paired effects for five audit-head seeds; red dashed line is the adjudicated ensemble effect | `Table_C3_seeds_strata_trim.csv` |
| (c) | Four marginal strata and trim 1%/5%; directional stability only | Same table; full values in Table S3 |
| (d) | Three OOD-profile qualifications, six absolute feature SMDs (threshold 0.5), domain-classifier AUC | `stage4d_ood/pilot/ood_profile_selection.json`; OOD report `decision.domain_shift` |

Reference scores `COMBINED_TOP2_MAX` and `CONSTANT_RANDOM` have no adjudication CIs. In panel (a), a longer bar means a weaker baseline. The strongest baseline depends on endpoint: `MARGIN_ONLY` has lowest AURC; `SEED_VOTE_ENTROPY` has lowest Regret@80.

Panel (c) is not independent confirmation; show any zero-crossing interval. Domain classification establishes distribution mismatch and never enters risk scoring. GEOMETRY_OOD and JOINT_OOD qualified; BEHAVIOR_OOD did not. JOINT_OOD was chosen by frozen priority. This does not establish arbitrary-OOD robustness.

Panels (a)–(c) use Merge confirmation only. Narrowest baseline margins differ by scope: Merge 26.0%, JOINT_OOD 18.6%, MetaDrive 10.1%. Do not attribute MetaDrive's 10.1% to Merge. The 95% intervals share paired resampling indices and are correlated; interval overlap/nonoverlap cannot replace a paired between-row difference analysis.

The frozen CSV has asymmetric Regret@80 CI column names: `i_regret80_ci95_lower` and `i_regret80_ci_upper`. `CI_COLS` explicitly maps these actual names; frozen files are unchanged. Domain classification used n = 5,000 / 2,000 and yielded AUC 0.983 [0.980, 0.986]; three of six SMDs shifted. Quantitative OOD effects are in Fig. 4 and Table 6, separate from this qualification panel.
