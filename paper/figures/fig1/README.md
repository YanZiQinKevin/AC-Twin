# Fig. 3: Factual-evidence value

Directory and asset names retain historical numbering. Current main-text number: Fig. 3.

The completed four-panel figure is 180 × 104 mm, intended at full two-column width. Outputs: `Fig1_evidence_value_structure.svg`, `.png` (600 dpi), and `.pdf`. `FIGURE_LEGEND.md` supplies the caption. `fig1_bins_plot_data.csv`, `fig1_contrasts_plot_data.csv` and `fig1_data.xlsx` contain the plotted data.

Run `python paper/figures/fig1/build_fig1.py` from the repository root. The script reads `experiments/stage2_confirmatory/merge/report.json` and asserts that bin `V_evt` values equal the frozen statistic points within atol 1e-12 before plotting.

| Panel | Content | Frozen fields |
| --- | --- | --- |
| (a) | Four response-bin definitions and thresholds | `statistics.response_bins`; counts from `statistics.bins.*.n_episodes` |
| (b) | Three mean losses per bin, logarithmic axis | `mean_l_no`, `mean_l_suffix`, `mean_l_shuffled` |
| (c) | Four `V_evt` points and 95% CIs | `statistics.v_zero / v_low / v_mid / v_high` |
| (d) | Three prospectively specified contrasts | `statistics.c_presence / c_high_low / c_high_mid`; `statistics.point_screen` |

The bin thresholds were frozen in Stage 1, not fitted to confirmation data. `experiments/stage1_evidence_value/merge/response_bins.json` has SHA256 `8f6aeb078b12007b84da55d3c682cfce861f74a199640e2e1cad4bf45e5cc15f`, registered in the Stage 2 protocol lock.

The three same-architecture models were trained separately and share this event's factual-trajectory reconstruction reference. Negative values describe relative model error, not intrinsically negative factual information. Episode counts may overlap across bins and must not be summed. Display the mid-bin negative CI `[-0.1129, -0.0199]` and low-bin zero-crossing CI `[-0.0110, 0.1697]` as observed. Do not infer a continuous curve, strict monotonicity, symmetric U shape, dual-edged evidence or a causal mechanism. `C_rebound` duplicates the `C_high_low` point without a separate adjudication CI and is not plotted again. Robustness analyses are reported separately.
