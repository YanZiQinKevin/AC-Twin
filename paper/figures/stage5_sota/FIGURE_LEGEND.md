# Fig. S-Stage5: Development risk–coverage curves

Plan: `AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3`. Terminal state: `STAGE5_MERGE_COMPARISON_UNDERPOWERED`.

Source: `figS_stage5_risk_coverage_data.csv`, exported by `paper/tables/table7/build_table7_details.py` from frozen `validation_comparison_report.json`, without recomputation, reselection or re-adjudication.

On Stage 4C-R2 validation data (1,500 ten-action groups), R(c) is mean decision regret among the lowest-risk c fraction, for coverage 5%–100%. Lower curves are better. Nine curves represent frozen AC-Twin Direct, selected adaptations of ConfidNet, SelectiveNet, RwR-kNN, Mondrian conformal and CCTR, plus MARGIN_ONLY, SEED_VOTE_ENTROPY and SEED_MARGIN_STD.

**Scope and limitations.** This is `ADAPTIVE_DEVELOPMENT_ONLY`. The Stage 5 comparison holdout was **never created**, because preregistered joint power at n=6000 under 75% effect shrinkage was 0.8354 < 0.90. No formal comparison was performed; these curves do not establish superiority over published methods. Against frozen `STRONGEST_PUBLISHED_BASELINE = CCTR_DRIVING_CALIBRATION_ADAPTED`, validation I_AURC was 0.1404 (95% CI [-0.0097, 0.2710]) and I_REGRET@80 was 0.1778 (95% CI [0.0807, 0.2789]), for development diagnosis only.
