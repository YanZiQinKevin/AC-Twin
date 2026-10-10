# Table 6 — AC-Twin audit across the three measured scopes

Corresponds to Fig. 4. All numbers come from frozen confirmation reports, without re-estimation, baseline reselection or re-adjudication.

## Table

| Scope | Method | AURC-regret ↓ | Regret@80 ↓ | PR−prevalence ↑ | I_AURC-regret [95% CI] | I_Regret@80 [95% CI] |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **Merge ID** | **AC-Twin audit** | **0.010783** | **0.014346** | **0.314627** | **0.4392 [0.3001, 0.5044]** | **0.2600 [0.1695, 0.3230]** |
|  | SEED_VOTE_ENTROPY *(frozen primary)* | 0.019228 | 0.019388 | 0.054195 | — | — |
|  | MARGIN_ONLY | 0.018671 | 0.020285 | 0.054327 | — | — |
|  | SEED_MARGIN_STD | 0.025405 | 0.022550 | −0.011707 | — | — |
| **JOINT_OOD** | **AC-Twin audit** | **0.024473** | **0.026289** | **0.215680** | **0.2077 [0.1276, 0.3050]** | **0.1855 [0.1274, 0.2366]** |
|  | SEED_VOTE_ENTROPY *(frozen primary)* | 0.030887 | 0.032275 | 0.046445 | — | — |
|  | MARGIN_ONLY | 0.030431 | 0.032628 | 0.062315 | — | — |
|  | SEED_MARGIN_STD | 0.036474 | 0.035893 | −0.025396 | — | — |
| **MetaDrive** | **AC-Twin audit** | **0.030011** | **0.045276** | **0.211717** | **0.1949 [0.1281, 0.2575]** | **0.1532 [0.1070, 0.1975]** |
|  | SEED_MARGIN_STD *(frozen primary)* | 0.037276 | 0.053470 | 0.075112 | — | — |
|  | MARGIN_ONLY | 0.065541 | 0.062677 | 0.008458 | — | — |
|  | SEED_VOTE_ENTROPY | 0.039973 | 0.050337 | 0.126450 | — | — |

n = 2,000 decision groups per scope in every row. Relative improvements appear only on the `AC-Twin audit` rows, against each scope's frozen primary baseline; other rows report absolute metrics.

## Main-text caption

**Table 6 | AC-Twin audit across the three measured scopes.** The three scopes are `Merge ID` (Merge in-distribution), `JOINT_OOD` (the prospectively specified generator-mismatch profile) and `MetaDrive` (MetaDrive/ScenarioNet/Waymo, retrained and frozen inside that platform); the same names are used in Fig. 4. Each scope is an independent confirmation set of 2,000 ten-action decision groups. AURC-regret and Regret@80 are lower-is-better; PR−prevalence is higher-is-better. Relative improvements are defined as I_Q = (Q_baseline − Q_audit)/Q_baseline against the scope's primary baseline frozen before confirmation, marked in italics; these are ratios, not absolute differences. `AC-Twin audit` is the risk ranking produced by the AC-Twin framework's Direct audit head (Fig. 2); Fig. 4 uses the same label. Merge ID selected `SEED_VOTE_ENTROPY` using the fixed development-validation rule; JOINT_OOD inherited that frozen baseline without selection on OOD validation data. MetaDrive selected `SEED_MARGIN_STD` on its platform-internal validation data. These comparison targets were frozen before confirmation and were not changed afterwards. All three standard baselines are reported per scope, in this table and in Fig. 4. The two metrics do not agree on one strongest standard baseline in any scope: on Merge ID and JOINT_OOD the lowest AURC-regret is `MARGIN_ONLY` while the lowest Regret@80 is `SEED_VOTE_ENTROPY`; on MetaDrive the lowest AURC-regret is `SEED_MARGIN_STD` while the lowest Regret@80 is `SEED_VOTE_ENTROPY`. The audit has lower point estimates than every evaluated standard baseline on both metrics in all three scopes, with the narrowest margin 10.1% (MetaDrive/Waymo, Regret@80 against `SEED_VOTE_ENTROPY`). The three scopes differ in task distribution, baseline and model relation — Merge and JOINT_OOD share frozen Merge weights, MetaDrive/Waymo uses a platform-internally retrained and frozen pipeline — so the rows are not comparable with one another and their effect sizes are not ranked, pooled or read as cross-platform performance. CIs are two-sided percentile intervals from 10,000 whole-group paired bootstrap resamples of the adjudicated confirmation sets. Full baseline, seed, stratum, trim and ablation results are in Table S3.

## Interpretation notes

- `AC-Twin audit` is the result-level method; `Direct audit head` is a component; AC-Twin denotes the framework.
- Merge and JOINT_OOD share primary `SEED_VOTE_ENTROPY`; MetaDrive uses `SEED_MARGIN_STD`. Qualify any strongest-baseline statement by endpoint: `MARGIN_ONLY` has lowest AURC in Merge/OOD; `SEED_VOTE_ENTROPY` has lowest Regret@80 in MetaDrive, differing from the frozen primary.
- Distributions, baselines and weight relationships differ; do not rank, pool or interpret rows as cross-platform performance differences.
- Fig. 4 curves are descriptive redraws of per-group quantities; table CIs come from formal adjudication. The script asserts agreement at 80% coverage.

## Supporting values: improvements against other evaluated baselines

These values support the caption's comparisons against all evaluated baselines and come from the same frozen baseline curves.

| Scope | Baseline | AURC-regret | Regret@80 | I_AURC-regret | I_Regret@80 |
| --- | --- | ---: | ---: | ---: | ---: |
| Merge ID | COMBINED_TOP2_MAX | 0.025972 | 0.024443 | +0.5848 | +0.4131 |
| Merge ID | CONSTANT_RANDOM | 0.022651 | 0.022429 | +0.5240 | +0.3604 |
| Merge ID | MARGIN_ONLY | 0.018671 | 0.020285 | +0.4225 | +0.2928 |
| Merge ID | SEED_MARGIN_STD | 0.025405 | 0.022550 | +0.5756 | +0.3638 |
| Merge ID | SEED_VOTE_ENTROPY | 0.019228 | 0.019388 | +0.4392 | +0.2600 |
| JOINT_OOD | COMBINED_TOP2_MAX | 0.034580 | 0.035467 | +0.2923 | +0.2588 |
| JOINT_OOD | CONSTANT_RANDOM | 0.035753 | 0.034443 | +0.3155 | +0.2367 |
| JOINT_OOD | MARGIN_ONLY | 0.030431 | 0.032628 | +0.1958 | +0.1943 |
| JOINT_OOD | SEED_MARGIN_STD | 0.036474 | 0.035893 | +0.3290 | +0.2676 |
| JOINT_OOD | SEED_VOTE_ENTROPY | 0.030887 | 0.032275 | +0.2077 | +0.1855 |
| MetaDrive | MARGIN_ONLY | 0.065541 | 0.062677 | +0.5421 | +0.2776 |
| MetaDrive | SEED_MARGIN_STD | 0.037276 | 0.053470 | +0.1949 | +0.1532 |
| MetaDrive | SEED_VOTE_ENTROPY | 0.039973 | 0.050337 | +0.2492 | +0.1005 |

## Numerical provenance

| Scope | Formal report | Directory |
| --- | --- | --- |
| Merge in distribution | `stage4cr2_confirmatory_report.json` | `experiments/stage4_reliability/stage4cr2_confirm/reports/` |
| Prospectively specified JOINT_OOD | `stage4d_ood_confirmatory_report.json` | `experiments/stage4_reliability/stage4d_ood/reports/` |
| MetaDrive / Waymo | `confirm_report.json` | `experiments/stage4_reliability/stage4d_d2f/confirm/` |

Primary endpoints use `metrics.point / ci_lower / ci_upper` (`decision.metrics` in Merge/OOD). Absolute baseline metrics and PR−prevalence use `decision.ablations.baseline_curves` (top-level `baseline_curves.curves` in MetaDrive). `table6_data.xlsx` contains the same values.

## Retrieval and consistency checks

`paper/figures/fig3/build_fig3.py` checks model/baseline curve values at 80% coverage against report values within atol 1e-12 before plotting. This table transcribes those frozen fields without re-estimation.
