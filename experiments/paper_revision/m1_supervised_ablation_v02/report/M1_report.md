# Supervised Controls and Feature Ablations: Development Diagnostic Report

- Plan ID: `AC-TWIN-RFER-M1-SUPERVISED-ABLATION-2026-10-07-V0.2`
- Scope: Merge component diagnostics, `ADAPTIVE_DEVELOPMENT_ONLY`; neither confirmation nor a new preregistered study.
- Data: R2 train 5,000 groups (main 4,500 / monitor 500) and validation 1,500 groups; no new collection or screen/power/confirm resources.
- Bootstrap: 10000 whole-group paired resamples, entropy 2026100701; primary C1–C5 AURC family uses 99% intervals (0.005/0.995, Bonferroni family-level 95%); the secondary-endpoint family uses the same specification. Uncorrected 95% intervals are provided for reference only.
- Continuity check: frozen Direct and three heuristics agree with R2 validation_gate_report within 1e-8 throughout (passed=True).
- **Historical manuscript-editing boundary: this report did not automatically update the abstract, main text, supplement, Word or manuscript figures/tables; manuscript changes required separate explicit author authorization.**

## 1. Absolute metrics (validation 1,500 groups; shared paired draws)

| Method | Information set | Active dimensions | Nominal parameters | AURC-regret | Regret@80 |
| --- | --- | ---: | ---: | ---: | ---: |
| FULL_MATCHED | Full 315-dimensional input | 315 | 15740 | 0.010970 | 0.014781 |
| BASELINE3_MATCHED | Three baseline scalars only | 3 | 15740 | 0.019369 | 0.020117 |
| NO_RISK_MATCHED | Five risk-channel families masked | 180 | 15740 | 0.012409 | 0.014493 |
| FULL_FROZEN_REFERENCE | Full 315-dimensional input | — | 15740 | 0.010970 | 0.014781 |
| MARGIN_ONLY | Fixed heuristic | — | 0 | 0.020824 | 0.021217 |
| SEED_VOTE_ENTROPY | Fixed heuristic | — | 0 | 0.018959 | 0.020542 |
| SEED_MARGIN_STD | Fixed heuristic | — | 0 | 0.021838 | 0.022366 |

## 2. C1–C5 primary contrasts (improvement = 1 − method/reference; family-adjusted 99% intervals)

| Contrast | Metric | Point estimate | 99% CI (family) | 95% CI (uncorrected) | Absolute difference (point) | Decision |
| --- | --- | ---: | --- | --- | ---: | --- |
| C1 | aurc_regret | 0.4336 | [0.2819, 0.5541] | [0.3250, 0.5274] | 0.008398 | CI lower bound >0 |
| C1 | regret_at_80 | 0.2653 | [0.1449, 0.3807] | [0.1733, 0.3509] | 0.005336 | CI lower bound >0 |
| C2 | aurc_regret | 0.1159 | [-0.0307, 0.2289] | [0.0066, 0.2038] | 0.001439 | CI crosses zero (not established) |
| C2 | regret_at_80 | -0.0199 | [-0.1177, 0.0696] | [-0.0907, 0.0499] | -0.000288 | CI crosses zero (not established) |
| C3 | aurc_regret | -0.0216 | [-0.1929, 0.1250] | [-0.1491, 0.0916] | -0.000409 | CI crosses zero (not established) |
| C3 | regret_at_80 | 0.0207 | [-0.0419, 0.0592] | [-0.0256, 0.0473] | 0.000425 | CI crosses zero (not established) |
| C4 | aurc_regret | 0.0699 | [-0.0887, 0.2061] | [-0.0505, 0.1766] | 0.001456 | CI crosses zero (not established) |
| C4 | regret_at_80 | 0.0518 | [-0.0296, 0.1480] | [-0.0069, 0.1246] | 0.001099 | CI crosses zero (not established) |
| C5 | aurc_regret | 0.1131 | [-0.1295, 0.3058] | [-0.0624, 0.2624] | 0.002469 | CI crosses zero (not established) |
| C5 | regret_at_80 | 0.1005 | [-0.0437, 0.2375] | [-0.0042, 0.2028] | 0.002248 | CI crosses zero (not established) |

## 3. Five-seed directions (descriptive point estimates)

| Contrast | Metric | Positive/negative/zero | Range | Median |
| --- | --- | --- | --- | ---: |
| C1 | aurc_regret | 5/0/0 | [0.3592, 0.4412] | 0.4086 |
| C1 | regret_at_80 | 5/0/0 | [0.2420, 0.2953] | 0.2615 |
| C2 | aurc_regret | 5/0/0 | [0.0079, 0.1616] | 0.1140 |
| C2 | regret_at_80 | 2/3/0 | [-0.0357, 0.0382] | -0.0057 |
| C3 | aurc_regret | 2/3/0 | [-0.0282, 0.0329] | -0.0109 |
| C3 | regret_at_80 | 5/0/0 | [0.0146, 0.0305] | 0.0229 |
| C4 | aurc_regret | 5/0/0 | [0.0639, 0.1195] | 0.0796 |
| C4 | regret_at_80 | 5/0/0 | [0.0459, 0.0613] | 0.0539 |
| C5 | aurc_regret | 5/0/0 | [0.1073, 0.1604] | 0.1223 |
| C5 | regret_at_80 | 5/0/0 | [0.0949, 0.1096] | 0.1025 |

## 4. Conclusion mapping (frozen specification, §8)

- C1 (full input versus supervised three-baseline combination, AURC): CI lower bound >0.
- C2 (conditional increment from explicit risk channels, AURC): CI crosses zero (not established).
- C3 (supervised combination versus frozen primary heuristic SEED_VOTE_ENTROPY, AURC): CI crosses zero (not established).
- C4 (supervised combination versus MARGIN_ONLY, AURC): CI crosses zero (not established).
- C5 (supervised combination versus SEED_MARGIN_STD, AURC): CI crosses zero (not established).

Interpretation boundaries: intervals are conditional on fitted models and historical development data. Even wholly positive intervals provide development-only incremental evidence. Do not subtract improvements to attribute components of the original confirmed 26.0% gain. A C1 interval crossing zero would not establish FULL–BASELINE3 equivalence; this is not an equivalence/noninferiority design. Report all negative or unstable results without adding candidates or modifying training to rescue them.
