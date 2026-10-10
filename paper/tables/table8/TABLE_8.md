# Table 8 — Supervised controls and feature ablations on development data

Corresponds to main-text §§3.5, 4.4 and Supplementary Methods SM11. All five contrasts use the same 1,500 development-validation groups.

## English table and caption

**Table 8 | Supervised controls and feature ablations on development data.**

| Comparison (method / reference) | AURC-regret reduction [99% CI] (%) | Regret@80 reduction [99% CI] (%) |
| --- | ---: | ---: |
| Complete input / Baseline signals only | 43.4 [28.2, 55.4] | 26.5 [14.5, 38.1] |
| Complete input / Explicit risk channels removed | 11.6 [-3.1, 22.9] | -2.0 [-11.8, 7.0] |
| Baseline signals only / SEED_VOTE_ENTROPY | -2.2 [-19.3, 12.5] | 2.1 [-4.2, 5.9] |
| Baseline signals only / MARGIN_ONLY | 7.0 [-8.9, 20.6] | 5.2 [-3.0, 14.8] |
| Baseline signals only / SEED_MARGIN_STD | 11.3 [-13.0, 30.6] | 10.1 [-4.4, 23.8] |

Note. Evaluation uses the same 1,500 complete ten-action development groups. Positive values indicate lower regret metrics for the evaluated method than for its reference. Five AURC-regret contrasts and five Regret@80 contrasts form separate families. Each 99% percentile interval corresponds to nominal 95% Bonferroni family coverage, using 10,000 whole-group paired bootstrap resamples; joint 95% coverage across both families is not claimed. All results are conditional development evidence.

## Comparison specification

Relative improvement is 1−method/reference. Machine-readable values are dimensionless ratios; the paper displays percentages rounded to one decimal. Five primary AURC-regret and five secondary Regret@80 contrasts form separate families. Each 99% percentile interval provides nominal 95% Bonferroni coverage within its five-comparison family; joint 95% coverage across both families is not claimed. Historical development-validation reuse does not constitute fresh confirmation.

Full inputs provide conditional gains over supervised use of three baseline scalars. Adjusted intervals for the scalar combination versus all three heuristics cross zero. Both risk-channel-removal intervals cross zero: independent channel increments are not established, nor are equivalence or uselessness. Equal nominal parameter counts do not imply equal effective capacity.

## Numerical sources

- `experiments/paper_revision/m1_supervised_ablation_v02/evaluation/contrasts.csv`: C1–C5 improvement rows, extracting aurc_regret/regret_at_80 point/ci99_lower/ci99_upper.
- `evaluation/bootstrap_receipt.json`: resample count, quantiles and families.
- `config.json` and `models/model_inventory.json`: inputs, training and parameter contract.
- Original manuscript §4.4: table and notes, maintained outside this code/data release.

This package organizes frozen results only, without training, inference, resampling or re-adjudication.
