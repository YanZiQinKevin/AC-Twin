# Stage 5 method registry (machine source: `method_registry.json`)

- Plan: `AC-TWIN-RFER-STAGE5-SOTA-COMPARISON-2026-09-27-V0.3`
- Frozen: 2026-09-27; access date for all citations: 2026-09-27
- Works: 16 (blocks: counterfactual driving / counterfactual outcome / failure & selective prediction / regression rejection & risk control)

## Experimental candidates (P1-P5) and supplemental

| Method ID | Source | Venue | Code / license | Status | Adaptation (one line) |
|---|---|---|---|---|---|
| `OOSR_DECISION_FOCUSED_ADAPTED` | Hao Zou et al. (2022) | ICML 2022 (PMLR 162:27693-27706) | NONE / NONE | SUPPLEMENTAL_NOT_STRONGEST_BASELINE_COMPETITION | Changes the action-consequence prediction/selection objective rather than auditing a frozen base selector; reported only in a supplemental decision-focused table if Phase L adaptation proves feasible without changing task/data roles (plan 2.3). |
| `CONFIDNET_TCP_ADAPTED` | Charles Corbiere et al. (2019) | NeurIPS 2019 (pp. 2902-2913) | https://github.com/valeoai/ConfidNet / Apache-2.0 | ELIGIBLE_ADAPTED_VERIFIED | Frozen base selector's ten-action predicted costs J_hat are converted to action probabilities by a train-only/validation-frozen temperature; the oracle action's probability at train time is the TCP supervision; a confidence network reads only legal features; inference risk = 1 - predicted_TCP. |
| `SELECTIVENET_REGRET_ADAPTED` | Yonatan Geifman et al. (2019) | ICML 2019 (PMLR 97:2151-2159) | https://github.com/geifmany/selectivenet / NONE | ELIGIBLE_ADAPTED_VERIFIED | Same legal-feature encoder with regret-regression, selection and auxiliary heads; trained with the paper's selective-risk + coverage penalty + auxiliary loss at target coverage 0.80; risk score = 1 - selection probability; full risk-coverage curve from this score. |
| `RWR_NN_KNN_REJECT_ADAPTED` | Xiaocheng Li et al. (2024) | AISTATS 2024 (PMLR 238:1126-1134) | https://github.com/hanzhao-wang/RwR / NONE | ELIGIBLE_ADAPTED_VERIFIED | Main regressor learns true selected-action regret on ALL train groups; rejector estimates local error/difficulty in the frozen learned representation via kNN averaging of reference residuals (kNNRej idea); k chosen from {16,32,64} on validation by lowest AURC. |
| `CONFORMAL_RISK_CONTROL_ADAPTED` | Anastasios N. Angelopoulos et al. (2024) | ICLR 2024 (OpenReview 33XGfHLtZg) | https://github.com/aangelopoulos/conformal-risk / MIT | REGISTERED_ALTERNATE_NOT_ACTIVATED | Outputs a calibrated guarantee, not a per-input risk ranking compatible with a full risk-coverage curve; activates only if P4 fails Phase L compatibility per plan 2.4 (P4 passed sanity, so not activated). |
| `CONFORMAL_REGRESSION_REJECT_ADAPTED` | Ulf Johansson et al. (2024) | COPA 2024 (PMLR 230:277-294) | NONE / NONE | ELIGIBLE_ADAPTED_VERIFIED | Regressor predicts group regret on a deterministic 80% fit split (group-seed hash); Mondrian split-conformal on the DISJOINT 20% calibration split with categories from a frozen difficulty rule; risk score = frozen difficulty/interval-width ordering; calibration data never overlaps the comparison holdout. |
| `CCTR_DRIVING_CALIBRATION_ADAPTED` | Chengtai Cao et al. (2024) | AAAI 2024 (38(19):20949-20957) | NONE / NONE | ELIGIBLE_ADAPTED_VERIFIED | Five-seed frozen action-consequence prediction disagreement plus legal context/history features feed a sample-specific temperature model calibrating action-consequence uncertainty; group risk = calibrated uncertainty of the selected action's consequence. May read frozen five-seed predictions; never counterfactual truth. |

## Literature-only positioning works (enter Table I, not the experiment)

| Work | Venue | arXiv/DOI | Status | Reason |
|---|---|---|---|---|
| Jiaru Zhang et al. (2026), *How Can Driving World Models Do Counterfactual Prediction?* | arXiv preprint (cs.CV) | arXiv:2608.11601 | LITERATURE_ONLY_NOT_SAME_ENDPOINT | Outputs counterfactual episode completions, not a pre-truth per-action-group scalar risk; adapting it to risk scoring would redefine its research question. |
| Siddhesh Khandelwal et al. (2020), *What-If Motion Prediction for Autonomous Driving* | arXiv preprint (cs.LG/stat.ML) | arXiv:2008.10587 | LITERATURE_ONLY_NOT_SAME_ENDPOINT | Trajectory distributions, not group-level decision-risk scores; no 2024+ paper with this title exists (the 2020 WIMP paper is the canonical one). |
| Shuhan Tan et al. (2024), *Promptable Closed-loop Traffic Simulation* | CoRL 2024 (arXiv 2024-09-09) | arXiv:2409.05863 | LITERATURE_ONLY_NOT_SAME_ENDPOINT | Simulation rollouts, not pre-truth failure-risk scores. |
| Keisuke Fujii et al. (2024), *Estimating counterfactual treatment outcomes over time in complex multiagent scenarios* | IEEE TNNLS (arXiv v1 2022-06) | arXiv:2206.01900 | LITERATURE_ONLY_NOT_SAME_ENDPOINT | Counterfactual outcome trajectories; a group-level action risk score is not native and would require redefining the endpoint. |
| Song Jiang et al. (2023), *CF-GODE: Continuous-Time Causal Inference for Multi-Agent Dynamical Systems* | KDD 2023 | arXiv:2306.11216; ACM DOI 10.1145/3580305.3599272 | LITERATURE_ONLY_NOT_SAME_ENDPOINT | Per-unit potential-outcome trajectories; group decision-risk score not native. |
| Hao Zou et al. (2023), *Factual Observation Based Heterogeneity Learning for Counterfactual Prediction* | CLeaR 2023 (PMLR 213:350-370) | PMLR v213/zou23a | LITERATURE_ONLY_NOT_SAME_ENDPOINT | Requires the factual outcome as input and emits counterfactual outcomes, not pre-truth action-group risk. |
| Yonatan Geifman et al. (2017), *Selective Classification for Deep Neural Networks* | NeurIPS 2017 (pp. 4878-4887) | arXiv:1705.08500 | LITERATURE_ONLY_POSITIONING | Post-hoc classification confidence thresholding; AC-Twin's frozen internal baselines already cover confidence-ranking controls, and the experimental slot for learned confidence is P1/P5 per the frozen family table. |
| Liu Ziyin et al. (2019), *Deep Gamblers: Learning to Abstain with Portfolio Theory* | NeurIPS 2019 (pp. 10622-10632) | arXiv:1907.00208 | LITERATURE_ONLY_NOT_SAME_ENDPOINT | Classification-only; the paper explicitly defers regression to future work, so it is not a same-endpoint candidate for a decision-regret regression task. |
| Ahmed Zaoui et al. (2020), *Regression with Reject Option and Application to kNN* | NeurIPS 2020 (pp. 20073-20082) | ACM DOI 10.5555/3495724.3497409 | LITERATURE_ONLY_POSITIONING | Same family as P3; the frozen candidate table assigns the regression-with-rejection slot to Li et al. 2024 (P3), the newer consistent no-rejection-learning route. |

## Verification notes

- All citation fields verified against fetched arXiv/PMLR/ACM/GitHub pages on 2026-09-27 by independent literature-research agents; nothing recorded from memory.
- GitHub HEAD commits fetched live from api.github.com or the public commits.atom feed on 2026-09-27.
- Formal comparison uses published methods adapted to the AC-Twin task/data/legal-input contract; original-paper numbers on other benchmarks are never copied into AC-Twin result tables.
- Correction vs plan draft wording: CCTR's original calibration is a context-conditioned post-processing network (CNN over BEV + GRU + Softplus temperature), not ensemble-based; the P5 adaptation uses five-seed frozen prediction disagreement + legal context as the temperature inputs (adapted implementation, deviation recorded).
- Correction vs plan draft wording: Li et al. 2024 (RwR) learns the rejector separately on an independent validation split via kernel/kNN conditional-loss estimation (kNNRej); the P3 adaptation uses in-sample train residuals as the kNN reference because validation is reserved for selection/power only (adaptation deviation recorded).
- Deep Gamblers (NeurIPS 2019) is classification-only and explicitly defers regression to future work: kept as positioning literature, not an experimental candidate.
- Grid amendment at Task 3 design time (pre-implementation, nothing trained/consumed): P1 depth fixed to 3, P2 hidden grid {64,96}, P5 grid {(64,3),(96,2)} so every grid item's parameter count falls inside the frozen 0.5x-2.0x Direct capacity band [7870, 31480]; the earlier draft grids contained items outside the band that the capacity rule would have excluded anyway.

Sanity evidence: `../phase_l/sanity/sanity_report.json` (5/5 PASS, ENGINEERING_SANITY_ONLY). Compatibility decisions: `compatibility_decision.json`.
