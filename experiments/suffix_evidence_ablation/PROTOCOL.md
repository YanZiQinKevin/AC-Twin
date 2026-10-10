# Factual-Evidence Source Ablation Protocol

Frozen on: 2026-09-04
Source data: `data/pilot_2000/twinworld_multi_action_2000.npz`

## Research question

The previous run found lower counterfactual-effect error with complete factual suffixes than with history alone. This study tests whether that gain requires evidence from the same episode and distinguishes complete trajectories, partial trajectories, terminal states and a dynamics-residual proxy.

## Representation controls

To prevent different absolute road positions from trivially revealing shuffled inputs, subtract the ego position at intervention from every vehicle's `x/y` coordinates before training-set standardization. Add a visibility-mask channel to factual evidence. Keep model architecture, parameter count, train/validation/test splits, optimization budget and training seeds identical.

## Eight conditions

1. `history_only`: preceding history only; factual-evidence context forced to 0.
2. `shuffled_full`: a complete suffix from another episode in the same split, matching factual action, counterfactual action and duration where possible.
3. `last_frame`: only the factual state at step 20.
4. `prefix_1`: only the first 1 factual-suffix step.
5. `prefix_5`: only the first 5 steps.
6. `prefix_10`: only the first 10 steps.
7. `full_20`: the complete 20-step factual suffix.
8. `residual_innovation_20`: fit a first-order dynamics model using training factual transitions only, then use observed minus predicted state increments as factual evidence. These residuals are an innovation proxy, not identified exogenous noise.

## Confirmatory primary test

Primary contrast: `shuffled_full error - full_20 error`. Primary endpoint: effect-FDE@20.

Compute errors separately for five training seeds and average errors per test episode. Use 300 test episodes as independent units for 10,000 paired bootstrap resamples and report 95% CIs.

- `STRONG_PASS`: at least 10% improvement over shuffled suffixes and 95% CI lower bound above 0.
- `PASS`: at least 5% improvement and 95% CI lower bound above 0.
- `STATISTICAL_BUT_SMALL`: CI lower bound above 0 but improvement below 5%.
- `INCONCLUSIVE`: CI crosses 0.
- `FAIL`: CI upper bound at or below 0.

## Exploratory analyses

- Information-length curve for `prefix_1/5/10/20`;
- `last_frame` versus `full_20`;
- `residual_innovation_20` versus the complete raw suffix;
- Action and duration strata.

Exploratory comparisons do not support uncorrected confirmatory significance claims.

## Interpretation boundaries

An advantage over action-matched shuffled suffixes supports episode-specific factual information, but does not alone establish causal identifiability. Better residual-proxy performance would motivate structural innovation inversion; first-order prediction residuals cannot be interpreted directly as true exogenous variables.

