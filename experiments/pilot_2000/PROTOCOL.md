# pilot_2000 Preregistered-Style Experimental Protocol

Frozen on: 2026-09-04

## Hypothesis

With identical pre-intervention histories, factual/counterfactual action sequences and model capacities, adding a factual suffix reduces individual-effect endpoint error on test episodes.

## Controls

- `history_only`: the preceding 10 history steps plus factual and counterfactual action sequences; the factual-suffix encoding is zeroed.
- `factual_suffix`: identical inputs to `history_only`, with an additional encoding of the subsequent 20 factual steps.
- Both use the same architecture, training split, optimization budget and five training seeds.
- `zero_effect`: always predicts no effect from the alternative action.
- `action_pair_mean`: uses the training-set mean effect conditioned only on factual action, counterfactual action and intervention duration.

## Data and independent unit

- 2,000 distinct episode seeds, each producing one factual/counterfactual pair.
- Branches share the seed, preceding actions and subsequent actions outside the replacement; only the first 1 or 3 steps are replaced in the counterfactual branch.
- Identity replay checks reproducibility under the same seed and actions.
- The independent statistical unit is the test episode. Five training seeds quantify algorithmic randomness and do not increase sample size.

## Primary outcome

`effect-FDE@20 = || predicted_delta_20 - true_delta_20 ||_2`。

The paired difference is `history_only error - factual_suffix error` per test episode. Compute errors separately under five training seeds, average the errors within each episode, then perform 10,000 episode-level paired bootstrap resamples to report the mean difference and 95% CI. Averaging predictions into an ensemble before computing the primary result is prohibited.

## Decision rules

- `STRONG_PASS`: factual-suffix improvement of at least 10%, with a mean-difference 95% CI lower bound above 0.
- `PASS`: improvement of at least 5%, with a 95% CI lower bound above 0.
- `STATISTICAL_BUT_SMALL`: CI lower bound above 0, but improvement below 5%.
- `INCONCLUSIVE`: CI crosses 0.
- `FAIL`: CI upper bound at or below 0.

These thresholds guide whether to continue developing the abduction module; they are not publication-level significance claims.

## Secondary outcomes

- effect-ADE@20；
- CF-ADE/CF-FDE from the factual trajectory plus the predicted effect, for description only;
- Stratification by counterfactual action and duration;
- Mean and standard deviation across five training seeds.

No confirmatory significance claims are made across multiple secondary outcomes.

## Exclusions

- Episodes terminated before intervention;
- Either branch cannot complete 20 steps;
- Inconsistent observations at the branching point;
- Non-finite numerical values;
- Identical factual and counterfactual action sequences.

Do not remove samples based on effect magnitude; zero effects are valid negative responses.

## Main limitations

- Highway-env reset-and-replay is not a general simulator's complete state snapshot.
- The observation matrix lacks cross-branch stable surrounding-vehicle IDs; only ego effects are primary targets here.
- The factual-suffix model supports retrospective analysis, not online planning before factual outcomes are observed.
