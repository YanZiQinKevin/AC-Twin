# replication_v2 protocol

## Objective

Test whether the episode-specific factual-suffix benefit replicates under (1) an independent collection seed in the original merge scene and (2) a second interaction topology in `roundabout-v0`.

## Frozen confirmatory comparison

- Reference: `shuffled_full`, matched within dataset split, counterfactual action, intervention duration, and—as far as strata permit—the first factual action.
- Candidate: `prefix_10`, using the first 10 observed steps of the same episode's factual suffix.
- Primary endpoint: episode-level effect-FDE at horizon 20.
- Independent unit: test episode pair.
- Inference: paired bootstrap with 10,000 resamples and 95% confidence interval.
- Training repeats: five initialization seeds; these do not increase the inferential sample size.

## Datasets

1. `merge-v0`, 2,000 accepted pairs, independent collection seed `20260914`.
2. `roundabout-v0`, 2,000 accepted pairs, collection seed `20260915`.

Each dataset is stratified by counterfactual action and intervention duration into train/validation/test splits of 70%/15%/15%.

## Decision gate

The replication gate passes only if both datasets satisfy:

1. `prefix_10` improves mean effect-FDE over `shuffled_full` by at least 5%.
2. The paired 95% confidence-interval lower bound is greater than zero.
3. Deterministic identity replay has zero numerical difference within the configured tolerance.
4. No major action family shows a systematic reversal that invalidates the aggregate conclusion.

Only after this gate passes will AC-Twin proceed to stable object identities, object-centric state encoding, and dynamic causal interaction graphs.
