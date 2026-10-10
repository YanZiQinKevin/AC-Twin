# Frozen Protocol for Fair Identity-Continuity Controls

Date: 2026-09-05
Status at protocol freezing: `SMOKE_PENDING`

This document was frozen before a short CPU smoke run. Smoke checks cover data flow, training, outputs and recovery only; they do not support scientific performance conclusions or authorize formal Merge/Roundabout training.

## 1. Four frozen conditions

Each parameter budget must contain all four conditions, without condition-specific hyperparameter overrides that change comparison fairness:

1. `stable_temporal`: stable vehicle slots within an episode; a shared per-vehicle temporal encoder encodes each vehicle over time before aggregating vehicle representations.
2. `distance_temporal`: preserve the ego slot and reorder other vehicles by Euclidean distance to ego in every frame; use the same encoder and prediction head as `stable_temporal`.
3. `permuted_temporal`: deterministically shuffle non-ego slots for every episode and frame using frozen seeds; preserve the ego slot and use the same encoder and head as `stable_temporal`.
4. `set_pool`: stable-slot inputs, with permutation-invariant masked set pooling per frame before temporal encoding.

## 2. Shared inputs and training constraints

All conditions use the same object-centric tensor, episodes, train/validation/test indices, object capacity, visibility range, history, factual suffix, factual/counterfactual action plans, targets, loss, optimizer, learning rate, batch size and maximum epoch. The shared input has 8 dimensions:

`presence, is_ego, x, y, vx, vy, cos_heading, sin_heading`

Sort or shuffle in original metric coordinates, then apply shared object and target normalization statistics fitted only on visible source-training samples. Anchor coordinates at the final historical ego position. Permutations may not add, remove or alter object features.

## 3. Frozen capacity, seeds and formal evaluation

- Formal budgets: 65,000 and 83,000 trainable parameters; absolute relative deviation from each target must not exceed 1.5%. Report exact counts.
- Five training seeds in fixed order: `11, 22, 33, 44, 55`.
- Smoke scope: 65k, seed 11, at most 2 epochs and limited samples.
- Merge and Roundabout use independent fresh frozen test sets. Do not inspect test metrics during training; evaluate each scene once after configurations and training choices are frozen.
- Old development test splits support diagnostics and smoke checks only, not final paper conclusions.

## 4. Frozen statistical analysis

The primary endpoint is ego `effect-FDE@20`; `effect-ADE` is secondary. Primary contrasts are `permuted_temporal - stable_temporal` and `distance_temporal - stable_temporal`; `set_pool - stable_temporal` is secondary. Positive differences indicate higher control error than stable identity.

Retain aligned `[training seed, test episode]` error arrays. For 10,000 two-level paired bootstrap resamples, resample training seeds and test episodes with replacement and report 95% CIs. Episode-only intervals describe episode uncertainty and do not capture full training randomness.

## 5. Frozen `PASS_IDENTITY_CONTINUITY` gate

Record `PASS_IDENTITY_CONTINUITY` only if all conditions hold:

1. `stable_temporal` improves the primary endpoint over `permuted_temporal` in both Merge and Roundabout.
2. `stable_temporal` improves the primary endpoint over `distance_temporal` in both scenes.
3. Two-level uncertainty analysis does not show a conclusion driven entirely by one training seed.
4. Fairness, object multiset, ego slot, permutation, data consistency, capacity, finite values and recovery QA all pass.

Passing one scene supports scene-limited evidence only. Beating distance sorting without beating ID permutation does not identify an identity-continuity gain. Prefer simpler set pooling when `set_pool` and `stable_temporal` show no reliable difference. No outcome automatically establishes explicit abduction, causal identifiability, multi-vehicle response, safety gains or cross-scene generalization.

## 6. Execution scope at protocol freezing

Only CPU smoke was in scope: four conditions × 65k × one seed 11 × 2 epochs. Hierarchical bootstrap is unavailable with one training seed and cannot replace formal two-level inference. Engineering outcomes are restricted to `SMOKE_PASS` or `SMOKE_FAIL`; the pre-run protocol status remains `SMOKE_PENDING`, with the scientific gate unevaluated.
