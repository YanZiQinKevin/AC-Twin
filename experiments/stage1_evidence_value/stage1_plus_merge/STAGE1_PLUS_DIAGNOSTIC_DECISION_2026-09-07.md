# Stage 1 Plus Shape Diagnostics and Power Analysis: Decision

> Document status: **DEVELOPMENT_DIAGNOSTIC_ONLY**
> Plan ID: `AC-TWIN-RFER-STAGE1-PLUS-2026-09-07-V1`
> Parent plan: `AC-TWIN-RFER-2026-09-07-V1`
> Date: 2026-09-07
> claim_boundary：`development_diagnostic_only; does_not_reopen_or_rejudge_stage1_gate_a; does_not_confirm_monotonic_decay_or_u_shape; no_mechanism_or_causal_claim; stage2_and_router_remain_blocked_pending_user_approval`

---

## 1. Scope and boundaries

- Only repeatedly inspected Merge development-test data were used; **this is not confirmatory evidence**.
- Stage 1 Gates S/A/B were not re-adjudicated; the recorded merge=`DEVELOPMENT_GO_TO_NEW_HOLDOUT` status remains unchanged.
- At this historical decision point, Stage 2 and router work remained blocked pending approval; this diagnostic did not authorize collection, training or a router.
- Run flags: `training_performed=False`, `inference_performed=False`, `stage1_gate_a_unchanged=True`.

## 2. Input integrity

- Scene: `merge`; seed order `[11, 22, 33, 44, 55]`; development-test episodes = 300.
- split：train=1400 / validation=300 / development_test=300。
- Source data, Stage 1 configuration, response bins, test manifest and 15 test-prediction SHA256 values were verified before execution; see `input_hashes.json`. All Stage 1 Merge file hashes were unchanged before/after execution.

## 3. Reproducing Stage 1 point estimates (tolerance 1e-10)

| Quantity | Recomputed | report.json | abs diff | Passed |
|---|---:|---:|---:|:---:|
| `v_evt_low` | 0.280775424031 | 0.280775424031 | 0.00e+00 | ✓ |
| `v_evt_low_minus_high` | 0.067795793923 | 0.067795793923 | 0.00e+00 | ✓ |
| `spearman_r_eta` | 0.287322123668 | 0.287322123668 | 0.00e+00 | ✓ |
| `trend_log_r` | 0.283038029154 | 0.283038029154 | 1.61e-15 | ✓ |

| Bin V_evt | Recomputed | report.json | abs diff | Passed |
|---|---:|---:|---:|:---:|
| zero | 0.338865307590 | 0.338865307590 | 0.00e+00 | ✓ |
| positive_low | 0.031953321589 | 0.031953321589 | 0.00e+00 | ✓ |
| positive_mid | -0.024744016359 | -0.024744016359 | 0.00e+00 | ✓ |
| positive_high | 0.212979630108 | 0.212979630108 | 0.00e+00 | ✓ |

Original CI endpoints are cited unchanged; the new diagnostic bootstrap seed need not reproduce them.

## 4. D1 zero/positive presence

- Zero agent-time fraction = 0.8434
- Zero fraction of original_low shuffled-suffix error sum = 0.8107
- Zero fraction of original_low correct-suffix error sum = 0.7452
- `V_evt_zero` = 0.3389 [0.2875, 0.3878]
- `V_evt_positive_all` = 0.1413 [0.0984, 0.1806]
- `C_presence` = 0.1976 [0.1313, 0.2617]
- leave-one-seed-out `C_presence` = [0.2096, 0.1911, 0.2065, 0.1797, 0.2014]
- **PRESENCE_DIFFERENCE_SUPPORTED = True**

## 5. D2 positive-only magnitude trend

- positive-only `Spearman(R, eta_evt)` = 0.0059 [-0.0676, 0.0774]
- Positive-only `log1p(R)` trend coefficient = -0.0814 [-0.1669, 0.0032]
- `V_evt_positive_low - V_evt_positive_high` = -0.1810 [-0.3491, -0.0283]
- leave-one-seed-out spearman = [0.0142, 0.0046, -0.0126, 0.0421, -0.0193]
- leave-one-seed-out trend = [-0.0797, -0.0961, -0.0979, -0.0304, -0.0872]
- leave-one-seed-out low-high = [-0.1811, -0.1848, -0.2008, -0.1328, -0.2084]
- **POSITIVE_MAGNITUDE_DECAY_SUPPORTED = False**

## 6. D3 high-response rebound

- `C_high_low` = 0.1810 [0.0268, 0.3499]
- `C_high_mid` = 0.2377 [0.1437, 0.3436]
- `C_rebound` = minimum contrast = 0.1810 [0.0270, 0.2930]
- leave-one-seed-out `C_rebound` = [0.1811, 0.1848, 0.2008, 0.1328, 0.2084]
- leave-one-action-type-out `C_rebound`：
  - action 0: C_rebound=0.1558 (240 episodes retained, 60 removed)
  - action 1: C_rebound=0.1717 (240 episodes retained, 60 removed)
  - action 2: C_rebound=0.1034 (240 episodes retained, 60 removed)
  - action 3: C_rebound=0.2707 (240 episodes retained, 60 removed)
  - action 4: C_rebound=0.1628 (240 episodes retained, 60 removed)
- leave-one-episode-out: min = 0.1487, reversals = 0
- High-error deletion drop_top_1pct: C_rebound = 0.1869 (297 episodes retained)
- High-error deletion drop_top_5pct: C_rebound = 0.1897 (285 episodes retained)
- Independent episodes per positive bin = {'positive_low': 188, 'positive_mid': 169, 'positive_high': 111}
- **ROBUST_HIGH_REBOUND_SUPPORTED = True**

## 7. Complete bin table

| Response bin | V_evt | rho_no | meanL_no | meanL_suffix | meanL_shuffled | agent-time | episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero | 0.338865 | 0.370779 | 0.116 | 0.073 | 0.111 | 20241 | 300 |
| positive_low | 0.031953 | 0.128836 | 0.432 | 0.377 | 0.389 | 1343 | 188 |
| positive_mid | -0.024744 | -0.004185 | 0.933 | 0.937 | 0.915 | 1249 | 169 |
| positive_high | 0.212980 | 0.236242 | 3.039 | 2.321 | 2.949 | 1167 | 111 |

## 8. Power analysis (marginal and joint)

- Candidate episode counts: [300, 450, 600, 750, 900, 1200, 1500, 1800, 2400]; Monte Carlo replicates = 10000; bootstrap pool = 300; z = 1.959963984540054
- Power is approximated from the empirical development distribution, not a guarantee of confirmation; it avoids carrying forward an underpowered n=300 design.

### 8.1 Marginal power of the original four numerical Gate A conditions (empirical)

| n | v_evt_low | v_evt_low_minus_high | spearman_r_eta | trend_log_r | joint |
|---|---|---|---|---|---|
| 300 | 1.000 | 0.550 | 1.000 | 1.000 | 0.550 |
| 450 | 1.000 | 0.739 | 1.000 | 1.000 | 0.739 |
| 600 | 1.000 | 0.844 | 1.000 | 1.000 | 0.844 |
| 750 | 1.000 | 0.910 | 1.000 | 1.000 | 0.910 |
| 900 | 1.000 | 0.948 | 1.000 | 1.000 | 0.948 |
| 1200 | 1.000 | 0.986 | 1.000 | 1.000 | 0.986 |
| 1500 | 1.000 | 0.996 | 1.000 | 1.000 | 0.996 |
| 1800 | 1.000 | 0.999 | 1.000 | 1.000 | 0.999 |
| 2400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### 8.2 Joint power (three effect scenarios)

| n | empirical | 75% | 50% |
|---|---:|---:|---:|
| 300 | 0.550 | 0.297 | 0.032 |
| 450 | 0.739 | 0.492 | 0.093 |
| 600 | 0.844 | 0.641 | 0.167 |
| 750 | 0.910 | 0.757 | 0.270 |
| 900 | 0.948 | 0.845 | 0.384 |
| 1200 | 0.986 | 0.940 | 0.605 |
| 1500 | 0.996 | 0.974 | 0.760 |
| 1800 | 0.999 | 0.994 | 0.867 |
| 2400 | 1.000 | 1.000 | 0.974 |

### 8.3 Weakest condition (empirical)

- n=300：`v_evt_low_minus_high` = 0.550
- n=450：`v_evt_low_minus_high` = 0.739
- n=600：`v_evt_low_minus_high` = 0.844
- n=750：`v_evt_low_minus_high` = 0.910
- n=900：`v_evt_low_minus_high` = 0.948
- n=1200：`v_evt_low_minus_high` = 0.986
- n=1500：`v_evt_low_minus_high` = 0.996
- n=1800：`v_evt_low_minus_high` = 0.999
- n=2400：`v_evt_low` = 1.000

### 8.4 Sample-size recommendations by scenario

| Scenario | smallest_n all marginal powers ≥0.90 | smallest_n joint power ≥0.80 | smallest_n joint power ≥0.90 |
|---|---:|---:|---:|
| empirical | 750 | 600 | 750 |
| 75pct | 1200 | 900 | 1200 |
| 50pct | 2400 | 1800 | 2400 |

## 9. Mechanical status

- `PRESENCE_DIFFERENCE_SUPPORTED` = True
- `POSITIVE_MAGNITUDE_DECAY_SUPPORTED` = False
- `ROBUST_HIGH_REBOUND_SUPPORTED` = True
- **Unique shape_recommendation = `REVISE_H2_TO_NONMONOTONIC_BEFORE_STAGE2`**

## 10. Recommendation for Stage 2

Development data showed a robust high-response rebound without monotonic attenuation within positive responses. Before collecting a fresh confirmation set, formally revise H2 to nonmonotonic response dependence and preregister the shape contrasts. This does not confirm a U shape. Revision of the parent plan and Stage 2 required user approval.

## 11. Unsupported claims

- Neither a U shape nor strict binwise monotonic attenuation is confirmed;
- No collision, lane-change or event-identity mechanism is established;
- Do not describe the response–evidence-value relationship as causal;
- Do not output `GATE_A_CONFIRMED` or authorize Stage 2 collection/router work;
- Do not use dual-edged evidence wording (Gate B remains NOT_SUPPORTED).

## 12. Verification

- Full tests: recorded in local QA/full-regression records;
- Stage 1 sealed-artifact hashes were unchanged (before_after_identical=True);
- `training_performed=False`、`inference_performed=False`。
- Run time 254.6s; Python 3.12.7.
