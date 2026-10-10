# Inspecting and reproducing the released results

Start with `release/summary.json` and `release/manifest.csv`. Large-file metadata are recorded in `release/large_data_inventory.json`. Contact the authors for access; no hosted download links are provided. Relative paths match the research workspace; verify SHA-256 after obtaining external files.

## Evidence-to-artifact map

| Evidence | Preserved artifacts |
| --- | --- |
| Merge factual-evidence value | `experiments/stage2_confirmatory/merge/`; datasets under `data/object_centric_stage/stage2_merge_1200/` |
| Response qualification | `experiments/stage3_routing/stage3a_plus/` |
| Merge Direct development | `experiments/stage4_reliability/stage4c_r2/` |
| Independent Merge decision audit | `experiments/stage4_reliability/stage4cr2_confirm/` |
| Specified generator mismatch | `experiments/stage4_reliability/stage4d_ood/` |
| MetaDrive repaired model stack | `experiments/stage4_reliability/stage4d_external/development/`, `experiments/stage4_reliability/stage4d_d2e/` |
| MetaDrive external audit | `experiments/stage4_reliability/stage4d_d2f/` |
| Adapted published methods | `experiments/stage5_sota_comparison/`; development evidence, no comparison holdout |
| Supervised controls and feature ablations | `experiments/paper_revision/m1_supervised_ablation_v02/`; development evidence |

Reports, configuration and weights are provided as historical artifacts. Inspect existing reports directly before attempting an expensive rerun. A copied lock or manifest does not authorize another confirmation adjudication. For independent reproduction, use a fresh workspace, fresh output namespace and an explicitly separate study protocol.

The release does not launch collectors, train models, run inference or recompute statistics. MetaDrive training data are excluded, so reconstructing its training requires obtaining provider scenarios and regenerating the study's training role. Historical absolute paths and protocol dependencies may need explicit reconciliation; never disable integrity checks just to make an old runner proceed.

## Figures and tables

Current figure order is: overview (`fig2`), audit head (`fig5_2`), evidence value (`fig1`), three audit scopes (`fig3`), MetaDrive cases (`fig5`), robustness (`fig4`). Directory names preserve established script paths.

Main tables are in `paper/tables/table1` through `table8`; supplementary tables are in `tableS1` through `tableS5`. Table 8 contains the supervised controls and feature ablations. Figure/source data and table exports retain their original values.

## Release validation

```powershell
python tools/check_release.py
```

This checks the supplied-file inventory, hashes, external-file exclusion, syntax, JSON records and credential patterns. It verifies packaging integrity, not scientific replication or fresh confirmation. Historical full-suite tests may require external data and separate platform environments.

To update the mapping from an authorized local research workspace:

```powershell
python tools/sync_release.py --source <research-workspace>
python tools/check_release.py
```

Synchronization is read-only toward the source. Review the inventory and exclusion report before committing. Large products remain outside Git; access is discussed directly with the authors.
