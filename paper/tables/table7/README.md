# Table 7: Adapted published-method comparison on development data

Scope: `ADAPTIVE_DEVELOPMENT_ONLY`, validation n=1,500. Terminal state: `STAGE5_MERGE_COMPARISON_UNDERPOWERED` (2026-09-28). No comparison holdout was created. Frozen strongest adaptation: `CCTR_DRIVING_CALIBRATION_ADAPTED`. Main manuscript comparison: `TABLE_7.md`; `TABLE_7_DETAILS.md` supplies adaptation, candidate, statistical, power and cost details. These results do not establish superiority over published methods. The independent timing receipt measures same-machine, batch=1, five-seed risk-head latency without truth access or gate changes.

## Files

- [build_stage5_manuscript_assets.py](build_stage5_manuscript_assets.py)
- [build_table7_details.py](build_table7_details.py)
- [stage5_inference_timing_receipt.json](stage5_inference_timing_receipt.json)
- [table7_comparison_main.csv](table7_comparison_main.csv)
- [table7_data.xlsx](table7_data.xlsx)
- [table7_source.json](table7_source.json)
- [TABLE_7.md](TABLE_7.md)
- [TABLE_7_DETAILS.md](TABLE_7_DETAILS.md)
- [Table_7_Published_Comparison.xlsx](Table_7_Published_Comparison.xlsx)

Build scripts organize existing frozen products only. They do not authorize training, inference, bootstrap reruns or re-adjudication. Some historical builders also require the manuscript source, maintained outside this code/data release.
