# AC-Twin

AC-Twin studies retrospective counterfactual traffic prediction in reactive multi-agent simulations. The code covers response-conditioned factual-evidence evaluation, multi-action prediction, decision-regret supervision, and risk-based auditing of a frozen action selector. Earlier ablation and architecture source is retained for implementation traceability.

This is a **code-only repository, currently private**. It contains source, tests, run configurations, dependencies and plotting scripts. Manuscripts, plans, discussion drafts, datasets, numerical result exports, model weights, reports and rendered figures are excluded. No open-source licence has been selected yet.

## Layout

```text
code/                 Collection, training, inference, statistics and staged runners
code/ac_twin/         Model, feature, objective and protocol implementations
config/               Standalone experiment configurations
experiments/          Source scripts and configuration JSON only; no data or results
legacy/               Earlier baseline source and its requirements
paper/figures/        Figure-generation source only
paper/tables/         Table-generation source only
tests/                Unit, integration and historical-artifact checks
tools/check_release.py  Code-only boundary, size, syntax and credential-pattern check
```

The `paper/` and `experiments/` paths preserve source imports and input/configuration locations; they do not distribute manuscripts or experiment products. Generated outputs remain local and ignored.

## Main environment

Observed on the packaging machine on 2026-10-06: Python 3.13.5, PyTorch 2.14.0+cu132, NumPy 2.1.3, SciPy 1.15.3, pandas 2.2.3, Gymnasium 1.2.3, highway-env 1.10.2, Matplotlib 3.10.0, scikit-learn 1.6.1, Pillow 11.3.0 and pytest 8.3.4. These are packaging-environment versions, not a claim that all historical experiments used this environment.

`requirements.txt` pins the installed non-PyTorch dependencies. Install the PyTorch CPU/CUDA build appropriate for your machine using the [official installer](https://pytorch.org/get-started/locally/). Keep the historical environment separate when reproducing a frozen run.

PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch
python -c "import torch, highway_env; print(torch.__version__); print(torch.cuda.is_available())"
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
```

## Data sources

- **Merge, intersection and roundabout simulation:** generated locally with [Farama highway-env](https://github.com/Farama-Foundation/HighwayEnv), using the collection configurations included here. No simulator episodes are distributed.
- **External scenarios:** [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/), scenario release v1.2.0, `training_20s`, converted with [ScenarioNet](https://github.com/metadriverse/scenarionet) and simulated with [MetaDrive](https://github.com/metadriverse/metadrive). Obtain the dataset from the provider and comply with its terms. Raw TFRecords and converted scenario files are not redistributed.
- **Analysis inputs:** checkpoints, predictions, paired branch outcomes, group metadata, result tables and figure CSVs are local products of the collection/training/analysis workflows, not additional public datasets. None are included.

The configurations specify expected local input and output paths. External-platform paths must be configured consistently with the corresponding environment checks. Figure icon/screenshot assets are also excluded from this code-only repository.

## Commands

These examples create local outputs and can run for a long time. Use a fresh reproduction workspace; do not overwrite an existing completed run.

```powershell
# Paired highway-env pilot and prediction.
python code/collect_pilot.py --config config/pilot_2000.json
python code/run_abduction_pilot.py --config config/pilot_2000.json

# History / factual-suffix ablation.
python code/run_suffix_ablation.py --config config/suffix_evidence_ablation.json
# This plotting script takes no CLI arguments and needs the ablation report.
python code/plot_suffix_ablation.py

# Small evidence-value run, after preparing its configured inputs.
python code/run_stage1_evidence_value.py --config config/stage1_evidence_value_smoke.json --device cpu
```

| Task | Entry point | Configuration / prerequisites |
|---|---|---|
| Evidence-value confirmation | `code/run_stage2_confirmatory.py` | `config/stage2_holdout_merge_confirm.json`; development inputs and protocol lock |
| Response prediction | `code/run_stage3a_plus.py` | Preserved `experiments/stage3_routing/stage3a_plus/` configurations |
| Multi-action Direct development | `code/run_stage4cr2.py` | `experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json`; upstream prediction and risk stack |
| Independent Merge audit | `code/run_stage4cr2_confirm.py` | `experiments/stage4_reliability/stage4cr2_confirm/stage4cr2_confirm_config.json`; frozen development stack |
| Generator mismatch | `code/run_stage4d_ood.py` | `experiments/stage4_reliability/stage4d_ood/config.json`; frozen Merge stack |
| MetaDrive audit | `code/drive_stage4d_d2f.py` | Separate platform environments, scenarios and D2/D2E upstream products |
| Published-method comparison | `code/ac_twin/stage5_*.py` | `config/stage5_sota_comparison_v0.3.json`; local development inputs |

```powershell
python code/run_stage2_confirmatory.py --help
python code/run_stage4cr2.py --help
python code/run_stage4cr2_confirm.py --help
python code/run_stage4d_ood.py --help
python code/drive_stage4d_d2f.py --help

# Runner syntax after preparing all upstream dependencies:
python code/run_stage4cr2.py run-all --config experiments/stage4_reliability/stage4c_r2/stage4cr2_config.json
```

Historical runners bind upstream hashes, split manifests, seed roles, protocol locks and sometimes internal protocol documents. Those dependencies are deliberately not shipped as datasets or manuscripts. This checkout alone is **not a complete reconstruction of a historical confirmation run**. Missing inputs must be prepared through the relevant upstream workflow; do not bypass fail-closed checks. Use fresh output namespaces for new studies, never reuse consumed confirmation resources and do not count reruns as independent confirmations.

## MetaDrive / Waymo environments

The external configuration records two environments:

- Conversion: Linux/WSL, Python 3.9, TensorFlow 2.11.0, Waymo conversion package 1.5.0. Install `requirements-waymo-conversion.txt` in that environment; install ScenarioNet separately.
- Simulation: Python 3.11, historical PyTorch target 2.9.1+cu130, MetaDrive revision `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`, ScenarioNet revision `d4acdb5f5a844744fc85cb2dc3880d7d4a6eb170`. TensorFlow conversion belongs in the other environment.

```powershell
# Inspect the wrapper after configuring its interpreter paths.
powershell -ExecutionPolicy Bypass -File code/run_stage4d_env.ps1 show
powershell -ExecutionPolicy Bypass -File code/run_stage4d_env.ps1 sim code/drive_stage4d_d2f.py --help
```

`code/run_stage4d_env.ps1`, platform verification modules and external configuration JSON retain historical absolute paths, revision checks and environment contracts. They are not portable installation scripts. Configure the corresponding paths consistently before use; the main requirements do not recreate the external environment.

## Figure and table source

Directory names retain older figure numbering. Current manuscript numbering and excluded prerequisites:

| Figure | Source | Required local input |
|---|---|---|
| Fig. 1: evidence value | `paper/figures/fig1/build_fig1.py` | Figure-folder bin and contrast CSV exports |
| Fig. 2: overview | `paper/figures/fig2/build_fig2.py` | `icon/` SVG/PNG assets, including the observed-prefix illustration |
| Fig. 3: audit head | `paper/figures/fig5_2/draw_direct_audit_v2.py` | None; drawn from code |
| Fig. 4: audit scopes | `paper/figures/fig3/build_fig3.py` | Local per-group outputs and confirmation reports |
| Fig. 5: MetaDrive cases | `paper/figures/fig5/build_fig5.py` | Scenario database, case metadata, render provenance and upstream outputs |
| Fig. 6: robustness | `paper/figures/fig4/build_fig4.py` | Confirmation tables and mismatch-selection products |

Additional source includes the graphical abstract, external-platform render/export helpers, the author-fixed Fig. 5 assembly helper and Stage 5 table builders. Original relative input paths are retained. Exact asset-dependent reproduction requires the excluded local assets; scripts do not substitute invented values or replacement images.

```powershell
# Self-contained schematic; creates ignored outputs beside the source.
python paper/figures/fig5_2/draw_direct_audit_v2.py

# Run only after supplying the inputs listed above.
python paper/figures/fig1/build_fig1.py
python paper/figures/fig2/build_fig2.py
python paper/figures/fig3/build_fig3.py
python paper/figures/fig4/build_fig4.py
python paper/figures/fig5/build_fig5.py
```

## Checks before committing

```powershell
python tools/check_release.py
python -m pytest tests/test_progress.py tests/test_checkpointing.py tests/test_stage1_model.py tests/test_stage1_statistics.py tests/test_stage4c_objective.py -q
```

The release check parses Python/configuration JSON, scans for credential patterns and rejects files of 1 MiB or larger. `.gitignore` uses an explicit file allowlist: new generated data/media stay excluded, and new source paths must be added deliberately.

The full test collection includes platform-specific and historical-artifact checks requiring local datasets, checkpoints, internal documents or external environments omitted here. A passing standalone test subset does not establish full scientific reproduction.

## Interpretation boundaries

- Statistical units are episodes or multi-action decision groups, not training seeds or trajectory time points.
- Keep train/validation/screen/confirmation roles separate and freeze methods before confirmation.
- Regret is relative to the fixed candidate actions and consequence function. Retention does not change the executed action.
- MetaDrive evaluation uses platform-internal retraining; it does not establish zero-shot weight transfer, real-road counterfactual ground truth or closed-loop safety improvement.
- Earlier routing, identity and architecture source is retained for traceability; its presence does not establish it as the current method.
