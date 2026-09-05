# AC-Twin

AC-Twin is a research codebase for episode-specific counterfactual traffic prediction in reactive multi-agent simulations. It contains data collection, stable-identity auditing, paired-effect prediction, factual-evidence ablation, and capacity-controlled interaction-encoder comparisons.

This repository contains source code, experiment configuration, and tests only. Datasets, checkpoints, predictions, logs, figures, and experiment reports are generated locally and are not included.

## Environment

Validated environment:

- Windows 11 / PowerShell
- Python 3.12.7
- PyTorch 2.9.1 with CUDA 13.0
- NumPy 1.26.4
- Gymnasium 1.2.x
- highway-env 1.10.2
- Matplotlib 3.x
- pytest 8.x

Create and activate a virtual environment, then install the required packages. Select the appropriate PyTorch command for your CPU or CUDA version from the official PyTorch installation guide.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas gymnasium highway-env matplotlib pytest
python -m pip install torch
```

Verify the installation:

```powershell
python -c "import torch, highway_env; print(torch.__version__); print(torch.cuda.is_available())"
python -m pytest tests -q
```

All commands below assume the repository root is the current directory. Configuration files use project-relative paths so the code can run from different computers and drive letters.

## Main workflow

### 1. Collect the 2,000-episode paired dataset

```powershell
python code/collect_pilot.py --config config/pilot_2000.json
```

### 2. Run the history-versus-factual-suffix pilot

```powershell
python code/run_abduction_pilot.py --config config/pilot_2000.json
```

### 3. Run the factual-evidence ablation

```powershell
python code/run_suffix_ablation.py --config config/suffix_evidence_ablation.json
```

### 4. Audit stable vehicle identities

```powershell
python code/audit_stable_id.py
```

### 5. Collect object-centric merge data

```powershell
python code/collect_pilot.py --config config/object_centric_merge_collect.json
```

### 6. Run the short architecture smoke test

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python code/run_architecture_comparison.py --config config/architecture_capacity_smoke.json
```

### 7. Run the full capacity-controlled architecture comparison

This experiment runs two capacity tiers, ten conditions, and five training seeds. It can take a long time; run it manually while keeping the computer awake.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python code/run_architecture_comparison.py --config config/architecture_capacity_merge.json
```

Generated files are written under `data/` and `experiments/`. Both directories are intentionally ignored by Git.

## Repository layout

```text
code/       Python source code
config/     Reproducible JSON experiment configurations
tests/      Model and configuration tests
README.md   Environment and command reference
```

## Reproducibility notes

- The statistical unit is an episode, not a trajectory time point or training seed.
- Confirmatory comparisons use five training seeds and paired episode-level bootstrap confidence intervals.
- Object-centric graph models use stable episode-local vehicle identities.
- Distance-gated graph edges are computed in metric coordinates even though node features are standardized.
- Simulator datasets must be regenerated locally before training.

