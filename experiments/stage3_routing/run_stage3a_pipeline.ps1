# AC-Twin Stage 3A overnight pipeline (user-launched, manual).
# Runs search -> fit-five -> select-final -> gate-r-dev in order, logs everything,
# and stops on a real crash. Gate exit codes 0/2/3 are all terminal outcomes
# (2 = gate adjudicated fail, 3 = single-use/tamper refusal), not crashes.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File experiments\stage3_routing\run_stage3a_pipeline.ps1
# To RESUME after an interruption, add -Resume (passes --resume to search/fit-five).

param(
    [switch]$Resume
)

$ErrorActionPreference = "Continue"
# Derive the repo root from this script's own location (avoids hardcoding any
# non-ASCII path and is independent of the launch directory).
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo

$config  = "experiments/stage3_routing/stage3a_config.json"
$source  = "data/object_centric_stage/merge_2000/object_twinworld_2000.npz"
$rundir  = "experiments/stage3_routing/stage3a"
$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$log     = "experiments/stage3_routing/stage3a_pipeline_$stamp.log"

$resumeArgs = @()
if ($Resume) { $resumeArgs = @("--resume") }

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$PyArgs,
        [int[]]$OkCodes = @(0)   # exit codes treated as "step finished, continue"
    )
    $line = "=" * 70
    "$line`n[$((Get-Date).ToString('HH:mm:ss'))] START $Name`n  python $($PyArgs -join ' ')`n$line" |
        Tee-Object -FilePath $log -Append
    & python @PyArgs 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    "[$((Get-Date).ToString('HH:mm:ss'))] END $Name exit=$code" | Tee-Object -FilePath $log -Append
    if ($OkCodes -notcontains $code) {
        "ABORT: $Name exited $code (expected one of $($OkCodes -join ',')). Halting pipeline." |
            Tee-Object -FilePath $log -Append
        exit $code
    }
    return $code
}

"AC-Twin Stage 3A pipeline started $stamp (Resume=$Resume). Log: $log" | Tee-Object -FilePath $log -Append

# 1. single-seed bounded screen (must exit 0)
Invoke-Step -Name "search" -PyArgs (@("code/run_stage3a.py","search","--config",$config,"--source",$source,"--run-dir",$rundir) + $resumeArgs)

# 2. five-seed fit of finalists (must exit 0)
Invoke-Step -Name "fit-five" -PyArgs (@("code/run_stage3a.py","fit-five","--config",$config,"--source",$source,"--run-dir",$rundir) + $resumeArgs)

# 3. freeze unique final config + validation-frozen baselines (must exit 0)
Invoke-Step -Name "select-final" -PyArgs @("code/run_stage3a.py","select-final","--config",$config,"--source",$source,"--run-dir",$rundir)

# 4. single development-test Go/No-Go (0/2/3 all terminal)
$gateCode = Invoke-Step -Name "gate-r-dev" -PyArgs @("code/run_stage3a.py","gate-r-dev","--config",$config,"--source",$source,"--run-dir",$rundir,"--n-resamples","2000") -OkCodes @(0,2,3)

$summary = switch ($gateCode) {
    0 { "Gate R_dev: PASSED" }
    2 { "Gate R_dev: ADJUDICATED FAIL (report written)" }
    3 { "Gate R_dev: REFUSED (single-use/tamper)" }
}
"=" * 70 | Tee-Object -FilePath $log -Append
"PIPELINE DONE. $summary" | Tee-Object -FilePath $log -Append
"Reports: $rundir\gate_r_dev_report_*.json ; Log: $log" | Tee-Object -FilePath $log -Append
