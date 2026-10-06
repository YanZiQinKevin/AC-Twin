[CmdletBinding(PositionalBinding = $true)]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('show', 'verify', 'convert', 'sim', 'run-all')]
    [string]$Mode,

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$SimEnvRoot = 'C:\ProgramData\anaconda3\envs\ac_twin_d2_sim_py311'
$SimPython = Join-Path $SimEnvRoot 'python.exe'
$WslDistribution = 'Ubuntu-22.04'
$ConversionPython = '/root/miniconda3/envs/ac_twin_waymo_convert_py39/bin/python'
$ConversionStagingRoot = '/root/ac_twin_stage4d_conversion_work'
$ConfigPath = 'experiments/stage4_reliability/stage4d_external/config.json'
$SimBootstrap = 'code/stage4d_sim_entry.py'

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 4D environment command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

if (-not (Test-Path -LiteralPath $SimPython -PathType Leaf)) {
    throw "Frozen Stage 4D simulation interpreter is missing: $SimPython"
}

# Directly invoking a conda interpreter does not activate its DLL search path.
# Freeze the required environment-local directories here instead of relying on
# whichever conda/base shell happened to launch this script.
$SimDllPaths = @(
    $SimEnvRoot,
    (Join-Path $SimEnvRoot 'Library\mingw-w64\bin'),
    (Join-Path $SimEnvRoot 'Library\usr\bin'),
    (Join-Path $SimEnvRoot 'Library\bin'),
    (Join-Path $SimEnvRoot 'Scripts'),
    (Join-Path $SimEnvRoot 'bin')
)
$env:PATH = (($SimDllPaths + @($env:PATH)) -join [IO.Path]::PathSeparator)

Push-Location -LiteralPath $ProjectRoot
try {
    switch ($Mode) {
        'show' {
            [ordered]@{
                project_root = $ProjectRoot
                sim_python = $SimPython
                wsl_distribution = $WslDistribution
                conversion_python = $ConversionPython
                conversion_staging_root = $ConversionStagingRoot
                config = $ConfigPath
                sim_path_prefix = $SimDllPaths
                sim_bootstrap = $SimBootstrap
            } | ConvertTo-Json
        }
        'verify' {
            Invoke-Checked -Executable $SimPython -Arguments @(
                $SimBootstrap,
                'code/stage4d_verify_sim_env.py'
            )
            Invoke-Checked -Executable 'wsl.exe' -Arguments @(
                '-d', $WslDistribution,
                '--cd', $ProjectRoot,
                '--', 'env', 'TF_CPP_MIN_LOG_LEVEL=3',
                "AC_TWIN_D2_CONVERSION_STAGING_ROOT=$ConversionStagingRoot",
                $ConversionPython,
                'code/stage4d_verify_conversion_env.py'
            )
            Write-Output 'STAGE4D_ENVIRONMENT_VERIFY_PASS'
        }
        'convert' {
            if (-not $CommandArgs -or $CommandArgs.Count -eq 0) {
                throw 'convert requires a Python script/module and its arguments.'
            }
            $convertArguments = @(
                '-d', $WslDistribution,
                '--cd', $ProjectRoot,
                '--', 'env', 'TF_CPP_MIN_LOG_LEVEL=3',
                "AC_TWIN_D2_CONVERSION_STAGING_ROOT=$ConversionStagingRoot",
                $ConversionPython
            ) + $CommandArgs
            Invoke-Checked -Executable 'wsl.exe' -Arguments $convertArguments
        }
        'sim' {
            if (-not $CommandArgs -or $CommandArgs.Count -eq 0) {
                throw 'sim requires a Python script/module and its arguments.'
            }
            $simArguments = @($SimBootstrap) + $CommandArgs
            Invoke-Checked -Executable $SimPython -Arguments $simArguments
        }
        'run-all' {
            Invoke-Checked -Executable $SimPython -Arguments @(
                $SimBootstrap,
                'code/run_stage4d_external.py',
                'run-all',
                '--config', $ConfigPath
            )
        }
    }
}
finally {
    Pop-Location
}
