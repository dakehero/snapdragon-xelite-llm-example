# Bootstrap script for fresh Windows 11 ARM64 machines (e.g. Qualcomm CRD / Cloud Device).
# Takes a blank Snapdragon X Elite box to first benchmark in ~40 minutes.
#
# Usage (open elevated PowerShell on the target machine):
#   iwr -useb https://raw.githubusercontent.com/dakehero/snapdragon-xelite-llm-example/main/scripts/bootstrap_cloud.ps1 | iex
#
# Or clone first and run locally:
#   git clone https://github.com/dakehero/snapdragon-xelite-llm-example
#   cd snapdragon-xelite-llm-example
#   pwsh -File scripts/bootstrap_cloud.ps1
#
# Time budget (approx, depends on bandwidth):
#   winget installs         :  2-3 min
#   pixi install            :  3-5 min
#   wheel install           :  1-2 min
#   Foundry model downloads : 10-15 min  (~7 GB total)
#   Smoke test              :  1 min
#   Full context sweep      : 22 min
#   --------------------------------
#   TOTAL                   : ~40-50 min

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"  # speeds up Invoke-WebRequest

function Step($msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Ensure-Command($cmd, $wingetId) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Step "Installing $cmd via winget ($wingetId)"
        winget install --id $wingetId --silent --accept-source-agreements --accept-package-agreements
        # winget modifies PATH; reload for current session
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [Environment]::GetEnvironmentVariable("Path", "User")
    } else {
        Write-Host "    $cmd already present"
    }
}

# -------------------------------------------------------------------------
# 1. Smart App Control warning
# -------------------------------------------------------------------------
Step "Checking Smart App Control status"
$sac = (Get-MpComputerStatus -ErrorAction SilentlyContinue).SmartAppControlState
if ($sac -eq "On") {
    Write-Warning "Smart App Control is ON. Pixi's Python may be blocked (error 4551)."
    Write-Warning "Manually turn it off: Settings > Privacy & Security > Windows Security > App & browser control > Smart App Control settings > Off"
    Write-Warning "Note: once OFF, SAC cannot be turned back on without a Windows reinstall."
    Write-Host "Continuing anyway; will fail later if SAC blocks python.exe..."
}

# -------------------------------------------------------------------------
# 2. Install prerequisites
# -------------------------------------------------------------------------
Ensure-Command "git"    "Git.Git"
Ensure-Command "pixi"   "prefix-dev.pixi"
Ensure-Command "foundry" "Microsoft.FoundryLocal"

# -------------------------------------------------------------------------
# 3. Clone or enter repo
# -------------------------------------------------------------------------
$RepoDir = "$env:USERPROFILE\snapdragon-xelite-llm-example"
if (-not (Test-Path "$RepoDir\pixi.toml")) {
    if (Test-Path ".\pixi.toml") {
        $RepoDir = (Get-Location).Path
    } else {
        Step "Cloning repo to $RepoDir"
        git clone https://github.com/dakehero/snapdragon-xelite-llm-example $RepoDir
    }
}
Set-Location $RepoDir
Write-Host "    Using repo at: $RepoDir"

# -------------------------------------------------------------------------
# 4. Pixi env
# -------------------------------------------------------------------------
Step "pixi install (Python 3.14 + deps)"
pixi install

# -------------------------------------------------------------------------
# 5. Install pre-built onnxruntime-genai wheel from GitHub Releases
# -------------------------------------------------------------------------
Step "Fetching latest onnxruntime-genai wheel"
$ReleaseApi = "https://api.github.com/repos/dakehero/snapdragon-xelite-llm-example/releases/latest"
$Release    = Invoke-RestMethod -Uri $ReleaseApi -Headers @{ "User-Agent" = "bootstrap" }
$WheelAsset = $Release.assets | Where-Object { $_.name -like "onnxruntime_genai-*-win_arm64.whl" } | Select-Object -First 1
if (-not $WheelAsset) {
    throw "No onnxruntime-genai win_arm64 wheel found in latest release. Fall back to building from source (see README)."
}
$WheelPath = Join-Path $env:TEMP $WheelAsset.name
if (-not (Test-Path $WheelPath)) {
    Invoke-WebRequest -Uri $WheelAsset.browser_download_url -OutFile $WheelPath
}
Write-Host "    Wheel: $($WheelAsset.name)"
pixi run python -m pip install --force-reinstall --no-deps $WheelPath
pixi run python scripts/install.py

# -------------------------------------------------------------------------
# 6. Download Foundry Local models (Qwen 2.5 7B NPU + CPU)
# -------------------------------------------------------------------------
Step "Downloading Qwen 2.5 7B models via Foundry Local (~7 GB total)"
foundry model download qwen2.5-7b-instruct-qnn-npu-2
foundry model download qwen2.5-7b-instruct-generic-cpu-4

# Resolve cache paths (Foundry Local puts models under ~\.foundry\cache\models\...)
$CacheRoot = "$env:USERPROFILE\.foundry\cache\models\Microsoft"
$NpuRoot   = Get-ChildItem "$CacheRoot\qwen2.5-7b-instruct-qnn-npu-2" | Select-Object -First 1
$CpuRoot   = Get-ChildItem "$CacheRoot\qwen2.5-7b-instruct-generic-cpu-4" | Select-Object -First 1
if (-not $NpuRoot -or -not $CpuRoot) {
    throw "Foundry Local model cache not found. Expected under $CacheRoot"
}
$NpuModel = $NpuRoot.FullName
$CpuModel = $CpuRoot.FullName
Write-Host "    NPU model: $NpuModel"
Write-Host "    CPU model: $CpuModel"

# -------------------------------------------------------------------------
# 7. Smoke test
# -------------------------------------------------------------------------
Step "Smoke test (CPU, 64-token prompt, 16-token decode)"
pixi run python llm_infer_ort_cpu.py $CpuModel --prompt-tokens 64 --decode-tokens 16

Step "Smoke test (NPU, 64-token prompt, 16-token decode)"
pixi run python llm_infer_ort_qnn.py $NpuModel --prompt-tokens 64 --decode-tokens 16

# -------------------------------------------------------------------------
# 8. Full context sweep
# -------------------------------------------------------------------------
Step "Running full context sweep (~22 minutes)"
pixi run python benchmark.py `
    --backend ort-qnn:llm_infer_ort_qnn.py:$NpuModel `
    --backend ort-cpu:llm_infer_ort_cpu.py:$CpuModel `
    --contexts 64,128,512,2048,8192 `
    --decode-tokens 128 `
    --warmup 1 --runs 3 `
    --output-md results/context_sweep_qwen7b.md

Step "Done. Results at: $RepoDir\results\context_sweep_qwen7b.md"
Get-Content "$RepoDir\results\context_sweep_qwen7b.md"
