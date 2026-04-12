# Build ONNX Runtime with QNN support on Windows ARM64
# Requirements: Visual Studio 2026/2022 with ARM64 build tools, QNN SDK installed

param(
    [switch]$SkipVSCheck,
    [string]$QnnSdkRoot = $env:QNN_SDK_ROOT
)

Write-Host "=== ONNX Runtime QNN Build for Windows ARM64 ===" -ForegroundColor Green

# Set QNN SDK environment variables
if (-not $QnnSdkRoot) {
    Write-Host "QNN SDK path not specified. Use -QnnSdkRoot or set QNN_SDK_ROOT env var." -ForegroundColor Red
    exit 1
}
$env:QNN_SDK_ROOT = $QnnSdkRoot
$env:ORT_QNN_EP_PATH = $QnnSdkRoot
$env:QNN_TARGET_ARCH = "ARM64"

Write-Host "QNN SDK Path: $env:QNN_SDK_ROOT" -ForegroundColor Cyan

# Verify QNN SDK
if (-not (Test-Path "$env:QNN_SDK_ROOT\lib\aarch64-windows-msvc\QnnSystem.dll")) {
    Write-Host "QNN SDK not found at $env:QNN_SDK_ROOT" -ForegroundColor Red
    exit 1
}
Write-Host "✓ QNN SDK verified" -ForegroundColor Green

# Check Visual Studio if not skipped
if (-not $SkipVSCheck) {
    Write-Host "`nChecking Visual Studio..." -ForegroundColor Yellow
    
    # VS2026 uses directory "18" (VS2022 version number)
    $vsPaths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\18\Community",
        "${env:ProgramFiles}\Microsoft Visual Studio\18\Professional",
        "${env:ProgramFiles}\Microsoft Visual Studio\18\Enterprise",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\18\Community",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\18\Professional",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\18\Enterprise"
    )
    
    $vsPath = $null
    foreach ($path in $vsPaths) {
        if (Test-Path $path) {
            $vcvarsall = "$path\VC\Auxiliary\Build\vcvarsall.bat"
            if (Test-Path $vcvarsall) {
                $vsPath = $path
                Write-Host "✓ Visual Studio 2026 found at: $vsPath" -ForegroundColor Green
                break
            }
        }
    }
    
    if (-not $vsPath) {
        Write-Host "✗ Visual Studio 2026 not found!" -ForegroundColor Red
        Write-Host "Please install VS2026 with C++ ARM64 build tools" -ForegroundColor Red
        Write-Host "Or run with -SkipVSCheck to proceed anyway" -ForegroundColor Yellow
        exit 1
    }
}

# Import VS development environment for ARM64
if ($vsPath) {
    $vcvarsall = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"
    if (Test-Path $vcvarsall) {
        Write-Host "`nImporting VS ARM64 build environment..." -ForegroundColor Yellow
        $output = cmd /c "`"$vcvarsall`" arm64 > nul 2>&1 && set" 2>$null
        if ($output) {
            $output | ForEach-Object {
                if ($_ -match "^([^=]+)=(.*)$") {
                    [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
                }
            }
            Write-Host "✓ VS ARM64 environment loaded" -ForegroundColor Green
        }
    }
}

# Clone onnxruntime-genai if not present (fixed to v0.13.0 for reproducible builds)
if (-not (Test-Path "onnxruntime-genai")) {
    Write-Host "`nCloning onnxruntime-genai (v0.13.0)..." -ForegroundColor Yellow
    git clone --branch v0.13.0 --depth 1 https://github.com/microsoft/onnxruntime-genai.git
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to clone onnxruntime-genai" -ForegroundColor Red
        exit 1
    }
}

# Navigate to build directory
$RepoRoot = Get-Location
Set-Location "onnxruntime-genai"
Write-Host "`nBuild directory: $(Get-Location)" -ForegroundColor Yellow

# Install dependencies
Write-Host "`nInstalling Python dependencies..." -ForegroundColor Yellow
pixi run --manifest-path "$RepoRoot\pixi.toml" python -m pip install numpy pybind11 requests wheel --upgrade --quiet

# Build command
Write-Host "`nStarting build..." -ForegroundColor Green
Write-Host "This will take 30-60 minutes..." -ForegroundColor Yellow

# First, check if onnxruntime-qnn is available
Write-Host "`nChecking for onnxruntime-qnn..." -ForegroundColor Yellow
$qnnCheck = pixi run --manifest-path "$RepoRoot\pixi.toml" python -m pip show onnxruntime-qnn 2>$null
if ($qnnCheck) {
    Write-Host "✓ onnxruntime-qnn already installed" -ForegroundColor Green
} else {
    Write-Host "⚠ onnxruntime-qnn not found" -ForegroundColor Yellow
    Write-Host "Install it first: pixi run pip install onnxruntime-qnn" -ForegroundColor Cyan
    exit 1
}

$buildArgs = @(
    "build.py",
    "--config", "RelWithDebInfo",
    "--parallel",
    "--update",
    "--build",
    "--cmake_generator", "Ninja",
    "--cmake_extra_defines", "CMAKE_CXX_FLAGS=/EHsc",
    "--skip_examples"
)

Write-Host "Command: pixi run python $($buildArgs -join ' ')" -ForegroundColor Cyan

# Execute build
$startTime = Get-Date
pixi run --manifest-path "$RepoRoot\pixi.toml" python $buildArgs
$buildTime = (Get-Date) - $startTime

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Build completed in $($buildTime.TotalMinutes.ToString('F1')) minutes!" -ForegroundColor Green

    # Copy NuGet + QNN DLLs to genai package directory
    Write-Host "`nCopying DLLs to genai package..." -ForegroundColor Yellow
    $genaiPkgDir = (pixi run --manifest-path "$RepoRoot\pixi.toml" python -m pip show onnxruntime-genai 2>$null) | Select-String -Pattern "Location: (.+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }
    if ($genaiPkgDir) {
        $genaiPkgDir = Join-Path $genaiPkgDir "onnxruntime_genai"
        $nugetDllDir = "build\Windows\RelWithDebInfo\_deps\ortlib-src\runtimes\win-arm64\native"
        if (Test-Path $nugetDllDir) {
            Copy-Item "$nugetDllDir\*.dll" $genaiPkgDir -Force
            Write-Host "Copied NuGet DLLs" -ForegroundColor Green
        }
        $qnnPkgDir = (pixi run --manifest-path "$RepoRoot\pixi.toml" python -m pip show onnxruntime-qnn 2>$null) | Select-String -Pattern "Location: (.+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }
        if ($qnnPkgDir) {
            $qnnPkgDir = Join-Path $qnnPkgDir "onnxruntime_qnn"
            if (Test-Path $qnnPkgDir) {
                Copy-Item "$qnnPkgDir\*.dll" $genaiPkgDir -Force
                Write-Host "Copied QNN provider DLLs" -ForegroundColor Green
            }
        }
    }
    
    # Find and install wheel
    $wheelPaths = @(
        ".\build\Windows\RelWithDebInfo\wheel\*.whl",
        ".\build\Windows\RelWithDebInfo\dist\*.whl",
        ".\build\Windows\RelWithDebInfo\*.whl"
    )

    $foundWheel = $false
    foreach ($path in $wheelPaths) {
        $wheel = Get-ChildItem -Path $path -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($wheel) {
            Write-Host "`nWheel created: $($wheel.Name)" -ForegroundColor Green
            Write-Host "Installing wheel..." -ForegroundColor Yellow
            pixi run --manifest-path "$RepoRoot\pixi.toml" python -m pip install --force-reinstall --no-deps $wheel.FullName
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Wheel installed successfully" -ForegroundColor Green
            } else {
                Write-Host "Wheel installation failed" -ForegroundColor Red
            }
            $foundWheel = $true
            break
        }
    }

    if (-not $foundWheel) {
        Write-Host "`nWheel not found in expected locations" -ForegroundColor Yellow
        Get-ChildItem -Path ".\build" -Filter "*.whl" -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "Found: $($_.FullName)" -ForegroundColor Cyan
        }
    }
} else {
    Write-Host "`n✗ Build failed!" -ForegroundColor Red
    Write-Host "Check the error messages above" -ForegroundColor Red
}

Write-Host "`n=== Build Complete ===" -ForegroundColor Green
