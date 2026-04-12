# Build ONNX Runtime with QNN EP from source
# This builds the base onnxruntime with QNN support

param(
    [string]$QnnSdkRoot = $env:QNN_SDK_ROOT
)

Write-Host "=== Building ONNX Runtime with QNN EP ===" -ForegroundColor Green

if (-not $QnnSdkRoot) {
    Write-Host "QNN SDK path not specified. Use -QnnSdkRoot or set QNN_SDK_ROOT env var." -ForegroundColor Red
    exit 1
}

$env:QNN_SDK_ROOT = $QnnSdkRoot
$env:ORT_QNN_EP_PATH = $QnnSdkRoot

Write-Host "QNN SDK Path: $env:QNN_SDK_ROOT" -ForegroundColor Cyan

# Check if onnxruntime directory exists
if (-not (Test-Path "onnxruntime")) {
    Write-Host "`nCloning ONNX Runtime..." -ForegroundColor Yellow
    git clone --recursive https://github.com/microsoft/onnxruntime.git
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Failed to clone ONNX Runtime" -ForegroundColor Red
        exit 1
    }
}

Set-Location "onnxruntime"
Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow

# Create build directory
$buildDir = "build\Windows\RelWithDebInfo"
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

Set-Location $buildDir

# Configure CMake
Write-Host "`nConfiguring CMake..." -ForegroundColor Yellow
$cmakeArgs = @(
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
    "-Donnxruntime_BUILD_SHARED_LIB=ON",
    "-Donnxruntime_BUILD_UNIT_TESTS=OFF",
    "-Donnxruntime_USE_QNN=ON",
    "-DQNN_SDK_ROOT=$env:QNN_SDK_ROOT",
    "-Donnxruntime_ENABLE_PYTHON=ON",
    "-DPython_EXECUTABLE=$(python -c 'import sys; print(sys.executable)')",
    "-G", "Ninja"
)

Write-Host "Running: cmake $($cmakeArgs -join ' ')" -ForegroundColor Cyan
cmake ..\.. $cmakeArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ CMake configuration failed" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "`nBuilding ONNX Runtime..." -ForegroundColor Yellow
cmake --build . --config RelWithDebInfo --parallel

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Build completed!" -ForegroundColor Green
    
    # Find the built wheel
    $wheel = Get-ChildItem -Path ".\dist\*.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($wheel) {
        Write-Host "Wheel created: $($wheel.Name)" -ForegroundColor Green
        Write-Host "Install with: pip install $($wheel.FullName)" -ForegroundColor Cyan
        
        # Copy to parent directory for easy access
        Copy-Item $wheel.FullName "..\..\..\onnxruntime_qnn.whl" -Force
        Write-Host "Copied to: onnxruntime\onnxruntime_qnn.whl" -ForegroundColor Green
    }
} else {
    Write-Host "✗ Build failed!" -ForegroundColor Red
}

Set-Location ..\..\..
Write-Host "`n=== Build Complete ===" -ForegroundColor Green
