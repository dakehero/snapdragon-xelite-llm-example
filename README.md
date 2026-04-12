# Snapdragon X Elite LLM Example

> Example project for running LLM inference on Qualcomm Snapdragon X Elite NPU via ONNX Runtime QNN Execution Provider on Windows ARM64.

## Result

Successfully ran **Qwen 2.5 7B Instruct** on NPU at **~10.86 tokens/s**.

## Prerequisites

- **OS**: Windows 11 ARM64 (Snapdragon X Elite / Plus)
- **Visual Studio 2026** with C++ ARM64 build tools
- **Python 3.14.3** (conda-forge win-arm64 only provides 3.14)
- **CMake 4.3+**, **Ninja 1.11+**
- **Pixi** for Python environment management
- **onnxruntime-qnn** (PyPI, provides QNN DLLs)
- **onnxruntime-genai** (must build from source, see below)

## Quick Start

### Option 1: Use Pre-built Wheel (Fastest)

For Windows ARM64 with Python 3.14, download the pre-built wheel from [GitHub Releases](https://github.com/dakehero/snapdragon-xelite-llm-example/releases):

```powershell
# 1. Setup environment
pixi install

# 2. Download and install pre-built wheel (includes QNN support)
#    Replace <VERSION> with the latest release tag, e.g., v0.14.0
$WheelUrl = "https://github.com/dakehero/snapdragon-xelite-llm-example/releases/download/<VERSION>/onnxruntime_genai-0.14.0.dev0-cp314-cp314-win_arm64.whl"
pixi run python -m pip install --force-reinstall --no-deps $WheelUrl

# 3. Copy DLLs (NuGet + QNN)
pixi run python scripts/install.py

# 4. Verify and run
make test
make run MODEL_DIR=/path/to/qnn-model
```

Or manually download the wheel from Releases and install locally.

### Option 2: Build from Source

If you need a different Python version or want to modify the source:

```powershell
# 1. Setup environment (installs onnxruntime-qnn and other deps via pixi)
pixi install

# 2. Build + install onnxruntime-genai with QNN (one command)
#    PyPI version does NOT include QNN - must build from source
#    This also auto-installs the wheel and copies required DLLs
make build-genai QNN_SDK_ROOT="C:\QNN\qairt\2.45.0.260326"

# 3. Verify
make test

# 4. Run inference
make run MODEL_DIR=/path/to/qnn-model
```

#### Build Details

There are two build paths available:

**Path A: Simple Build (Default)** — `make build` or `make build-genai`
- `make build` is an alias for `make build-genai` (default behavior)
- Downloads pre-built `Microsoft.ML.OnnxRuntime.QNN` from **NuGet** automatically
- Links genai against the NuGet QNN-enabled `onnxruntime.dll`
- Fastest option, no need to build onnxruntime from scratch

**Path B: Full Source Build** — `make build-ort` then `make build-genai`
- First builds onnxruntime base from source with QNN EP (`make build-ort`)
- Then builds genai linked against your custom onnxruntime (`make build-genai`)
- Use this if you need to modify onnxruntime itself or want complete control

> **Note**: `onnxruntime-genai` is intentionally excluded from `pixi.toml` to prevent `pixi install` from overwriting the custom-built QNN wheel with the standard PyPI version.

## Pitfalls & Solutions

These are the problems we encountered and how we solved them. If you're doing the same thing, these will save you hours.

### 1. PyPI `onnxruntime-genai` does NOT include QNN support

**Symptom**: `og.Model(config)` throws `RuntimeError: QNN execution provider is not supported in this build.`

**Root cause**: The standard `onnxruntime-genai` wheel on PyPI is built without QNN. Even though `og.is_qnn_available()` returns `True` (it's hardcoded in the source), the actual EP registration fails because the underlying `onnxruntime.dll` linked by genai doesn't know about QNN.

**Solution**: Build `onnxruntime-genai` from source. On Windows ARM64, the build system automatically downloads `Microsoft.ML.OnnxRuntime.QNN` NuGet package which contains QNN-enabled `onnxruntime.dll`.

### 2. QNN EP registration name must be `QNNExecutionProvider`, NOT `qnn`

**Symptom**: Same error as above, even after building from source and calling `og.register_execution_provider_library('qnn', ...)`.

**Root cause**: `onnxruntime-genai` internally calls `FindRegisteredEpDevices("QNNExecutionProvider")` to look up the registered EP. If you register with the name `"qnn"`, the lookup fails, and genai falls back to the legacy V1 API (`AppendExecutionProvider`), which also fails because the NuGet `onnxruntime.dll` doesn't have QNN built-in — it expects it as a plugin registered under the correct name.

**Solution**:
```python
# WRONG
og.register_execution_provider_library('qnn', onnxruntime_qnn.get_library_path())

# CORRECT
og.register_execution_provider_library('QNNExecutionProvider', onnxruntime_qnn.get_library_path())
```

### 3. MSVC requires `/EHsc` flag for C++ exception handling

**Symptom**: Build fails with `error C2220: warnings treated as errors` and `warning C4530: C++ exception handler used, but unwind semantics not enabled`

**Root cause**: The genai source code uses C++ exceptions, but the Ninja + MSVC build doesn't enable exception handling by default.

**Solution**: Pass `--cmake_extra_defines CMAKE_CXX_FLAGS=/EHsc` to the build command.

### 4. DLL version mismatch between PyPI onnxruntime and NuGet onnxruntime

**Symptom**: `og.Model(config)` still fails after building genai from source, even with correct registration name.

**Root cause**: The build links against NuGet's `onnxruntime.dll` (1.25.0-dev), but at runtime Python loads PyPI's `onnxruntime.dll` (1.24.4) from `onnxruntime/capi/`. The two DLLs have different ABIs — the QNN provider DLL built for 1.24.4 cannot be loaded by the 1.25.0-dev DLL, and vice versa.

**Solution**: Copy the NuGet `onnxruntime.dll` and all QNN DLLs into the `onnxruntime_genai` package directory so genai loads the correct version:
```powershell
$genaiDir = ".pixi\envs\default\Lib\site-packages\onnxruntime_genai"
# NuGet onnxruntime DLLs (linked by genai)
Copy-Item "build\Windows\RelWithDebInfo\_deps\ortlib-src\runtimes\win-arm64\native\*.dll" $genaiDir -Force
# QNN provider DLLs (from onnxruntime-qnn package)
Copy-Item ".pixi\envs\default\Lib\site-packages\onnxruntime_qnn\*.dll" $genaiDir -Force
```

### 5. Python 3.12 not available for win-arm64 on conda-forge

**Symptom**: `pixi install` fails because `python = "3.12"` has no win-arm64 build on conda-forge.

**Solution**: Use `python = "3.14.*"` in `pixi.toml`. As of 2026-04, conda-forge only provides Python 3.14.3 for win-arm64.

### 6. GenAI API change: `get_next_tokens()` replaces `get_next_token()`

**Symptom**: `AttributeError: 'Generator' object has no attribute 'get_next_token'`

**Solution**: Use `generator.get_next_tokens()` (plural) which returns a list of token IDs.

## Working Inference Script Pattern

```python
import os
import onnxruntime_qnn
import onnxruntime_genai as og

# 1. Add DLL directories BEFORE any onnxruntime imports
qnn_dir = os.path.dirname(onnxruntime_qnn.__file__)
genai_dir = os.path.dirname(og.__file__)
os.add_dll_directory(qnn_dir)
os.add_dll_directory(genai_dir)
os.environ["PATH"] = genai_dir + os.pathsep + qnn_dir + os.pathsep + os.environ.get("PATH", "")

# 2. Register QNN EP with the CORRECT name
og.register_execution_provider_library('QNNExecutionProvider', onnxruntime_qnn.get_library_path())

# 3. Load model with QNN provider
model_dir = r"C:\Users\dake_\.foundry\cache\models\Microsoft\qwen2.5-7b-instruct-qnn-npu-2\v2"
config = og.Config(model_dir)
config.clear_providers()
config.append_provider('qnn')
model = og.Model(config)

# 4. Run inference
tokenizer = og.Tokenizer(model)
prompt = "Hello"
input_tokens = tokenizer.encode(prompt)
params = og.GeneratorParams(model)
params.set_search_options(max_length=128)
params.input_ids = input_tokens
generator = og.Generator(model, params)

while not generator.is_done():
    generator.generate_next_token()
    tokens = generator.get_next_tokens()

output = tokenizer.decode(tokens)
print(output)
```

## Verification

```powershell
make check                      # Check ORT environment + QNN EP registration
make test                      # Full QNN integration test
make run MODEL_DIR=/path/to/qnn-model  # NPU inference
make benchmark NPU_MODEL_DIR=/path/to/npu-model CPU_MODEL_DIR=/path/to/cpu-model  # Compare NPU vs CPU
```

## Pre-built Wheels

Pre-built wheels for Windows ARM64 are available on [GitHub Releases](https://github.com/dakehero/snapdragon-xelite-llm-example/releases).

## File Structure

```
qnn/
├── Makefile                    # Unified entry point (make build/test/run)
├── scripts/
│   ├── windows/
│   │   ├── build_onnxruntime_qnn.ps1   # Build ONNX Runtime with QNN EP
│   │   └── build_qnn.ps1              # Build ONNX Runtime GenAI
│   ├── check_ort.py            # Check ORT environment
│   ├── check_qnn.py            # Check QNN provider integration
│   ├── install.py              # Install built wheel + DLLs
│   ├── clean.py                # Clean build artifacts
│   └── benchmark.py            # NPU vs CPU benchmark
├── onnxruntime/                 # ONNX Runtime source (submodule)
├── onnxruntime-genai/           # ONNX Runtime GenAI source (submodule)
├── llm_infer_npu.py             # NPU inference with benchmark
├── llm_infer_cpu.py             # CPU inference for comparison
├── pixi.toml                    # Pixi environment config
└── README.md                    # This file
```
