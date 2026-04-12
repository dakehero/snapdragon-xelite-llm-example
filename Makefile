# QNN NPU Inference - Makefile entry point
# Usage: make <target>
#
# Platform-specific build scripts live in scripts/<platform>/
# All cross-platform operations use Python via pixi

.PHONY: help build build-ort build-genai install test run run-npu run-cpu benchmark check clean

# Default
help:
	@echo "QNN NPU Inference"
	@echo ""
	@echo "Targets:"
	@echo "  build        Build onnxruntime-genai with QNN support"
	@echo "  build-ort    Build onnxruntime with QNN EP (set QNN_SDK_ROOT)"
	@echo "  build-genai  Build onnxruntime-genai (set QNN_SDK_ROOT)"
	@echo "  install      Install the built genai wheel + DLLs"
	@echo "  test         Run integration tests"
	@echo "  run          Run NPU inference (MODEL_DIR required)"
	@echo "  run-npu      Run NPU inference (MODEL_DIR required)"
	@echo "  run-cpu      Run CPU inference (MODEL_DIR required)"
	@echo "  benchmark    Compare NPU vs CPU performance (requires NPU_MODEL_DIR and CPU_MODEL_DIR)"
	@echo "  check        Check ORT environment and QNN EP availability"
	@echo "  clean        Clean build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL_DIR    Path to model directory (required for run/run-npu/run-cpu)"
	@echo "  NPU_MODEL_DIR Path to NPU model directory (required for benchmark)"
	@echo "  CPU_MODEL_DIR Path to CPU model directory (required for benchmark)"
	@echo "  QNN_SDK_ROOT Path to QNN SDK (required for build-ort/build-genai)"
	@echo "  PROMPT       Input prompt (optional)"
	@echo "  MAX_LENGTH   Max generation length (default: 512)"

# --- Build ---

build-ort:
	@echo "[build-ort] Building onnxruntime with QNN EP..."
	pixi run pwsh -File scripts/windows/build_onnxruntime_qnn.ps1 -QnnSdkRoot "$(QNN_SDK_ROOT)"

build-genai:
	@echo "[build-genai] Building onnxruntime-genai with QNN..."
	pixi run pwsh -File scripts/windows/build_qnn.ps1 -QnnSdkRoot "$(QNN_SDK_ROOT)"

build: build-genai

# --- Install ---

install:
	pixi run python scripts/install.py

# --- Run ---

check:
	pixi run python scripts/check_ort.py

test:
	pixi run python scripts/check_qnn.py

run run-npu:
	pixi run python llm_infer_npu.py "$(MODEL_DIR)" --prompt "$(PROMPT)" --max-length $(or $(MAX_LENGTH),512)

run-cpu:
	pixi run python llm_infer_cpu.py "$(MODEL_DIR)" --prompt "$(PROMPT)" --max-length $(or $(MAX_LENGTH),512)

benchmark:
	pixi run python scripts/benchmark.py \
		$(if $(NPU_MODEL_DIR),--npu-model-dir "$(NPU_MODEL_DIR)") \
		$(if $(CPU_MODEL_DIR),--cpu-model-dir "$(CPU_MODEL_DIR)") \
		--prompt "$(or $(PROMPT),Briefly describe the advantages of NPU.)" \
		--max-length $(or $(MAX_LENGTH),512)

# --- Clean ---

clean:
	pixi run python scripts/clean.py
