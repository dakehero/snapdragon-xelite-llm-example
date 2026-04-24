# QNN NPU Inference - Makefile entry point
# Usage: make <target>
#
# Core entry points (benchmark.py, verify.py, profile.py, llm_infer_*.py)
# live at the repo root. scripts/ holds only environment/build tooling.

.PHONY: help build build-ort build-genai install test run run-npu run-cpu run-ort-qnn run-ort-cpu benchmark benchmark-context plot profile verify download-model build-onnx check clean

# Default
help:
	@echo "QNN NPU Inference"
	@echo ""
	@echo "Targets:"
	@echo "  build              Build onnxruntime-genai with QNN support"
	@echo "  build-ort          Build onnxruntime with QNN EP (set QNN_SDK_ROOT)"
	@echo "  build-genai        Build onnxruntime-genai (set QNN_SDK_ROOT)"
	@echo "  install            Install the built genai wheel + DLLs"
	@echo "  check / test       Check ORT environment and QNN EP availability"
	@echo "  download-model     Download a reference ONNX model from HuggingFace"
	@echo "  build-onnx         Build CPU-int4 ONNX from HuggingFace (cloud VM)"
	@echo "  run-ort-qnn        Run ORT-GenAI + QNN EP inference (MODEL_DIR)"
	@echo "  run-ort-cpu        Run ORT-GenAI + CPU EP inference (MODEL_DIR)"
	@echo "  benchmark          Single-prompt benchmark (warmup + multi-run stats)"
	@echo "  benchmark-context  Context-length sweep (prefill/decode TPS vs ctx size)"
	@echo "  profile            ORT per-op profiling (Chrome trace JSON)"
	@echo "  verify             Verify NPU output matches CPU (greedy, token-level)"
	@echo "  plot               Plot context-sweep results (INPUTS=results/*.md)"
	@echo "  clean              Clean build artifacts"
	@echo ""
	@echo "Variables for benchmark (slot style; set the ones you want to include):"
	@echo "  ORT_QNN_MODEL      Path to ONNX model for ORT-GenAI + QNN EP backend"
	@echo "  ORT_CPU_MODEL      Path to ONNX model for ORT-GenAI + CPU EP backend"
	@echo "  GENIE_MODEL        Path to Genie QAIRT .bin for Genie backend"
	@echo "  BACKENDS           Raw pass-through; overrides the slot vars if set"
	@echo "                     e.g. BACKENDS='--backend name:script.py:/path'"
	@echo ""
	@echo "Variables for other targets:"
	@echo "  MODEL_DIR          Path to model directory (run-ort-qnn / run-ort-cpu / profile)"
	@echo "  NPU_MODEL_DIR      Path to NPU model directory (verify)"
	@echo "  CPU_MODEL_DIR      Path to CPU model directory (verify)"
	@echo "  QNN_SDK_ROOT       Path to QNN SDK (build-ort / build-genai)"
	@echo "  PROMPT             Input prompt (optional)"
	@echo "  MAX_LENGTH         Max generation length (default: 512)"
	@echo "  WARMUP             Benchmark warmup runs (default: 1)"
	@echo "  RUNS               Benchmark measured runs (default: 3)"
	@echo "  NUM_TOKENS         Tokens to compare in verify (default: 20)"
	@echo "  CONTEXTS           Ctx sizes for benchmark-context (default: 64,128,256,512,1024,2048,4096,8192)"
	@echo "  DECODE_TOKENS      Decode tokens per run for benchmark-context (default: 128)"
	@echo "  OUTPUT_MD          Path to save benchmark-context markdown table (default: results/context_sweep_qwen7b.md; override per model)"

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

# --- Environment checks ---

check test:
	pixi run python scripts/check.py

# --- Inference (single run) ---

run run-ort-qnn run-npu:
	pixi run python llm_infer_ort_qnn.py "$(MODEL_DIR)" --prompt "$(PROMPT)" --max-length $(or $(MAX_LENGTH),512)

run-ort-cpu run-cpu:
	pixi run python llm_infer_ort_cpu.py "$(MODEL_DIR)" --prompt "$(PROMPT)" --max-length $(or $(MAX_LENGTH),512)

# --- Benchmarks (unified entry: benchmark.py with two modes) ---
# Example:
#   make benchmark ORT_QNN_MODEL=/path/qnn-model ORT_CPU_MODEL=/path/cpu-model
#   make benchmark-context ORT_QNN_MODEL=... ORT_CPU_MODEL=... CONTEXTS=128,1024,4096

benchmark:
	pixi run python benchmark.py \
		$(if $(ORT_QNN_MODEL),--backend ort-qnn:llm_infer_ort_qnn.py:"$(ORT_QNN_MODEL)") \
		$(if $(ORT_CPU_MODEL),--backend ort-cpu:llm_infer_ort_cpu.py:"$(ORT_CPU_MODEL)") \
		$(if $(GENIE_MODEL),--backend genie:llm_infer_genie.py:"$(GENIE_MODEL)") \
		$(BACKENDS) \
		--prompt "$(or $(PROMPT),Briefly describe the advantages of NPU.)" \
		--max-length $(or $(MAX_LENGTH),512) \
		--warmup $(or $(WARMUP),1) \
		--runs $(or $(RUNS),3)

benchmark-context:
	pixi run python benchmark.py \
		$(if $(ORT_QNN_MODEL),--backend ort-qnn:llm_infer_ort_qnn.py:"$(ORT_QNN_MODEL)") \
		$(if $(ORT_CPU_MODEL),--backend ort-cpu:llm_infer_ort_cpu.py:"$(ORT_CPU_MODEL)") \
		$(if $(GENIE_MODEL),--backend genie:llm_infer_genie.py:"$(GENIE_MODEL)") \
		$(BACKENDS) \
		--contexts $(or $(CONTEXTS),64,128,256,512,1024,2048,4096,8192) \
		--decode-tokens $(or $(DECODE_TOKENS),128) \
		--warmup $(or $(WARMUP),1) \
		--runs $(or $(RUNS),3) \
		--output-md "$(or $(OUTPUT_MD),results/context_sweep_qwen7b.md)"

# --- Plotting ---
# Plot one or more context-sweep markdown files.
# Usage:
#   make plot INPUTS=results/context_sweep_qwen7b.md
#   make plot INPUTS="results/context_sweep_qwen7b.md results/context_sweep_qwen1.5b.md" OUT=results/compare.png
plot:
	pixi run python plot.py $(or $(INPUTS),results/context_sweep_qwen7b.md) \
		$(if $(OUT),--out "$(OUT)") \
		$(if $(LABELS),--labels $(LABELS))

# --- Profiling / verification ---

# Per-op profile via ORT's enable_profiling. Chrome-trace JSON under profile_output/.
# Usage: make profile MODEL_DIR=/path/model BACKEND=ort-qnn   (or BACKEND=ort-cpu)
profile:
	pixi run python profile.py \
		--model-dir "$(MODEL_DIR)" \
		--backend $(or $(BACKEND),ort-qnn) \
		--prompt "$(or $(PROMPT),Briefly describe the advantages of NPU.)" \
		--max-length $(or $(MAX_LENGTH),256) \
		--output-dir $(or $(PROFILE_OUT),profile_output)

verify:
	pixi run python verify.py \
		$(if $(NPU_MODEL_DIR),--npu-model-dir "$(NPU_MODEL_DIR)") \
		$(if $(CPU_MODEL_DIR),--cpu-model-dir "$(CPU_MODEL_DIR)") \
		$(if $(PROMPT),--prompt "$(PROMPT)") \
		--num-tokens $(or $(NUM_TOKENS),20)

# --- Data ---

download-model:
	pixi run python scripts/download_model.py

# Build CPU-int4 ONNX from HuggingFace. Run on a 64+ GB Linux VM, not X Elite.
#   make build-onnx HF_ID=Qwen/Qwen2.5-14B-Instruct OUT=./qwen2.5-14b-cpu-int4
build-onnx:
	python scripts/build_onnx_model.py \
		--hf-id "$(HF_ID)" \
		--out "$(OUT)" \
		--precision $(or $(PRECISION),int4) \
		--execution-provider $(or $(EP),cpu)

# --- Clean ---

clean:
	pixi run python scripts/clean.py
