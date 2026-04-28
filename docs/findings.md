# Findings log

This is the long-form findings and planning log. The README keeps only the headline conclusions.

## Key findings

### F1. Chat template unlocks bit-exact NPU/CPU agreement

Qwen 2.5 7B int4, NPU (QNN) vs CPU (generic) on the same prompt:

| Prompt style | Result |
|---|---|
| Bare: `"The capital of France is"` | Diverge at token 2; NPU falls into multiple-choice exam format |
| Chat-templated (`<\|im_start\|>user\n...`) | **Bit-exact identical** output across NPU/CPU |

Quantization fidelity is fine **when the prompt puts the model in the high-confidence regime**. Low-confidence prompts let quantization noise tip the top-1.

### F2. Phase × size: NPU prefill scales up, CPU decode compresses

Four models, chat-templated, int4, median of 3 runs, short prompt, `max_length=512`:

| Size | NPU prefill TPS | CPU prefill TPS | NPU prefill speedup | NPU decode TPS | CPU decode TPS | CPU decode speedup |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 1.5B | 213.75 | **224.68** | 0.95x (~tied) | 37.28 | **99.23** | **2.66x** |
| Phi 3.8B | 75.40 | **90.36** | 0.83x | 16.78 | **27.11** | 1.62x |
| Qwen 7B | **84.75** | 68.43 | 1.24x | 13.96 | **23.59** | 1.69x |
| R1-Distill 14B | **22.72** | 12.33 | **1.84x** | 6.11 | **8.75** | 1.43x |

- **Prefill crossover is in the 3.8B-7B band.** NPU advantage grows monotonically from 3.8B upward; at 14B it is nearly 2x.
- **CPU decode wins at every tested size, but the lead compresses** as workloads become bandwidth-bound.

> **Correction (2026-04-25)**: the single-point numbers above use a short prompt that under-utilizes the NPU's AR-64 prefill graph. See F7 for the corrected phase × context picture.

This motivates heterogeneous prefill-on-NPU / decode-on-CPU, with maximum theoretical ROI at the largest model that fits in memory.

### F3. QNN LLM pipeline structure

Every QNN LLM export inspected follows the same 4-stage pipeline:

```text
embeddings.onnx          ->  context_ctx.onnx       ->  iterator_ctx.onnx      ->  lm_head.onnx
(CPU, Gather)                (QNN, AR-128 prefill)      (QNN, AR-1 decode)         (CPU, MatMulNBits)
                             run_on_token_gen:false     run_on_prompt:false
                             [ shared weights across AR variants, packed into 4-8 × ~825MB .bin shards ]
```

Official Qualcomm terminology:

| Our finding | Qualcomm official term |
|---|---|
| "prefill graph" | **AR-128** autoregressive seq_len=128 variant; Microsoft uses AR-64 |
| "decode graph" | **AR-1** |
| "prefill/decode share weights, shape-specialized recompile" | AR-1 and AR-128 variants linked together to form a context binary with shared weights |
| "HTP per-binary cap ~825-833 MB" | NPU is a 32-bit co-processor; individual compiled models around **1-2 GB** |
| "Parts N/M layer split" | **`NUM_SPLITS` / `NUM_LAYERS_PER_SPLIT`** in `model.py` |

Cross-model verification:

| Model | Exporter | embeddings | lm_head | QNN shards |
|---|---|---:|---:|---|
| Qwen 1.5B | Microsoft | 146 MB | 146 MB | 4 × 168 MB |
| Phi 3.8B | Microsoft | 62 MB | 62 MB | 4 × 460 MB |
| Qwen 7B | Microsoft | 341 MB | 341 MB | 4 × 825 MB |
| R1-Distill 14B | Qualcomm | 487 MB | 487 MB | **8 × 833 MB** |

14B cannot fit 4 shards, so the export doubles to 8. Embeddings and lm_head are equal-sized in the exported artifact even when the upstream HF model ties them.

Qualcomm's Genie-native deployment keeps lm_head inside the final QNN binary. The ORT-GenAI deployment pulls lm_head out to CPU. Same AIMET-quantized weights, two runtimes, different end-to-end execution.

### F4. lm_head CPU fallback on ORT-GenAI scales with `hidden × vocab`

Profiling via `ort_genai` session profiler across 4 models:

| Model | vocab | hidden | lm_head ms/tok | NPU decode TPS | Share |
|---|---:|---:|---:|---:|---:|
| Qwen 1.5B | 151,936 | 1,536 | 1.91 | 29.26 | 5.6% |
| Phi 3.8B | 32,064 | 3,072 | 0.95 | 17.90 | 1.7% |
| Qwen 7B | 151,936 | 3,584 | 3.70 | 9.88 | 3.7% |
| R1-Distill 14B | 152,064 | 5,120 | 5.00 | 6.22 | 3.1% |

- Scales proportionally with `hidden × vocab`.
- Fallback is structural, not shape-dependent.
- lm_head is 1.7-5.6% of NPU-mode decode wall time.
- CPU-mode decode op breakdown on Qwen 7B: MatMulNBits 90.8%, GroupQueryAttention 3.9%, LayerNorm 2.2%.

### F5. NPU-decode loss is about 90% intrinsic, 10% lm_head

Decomposition for Phi 3.8B:

- NPU decode 16.78 TPS = 59.6 ms/token.
- CPU decode 27.11 TPS = 36.9 ms/token.
- NPU is 22.7 ms/token slower.
- lm_head CPU tax accounts for only 0.95 ms.
- Remaining ~21.7 ms is NPU transformer forward itself being slower than CPU at batch=1 decode.

Root cause: NPU is optimized for large-batch parallel matmul (prefill AR-128). Decode AR-1 has no parallelism to exploit, so HTP kernels compiled for batch=1 underperform Oryon CPU.

### F6. Plan A KV-cache handoff is plug-and-play

`inspect_kv_dtype.py` verified `past_key_values.{i}.key/value` and `present.{i}.key/value` I/O signatures on NPU `context_ctx.onnx`, NPU `iterator_ctx.onnx`, and CPU `model.onnx`:

```text
All three:  FLOAT32  [batch_or_1, 4, past_sequence_length, 128]
```

Identical shape and dtype. No requantization bridge needed. KV cache hands off naturally via ONNX tensor IO. This obsoletes the original Plan B, which assumed patching onnxruntime-genai to expose KV IO.

### F7. Context-length sweep: CPU's decode advantage erodes, not reverses

Qwen 7B, synthetic prompt of exact token count, decode-tokens=128, 1 warmup + 3 measured runs:

| ctx | NPU prefill | CPU prefill | NPU prefill speedup | NPU decode | CPU decode | CPU decode lead |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 353 | 82 | **4.3x** | 11.7 | 19.0 | +62% |
| 128 | 356 | 76 | **4.7x** | 11.8 | 17.0 | +44% |
| 256 | 346 | 81 | 4.3x | 11.2 | 18.3 | +63% |
| 512 | 337 | 76 | 4.4x | 11.2 | 17.2 | +54% |
| 1024 | 303 | 71 | 4.3x | 10.6 | 15.5 | +46% |
| 2048 | 248 | 66 | 3.8x | 9.0 | 10.5 | +17% |
| 4096 | 189 | 53 | 3.6x | 7.4 | 9.8 | +32% |
| 8192 | 114 | 40 | 2.9x | **4.85** ± 0.11 | **4.34** ± 0.67 | **~0%** within noise |

Key observations:

- **NPU prefill peaks at ctx~128**, around 356 t/s.
- **Prefill follows clean power-law degradation** as attention becomes O(n²)-heavy.
- **CPU decode advantage erodes from +63% to roughly 0%.**
- **TTFT gap stays 3-5x NPU-favored across all contexts.**
- **CPU decode variance increases at ctx>=2048**, while NPU stays tight.

### F8. Size × context: decode crossover moves left, but feasibility bites

Multi-model context sweeps, synthetic prompt of exact token count, decode-tokens=128, 1 warmup + 3 measured runs:

| Model | NPU prefill peak | ctx=64 decode winner | Long-context decode | NPU feasibility |
|---|---:|---|---|---|
| Qwen 1.5B | 921 t/s @ ctx=256 | CPU 1.9x | CPU still 1.55x at ctx=8192 | Runs through ctx=8192 |
| Qwen 7B | 356 t/s @ ctx=128 | CPU 1.6x | Within noise at ctx=8192 | Runs through ctx=8192 |
| R1-Distill 14B | 166 t/s @ ctx=64 | CPU 1.4x | NPU roughly tied/slightly ahead by ctx=1024-2048 | QNN fails at ctx=4096 |

Key observations:

- **CPU decode advantage erodes faster as model size grows.**
- **NPU prefill remains the TTFT engine when it fits.**
- **Large-model long-context execution has a feasibility boundary.**
- **The policy problem is feasibility-aware, not just speed-aware.**

## Plans

### Plan A: NPU prefill + CPU decode

Hand off KV cache from QNN prefill (`context_ctx.onnx` + embeddings) to a CPU ORT session running the same logical decoder.

Prototype flow:

1. Direct `onnxruntime.InferenceSession` to drive QNN prefill, bypassing onnxruntime-genai's pipeline wrapper to get raw `present.*.key/value`.
2. Load Microsoft Foundry Local Qwen 7B CPU model.
3. Inject KV cache as `past_key_values.*`.
4. Decode on CPU to EOS.
5. Compare end-to-end latency on short, medium, and long prompts against all-NPU and all-CPU baselines.
6. Eyeball output quality and optionally compute perplexity delta.

Risk: Microsoft's NPU and CPU Qwen 7B models are independently AIMET/genai-builder quantized. Weight values differ, so KV cache numerical drift is the unknown.

| Outcome | Next action |
|---|---|
| Quality holds | Write up results, Plan A done |
| Mild quality drop | Report as-is, motivates Phase 2 |
| Garbled output | Go to Phase 2 with unified quantization |

Predicted gain for Qwen 7B, 2000-token prompt + 500-token generation:

- All-NPU: 59 s
- All-CPU: 77 s
- Plan A: about 45 s
- Estimated improvement: about 24% vs best baseline

### Phase 2: custom hybrid-aware quantized model

If Plan A shows quality drift:

1. AIMET PTQ from HF FP16 weights.
2. Export ONNX for multiple AR variants.
3. Split parts to fit 825 MB / shard cap.
4. Compile and link through AI Hub.
5. Feed the same quantized weights to both ORT-GenAI and Genie runtimes.

This enables a clean Genie vs ORT-GenAI ablation.

### Plan A': speculative decoding

Shelved.

Measured acceptance α = 14% overall with Qwen 1.5B Instruct draft + DeepSeek R1 Distill 14B target. Break-even for throughput win is α = 0.77, so this pair is 5.5x below break-even.

Root cause: R1 distillation emits `<think>...</think>` reasoning traces at token 0; non-reasoning Qwen draft cannot match.

Revival path:

- Convert `Qwen/Qwen2.5-14B-Instruct` non-distill to CPU int4 ONNX on a 64-120 GB cloud VM.
- Re-run `estimate_alpha`.
- Gate: α > 0.85.

### Plan C: graph-level fusion

Deferred. Per-layer dispatch decisions at compile time are a large engineering scope.

## Hardware trajectory: X Elite to X2 Elite

External Foundry Local data at 4096 ctx:

| Model | Metric | X Elite | X2 Elite | X2/X1 |
|---|---|---:|---:|---:|
| Falcon3-7B | TTFT | 0.20-6.32s | 0.12-3.73s | ~1.68x |
| Falcon3-7B | Decode | 9.97 t/s | 22.6 t/s | **2.27x** |
| Qwen3-4B | TTFT | 0.11-3.64s | 0.08-2.43s | ~1.44x |
| Qwen3-4B | Decode | 18.7 t/s | 32.2 t/s | **1.72x** |

Decode speedup is much larger than prefill speedup, suggesting X2's main upgrade is DRAM bandwidth, not NPU TOPS. This is consistent with decode being memory-bound.

Implication: Plan A's strongest window is X Elite. On X2, the CPU/NPU decode asymmetry likely compresses.
