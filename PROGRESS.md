# Progress & Plan

> Working notes for session-to-session continuity. Not a polished roadmap.
> Positioning: blog-level technical curiosity, not paper or product.
> Last updated: **2026-04-25** — Qwen 7B full context-length sweep shipped
> with PNG plot; README pivoted to context-sweep headline. Plan A (NPU
> prefill + CPU decode) is still a weekend prototype target.

## Key Findings

### F1. Chat template unlocks bit-exact NPU/CPU agreement

Qwen 2.5 7B int4, NPU (QNN) vs CPU (generic) on the same prompt:

| Prompt style | Result |
|---|---|
| Bare: `"The capital of France is"` | Diverge at token 2; NPU falls into multiple-choice exam format |
| Chat-templated (`<\|im_start\|>user\n...`) | **Bit-exact identical** output across NPU/CPU |

Quantization fidelity is fine **when the prompt puts the model in the
high-confidence regime**. Low-confidence prompts let quantization noise
tip the top-1.

### F2. Phase × size: NPU prefill scales up, CPU decode compresses

Four models, chat-templated, int4, median of 3 runs (short prompt,
`max_length=512`):

| Size | NPU prefill TPS | CPU prefill TPS | NPU prefill speedup | NPU decode TPS | CPU decode TPS | CPU decode speedup |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 1.5B | 213.75 | **224.68** | 0.95x (~tied) | 37.28 | **99.23** | **2.66x** |
| Phi 3.8B | 75.40 | **90.36** | 0.83x | 16.78 | **27.11** | 1.62x |
| Qwen 7B | **84.75** | 68.43 | 1.24x | 13.96 | **23.59** | 1.69x |
| R1-Distill 14B | **22.72** | 12.33 | **1.84x** | 6.11 | **8.75** | 1.43x |

- **Prefill crossover is in the 3.8B-7B band.** NPU advantage grows
  monotonically from 3.8B upward; at 14B it's nearly 2x. This is where
  NPU earns its transistors.
- **CPU decode wins at every tested size, but the lead compresses**
  (2.66x -> 1.62x -> 1.69x -> 1.43x) as workloads become bandwidth-bound.

> **Correction (2026-04-25)**: the single-point numbers above use a short
> prompt (~15 tokens) that under-utilizes the NPU's AR-64 prefill graph
> (effective prefill throughput is padding-diluted). See **F7** for the
> corrected phase × context picture from a proper context-length sweep.

This motivates **heterogeneous prefill-on-NPU / decode-on-CPU** (Plan A),
with maximum theoretical ROI at the largest model that fits in memory.

### F3. QNN LLM pipeline structure (verified against Qualcomm official docs)

Every QNN LLM export we've inspected (3 Microsoft + 1 Qualcomm-native)
follows the same 4-stage pipeline:

```
embeddings.onnx          →  context_ctx.onnx       →  iterator_ctx.onnx      →  lm_head.onnx
(CPU, Gather)               (QNN, AR-128 prefill)     (QNN, AR-1 decode)        (CPU, MatMulNBits)
                            run_on_token_gen:false    run_on_prompt:false
                            [ shared weights across AR variants, packed into 4–8 × ~825MB .bin shards ]
```

Official Qualcomm terminology (from `ai-hub-models/tutorials/llm/onboarding.md`):

| Our finding | Qualcomm official term |
|---|---|
| "prefill graph" | **AR-128** (autoregressive seq_len=128 variant; Microsoft uses AR-64) |
| "decode graph" | **AR-1** |
| "prefill/decode share weights, shape-specialized recompile" | "AR-1 and AR-128 variants linked together to form a context binary with shared weights" |
| "HTP per-binary cap ~825-833 MB" | "NPU is a 32-bit co-processor; individual compiled models ~**1-2 GB**" |
| "Parts N/M layer split" | **`NUM_SPLITS` / `NUM_LAYERS_PER_SPLIT`** in `model.py` |

**Cross-model verification:**

| Model | Exporter | embeddings | lm_head | QNN shards |
|---|---|---:|---:|---|
| Qwen 1.5B | Microsoft | 146 MB | 146 MB | 4 × 168 MB |
| Phi 3.8B | Microsoft | 62 MB | 62 MB | 4 × 460 MB |
| Qwen 7B | Microsoft | 341 MB | 341 MB | 4 × 825 MB |
| R1-Distill 14B | Qualcomm | 487 MB | 487 MB | **8 × 833 MB** |

14B cannot fit 4 shards; export doubles to 8. embeddings and lm_head are
equal-sized (no weight tying in exported artifact even when upstream HF
model ties them).

**Deployment-path divergence:** Qualcomm's
Genie-native deployment keeps lm_head **inside** the final QNN binary
("LM head ... always belongs to the final part"). Our ORT-GenAI
deployment **pulls lm_head out to CPU**. Same AIMET-quantized weights,
two runtimes, different end-to-end execution.

### F4. lm_head CPU fallback on ORT-GenAI: scales with `hidden × vocab`

Profiling via `ort_genai` session profiler across 4 models (192 decode tokens):

| Model | vocab | hidden | lm_head ms/tok | NPU decode TPS | Share |
|---|---:|---:|---:|---:|---:|
| Qwen 1.5B | 151,936 | 1,536 | 1.91 | 29.26 | 5.6% |
| Phi 3.8B | 32,064 | 3,072 | 0.95 | 17.90 | 1.7% |
| Qwen 7B | 151,936 | 3,584 | 3.70 | 9.88 | 3.7% |
| R1-Distill 14B | 152,064 | 5,120 | 5.00 | 6.22 | 3.1% |

- Scales proportionally with `hidden × vocab` (matmul FLOPs). Cross-model
  ratios track within 15%.
- Fallback is **structural, not shape-dependent**: Phi with 32k vocab
  also falls back.
- lm_head is **1.7–5.6% of NPU-mode decode wall time** — noticeable tax,
  not dominant.
- CPU-mode decode op breakdown (Qwen 7B): **MatMulNBits 90.8%,
  GroupQueryAttention 3.9%, LayerNorm 2.2%** — decode is
  weight-memory-bound on CPU; attention is fused and small.

### F5. NPU-decode loss is ~90% intrinsic, ~10% lm_head

Decomposition for Phi 3.8B (clearest case):
- NPU decode 16.78 TPS → 59.6 ms/token; CPU decode 27.11 TPS → 36.9
  ms/token. NPU is 22.7 ms/token slower.
- lm_head CPU tax accounts for only 0.95 ms (4% of the gap).
- **Remaining ~21.7 ms is NPU transformer forward itself** being slower
  than CPU at batch=1 decode.

Root cause: NPU is optimized for large-batch parallel matmul (=prefill
AR-128). Decode AR-1 has no parallelism to exploit; HTP kernels compiled
for batch=1 underperform Oryon CPU. This is the dominant term in F2's
"CPU wins decode" observation — lm_head is secondary.

### F6. Plan A KV-cache handoff is plug-and-play

`scripts/inspect_kv_dtype.py` verified `past_key_values.{i}.key/value` and
`present.{i}.key/value` I/O signatures on NPU `context_ctx.onnx`,
NPU `iterator_ctx.onnx`, and CPU `model.onnx`:

```
All three:  FLOAT32  [batch_or_1, 4, past_sequence_length, 128]
```

Identical shape and dtype. No requantization bridge needed. KV cache
hands off naturally via ONNX tensor IO. **This obsoletes the original
Plan B** (which assumed patching onnxruntime-genai to expose KV IO) —
everything is already exposed at the pipeline stage boundaries.

### F7. Context-length sweep: CPU's decode advantage erodes, not reverses

Qwen 7B, synthetic prompt of exact token count, decode-tokens=128,
1 warmup + 3 measured runs, `results/context_sweep_qwen7b.md`:

| ctx | NPU prefill | CPU prefill | NPU prefill speedup | NPU decode | CPU decode | CPU decode lead |
|---:|---:|---:|---:|---:|---:|---:|
|   64 | 353 | 82 | **4.3x** | 11.7 | 19.0 | +62% |
|  128 | 356 | 76 | **4.7x** | 11.8 | 17.0 | +44% |
|  256 | 346 | 81 | 4.3x | 11.2 | 18.3 | +63% |
|  512 | 337 | 76 | 4.4x | 11.2 | 17.2 | +54% |
| 1024 | 303 | 71 | 4.3x | 10.6 | 15.5 | +46% |
| 2048 | 248 | 66 | 3.8x | 9.0 | 10.5 | +17% |
| 4096 | 189 | 53 | 3.6x | 7.4 | 9.8 | +32% |
| 8192 | 114 | 40 | 2.9x | **4.85** ± 0.11 | **4.34** ± 0.67 | **~0%** (within noise) |

Key observations (replaces F2's short-prompt narrative):

- **NPU prefill peaks at ctx~128 (356 t/s)**, 4x higher than our earlier
  single-point 84.75 figure. That number was dominated by AR-64 padding
  waste from a ~15-token prompt. The corrected peak is the NPU's real
  prefill capability.

- **Prefill follows clean power-law degradation** as attention becomes
  O(n²)-heavy. NPU 353 → 114 t/s across 128x context range. CPU 82 → 40.
  NPU's relative advantage shrinks (4.7x → 2.9x) but never disappears.

- **CPU decode advantage erodes from +63% to ~0%**. At ctx=8192 the two
  engines are statistically indistinguishable (4.34 ± 0.67 vs 4.85 ±
  0.11, intervals overlap). **CPU never loses mean, but the cliff is
  real: its lead goes from decisive to within-noise.**

- **TTFT gap stays ~3-5x NPU-favored across all contexts.** This is the
  most actionable production finding: any interactive app with >1K-token
  system prompts should route prefill through the NPU regardless of how
  decode is handled.

- **Variance signal: CPU decode stdev explodes at ctx≥2048** (1.00, 0.67
  t/s respectively) while NPU stays tight (0.11-0.28). CPU decode
  transitions out of its steady-state regime into a memory-bound,
  scheduler-sensitive mode.

**Revised narrative for F2**: the "CPU always wins decode" claim holds
as-stated, but the magnitude is context-dependent. The phase × size
story from F2 is *correct in direction*; F7 adds the phase × context
dimension that makes it actionable.

## Plans

### Plan A — NPU prefill + CPU decode (weekend prototype, CURRENT FOCUS)

Hand off KV cache from QNN prefill (`context_ctx.onnx` + embeddings) to a
CPU ORT session running the same logical decoder. Uses each engine where
it is strong per F2.

**Prototype flow (1-2 days):**
1. Direct `onnxruntime.InferenceSession` to drive QNN prefill, bypass
   onnxruntime-genai's pipeline wrapper to get raw `present.*.key/value`.
2. Load Microsoft Foundry Local Qwen 7B CPU model. Inject KV cache as
   `past_key_values.*`.
3. Decode on CPU to EOS.
4. Compare end-to-end latency on short / medium / long prompts vs all-NPU
   and all-CPU baselines.
5. Eyeball output quality; optionally compute perplexity delta.

**Risk:** Microsoft's NPU and CPU Qwen 7B models are **independently
AIMET/genai-builder quantized**. Weight values differ. KV cache
numerical drift from "NPU-quantized attention" into "CPU-quantized
decoder" is the only unknown. Three possible outcomes:

| Outcome | Next action |
|---|---|
| Quality holds | Write up results, Plan A done |
| Mild quality drop | Report as-is, motivates Phase 2 |
| Garbled output | Go to Phase 2 (unified quantization) |

**Predicted gain** for Qwen 7B, 2000-token prompt + 500-token gen:
all-NPU 59s, all-CPU 77s, Plan A ~45s → ~24% speedup vs best baseline.

### Phase 2 (conditional) — Custom hybrid-aware quantized model

If Plan A shows quality drift. Path per Qualcomm tutorial:
1. AIMET PTQ from HF FP16 weights.
2. Export ONNX for multiple AR variants (AR-1, AR-128).
3. Parts splitting to fit 825 MB / shard cap.
4. AI Hub compile + link, produce shared-weight context binaries.
5. Same quantized weights feed both ORT-GenAI and Genie runtimes →
   enables clean **Genie vs ORT-GenAI** ablation ("what does the
   ORT-GenAI lm_head CPU fallback actually cost?").

All scaffolding exists in `qualcomm/ai-hub-models`. Needs 64-120 GB RAM
cloud VM for ONNX export step and Qualcomm AI Hub account.

### Plan A' — Speculative decoding (SHELVED)

Measured α = 14% overall (0.22 / 8 tokens) with Qwen 1.5B Instruct draft
+ DeepSeek R1 Distill 14B target. Break-even for throughput win is α =
0.77. **5.5x below break-even; dead for this pair.**

Root cause: R1 distillation emits `<think>...</think>` reasoning traces
at token 0; non-reasoning Qwen draft cannot match. Useful negative
result: **R1-style reasoning distillation is structurally incompatible
with non-reasoning draft models for speculative decoding, even with
identical tokenizer and architecture.**

Revival path (deferred): convert `Qwen/Qwen2.5-14B-Instruct` (non-distill)
to CPU int4 ONNX on a 64-120 GB cloud VM via `scripts/build_onnx_model.py`,
re-run `estimate_alpha`. Gate: α > 0.85. Only worth pursuing if Plan A
fails or if we want spec-dec as a separate contribution.

Second wall (if α check passes): **LPDDR5x bandwidth contention**. NPU
draft + CPU target in pipelined mode needs ~100 GB/s aggregate vs
sustained 85-95 GB/s. Pushes effective break-even α to ~0.85+.

### Plan C — Graph-level fusion (deferred)

Per-layer dispatch decisions at compile time. Large engineering scope; not pursued here.

## Hardware trajectory — X Elite → X2 Elite

External data (Foundry Local, 4096 ctx):

| Model | Metric | X Elite | X2 Elite | X2/X1 |
|---|---|---:|---:|---:|
| Falcon3-7B | TTFT | 0.20-6.32s | 0.12-3.73s | ~1.68x |
| Falcon3-7B | Decode | 9.97 t/s | 22.6 t/s | **2.27x** |
| Qwen3-4B | TTFT | 0.11-3.64s | 0.08-2.43s | ~1.44x |
| Qwen3-4B | Decode | 18.7 t/s | 32.2 t/s | **1.72x** |

Decode speedup (1.7-2.3x) >> prefill speedup (1.4-1.7x) → X2's main
upgrade is **DRAM bandwidth**, not NPU TOPS. Consistent with decode being
memory-bound. Larger model gets bigger decode speedup (7B 2.27x vs 4B
1.72x) — textbook memory-bound signature.

**Implication for this project:** F2's "CPU decode wins" is partly a
bandwidth-contention artifact specific to X Elite. On X2, NPU decode
likely closes or crosses the gap at 7B+. **Plan A's window is X Elite
only; on X2 the asymmetry compresses.** Pitch accordingly.

## TODO

### High (this week)
- [ ] **Plan A prototype**: `scripts/plan_a_prototype.py` driving
  `embeddings.onnx` + `context_ctx.onnx` (QNN) then feeding KV cache to
  CPU `model.onnx`. Measure end-to-end vs all-NPU / all-CPU on Qwen 7B.
  Three prompt lengths (short/medium/long). Quality eyeballed + optional
  perplexity.

### Medium (after Plan A lands)
- [x] **Context-length sweep benchmark** — shipped as F7 (Qwen 7B only).
  Hypothesis confirmed: CPU decode TPS drops sharply when KV cache
  exceeds cache-resident size; NPU curve flatter.
- [ ] **Genie vs ORT-GenAI ablation** on same Qwen 2.5 7B weights.
  Quantifies the lm_head-on-CPU cost. Clean second data point.
- [ ] **QNN SDK profiling** (`QNN_PROFILING_LEVEL=detailed`) to open the
  AR-1 / AR-128 kernel black box. Get per-layer HTP times to explain
  F5's "21.7 ms/token intrinsic NPU penalty".
- [ ] Energy measurement. Open question: Windows ARM64 power counters?
  Perfmon? Qualcomm Performance Toolkit?
- [ ] Convert `Qwen/Qwen2.5-14B-Instruct` (non-distill) on cloud VM via
  `make build-onnx` — only if Plan A' is revived.

### Low
- [ ] `llm_infer_genie.py` — Genie SDK path (blocks Genie ablation above).
- [ ] `tests/test_sanity.py` — tokenizer round-trip + DLL setup.
- [ ] CI with x86 static checks only (GitHub Actions has no Win ARM64 runner).

## Open Questions

1. Does `onnxruntime-genai` support running two `Generator` objects
   concurrently in one Python process? (Relevant only if Plan A' revives.)
2. Do NPU/CPU outputs remain bit-exact across long generations (200+
   tokens)? (F1 verified on short only.)
3. Windows ARM64 NPU power measurement path — does Perfmon expose HTP
   counters? Any Qualcomm-provided tool?

## Session Handoff Notes

- **`make verify` default prompt is intentionally bare-completion** — do not
  "fix" it; it stress-tests Pitfall #7 (chat template requirement).
- If `verify.py` crashes with `-1073740791` (STATUS_STACK_BUFFER_OVERRUN),
  the fix is in `set_search_options` — the QNN model has KV-cache shape
  constraints; avoid `max_length < ~128`.
- Profile output prefix config: `enable_profiling` in `genai_config.json`
  must be a **non-empty string**, not a bool (genai schema quirk).
- `build_release.log` / `build_wheel.log` in repo root are `.gitignore`d
  but physically present; leave alone.
