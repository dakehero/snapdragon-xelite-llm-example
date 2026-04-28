# Progress & Plan

> Working notes for session-to-session continuity. Full findings moved to [`docs/findings.md`](docs/findings.md). Setup and troubleshooting moved to [`docs/setup.md`](docs/setup.md).

## Current status

- **Positioning**: strong technical blog / reproducible measurement artifact.
- **Latest result**: README now frames the project around **size × context × feasibility**, not a Qwen 7B single-point story.
- **Main finding**: NPU is the prefill / TTFT engine when feasible; CPU decode wins at short context, but the lead erodes with context length and model size.
- **Current focus**: Plan A, NPU prefill + CPU decode via KV-cache handoff.

## Current plan

### High

- [ ] **Plan A prototype**: drive `embeddings.onnx` + `context_ctx.onnx` on QNN, extract `present.*.key/value`, feed them to CPU `model.onnx` as `past_key_values.*`, then decode on CPU.
- [ ] Compare end-to-end latency against all-NPU and all-CPU on Qwen 7B for short, medium, and long prompts.
- [ ] Check output quality by eyeballing generations and optionally computing perplexity delta.

### Medium

- [x] **Context-length sweep benchmark**: shipped for Qwen 1.5B, Qwen 7B, and R1-Distill 14B.
- [ ] **Genie vs ORT-GenAI ablation** on same Qwen 2.5 7B weights.
- [ ] **QNN SDK profiling** with `QNN_PROFILING_LEVEL=detailed` to inspect AR-1 / AR-128 kernel timings.
- [ ] Energy measurement path on Windows ARM64.

### Low

- [ ] `llm_infer_genie.py` for Genie SDK path.
- [ ] `tests/test_sanity.py` for tokenizer round-trip and DLL setup.
- [ ] CI with x86 static checks only.

## Open questions

1. Does `onnxruntime-genai` support running two `Generator` objects concurrently in one Python process?
2. Do NPU/CPU outputs remain bit-exact across long generations?
3. What is the best Windows ARM64 NPU power measurement path?

## Session handoff notes

- **`make verify` default prompt is intentionally bare-completion**. Do not "fix" it; it stress-tests the chat-template requirement.
- If `verify.py` crashes with `-1073740791` (`STATUS_STACK_BUFFER_OVERRUN`), the fix is in `set_search_options`; the QNN model has KV-cache shape constraints, so avoid `max_length < ~128`.
- Profile output prefix config: `enable_profiling` in `genai_config.json` must be a **non-empty string**, not a bool.
- `build_release.log` and `build_wheel.log` in repo root are `.gitignore`d but physically present; leave them alone.
