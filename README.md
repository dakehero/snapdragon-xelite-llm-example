# Snapdragon X Elite LLM Benchmarks

> Running and benchmarking int4 LLM inference on Qualcomm Snapdragon X Elite NPU via ONNX Runtime QNN Execution Provider on Windows ARM64.

> ⚠ **Platform**: Windows 11 ARM64 (Snapdragon X Elite / Plus) + Python 3.14. Not compatible with x86_64, Linux, or macOS.

## Main finding

The useful question is not **"NPU or CPU?"** It is:

> **For a given model size and context length, which engine is fast enough, and which engine is feasible at all?**

Across Qwen 1.5B, Qwen 7B, and R1-Distill 14B, the pattern is:

1. **NPU is the prefill / TTFT engine when it fits.**
   It wins prefill by roughly 3-6x on the tested long-prompt workloads.

2. **CPU is often the short-context decode engine.**
   At ctx=64, CPU decode is faster for all tested models.

3. **CPU's decode lead erodes with both context length and model size.**
   Qwen 1.5B still favors CPU at ctx=8192. Qwen 7B reaches parity at ctx=8192. R1-Distill 14B is already near parity around ctx=1024-2048.

4. **Large-model long-context routing is feasibility-aware, not just speed-aware.**
   R1-Distill 14B QNN fails at ctx=4096 while CPU continues, so a runtime must first ask whether a model × context point can run on NPU at all.

## Size × context × feasibility

![All models context-length sweep](results/context_sweep_all_models.png)

Full data:

- **Qwen 1.5B**: [`results/context_sweep_qwen1.5b.md`](results/context_sweep_qwen1.5b.md)
- **Qwen 7B**: [`results/context_sweep_qwen7b.md`](results/context_sweep_qwen7b.md)
- **R1-Distill 14B**: [`results/context_sweep_r1distill14b.md`](results/context_sweep_r1distill14b.md)

| Model | NPU prefill peak | ctx=64 decode winner | Long-context decode | NPU feasibility |
|---|---:|---|---|---|
| Qwen 1.5B | 921 t/s @ ctx=256 | CPU 1.9x | CPU still 1.55x at ctx=8192 | Runs through ctx=8192 |
| Qwen 7B | 356 t/s @ ctx=128 | CPU 1.6x | Within noise at ctx=8192 | Runs through ctx=8192 |
| R1-Distill 14B | 166 t/s @ ctx=64 | CPU 1.4x | NPU roughly tied/slightly ahead by ctx=1024-2048 | QNN fails at ctx=4096 |

## Qwen 7B detail

Qwen 2.5 7B Instruct int4, context-length sweep from 64 to 8192 tokens, median of 3 runs:

![Qwen 7B context-length sweep](results/context_sweep_qwen7b.png)

Three observations:

- **NPU prefill wins across every context length.**
  Peak NPU prefill is 356 t/s at ctx=128. The advantage degrades from 4.7x at short context to 2.9x at ctx=8192, but it never disappears.

- **NPU TTFT wins by the same factor.**
  Since `TTFT = ctx / prefill_tps`, long prompts expose the gap directly: 72 s on NPU vs 206 s on CPU for an 8192-token prompt.

- **CPU decode advantage shrinks to within noise at ctx=8192.**
  At short context, CPU decode is clearly faster. At ctx=8192, NPU and CPU are statistically indistinguishable: NPU 4.85 ± 0.11 t/s vs CPU 4.34 ± 0.67 t/s.

## Practical routing rule

| Workload | Suggested route | Why |
|---|---|---|
| Short prompt, short answer | CPU | Decode dominates and CPU is faster at short KV-cache sizes |
| Long system prompt / RAG prompt | NPU prefill | TTFT is 3-6x better when NPU fits |
| Long-context summarization | NPU or hybrid | CPU decode lead erodes as KV cache grows |
| Large model + long context | Feasibility check first | NPU may OOM/fail even when it is faster on smaller contexts |

This motivates a heterogeneous policy:

> **Use NPU for prefill when feasible, then optionally decode on CPU when CPU has the decode advantage.**

The current prototype target is **NPU prefill + CPU decode** using KV-cache handoff. The KV tensors have matching ONNX I/O shape and dtype across the tested QNN and CPU models, so the handoff appears mechanically feasible. See [`docs/findings.md`](docs/findings.md#f6-plan-a-kv-cache-handoff-is-plug-and-play).

## Reproduce

### One-shot bootstrap

For a fresh Windows ARM64 Qualcomm device:

```powershell
iwr -useb https://raw.githubusercontent.com/dakehero/snapdragon-xelite-llm-example/main/scripts/bootstrap_cloud.ps1 | iex
```

This installs the environment, downloads the pre-built wheel and Foundry Local models, then runs the Qwen 7B context sweep.

### Manual benchmark

```powershell
pixi install
pixi run python scripts/install.py

make benchmark-context `
  ORT_QNN_MODEL="C:\Users\<you>\.foundry\cache\models\Microsoft\qwen2.5-7b-instruct-qnn-npu-2\v2" `
  ORT_CPU_MODEL="C:\Users\<you>\.foundry\cache\models\Microsoft\qwen2.5-7b-instruct-generic-cpu-4\v4"

make plot INPUTS=results/context_sweep_qwen7b.md
```

For build-from-source instructions, model setup, troubleshooting, and runtime pitfalls, see [`docs/setup.md`](docs/setup.md).

## More documentation

- **Setup and troubleshooting**: [`docs/setup.md`](docs/setup.md)
- **Full findings log**: [`docs/findings.md`](docs/findings.md)
- **Current working plan**: [`PROGRESS.md`](PROGRESS.md)
- **Raw benchmark tables**: [`results/`](results/)

## Repository map

```text
qnn/
├── README.md                         # Conclusion-first project overview
├── PROGRESS.md                       # Current plan and short handoff notes
├── docs/
│   ├── setup.md                      # Build, install, models, pitfalls
│   └── findings.md                   # F1-F8 findings and longer research notes
├── results/                          # Benchmark markdown tables and plots
├── benchmark.py                      # Multi-backend benchmark harness
├── plot.py                           # Context-sweep plotting
├── profile.py                        # ORT per-op profiling
├── verify.py                         # NPU vs CPU correctness check
├── llm_infer_ort_qnn.py              # ORT-GenAI + QNN EP inference
├── llm_infer_ort_cpu.py              # ORT-GenAI + CPU EP inference
└── scripts/                          # Setup, install, model, and build helpers
```
