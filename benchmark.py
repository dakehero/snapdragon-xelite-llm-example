"""Benchmark inference backends on Snapdragon X Elite.

Two modes:

1. Single-prompt mode (default): run each backend with one prompt, report
   median +/- stdev of decode TPS / prefill TPS / TTFT over N runs.

     python benchmark.py \\
         --backend ort-qnn:llm_infer_ort_qnn.py:C:/models/qwen-npu \\
         --backend ort-cpu:llm_infer_ort_cpu.py:C:/models/qwen-cpu \\
         --prompt "Hello" --runs 3

2. Context-sweep mode (triggered by --contexts): for each context length,
   run each backend with a synthetic prompt of exactly that many tokens plus
   a fixed decode budget. Produces a markdown table.

     python benchmark.py \\
         --backend ort-qnn:llm_infer_ort_qnn.py:C:/models/qwen-npu \\
         --backend ort-cpu:llm_infer_ort_cpu.py:C:/models/qwen-cpu \\
         --contexts 64,128,512,2048,8192 --decode-tokens 128

Backends are supplied as NAME:SCRIPT:MODEL_DIR. The harness is agnostic
to the engine (ORT-GenAI+QNN, ORT-GenAI+CPU, Qualcomm Genie SDK, ...) as
long as the backend script prints "Prefill speed:", "Decode speed:" and
"TTFT:" lines on stdout.
"""

import argparse
import os
import re
import statistics
import subprocess
import sys


# --- stdout parsing ---------------------------------------------------------

def parse_metrics(output):
    metrics = {"decode_tps": None, "prefill_tps": None, "tokens": None, "ttft": None}
    for line in output.splitlines():
        if m := re.search(r"Decode speed:\s*([\d.]+)", line):
            metrics["decode_tps"] = float(m.group(1))
        if m := re.search(r"Prefill speed:\s*([\d.]+)", line):
            metrics["prefill_tps"] = float(m.group(1))
        # Legacy fallbacks
        if metrics["decode_tps"] is None:
            if m := re.search(r"^\s*Speed:\s*([\d.]+)", line):
                metrics["decode_tps"] = float(m.group(1))
            elif m := re.search(r"^\s*TPS:\s*([\d.]+)", line):
                metrics["decode_tps"] = float(m.group(1))
        if m := re.search(r"Generated tokens:\s*(\d+)", line):
            metrics["tokens"] = int(m.group(1))
        elif metrics["tokens"] is None:
            if m := re.search(r"^\s*Tokens:\s*(\d+)", line):
                metrics["tokens"] = int(m.group(1))
            elif m := re.search(r"Total tokens:\s*(\d+)", line):
                metrics["tokens"] = int(m.group(1))
        if m := re.search(r"TTFT:\s*([\d.]+)\s*ms", line):
            metrics["ttft"] = float(m.group(1)) / 1000
        elif m := re.search(r"TTFT:\s*([\d.]+)\s*s\b", line):
            metrics["ttft"] = float(m.group(1))
    return metrics


# --- runner -----------------------------------------------------------------

def run_inference(script, model_dir, prompt=None, max_length=512,
                  prompt_tokens=None, decode_tokens=None):
    cmd = [sys.executable, script, model_dir]
    if prompt_tokens is not None:
        cmd += ["--prompt-tokens", str(prompt_tokens)]
    elif prompt is not None:
        cmd += ["--prompt", prompt]
    if decode_tokens is not None:
        cmd += ["--decode-tokens", str(decode_tokens)]
    cmd += ["--max-length", str(max_length)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    m = parse_metrics(result.stdout + result.stderr)
    m["returncode"] = result.returncode
    m["output"] = result.stdout + result.stderr
    return m


def run_multiple(label, script, model_dir, warmup, runs, verbose=False, **kwargs):
    print(f"--- {label}: {warmup} warmup + {runs} measured runs ---")
    for i in range(warmup):
        print(f"  [warmup {i+1}/{warmup}] ...", end="", flush=True)
        r = run_inference(script, model_dir, **kwargs)
        if r["returncode"] != 0:
            print(f" FAILED (rc={r['returncode']})")
            if verbose:
                print(r["output"])
            return None
        print(
            f" prefill {r['prefill_tps'] or 0:.2f} t/s, "
            f"decode {r['decode_tps'] or 0:.2f} t/s (full run, discarded)"
        )

    results = []
    for i in range(runs):
        print(f"  [run {i+1}/{runs}] ...", end="", flush=True)
        r = run_inference(script, model_dir, **kwargs)
        if r["returncode"] != 0:
            print(f" FAILED (rc={r['returncode']})")
            if verbose:
                print(r["output"])
            continue
        print(
            f" prefill {r['prefill_tps'] or 0:.2f} t/s, "
            f"decode {r['decode_tps'] or 0:.2f} t/s, "
            f"ttft {(r['ttft'] or 0)*1000:.0f} ms"
        )
        results.append(r)
    return results


# --- aggregation / formatting ----------------------------------------------

def aggregate(results, key):
    vals = [r[key] for r in results if r.get(key) is not None]
    if not vals:
        return None, None
    med = statistics.median(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return med, stdev


def fmt_stat(median, stdev, unit="", prec=2):
    if median is None:
        return "N/A"
    if stdev is None or stdev == 0:
        return f"{median:.{prec}f}{unit}"
    return f"{median:.{prec}f} +/- {stdev:.{prec}f}{unit}"


def md_table(rows):
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]

    def fmt_row(r):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"

    out = [fmt_row(rows[0])]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows[1:]:
        out.append(fmt_row(r))
    return "\n".join(out)


# --- CLI --------------------------------------------------------------------

def parse_backend(spec):
    """Parse NAME:SCRIPT:MODEL_DIR. MODEL_DIR can contain colons on Windows (C:\\...)."""
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--backend expects NAME:SCRIPT:MODEL_DIR, got: {spec!r}"
        )
    name, script, model_dir = parts
    if not name or not script or not model_dir:
        raise argparse.ArgumentTypeError(f"--backend has empty field in: {spec!r}")
    return {"name": name, "script": script, "model_dir": model_dir}


def run_single_mode(args):
    print(f"Prompt: {args.prompt}")
    print(f"Max length: {args.max_length}")
    print(f"Warmup: {args.warmup}   Runs: {args.runs}")
    print(f"Backends: {', '.join(b['name'] for b in args.backend)}")
    print()

    all_results = []
    for b in args.backend:
        results = run_multiple(
            b["name"], b["script"], b["model_dir"],
            args.warmup, args.runs, args.verbose,
            prompt=args.prompt, max_length=args.max_length,
        )
        all_results.append((b["name"], results or []))
        print()

    # Summary table
    col_width = max(20, max(len(name) for name, _ in all_results) + 4)
    sep_width = 18 + (col_width + 1) * len(all_results)
    print("=" * sep_width)
    header = f"{'Metric':<18}" + "".join(f"{name:>{col_width}}" for name, _ in all_results)
    print(header)
    print(f"{'-'*18}" + "".join(f" {'-'*(col_width-1)}" for _ in all_results))

    for label, key, unit in [
        ("Decode TPS", "decode_tps", " t/s"),
        ("Prefill TPS", "prefill_tps", " t/s"),
        ("TTFT", "ttft", " s"),
    ]:
        row = f"{label:<18}"
        for _, results in all_results:
            med, std = aggregate(results, key)
            row += f"{fmt_stat(med, std, unit):>{col_width}}"
        print(row)

    # Pairwise decode speedups vs first backend
    if len(all_results) >= 2:
        ref_name, ref_results = all_results[0]
        ref_med, _ = aggregate(ref_results, "decode_tps")
        if ref_med:
            print()
            for name, results in all_results[1:]:
                med, _ = aggregate(results, "decode_tps")
                if med:
                    ratio = med / ref_med
                    direction = "faster" if ratio > 1 else "slower"
                    print(f">> {name} decode is {ratio:.2f}x {direction} than {ref_name}")
    print("=" * sep_width)


def run_sweep_mode(args):
    contexts = [int(x.strip()) for x in args.contexts.split(",") if x.strip()]
    print(f"Context sweep: {contexts}")
    print(f"Decode tokens per run: {args.decode_tokens}")
    print(f"Warmup: {args.warmup}   Runs: {args.runs}")
    print(f"Backends: {', '.join(b['name'] for b in args.backend)}")
    print()

    # sweep_results[ctx][backend_name] = list of per-run metric dicts
    sweep_results = {}
    for ctx in contexts:
        sweep_results[ctx] = {}
        for b in args.backend:
            label = f"{b['name']} @ ctx={ctx}"
            results = run_multiple(
                label, b["script"], b["model_dir"],
                args.warmup, args.runs, args.verbose,
                prompt_tokens=ctx, decode_tokens=args.decode_tokens,
            ) or []
            sweep_results[ctx][b["name"]] = results
            print()

    # Markdown summary
    backends = [b["name"] for b in args.backend]
    header = ["Ctx tokens"]
    for name in backends:
        header += [f"{name} prefill (t/s)", f"{name} decode (t/s)", f"{name} TTFT (s)"]
    rows = [header]
    for ctx in contexts:
        row = [str(ctx)]
        for name in backends:
            results = sweep_results[ctx][name]
            p_med, p_std = aggregate(results, "prefill_tps")
            d_med, d_std = aggregate(results, "decode_tps")
            t_med, t_std = aggregate(results, "ttft")
            row += [
                fmt_stat(p_med, p_std),
                fmt_stat(d_med, d_std),
                fmt_stat(t_med, t_std, prec=3),
            ]
        rows.append(row)

    md = md_table(rows)
    print("=" * 60)
    print("Summary (markdown):")
    print(md)
    print("=" * 60)

    if args.output_md:
        out_dir = os.path.dirname(args.output_md)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"Saved markdown table to {args.output_md}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark one or more inference backends (single prompt or context sweep).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backend", type=parse_backend, action="append", required=True,
        metavar="NAME:SCRIPT:MODEL_DIR",
        help="Inference backend to benchmark. Repeat for multiple backends.",
    )
    # Single-prompt mode args
    parser.add_argument("--prompt", default="Briefly describe the advantages of NPU.",
                        help="Input prompt (single-prompt mode)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max generation length (default: 512)")
    # Sweep-mode args
    parser.add_argument("--contexts", default=None,
                        help="Comma-separated context lengths (triggers sweep mode). "
                             "Example: 64,128,512,2048,8192")
    parser.add_argument("--decode-tokens", type=int, default=128,
                        help="Tokens to decode per run in sweep mode (default: 128)")
    parser.add_argument("--output-md", default=None,
                        help="Optional path to save sweep-mode markdown table.")
    # Shared
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (default: 1)")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Print full output on failure")
    args = parser.parse_args()

    if args.contexts:
        run_sweep_mode(args)
    else:
        run_single_mode(args)


if __name__ == "__main__":
    main()
