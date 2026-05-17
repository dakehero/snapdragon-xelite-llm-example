"""Run a reproducible context-length sweep with Foundry Local model paths.

Examples:
    pixi run python scripts/context_sweep.py --model qwen7b --backends both

    pixi run python scripts/context_sweep.py --model qwen7b --backends qnn \
        --contexts 64,128,256,512,2048 \
        --output-md results/context_sweep_qwen7b_x2elite_qnn.md --overwrite

    pixi run python scripts/context_sweep.py \
        --qnn-model-id qwen2.5-7b-instruct-qnn-npu:2 \
        --cpu-model-id qwen2.5-7b-instruct-generic-cpu:4
"""

import argparse
import os
import subprocess
import sys


MODEL_ALIASES = {
    "qwen1.5b": {
        "qnn": "qwen2.5-1.5b-instruct-qnn-npu:2",
        "cpu": "qwen2.5-1.5b-instruct-generic-cpu:4",
        "output": "results/context_sweep_qwen1.5b.md",
    },
    "qwen7b": {
        "qnn": "qwen2.5-7b-instruct-qnn-npu:2",
        "cpu": "qwen2.5-7b-instruct-generic-cpu:4",
        "output": "results/context_sweep_qwen7b.md",
    },
    "r1distill14b": {
        "qnn": "deepseek-r1-distill-qwen-14b-qnn-npu:2",
        "cpu": "deepseek-r1-distill-qwen-14b-generic-cpu:4",
        "output": "results/context_sweep_r1distill14b.md",
    },
}


def foundry_cache_root():
    return os.path.join(
        os.path.expanduser("~"), ".foundry", "cache", "models", "Microsoft"
    )


def newest_child_dir(path):
    if not os.path.isdir(path):
        return None
    children = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]
    if not children:
        return path
    return max(children, key=os.path.getmtime)


def resolve_foundry_model(model_id_or_path):
    if not model_id_or_path:
        return None
    expanded = os.path.abspath(os.path.expanduser(model_id_or_path))
    if os.path.isdir(expanded):
        return expanded

    candidates = [
        model_id_or_path,
        model_id_or_path.replace(":", "-"),
    ]
    for candidate in dict.fromkeys(candidates):
        model_root = os.path.join(foundry_cache_root(), candidate)
        resolved = newest_child_dir(model_root)
        if resolved:
            return resolved
    return None


def require_model(label, model_id_or_path):
    resolved = resolve_foundry_model(model_id_or_path)
    if resolved:
        print(f"{label}: {resolved}")
        return resolved

    print(f"ERROR: Could not resolve {label}: {model_id_or_path}")
    print(f"Expected either an existing path or a Foundry cache entry under:")
    print(f"  {foundry_cache_root()}")
    print()
    print("Download with Foundry Local, for example:")
    print(f"  foundry model download {model_id_or_path}")
    sys.exit(2)


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark.py context sweep using Foundry model aliases."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_ALIASES),
        default="qwen7b",
        help="Known model alias (default: qwen7b).",
    )
    parser.add_argument(
        "--backends",
        choices=["both", "qnn", "cpu"],
        default="both",
        help="Which backend(s) to run (default: both).",
    )
    parser.add_argument("--qnn-model-id", default=None,
                        help="Foundry model id or model directory for QNN.")
    parser.add_argument("--cpu-model-id", default=None,
                        help="Foundry model id or model directory for CPU.")
    parser.add_argument(
        "--contexts",
        default="64,128,256,512,1024,2048,4096,8192",
        help="Comma-separated context sizes.",
    )
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing output markdown file.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    alias = MODEL_ALIASES[args.model]
    qnn_model_id = args.qnn_model_id or alias["qnn"]
    cpu_model_id = args.cpu_model_id or alias["cpu"]
    output_md = args.output_md or alias["output"]

    if output_md and os.path.exists(output_md) and not args.overwrite:
        print(f"ERROR: Output file already exists: {output_md}")
        print("Pass --overwrite, or choose a new --output-md such as:")
        stem, ext = os.path.splitext(output_md)
        print(f"  {stem}_x2elite{ext}")
        sys.exit(2)

    cmd = [sys.executable, "benchmark.py"]
    if args.backends in ("both", "qnn"):
        qnn_model = require_model("QNN model", qnn_model_id)
        cmd += ["--backend", f"ort-qnn:llm_infer_ort_qnn.py:{qnn_model}"]
    if args.backends in ("both", "cpu"):
        cpu_model = require_model("CPU model", cpu_model_id)
        cmd += ["--backend", f"ort-cpu:llm_infer_ort_cpu.py:{cpu_model}"]

    cmd += [
        "--contexts", args.contexts,
        "--decode-tokens", str(args.decode_tokens),
        "--max-length", str(args.max_length),
        "--warmup", str(args.warmup),
        "--runs", str(args.runs),
        "--output-md", output_md,
    ]
    if args.verbose:
        cmd.append("--verbose")

    print("Running:", flush=True)
    print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    print(flush=True)

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
