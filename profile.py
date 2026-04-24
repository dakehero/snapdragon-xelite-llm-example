"""Per-op profiling via onnxruntime-genai's session_options.enable_profiling.

Patches the model's genai_config.json in place to turn profiling on, runs the
selected inference backend once, restores the original config, then parses the
Chrome-trace JSON that ORT dropped on disk and prints a summary (top ops by
total time, grouped by op kind).

Notes on limitations:
- With QNN EP, the QNN-delegated subgraph typically shows up as a single
  compound op in the trace ("QNNExecutionProvider_node_...") — ORT only sees
  the boundary, not what happens inside the NPU. Use this profile to find
  non-fused ops and CPU fallback hotspots; use QNN SDK profiling separately
  for HTP kernel-level insight.
- Profiling adds overhead; TPS numbers from this run are not benchmark-grade.

Usage:
    python profile.py --model-dir /path/to/model --backend ort-qnn
    python profile.py --model-dir /path/to/model --backend ort-cpu \
        --output-dir profile_output --max-length 128
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


BACKEND_SCRIPTS = {
    "ort-qnn": "llm_infer_ort_qnn.py",
    "ort-cpu": "llm_infer_ort_cpu.py",
}


def find_config(model_dir: Path) -> Path:
    candidates = [model_dir / "genai_config.json", model_dir / "genai" / "genai_config.json"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No genai_config.json under {model_dir}. Checked: {candidates}"
    )


def patch_config(config_path: Path, profile_prefix: str):
    """Enable profiling in genai_config.json. Returns the original content for restore."""
    original = config_path.read_text(encoding="utf-8")
    data = json.loads(original)

    # genai_config shape: model.decoder.session_options (per Microsoft's ORT-GenAI schema).
    # If the schema differs for a given model, we walk the tree defensively.
    decoder = data.get("model", {}).get("decoder", {})
    if not isinstance(decoder, dict):
        raise RuntimeError("Unexpected config shape: model.decoder is not an object")

    so = decoder.setdefault("session_options", {})
    # ORT-GenAI's JSON schema takes a single string: non-empty enables profiling
    # and acts as the output-file prefix. (Unlike the ORT C++ API which splits
    # these into a bool + a string.) Setting this to True fails with "Expected
    # a string but saw a bool".
    so["enable_profiling"] = profile_prefix

    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return original


def restore_config(config_path: Path, original: str):
    config_path.write_text(original, encoding="utf-8")


def run_backend(script: str, model_dir: Path, prompt: str, max_length: int, cwd: Path) -> int:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / script),
        str(model_dir),
        "--prompt", prompt,
        "--max-length", str(max_length),
    ]
    print(f"[profile] running: {' '.join(cmd)}")
    print(f"[profile] cwd:     {cwd}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def find_latest_profile(output_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(output_dir.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def summarize_profile(profile_path: Path, top_n: int = 20):
    """Parse ORT Chrome-trace JSON and print top ops by total duration."""
    with profile_path.open(encoding="utf-8") as f:
        events = json.load(f)

    # ORT emits a list of events, each with {"cat", "name", "dur" (microseconds),
    # "args": {"op_name", "provider", ...}, ...}.
    # We care about per-op events; filter by cat == "Node" (the per-op timing
    # category ORT uses).
    per_op_dur = defaultdict(lambda: {"count": 0, "total_us": 0})
    per_provider_dur = defaultdict(lambda: {"count": 0, "total_us": 0})
    per_instance = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("cat") != "Node":
            continue
        dur = ev.get("dur")
        if dur is None:
            continue
        args = ev.get("args", {}) if isinstance(ev.get("args"), dict) else {}
        op_name = args.get("op_name") or ev.get("name", "?")
        provider = args.get("provider", "?")

        per_op_dur[op_name]["count"] += 1
        per_op_dur[op_name]["total_us"] += dur
        per_provider_dur[provider]["count"] += 1
        per_provider_dur[provider]["total_us"] += dur
        per_instance.append((dur, op_name, provider, ev.get("name", "")))

    total_us = sum(v["total_us"] for v in per_op_dur.values()) or 1

    print()
    print("=" * 78)
    print(f"Profile: {profile_path}")
    print(f"Events analyzed: {sum(v['count'] for v in per_op_dur.values())}")
    print(f"Total node time: {total_us / 1000:.1f} ms")
    print("=" * 78)

    print(f"\nBy provider:")
    print(f"  {'Provider':<40} {'Count':>8} {'Total (ms)':>14} {'Share':>8}")
    for prov, v in sorted(per_provider_dur.items(), key=lambda kv: -kv[1]["total_us"]):
        share = v["total_us"] / total_us * 100
        print(f"  {prov:<40} {v['count']:>8} {v['total_us']/1000:>14.2f} {share:>7.1f}%")

    print(f"\nTop {top_n} op kinds by total time:")
    print(f"  {'Op':<40} {'Count':>8} {'Total (ms)':>14} {'Share':>8} {'Avg (us)':>10}")
    ranked = sorted(per_op_dur.items(), key=lambda kv: -kv[1]["total_us"])
    for op, v in ranked[:top_n]:
        share = v["total_us"] / total_us * 100
        avg = v["total_us"] / v["count"] if v["count"] else 0
        print(f"  {op:<40} {v['count']:>8} {v['total_us']/1000:>14.2f} {share:>7.1f}% {avg:>10.1f}")

    # Single-slowest invocations, useful for spotting that one QNN subgraph
    # that eats 80% of the time
    print(f"\nTop 10 single slowest event invocations:")
    per_instance.sort(key=lambda t: -t[0])
    for dur, op_name, prov, label in per_instance[:10]:
        print(f"  {dur/1000:>10.2f} ms   {op_name:<30} {prov:<25} {label[:40]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="Path to model directory (containing genai_config.json)")
    parser.add_argument("--backend", choices=BACKEND_SCRIPTS.keys(), default="ort-qnn",
                        help="Which inference backend to profile")
    parser.add_argument("--prompt", default="Briefly describe the advantages of NPU.",
                        help="Input prompt")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max generation length (kept modest; profile runs slower than normal)")
    parser.add_argument("--output-dir", default="profile_output", help="Directory for profile JSON and summary")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N ops to print")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = find_config(model_dir)
    print(f"[profile] model-dir: {model_dir}")
    print(f"[profile] config:    {config_path}")
    print(f"[profile] backend:   {args.backend}")
    print(f"[profile] output:    {output_dir}")

    # Prefix embeds backend + timestamp so successive runs don't collide
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"ort_profile_{args.backend}_{timestamp}_"

    # Back up original config to the output dir for auditability
    backup_path = output_dir / f"genai_config.before_{timestamp}.json"
    shutil.copy2(config_path, backup_path)
    original = patch_config(config_path, prefix)
    print(f"[profile] patched config; original backed up to {backup_path}")

    try:
        script = BACKEND_SCRIPTS[args.backend]
        rc = run_backend(script, model_dir, args.prompt, args.max_length, output_dir)
        if rc != 0:
            print(f"[profile] inference returned rc={rc}; continuing to parse whatever profile landed")
    finally:
        restore_config(config_path, original)
        print(f"[profile] restored original config")

    # ORT writes the profile into the cwd we passed (output_dir) with the prefix we set.
    profile_file = find_latest_profile(output_dir, prefix)
    if profile_file is None:
        print(f"[profile] ERROR: no {prefix}*.json written under {output_dir}")
        print(f"[profile] possible causes:")
        print(f"         - this onnxruntime-genai build ignores session_options.enable_profiling")
        print(f"         - the model config schema differs from model.decoder.session_options")
        print(f"         - inference crashed before any op ran")
        sys.exit(2)

    summarize_profile(profile_file, top_n=args.top_n)


if __name__ == "__main__":
    main()
