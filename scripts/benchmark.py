"""Benchmark NPU vs CPU inference and print comparison table."""

import argparse
import subprocess
import sys
import re


def run_inference(script, model_dir, prompt, max_length):
    cmd = [sys.executable, script, model_dir, "--prompt", prompt, "--max-length", str(max_length)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    tps = None
    tokens = None
    ttft = None

    for line in output.splitlines():
        if m := re.search(r"Speed:\s*([\d.]+)", line):
            tps = float(m.group(1))
        if m := re.search(r"TPS:\s*([\d.]+)", line):
            tps = float(m.group(1))
        if m := re.search(r"Tokens:\s*(\d+)", line):
            tokens = int(m.group(1))
        if m := re.search(r"Total tokens:\s*(\d+)", line):
            tokens = int(m.group(1))
        if m := re.search(r"TTFT:\s*([\d.]+)\s*ms", line):
            ttft = float(m.group(1)) / 1000
        if m := re.search(r"TTFT:\s*([\d.]+)s", line):
            ttft = float(m.group(1))

    return {"tps": tps, "tokens": tokens, "ttft": ttft, "output": output}


def main():
    parser = argparse.ArgumentParser(description="Benchmark NPU vs CPU inference")
    parser.add_argument("--npu-model-dir", required=True, help="Path to NPU model directory")
    parser.add_argument("--cpu-model-dir", required=True, help="Path to CPU model directory")
    parser.add_argument("--prompt", default="Briefly describe the advantages of NPU.", help="Input prompt")
    parser.add_argument("--max-length", type=int, default=512, help="Max generation length")
    args = parser.parse_args()

    print(f"Prompt: {args.prompt}")
    print(f"Max length: {args.max_length}")
    print()

    print("--- Running NPU inference ---")
    npu = run_inference("llm_infer_npu.py", args.npu_model_dir, args.prompt, args.max_length)
    print(npu["output"])

    print("--- Running CPU inference ---")
    cpu = run_inference("llm_infer_cpu.py", args.cpu_model_dir, args.prompt, args.max_length)
    print(cpu["output"])

    print("=" * 50)
    print(f"{'Metric':<12} {'NPU':>12} {'CPU':>12}")
    print(f"{'-'*12} {'-'*12} {'-'*12}")

    def fmt(v, unit=""):
        return f"{v:.2f}{unit}" if v is not None else "N/A"

    print(f"{'TPS':<12} {fmt(npu['tps'], ' t/s'):>12} {fmt(cpu['tps'], ' t/s'):>12}")
    print(f"{'TTFT':<12} {fmt(npu['ttft'], 's'):>12} {fmt(cpu['ttft'], 's'):>12}")
    print(f"{'Tokens':<12} {str(npu['tokens'] or 'N/A'):>12} {str(cpu['tokens'] or 'N/A'):>12}")

    if npu["tps"] and cpu["tps"]:
        speedup = cpu["tps"] / npu["tps"]
        faster = "NPU" if npu["tps"] > cpu["tps"] else "CPU"
        ratio = npu["tps"] / cpu["tps"] if npu["tps"] > cpu["tps"] else speedup
        print(f"\n{faster} is {ratio:.2f}x faster")

    print("=" * 50)


if __name__ == "__main__":
    main()
