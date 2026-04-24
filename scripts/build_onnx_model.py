"""Convert a HuggingFace PyTorch LLM to CPU-int4 ONNX via onnxruntime-genai.

Run on a Linux VM with 64+ GB RAM (not on a 16 GB X Elite). Output ONNX is
portable across CPU architectures.

    pip install onnxruntime-genai transformers torch huggingface_hub
    python scripts/build_onnx_model.py --hf-id Qwen/Qwen2.5-14B-Instruct --out ./out
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-id", required=True, help="HuggingFace model id")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--precision", default="int4", choices=["int4", "int8", "fp16", "fp32"])
    parser.add_argument("--execution-provider", default="cpu", choices=["cpu", "cuda", "dml", "webgpu"])
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--extra-options", default=None, help="Forwarded to builder as key=value,key=value")
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "onnxruntime_genai.models.builder",
        "-m", args.hf_id,
        "-o", str(out_path),
        "-p", args.precision,
        "-e", args.execution_provider,
    ]
    if args.cache_dir:
        cmd.extend(["-c", args.cache_dir])
    if args.extra_options:
        cmd.extend(["--extra_options", args.extra_options])

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print(f"\nBuild complete at: {out_path}")
    for p in sorted(out_path.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:<40} {size_mb:>10.1f} MB")


if __name__ == "__main__":
    main()
