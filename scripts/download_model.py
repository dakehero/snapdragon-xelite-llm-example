"""Download ONNX models from Hugging Face for local inference.

Known working models on Snapdragon X Elite NPU (via QNN EP):
  - microsoft/Phi-3.5-mini-instruct-onnx (subfolder: cpu-int4-rtn-block-32-acc-level-4 for CPU)
  - onnx-community/Qwen2.5-0.5B-Instruct (for CPU verification)

For QNN-specific models, Microsoft publishes them through Foundry Local
(which caches to ~/.foundry/cache/models/). HuggingFace alternatives for QNN are limited.

Usage:
    # Download default (Phi-3.5 CPU int4, for verify.py and CPU benchmarking)
    pixi run python scripts/download_model.py

    # Custom HF repo + subfolder
    pixi run python scripts/download_model.py --repo microsoft/Phi-3.5-mini-instruct-onnx \\
        --subfolder cpu-int4-rtn-block-32-acc-level-4 --dest ./models/phi-3.5-cpu
"""

import argparse
import os
import sys


DEFAULT_REPO = "microsoft/Phi-3.5-mini-instruct-onnx"
DEFAULT_SUBFOLDER = "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
DEFAULT_DEST = os.path.join("models", "phi-3.5-mini-cpu-int4")


def main():
    parser = argparse.ArgumentParser(description="Download ONNX model from Hugging Face")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO})")
    parser.add_argument("--subfolder", default=DEFAULT_SUBFOLDER,
                        help=f"Subfolder inside repo (default: {DEFAULT_SUBFOLDER}). Empty string for root.")
    parser.add_argument("--dest", default=DEFAULT_DEST, help=f"Local destination (default: {DEFAULT_DEST})")
    parser.add_argument("--revision", default=None, help="Git revision (branch/tag/commit)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Install with: pixi run pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(args.dest, exist_ok=True)
    subfolder = args.subfolder.strip("/") or None

    print(f"Downloading {args.repo}" + (f" / {subfolder}" if subfolder else ""))
    print(f"Destination: {os.path.abspath(args.dest)}")
    print()

    # Explicitly list repo files then filter by subfolder prefix. This is more
    # reliable than allow_patterns with nested paths (which can silently match 0
    # files on some huggingface_hub versions).
    api = HfApi()
    try:
        all_files = api.list_repo_files(repo_id=args.repo, revision=args.revision)
    except Exception as e:
        print(f"Failed to list repo files: {e}")
        sys.exit(2)

    if subfolder:
        target_files = [f for f in all_files if f.startswith(subfolder + "/")]
    else:
        target_files = all_files

    if not target_files:
        print(f"WARNING: No files matched subfolder {subfolder!r} in {args.repo}.")
        print("Available top-level entries in repo (first 40):")
        tops = sorted({f.split("/", 1)[0] for f in all_files})[:40]
        for t in tops:
            print(f"  {t}")
        print()
        print(f"Browse: https://huggingface.co/{args.repo}/tree/main")
        sys.exit(3)

    print(f"Found {len(target_files)} file(s) to download.")
    for i, f in enumerate(target_files, 1):
        print(f"  [{i}/{len(target_files)}] {f}")
        try:
            hf_hub_download(
                repo_id=args.repo,
                filename=f,
                revision=args.revision,
                local_dir=args.dest,
            )
        except Exception as e:
            print(f"    FAILED: {e}")
            sys.exit(2)

    final_path = os.path.join(args.dest, subfolder) if subfolder else args.dest
    print()
    print(f"Downloaded to: {final_path}")
    print()
    print("Next steps:")
    print(f"  make run-cpu MODEL_DIR=\"{final_path}\"")
    print(f"  make verify NPU_MODEL_DIR=<your-qnn-model> CPU_MODEL_DIR=\"{final_path}\"")


if __name__ == "__main__":
    main()
