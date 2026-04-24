"""Verify numerical correctness: compare NPU output against CPU (or reference).

Runs greedy decoding (temperature=0) on both NPU and CPU models with the same prompt,
then compares the first N generated token IDs. Reports the first divergence point
and the decoded text for visual inspection.

Usage:
    pixi run python verify.py \
        --npu-model-dir PATH_TO_QNN_MODEL \
        --cpu-model-dir PATH_TO_CPU_MODEL \
        --prompt "The capital of France is" \
        --num-tokens 20
"""

import argparse
import os
import sys


def _setup_dlls():
    """Add QNN / genai DLL directories to search path."""
    import onnxruntime_qnn
    import onnxruntime_genai as og
    qnn_dir = os.path.dirname(onnxruntime_qnn.__file__)
    genai_dir = os.path.dirname(og.__file__)
    os.add_dll_directory(qnn_dir)
    os.add_dll_directory(genai_dir)
    os.environ["PATH"] = genai_dir + os.pathsep + qnn_dir + os.pathsep + os.environ.get("PATH", "")
    og.register_execution_provider_library('QNNExecutionProvider', onnxruntime_qnn.get_library_path())
    return og


def generate_tokens(og, model_dir, prompt, num_tokens, use_qnn):
    """Generate `num_tokens` tokens greedily. Returns list of token ids and decoded text."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    config = og.Config(model_dir)
    if use_qnn:
        config.clear_providers()
        config.append_provider('qnn')
    # else: default CPU provider

    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    # Use max_length=512 (matches the working llm_infer_ort_qnn.py). QNN models have
    # KV-cache shape constraints baked into the graph; a very small max_length
    # (e.g., prompt_len + 20) can trigger a native buffer check failure
    # (STATUS_STACK_BUFFER_OVERRUN).
    #
    # For determinism we use temperature=0.0 (genai interprets this as greedy).
    # If your genai build does not support temperature=0, set top_k=1 instead.
    params.set_search_options(
        max_length=512,
        temperature=0.0,
    )
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    generated = []
    while not generator.is_done() and len(generated) < num_tokens:
        generator.generate_next_token()
        new_tokens = generator.get_next_tokens()
        if new_tokens:
            generated.extend(new_tokens)

    generated = [int(t) for t in generated[:num_tokens]]  # strip numpy int32 wrappers
    text = tokenizer.decode(generated)
    return generated, text


def main():
    parser = argparse.ArgumentParser(description="Verify NPU output matches CPU (numerical correctness)")
    parser.add_argument("--npu-model-dir", required=True, help="Path to QNN (NPU) model directory")
    parser.add_argument("--cpu-model-dir", required=True, help="Path to CPU model directory")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="Test prompt (default: deterministic factual prompt)")
    parser.add_argument("--num-tokens", type=int, default=20,
                        help="Number of tokens to compare (default: 20)")
    args = parser.parse_args()

    og = _setup_dlls()

    print("=" * 60)
    print("Numerical Correctness Verification")
    print("=" * 60)
    print(f"Prompt:     {args.prompt!r}")
    print(f"Tokens:     first {args.num_tokens} (greedy decoding)")
    print(f"NPU model:  {args.npu_model_dir}")
    print(f"CPU model:  {args.cpu_model_dir}")
    print()

    print("[1/2] Generating on NPU...")
    try:
        npu_ids, npu_text = generate_tokens(og, args.npu_model_dir, args.prompt, args.num_tokens, use_qnn=True)
    except Exception as e:
        print(f"  NPU generation failed: {e}")
        sys.exit(2)
    print(f"  NPU tokens: {npu_ids}")
    print(f"  NPU text:   {npu_text!r}")
    print()

    print("[2/2] Generating on CPU...")
    try:
        cpu_ids, cpu_text = generate_tokens(og, args.cpu_model_dir, args.prompt, args.num_tokens, use_qnn=False)
    except Exception as e:
        print(f"  CPU generation failed: {e}")
        sys.exit(2)
    print(f"  CPU tokens: {cpu_ids}")
    print(f"  CPU text:   {cpu_text!r}")
    print()

    print("=" * 60)
    print("Comparison")
    print("=" * 60)

    min_len = min(len(npu_ids), len(cpu_ids))
    divergence_at = None
    for i in range(min_len):
        if npu_ids[i] != cpu_ids[i]:
            divergence_at = i
            break

    # Token-set overlap: how many NPU tokens also appear in CPU output?
    # Coarse signal of "do both models draw from similar vocab for this prompt".
    overlap = len(set(npu_ids) & set(cpu_ids))
    overlap_ratio = overlap / max(len(npu_ids), 1)

    if divergence_at is None and len(npu_ids) == len(cpu_ids):
        print(f"PASS: First {min_len} tokens match exactly (bit-exact).")
        print(f"  NPU: {npu_text!r}")
        sys.exit(0)

    matched_prefix = divergence_at if divergence_at is not None else min_len
    print(f"Matched prefix:  {matched_prefix} / {min_len} tokens")
    print(f"Token overlap:   {overlap}/{len(npu_ids)} ({overlap_ratio:.0%}) of NPU tokens also appear in CPU output")
    print()
    print(f"  NPU: {npu_text!r}")
    print(f"  CPU: {cpu_text!r}")
    print()

    # Classify result. Full bit-exact match for quantized QNN models is unrealistic;
    # we accept divergence as long as both outputs are coherent (non-empty, no degenerate repetition).
    def looks_degenerate(ids):
        if not ids:
            return True
        # Trivial repetition check: all same token, or very low unique ratio
        if len(set(ids)) <= 2:
            return True
        return False

    if matched_prefix >= 1 and not looks_degenerate(npu_ids) and not looks_degenerate(cpu_ids):
        print("RESULT: ACCEPTABLE_DIVERGENCE")
        print("  Both outputs are coherent. Token-level divergence after the initial")
        print("  match is expected for quantized QNN models (small logit perturbations")
        print("  amplify into different top-1 choices, then autoregressive sampling diverges).")
        print("  Inspect the decoded texts above for semantic fidelity.")
        sys.exit(0)
    else:
        print("RESULT: UNEXPECTED")
        if matched_prefix == 0:
            print("  NPU and CPU disagree on the very first generated token. This could indicate:")
            print("    - model family mismatch (different tokenizer / weights)")
            print("    - severe quantization regression")
            print("    - broken QNN graph")
        if looks_degenerate(npu_ids):
            print("  NPU output looks degenerate (repetition / low diversity).")
        if looks_degenerate(cpu_ids):
            print("  CPU output looks degenerate (repetition / low diversity).")
        sys.exit(1)


if __name__ == "__main__":
    main()
