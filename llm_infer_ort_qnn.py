import argparse
import onnxruntime_genai as og
import onnxruntime_qnn
import time
import os
import sys

# Add DLL directories
qnn_dir = os.path.dirname(onnxruntime_qnn.__file__)
genai_dir = os.path.dirname(og.__file__)
os.add_dll_directory(qnn_dir)
os.add_dll_directory(genai_dir)
os.environ["PATH"] = genai_dir + os.pathsep + qnn_dir + os.pathsep + os.environ.get("PATH", "")

# Register QNN provider (must use 'QNNExecutionProvider' as registration name)
og.register_execution_provider_library('QNNExecutionProvider', onnxruntime_qnn.get_library_path())


_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
)


def make_input_tokens(tokenizer, prompt_str, prompt_tokens):
    """Return token IDs. If prompt_tokens is set, synthesize exactly that many
    tokens by repeat-encoding filler text and truncating."""
    if prompt_tokens is None:
        return tokenizer.encode(prompt_str)
    # Estimate filler repetitions; _FILLER is ~20 BPE tokens for most models.
    reps = max(1, (prompt_tokens // 15) + 2)
    text = _FILLER * reps
    tokens = tokenizer.encode(text)
    while len(tokens) < prompt_tokens:
        text += _FILLER
        tokens = tokenizer.encode(text)
    return list(tokens)[:prompt_tokens]


def run_benchmark(model_dir, prompt=None, max_length=512, prompt_tokens=None, decode_tokens=None):
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return
    
    try:
        print("--- Initializing QNN engine ---")
        print(f"QNN available: {og.is_qnn_available()}")
        
        load_start = time.perf_counter()
        
        # Configure QNN provider via Config
        config = og.Config(model_dir)
        config.clear_providers()
        config.append_provider('qnn')
        
        model = og.Model(config)
        tokenizer = og.Tokenizer(model)
        
        load_end = time.perf_counter()
        print(f"Model loaded in: {load_end - load_start:.2f}s")

        # Prepare input
        if prompt is None:
            prompt = "Briefly describe the advantages of NPU."
        tokenizer_start = time.perf_counter()
        input_tokens = make_input_tokens(tokenizer, prompt, prompt_tokens)
        tokenizer_end = time.perf_counter()
        print(f"Tokenizer time: {tokenizer_end-tokenizer_start:.2f}s")

        prompt_len = len(input_tokens)
        # When decode_tokens is set, bump max_length so generator doesn't stop early.
        effective_max_length = max(max_length, prompt_len + (decode_tokens or 0) + 8)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=effective_max_length, temperature=0.7)
        
        generator = og.Generator(model, params)
        
        print("\n--- Running inference ---")
        
        # Record prefill time
        prefill_start = time.perf_counter()
        generator.append_tokens(input_tokens)
        
        # Generate first token
        generator.generate_next_token()
        first_token_time = time.perf_counter()
        
        ttft = first_token_time - prefill_start
        prefill_tps = prompt_len / ttft if ttft > 0 else 0
        print(f"TTFT: {ttft:.2f}s  ({prompt_len} prompt tokens, prefill {prefill_tps:.2f} tokens/s)")

        # Generate remaining tokens with streaming
        gen_start = time.perf_counter()
        tokenizer_stream = tokenizer.create_stream()
        print(f"\nOutput:\n", end='', flush=True)

        tokens_count = 0

        # Stream first token (already generated)
        new_tokens = generator.get_next_tokens()
        if new_tokens:
            print(tokenizer_stream.decode(new_tokens[0]), end='', flush=True)
            tokens_count += 1

        # Continue streaming; stop at decode_tokens if provided, else EOS / max_length.
        decode_target = decode_tokens  # None => rely on generator.is_done()
        while not generator.is_done():
            if decode_target is not None and tokens_count >= decode_target:
                break
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if new_tokens:
                print(tokenizer_stream.decode(new_tokens[0]), end='', flush=True)
                tokens_count += 1

        gen_end = time.perf_counter()
        print()

        total_tokens = tokens_count
        total_time = gen_end - gen_start
        decode_tps = total_tokens / total_time if total_time > 0 else 0
        
        print(f"\nGeneration stats:")
        print(f"   Prompt tokens: {prompt_len}")
        print(f"   Generated tokens: {total_tokens}")
        print(f"   TTFT: {ttft*1000:.2f} ms")
        print(f"   Prefill speed: {prefill_tps:.2f} tokens/s")
        print(f"   Decode speed: {decode_tps:.2f} tokens/s")
        print(f"   Speed: {decode_tps:.2f} tokens/s")  # kept for benchmark.py parsing compat

    except Exception as e:
        print(f"\nRuntime error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QNN NPU inference benchmark")
    parser.add_argument("model_dir", help="Path to QNN model directory")
    parser.add_argument("--prompt", default=None, help="Input prompt (ignored if --prompt-tokens is set)")
    parser.add_argument("--max-length", type=int, default=512, help="Max generation length")
    parser.add_argument("--prompt-tokens", type=int, default=None,
                        help="Synthesize a prompt of exactly this many tokens (overrides --prompt)")
    parser.add_argument("--decode-tokens", type=int, default=None,
                        help="Stop decoding after this many generated tokens (for controlled benchmarks)")
    args = parser.parse_args()
    run_benchmark(args.model_dir, args.prompt, args.max_length,
                  prompt_tokens=args.prompt_tokens, decode_tokens=args.decode_tokens)
