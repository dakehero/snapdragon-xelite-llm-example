import argparse
import onnxruntime_genai as og
import time
import os


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
    reps = max(1, (prompt_tokens // 15) + 2)
    text = _FILLER * reps
    tokens = tokenizer.encode(text)
    while len(tokens) < prompt_tokens:
        text += _FILLER
        tokens = tokenizer.encode(text)
    return list(tokens)[:prompt_tokens]


def run_cpu_benchmark(model_dir, prompt=None, max_length=512, prompt_tokens=None, decode_tokens=None):
    if not os.path.exists(model_dir):
        print(f"CPU model directory not found: {model_dir}")
        return

    try:
        print("--- Initializing CPU engine (Oryon Cores) ---")
        load_start = time.perf_counter()
        # CPUExecutionProvider is used automatically
        model = og.Model(model_dir)
        tokenizer = og.Tokenizer(model)
        load_end = time.perf_counter()
        print(f"CPU model loaded in: {load_end - load_start:.2f}s")

        if prompt is None:
            prompt = "Briefly describe the advantages of Snapdragon X Elite NPU in about 200 words."
        tokenizer_start = time.perf_counter()
        input_tokens = make_input_tokens(tokenizer, prompt, prompt_tokens)
        tokenizer_end = time.perf_counter()
        print(f"Tokenizer time: {tokenizer_end-tokenizer_start:.2f}s")

        prompt_len = len(input_tokens)
        effective_max_length = max(max_length, prompt_len + (decode_tokens or 0) + 8)

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=effective_max_length, temperature=0.7)
        
        generator = og.Generator(model, params)
        
        print("\n--- Running CPU inference ---")
        
        prefill_start = time.perf_counter()
        generator.append_tokens(input_tokens)
        generator.generate_next_token()
        first_token_time = time.perf_counter()
        
        tokens_count = 0
        decode_start = time.perf_counter()
        tokenizer_stream = tokenizer.create_stream()
        
        # Print first token
        print(tokenizer_stream.decode(generator.get_next_tokens()[0]), end='', flush=True)
        tokens_count += 1

        decode_target = decode_tokens
        while not generator.is_done():
            if decode_target is not None and tokens_count >= decode_target:
                break
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if new_tokens:
                print(tokenizer_stream.decode(new_tokens[0]), end='', flush=True)
                tokens_count += 1
        
        decode_end = time.perf_counter()
        
        ttft_s = first_token_time - prefill_start
        ttft_ms = ttft_s * 1000
        total_decode_time = decode_end - decode_start
        decode_tps = tokens_count / total_decode_time if total_decode_time > 0 else 0
        prefill_tps = prompt_len / ttft_s if ttft_s > 0 else 0

        print("\n\n" + "="*30)
        print(f"Performance report (Snapdragon X Elite CPU-Oryon)")
        print(f"Prompt tokens: {prompt_len}")
        print(f"TTFT: {ttft_ms:.2f} ms")
        print(f"Prefill speed: {prefill_tps:.2f} tokens/s")
        print(f"Decode speed: {decode_tps:.2f} tokens/s")
        print(f"TPS: {decode_tps:.2f} tokens/s")  # kept for benchmark.py parsing compat
        print(f"Total tokens: {tokens_count}")
        print("="*30)

    except Exception as e:
        print(f"\nRuntime error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU inference benchmark")
    parser.add_argument("model_dir", help="Path to CPU model directory")
    parser.add_argument("--prompt", default=None, help="Input prompt (ignored if --prompt-tokens is set)")
    parser.add_argument("--max-length", type=int, default=512, help="Max generation length")
    parser.add_argument("--prompt-tokens", type=int, default=None,
                        help="Synthesize a prompt of exactly this many tokens (overrides --prompt)")
    parser.add_argument("--decode-tokens", type=int, default=None,
                        help="Stop decoding after this many generated tokens (for controlled benchmarks)")
    args = parser.parse_args()
    run_cpu_benchmark(args.model_dir, args.prompt, args.max_length,
                      prompt_tokens=args.prompt_tokens, decode_tokens=args.decode_tokens)