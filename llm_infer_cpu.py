import argparse
import onnxruntime_genai as og
import time
import os

def run_cpu_benchmark(model_dir, prompt=None, max_length=512):
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
        input_tokens = tokenizer.encode(prompt)
        tokenizer_end = time.perf_counter()
        print(f"Tokenizer time: {tokenizer_end-tokenizer_start:.2f}s")
        
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=max_length, temperature=0.7)
        
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

        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if new_tokens:
                print(tokenizer_stream.decode(new_tokens[0]), end='', flush=True)
                tokens_count += 1
        
        decode_end = time.perf_counter()
        
        ttft = (first_token_time - prefill_start) * 1000
        total_decode_time = decode_end - decode_start
        tps = tokens_count / total_decode_time if total_decode_time > 0 else 0

        print("\n\n" + "="*30)
        print(f"Performance report (Snapdragon X Elite CPU-Oryon)")
        print(f"TTFT: {ttft:.2f} ms")
        print(f"TPS: {tps:.2f} tokens/s")
        print(f"Total tokens: {tokens_count}")
        print("="*30)

    except Exception as e:
        print(f"\nRuntime error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU inference benchmark")
    parser.add_argument("model_dir", help="Path to CPU model directory")
    parser.add_argument("--prompt", default=None, help="Input prompt")
    parser.add_argument("--max-length", type=int, default=512, help="Max generation length")
    args = parser.parse_args()
    run_cpu_benchmark(args.model_dir, args.prompt, args.max_length)