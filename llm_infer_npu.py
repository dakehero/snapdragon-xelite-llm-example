import argparse
import onnxruntime_genai as og
import onnxruntime_qnn
import time
import os

# Add DLL directories
qnn_dir = os.path.dirname(onnxruntime_qnn.__file__)
genai_dir = os.path.dirname(og.__file__)
os.add_dll_directory(qnn_dir)
os.add_dll_directory(genai_dir)
os.environ["PATH"] = genai_dir + os.pathsep + qnn_dir + os.pathsep + os.environ.get("PATH", "")

# Register QNN provider (must use 'QNNExecutionProvider' as registration name)
og.register_execution_provider_library('QNNExecutionProvider', onnxruntime_qnn.get_library_path())

def run_benchmark(model_dir, prompt=None, max_length=512):
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
        input_tokens = tokenizer.encode(prompt)
        tokenizer_end = time.perf_counter()
        print(f"Tokenizer time: {tokenizer_end-tokenizer_start:.2f}s")
        
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=max_length, temperature=0.7)
        
        generator = og.Generator(model, params)
        
        print("\n--- Running inference ---")
        
        # Record prefill time
        prefill_start = time.perf_counter()
        generator.append_tokens(input_tokens)
        
        # Generate first token
        generator.generate_next_token()
        first_token_time = time.perf_counter()
        
        ttft = first_token_time - prefill_start
        print(f"TTFT: {ttft:.2f}s")
        
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

        # Continue streaming
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if new_tokens:
                print(tokenizer_stream.decode(new_tokens[0]), end='', flush=True)
                tokens_count += 1

        gen_end = time.perf_counter()
        print()

        total_tokens = tokens_count
        total_time = gen_end - gen_start
        tps = total_tokens / total_time if total_time > 0 else 0
        
        print(f"\nGeneration stats:")
        print(f"   Tokens: {total_tokens}")
        print(f"   Time: {total_time:.2f}s")
        print(f"   Speed: {tps:.2f} tokens/s")

    except Exception as e:
        print(f"\nRuntime error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QNN NPU inference benchmark")
    parser.add_argument("model_dir", help="Path to QNN model directory")
    parser.add_argument("--prompt", default=None, help="Input prompt")
    parser.add_argument("--max-length", type=int, default=512, help="Max generation length")
    args = parser.parse_args()
    run_benchmark(args.model_dir, args.prompt, args.max_length)
