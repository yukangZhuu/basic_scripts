import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_model(model_path: str, num_iterations: int = 3):
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"{'='*60}\n")
    
    print(f"Loading model from {model_path}...")
    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s\n")
    
    test_prompts = [
        "What is 2 + 2?",
        "Solve: 3 * 4 + 5 = ?",
        "If x = 5, what is x^2 + 2x?",
    ]
    
    print(f"Running {len(test_prompts)} test prompts x {num_iterations} iterations...")
    print(f"{'='*60}\n")
    
    all_times = []
    all_tokens = []
    
    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"Prompt {prompt_idx + 1}: {prompt}")
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        prompt_times = []
        prompt_tokens = []
        
        for i in range(num_iterations):
            model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    use_cache=True
                )
            
            end_time = time.time()
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            num_generated = len(generated_ids[0])
            elapsed = end_time - start_time
            tokens_per_second = num_generated / elapsed
            
            prompt_times.append(elapsed)
            prompt_tokens.append(num_generated)
            
            print(f"  Iteration {i+1}: {num_generated} tokens in {elapsed:.2f}s ({tokens_per_second:.1f} tokens/s)")
        
        avg_time = sum(prompt_times) / len(prompt_times)
        avg_tokens = sum(prompt_tokens) / len(prompt_tokens)
        avg_tps = avg_tokens / avg_time
        
        print(f"  Average: {avg_tokens:.0f} tokens in {avg_time:.2f}s ({avg_tps:.1f} tokens/s)\n")
        
        all_times.extend(prompt_times)
        all_tokens.extend(prompt_tokens)
    
    overall_avg_time = sum(all_times) / len(all_times)
    overall_avg_tokens = sum(all_tokens) / len(all_tokens)
    overall_tps = overall_avg_tokens / overall_avg_time
    
    print(f"{'='*60}")
    print(f"Overall Performance:")
    print(f"  Total iterations: {len(all_times)}")
    print(f"  Average time per generation: {overall_avg_time:.2f}s")
    print(f"  Average tokens per generation: {overall_avg_tokens:.0f}")
    print(f"  Average speed: {overall_tps:.1f} tokens/second")
    print(f"{'='*60}")


if __name__ == "__main__":
    benchmark_model("./models/qwen3-0.6b", num_iterations=3)