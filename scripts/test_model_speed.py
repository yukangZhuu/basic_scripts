import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_generation_speed(model_path: str, num_iterations: int = 5):
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    print("Model loaded successfully\n")
    
    test_prompt = "What is 2 + 2? Let's think step by step."
    messages = [{"role": "user", "content": test_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print(f"Test prompt: {test_prompt}")
    print(f"Tokenized length: {len(tokenizer(text)['input_ids'])} tokens\n")
    
    times = []
    total_tokens = 0
    
    for i in range(num_iterations):
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=100,
                do_sample=False,
                use_cache=True
            )
        
        end_time = time.time()
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        num_generated = len(generated_ids[0])
        total_tokens += num_generated
        elapsed = end_time - start_time
        times.append(elapsed)
        
        tokens_per_second = num_generated / elapsed
        print(f"Iteration {i+1}: {num_generated} tokens in {elapsed:.2f}s ({tokens_per_second:.1f} tokens/s)")
    
    avg_time = sum(times) / len(times)
    avg_tokens_per_second = total_tokens / sum(times)
    
    print(f"\n{'='*60}")
    print(f"Average over {num_iterations} iterations:")
    print(f"  Time: {avg_time:.2f}s")
    print(f"  Speed: {avg_tokens_per_second:.1f} tokens/second")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_generation_speed("./models/qwen3-0.6b", num_iterations=5)