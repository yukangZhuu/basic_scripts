# Local Model Evaluation

This guide explains how to download and use local models for evaluation.

## Overview

- **Models Directory**: `./models/` - Stores downloaded models (gitignored)
- **Download Script**: `scripts/download_qwen_model.py` - Downloads models from HuggingFace
- **Evaluation Script**: `scripts/evaluate_gsm8k_local.py` - Evaluates local models on GSM8K

## Step 1: Download Qwen3-0.6B Model

Download the model from HuggingFace and save it locally:

```bash
conda run -n eval python scripts/download_qwen_model.py
```

This will:

- Download `Qwen/Qwen3-0.6B` from HuggingFace
- Save it to `./models/qwen3-0.6b/`
- Include both model weights and tokenizer

### Custom Download Options

```bash
# Download specific model
python scripts/download_qwen_model.py --model-name Qwen/Qwen3-0.6B

# Save to custom location
python scripts/download_qwen_model.py --output-dir ./models/my-model
```

## Step 2: Evaluate with Local Model

Once the model is downloaded, evaluate it on GSM8K:

```bash
# Using default path (./models/qwen3-0.6b)
bash scripts/gsm8k_qwen3_0.6b_local.sh

# Using custom model path
LOCAL_MODEL_PATH=./models/qwen3-0.6b bash scripts/gsm8k_qwen3_0.6b_local.sh

# Evaluate with specific number of samples
NUM_SAMPLES=20 bash scripts/gsm8k_qwen3_0.6b_local.sh
```

### Programmatic Usage

```python
from scripts.evaluate_gsm8k_local import evaluate_gsm8k_local

# Evaluate with default settings
results = evaluate_gsm8k_local(num_samples=10)

# Evaluate with custom model path
results = evaluate_gsm8k_local(
    num_samples=50,
    local_model_path="./models/qwen3-0.6b"
)

print(f"Accuracy: {results['accuracy']:.2%}")
```

## Configuration

### Local Model Configuration

The local model evaluation uses `GSM8KQwenLocalConfig`:

```python
from configs.gsm8k_qwen_local_config import GSM8KQwenLocalConfig

config = GSM8KQwenLocalConfig()

# Data source: HuggingFace GSM8K test set
config.data_source.dataset_name = "openai/gsm8k"
config.data_source.split = "test"

# Model: Local Qwen3-0.6B
config.model.model_type = "local"
config.model.local_model_path = "./models/qwen3-0.6b"
config.model.enable_thinking = False

# Evaluation settings
config.evaluation.num_samples = 10
config.evaluation.output_dir = "./results/gsm8k/qwen3_0.6b_local"
```

### Device Configuration

The model automatically detects and uses the best available device:

- **CUDA**: If NVIDIA GPU is available (Linux/Windows)
- **MPS**: If Apple Silicon GPU is available (Mac)
- **CPU**: Fallback if no GPU is available

Device detection is handled in `core/model_client.py`:

```python
def _get_device(self):
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

### Performance Optimizations

The local model client includes several performance optimizations:

1. **Automatic Device Selection**: Uses GPU/MPS when available
2. **Half Precision**: Uses `torch.float16` for faster inference
3. **Greedy Decoding**: `do_sample=False` for deterministic, faster generation
4. **KV Cache**: `use_cache=True` to cache key-value states
5. **Attention Mask**: Properly handles padding for batch processing

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Half precision
    device_map=device            # Auto-detect best device
)

generated_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_tokens,
    do_sample=False,    # Greedy decoding (faster)
    use_cache=True      # Enable KV cache
)
```

### Expected Performance

On different hardware:

| Device              | Model      | Expected Speed   |
| ------------------- | ---------- | ---------------- |
| Apple Silicon (MPS) | Qwen3-0.6B | ~25-30 tokens/s  |
| NVIDIA GPU (CUDA)   | Qwen3-0.6B | ~50-100 tokens/s |
| CPU                 | Qwen3-0.6B | ~5-10 tokens/s   |

_Actual performance may vary based on hardware and model size_

## Model Client Improvements

The local model client now uses `apply_chat_template` for better chat formatting:

```python
# Old approach (simple text)
full_prompt = f"{system_prompt}\n\n{prompt}"

# New approach (chat template)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)
```

This provides:

- Better handling of chat-style prompts
- Proper tokenization for chat models
- Improved response quality

## Advantages of Local Models

1. **No API Costs**: Run evaluations without incurring API charges
2. **Faster Inference**: No network latency
3. **Privacy**: Data stays on your machine
4. **Customizable**: Modify model weights if needed
5. **Offline**: Run without internet connection

## Troubleshooting

### Model Not Found

```
Error: Model path does not exist: ./models/qwen3-0.6b
```

**Solution**: Download the model first:

```bash
python scripts/download_qwen_model.py
```

### Out of Memory

If you encounter CUDA out of memory errors:

1. Use a smaller model (e.g., Qwen3-0.5B)
2. Reduce `max_tokens` in configuration
3. Use CPU inference (slower but less memory)
4. Enable gradient checkpointing

### Slow Inference

For faster inference:

1. Use GPU if available
2. Reduce `max_tokens` limit
3. Use quantization (e.g., `torch_dtype=torch.float16`)
4. Batch multiple samples (not yet implemented)

## Model Storage

Models are stored in `./models/` directory:

```
models/
└── qwen3-0.6b/
    ├── config.json
    ├── generation_config.json
    ├── model-00001-of-00002.safetensors
    ├── model-00002-of-00002.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.json
```

This directory is gitignored to prevent large model files from being committed.

## Next Steps

1. Download the model: `python scripts/download_qwen_model.py`
2. Run a quick test: `bash scripts/gsm8k_qwen3_0.6b_local.sh`
3. Evaluate full dataset: `NUM_SAMPLES=1319 bash scripts/gsm8k_qwen3_0.6b_local.sh`
4. Compare with API model results

## References

- [Qwen3-0.6B on HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
