# GSM8K Evaluation Scripts

This directory contains bash scripts for running GSM8K evaluations with different configurations.

## Available Scripts

### 1. Quick Test Script

**File**: `gsm8k_quick_test.sh`

Runs a quick test with 3 samples to verify your setup.

```bash
bash scripts/gsm8k_quick_test.sh
```

### 2. Qwen3 32B API Evaluation

**File**: `gsm8k_qwen3_32b_api.sh`

Evaluates the Qwen3 32B model on GSM8K using remote API.

First, set up your API key in `.env` file (see [API_KEYS.md](../API_KEYS.md)):

```bash
cp .env.example .env
# Edit .env and add your DASHSCOPE_API_KEY
```

Then run the evaluation:

```bash
# Run with default 10 samples
bash scripts/gsm8k_qwen3_32b_api.sh

# Or specify number of samples
NUM_SAMPLES=50 bash scripts/gsm8k_qwen3_32b_api.sh
```

### 3. Local Model Evaluation

**File**: `gsm8k_local_model.sh`

Evaluates a local model on GSM8K dataset.

```bash
# Set your local model path
export LOCAL_MODEL_PATH=/path/to/your/model

# Run evaluation
bash scripts/gsm8k_local_model.sh

# Or specify number of samples
NUM_SAMPLES=100 bash scripts/gsm8k_local_model.sh
```

### 4. Full Dataset Evaluation

**File**: `gsm8k_full_dataset.sh`

Evaluates on the entire GSM8K test set (~1319 samples).

First, set up your API key in `.env` file (see [API_KEYS.md](../API_KEYS.md)):

```bash
cp .env.example .env
# Edit .env and add your DASHSCOPE_API_KEY
```

Then run the evaluation:

```bash
# Run full evaluation
bash scripts/gsm8k_full_dataset.sh
```

## Environment Variables

### Required Variables

- `DASHSCOPE_API_KEY`: Your API key for remote model access (required for API-based evaluation)
  - Set this in your `.env` file (recommended)
  - Or set as environment variable: `export DASHSCOPE_API_KEY=your-api-key`

### Optional Variables

- `NUM_SAMPLES`: Number of samples to evaluate (default: 10)
- `MODEL_NAME`: Model name for API calls (default: qwen3-32b)
- `LOCAL_MODEL_PATH`: Path to local model (required for local model evaluation)
- `OUTPUT_DIR`: Directory to save results (default varies by script)

## Examples

### Quick Test (Recommended First Step)

```bash
export DASHSCOPE_API_KEY=your-api-key
bash scripts/gsm8k_quick_test.sh
```

### Evaluate with 50 Samples

```bash
export DASHSCOPE_API_KEY=your-api-key
NUM_SAMPLES=50 bash scripts/gsm8k_qwen3_32b_api.sh
```

### Evaluate with Different Model

```bash
export DASHSCOPE_API_KEY=your-api-key
MODEL_NAME=gpt-4 bash scripts/gsm8k_qwen3_32b_api.sh
```

### Evaluate Local Model

```bash
export LOCAL_MODEL_PATH=/path/to/your/model
NUM_SAMPLES=20 bash scripts/gsm8k_local_model.sh
```

### Full Dataset Evaluation

```bash
export DASHSCOPE_API_KEY=your-api-key
bash scripts/gsm8k_full_dataset.sh
```

## Output

Results are saved in the `results/gsm8k/` directory with the following structure:

```
results/gsm8k/
├── qwen3_32b_api/
│   └── evaluation_results.json
├── local_model/
│   └── evaluation_results.json
├── full_dataset/
│   └── evaluation_results.json
└── quick_test/
    └── evaluation_results.json
```

## Troubleshooting

### API Key Not Set

```
Error: DASHSCOPE_API_KEY environment variable is not set
```

**Solution**: Set the API key using `export DASHSCOPE_API_KEY=your-api-key`

### Local Model Path Not Set

```
Error: LOCAL_MODEL_PATH environment variable is not set
```

**Solution**: Set the model path using `export LOCAL_MODEL_PATH=/path/to/your/model`

### Model Path Does Not Exist

```
Error: Model path does not exist: /path/to/your/model
```

**Solution**: Verify the path is correct and the model files exist

## Creating Custom Scripts

To create a custom evaluation script, copy one of the existing scripts and modify the configuration variables:

```bash
#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration
NUM_SAMPLES="${NUM_SAMPLES:-10}"
MODEL_NAME="${MODEL_NAME:-your-model-name}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/gsm8k/custom}"

# Check if API key is set
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "Error: DASHSCOPE_API_KEY environment variable is not set"
    echo "Please create a .env file based on .env.example and set your API key"
    exit 1
fi

# Run evaluation
python scripts/evaluate_gsm8k.py \
    --num-samples "$NUM_SAMPLES" \
    --model-type api \
    --model-name "$MODEL_NAME"
```

Make sure to:

1. Load environment variables from `.env` file
2. Check if required API keys are set
3. Don't hardcode API keys in the script
