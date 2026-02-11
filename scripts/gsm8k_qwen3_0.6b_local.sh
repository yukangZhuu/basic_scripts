#!/bin/bash

# GSM8K Evaluation Script for Local Qwen3-0.6B Model
# This script evaluates the local Qwen3-0.6B model on GSM8K dataset

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration, full test: 1319
NUM_SAMPLES="${NUM_SAMPLES:-1319}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-./models/qwen3-0.6b}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/gsm8k/qwen3_0.6b_local}"
# Optional: use vLLM for faster inference (and batched generation). Set USE_VLLM=1 to enable.
USE_VLLM="${USE_VLLM:-1}"
USE_VLLM_BATCH="${USE_VLLM_BATCH:-1}"
# When USE_VLLM_BATCH=1: max samples per batch (default 64). Tune down if OOM, up for speed.
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-64}"

# Check if local model path is set
if [ -z "$LOCAL_MODEL_PATH" ]; then
    echo "Error: LOCAL_MODEL_PATH environment variable is not set"
    echo "Please set it using: export LOCAL_MODEL_PATH=/path/to/your/model"
    echo "Or use the default path: ./models/qwen3-0.6b"
    exit 1
fi

# Check if model path exists
if [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $LOCAL_MODEL_PATH"
    echo "Please download the model first using: python scripts/download_qwen_model.py"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=========================================="
echo "GSM8K Evaluation Configuration"
echo "=========================================="
echo "Model: Local Qwen3-0.6B at $LOCAL_MODEL_PATH"
echo "Dataset: GSM8K"
echo "Number of samples: $NUM_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "Use vLLM: $USE_VLLM"
[ "$USE_VLLM" = "1" ] && echo "Use vLLM batch: $USE_VLLM_BATCH"
[ "$USE_VLLM" = "1" ] && [ "$USE_VLLM_BATCH" = "1" ] && echo "vLLM batch size: $VLLM_BATCH_SIZE"
echo "=========================================="
echo ""

# Build optional vLLM args (optional: set USE_VLLM=1 for faster inference)
VLLM_ARGS=""
if [ "$USE_VLLM" = "1" ]; then
    VLLM_ARGS="--use-vllm"
    [ "$USE_VLLM_BATCH" = "0" ] && VLLM_ARGS="$VLLM_ARGS --no-vllm-batch"
    [ -n "$VLLM_BATCH_SIZE" ] && VLLM_ARGS="$VLLM_ARGS --vllm-batch-size $VLLM_BATCH_SIZE"
fi

# Run evaluation
python scripts/evaluate_gsm8k_local.py \
    --num-samples "$NUM_SAMPLES" \
    --local-model-path "$LOCAL_MODEL_PATH" \
    $VLLM_ARGS

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with exit code $?"
    echo "=========================================="
    exit 1
fi
