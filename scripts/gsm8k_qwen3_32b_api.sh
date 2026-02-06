#!/bin/bash

# GSM8K Evaluation Script for Qwen3 32B API Model
# This script evaluates the Qwen3 32B model on GSM8K dataset using remote API

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration
NUM_SAMPLES="${NUM_SAMPLES:-3}"
MODEL_NAME="${MODEL_NAME:-qwen3-32b}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/gsm8k/qwen3_32b_api}"

# Check if API key is set
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "Error: DASHSCOPE_API_KEY environment variable is not set"
    echo "Please create a .env file based on .env.example and set your API key"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=========================================="
echo "GSM8K Evaluation Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME (API)"
echo "Dataset: GSM8K"
echo "Number of samples: $NUM_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run evaluation
python scripts/evaluate_gsm8k.py \
    --num-samples "$NUM_SAMPLES" \
    --model-type api \
    --model-name "$MODEL_NAME"

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
