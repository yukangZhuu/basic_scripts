#!/bin/bash

# GSM8K Evaluation Script for Local Model
# This script evaluates a local model on the GSM8K dataset

# Configuration
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/gsm8k/local_model}"

# Check if local model path is set
if [ -z "$LOCAL_MODEL_PATH" ]; then
    echo "Error: LOCAL_MODEL_PATH environment variable is not set"
    echo "Please set it using: export LOCAL_MODEL_PATH=/path/to/your/model"
    exit 1
fi

# Check if model path exists
if [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $LOCAL_MODEL_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=========================================="
echo "GSM8K Evaluation Configuration"
echo "=========================================="
echo "Model: Local model at $LOCAL_MODEL_PATH"
echo "Dataset: GSM8K"
echo "Number of samples: $NUM_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run evaluation
python scripts/evaluate_gsm8k.py \
    --num-samples "$NUM_SAMPLES" \
    --model-type local \
    --local-model-path "$LOCAL_MODEL_PATH"

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
