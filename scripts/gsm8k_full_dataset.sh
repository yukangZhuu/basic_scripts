#!/bin/bash

# GSM8K Evaluation Script for Full Dataset
# This script evaluates the model on the entire GSM8K test set

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration
MODEL_NAME="${MODEL_NAME:-qwen3-32b}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/gsm8k/full_dataset}"

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
echo "GSM8K Full Dataset Evaluation"
echo "=========================================="
echo "Model: $MODEL_NAME (API)"
echo "Dataset: GSM8K (full test set)"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Warning: This will evaluate on the entire GSM8K test set (~1319 samples)"
echo "This may take a long time and incur significant API costs."
echo ""
read -p "Do you want to continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled."
    exit 0
fi

# Run evaluation (no num-samples argument means full dataset)
python scripts/evaluate_gsm8k.py \
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
