#!/usr/bin/env python3
"""
Download Qwen3-0.6B model from HuggingFace and save it locally.
This script downloads the model and tokenizer to the models directory.
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def download_qwen_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "./models/qwen3-0.6b"
):
    """
    Download Qwen model from HuggingFace and save it locally.
    
    Args:
        model_name: HuggingFace model name
        output_dir: Local directory to save the model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model: {model_name}")
    print(f"Saving to: {output_path.absolute()}")
    print()
    
    try:
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        tokenizer.save_pretrained(output_path)
        print(f"✓ Tokenizer saved to {output_path}")
        print()
        
        print("Downloading model...")
        print("This may take a while depending on your internet connection...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        model.save_pretrained(output_path)
        print(f"✓ Model saved to {output_path}")
        print()
        
        print("=" * 60)
        print("Download completed successfully!")
        print(f"Model location: {output_path.absolute()}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Qwen model from HuggingFace")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/qwen3-0.6b",
        help="Local directory to save the model"
    )
    
    args = parser.parse_args()
    
    download_qwen_model(args.model_name, args.output_dir)
