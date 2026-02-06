#!/usr/bin/env python3
"""
Simple test script to verify local model is working correctly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_client import ModelClient


def test_local_model():
    print("Testing local model...")
    print("=" * 60)
    
    try:
        model_client = ModelClient(
            model_type="local",
            local_model_path="./models/qwen3-0.6b",
            temperature=0.7,
            max_tokens=100
        )
        
        print("✓ Model client created successfully")
        print()
        
        test_prompt = "What is 2 + 2? Let's think step by step and output the final answer after \"####\"."
        system_prompt = "You are a helpful assistant that solves math problems step by step."
        
        print("Test prompt:")
        print(test_prompt)
        print()
        print("Generating response...")
        print("-" * 60)
        
        reasoning, answer = model_client.generate(test_prompt, system_prompt)
        
        print("-" * 60)
        print()
        print("✓ Generation completed!")
        print()
        print("Reasoning:")
        print(reasoning if reasoning else "(No reasoning)")
        print()
        print("Answer:")
        print(answer)
        print()
        print("=" * 60)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_local_model()
    sys.exit(0 if success else 1)
