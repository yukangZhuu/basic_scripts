from datasets import load_dataset
import json


def check_gsm8k_dataset():
    print("=" * 60)
    print("GSM8K Dataset Information")
    print("=" * 60)
    
    dataset = load_dataset("openai/gsm8k", "main")
    
    print(f"\nDataset splits available: {list(dataset.keys())}")
    
    for split_name, split_data in dataset.items():
        print(f"\n{split_name.upper()} split:")
        print(f"  - Number of samples: {len(split_data)}")
        print(f"  - Features: {split_data.features}")
        
        if len(split_data) > 0:
            print(f"\n  Sample data structure:")
            sample = split_data[0]
            for key, value in sample.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    print(f"    {key}: {preview}")
                else:
                    print(f"    {key}: {value}")
    
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  - Train set: {len(dataset['train'])} samples")
    print(f"  - Test set: {len(dataset['test'])} samples")
    print(f"  - Total: {len(dataset['train']) + len(dataset['test'])} samples")
    print(f"{'=' * 60}")
    
    return dataset


if __name__ == "__main__":
    check_gsm8k_dataset()