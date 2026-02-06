# Basic Evaluator

A flexible and extensible framework for evaluating large language models on various benchmarks.

## Features

- **Multiple Data Sources**: Support for HuggingFace datasets and local files (JSON/JSONL)
- **Flexible Model Support**: API-based models and local model inference
- **Configurable Evaluation**: Sample-based or full dataset evaluation
- **Extensible Architecture**: Easy to add new benchmarks and evaluation metrics
- **Result Management**: Automatic result saving and reporting

## Installation

```bash
pip install -r requirements.txt
```

## API Key Setup

Before running evaluations, you need to set up your API keys:

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:

```bash
DASHSCOPE_API_KEY=your-dashscope-api-key-here
```

For detailed API key management instructions, see [API_KEYS.md](API_KEYS.md).

## UsageProject Structure

```
basic-scripts/
├── configs/              # Configuration files for different benchmarks
│   ├── base_config.py   # Base configuration classes
│   └── gsm8k_config.py  # GSM8K specific configuration
├── core/                # Core evaluation components
│   ├── data_loader.py   # Data loading utilities
│   ├── model_client.py  # Model client (API and local)
│   ├── answer_extractor.py # Answer extraction and comparison
│   └── evaluator.py     # Main evaluation logic
├── scripts/             # Evaluation scripts
│   └── evaluate_gsm8k.py # GSM8K evaluation script
├── results/             # Evaluation results (auto-created)
└── requirements.txt     # Python dependencies
```

## Usage

### Quick Start with GSM8K

Evaluate using the default API model:

```bash
python scripts/evaluate_gsm8k.py --num-samples 10
```

### Using a Local Model

Download and evaluate with a local model:

```bash
# Download Qwen3-0.6B model
python scripts/download_qwen_model.py

# Evaluate with local model
bash scripts/gsm8k_qwen3_0.6b_local.sh
```

For detailed local model instructions, see [LOCAL_MODEL.md](LOCAL_MODEL.md).

### Programmatic Usage

```python
from scripts.evaluate_gsm8k import evaluate_gsm8k

results = evaluate_gsm8k(num_samples=10)
print(f"Accuracy: {results['accuracy']:.2%}")
```

## Configuration

### Data Source Configuration

- **HuggingFace**: Automatically downloads from HuggingFace Hub
- **Local**: Load from local JSON/JSONL files

### Model Configuration

- **API Models**: Support for OpenAI-compatible APIs (e.g., Qwen, GPT)
- **Local Models**: Support for HuggingFace Transformers models

### Evaluation Configuration

- **Sampling**: Evaluate on a subset or full dataset
- **Output**: Automatic result saving with detailed metrics

## Adding New Benchmarks

1. Create a new config file in `configs/` (e.g., `my_benchmark_config.py`)
2. Implement the configuration class inheriting from `BaseConfig`
3. Create an evaluation script in `scripts/`
4. Define prompt templates and answer extraction logic

Example:

```python
from configs.base_config import BaseConfig

class MyBenchmarkConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.data_source.dataset_name = "my-benchmark"
        self.evaluation.num_samples = 100

    def get_prompt_template(self):
        return "Your custom prompt template"
```

## Requirements

- Python 3.8+
- openai>=1.0.0
- datasets>=2.0.0
- transformers>=4.30.0 (for local models)
- torch>=2.0.0 (for local models)
- python-dotenv>=1.0.0

## Documentation

- [API Keys Management](API_KEYS.md) - How to configure and manage API keys
- [Local Model Evaluation](LOCAL_MODEL.md) - Download and use local models for evaluation
- [Evaluation Scripts](scripts/README.md) - Available evaluation scripts and usage

## License

MIT License
