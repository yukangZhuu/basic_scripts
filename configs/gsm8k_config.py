from .base_config import BaseConfig, DataSourceConfig, ModelConfig, EvaluationConfig
from typing import Dict, Any


class GSM8KConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.data_source = DataSourceConfig(
            source_type="huggingface",
            dataset_name="openai/gsm8k",
            split="test"
        )
        self.model = ModelConfig(
            model_type="api",
            model_name="qwen3-32b",
            enable_thinking=False
        )
        self.evaluation = EvaluationConfig(
            num_samples=10,
            output_dir="./results/gsm8k"
        )
    
    def get_prompt_template(self) -> str:
        return "Let's think step by step and output the final answer after \"####\"."
    
    def get_system_prompt(self) -> str:
        return "You are a helpful assistant that solves math problems step by step."
