from .base_config import BaseConfig, DataSourceConfig, ModelConfig, EvaluationConfig


class GSM8KQwenLocalConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.data_source = DataSourceConfig(
            source_type="huggingface",
            dataset_name="openai/gsm8k",
            split="test"
        )
        self.model = ModelConfig(
            model_type="local",
            local_model_path="./models/qwen3-0.6b",
            enable_thinking=False
        )
        self.evaluation = EvaluationConfig(
            num_samples=10,
            output_dir="./results/gsm8k/qwen3_0.6b_local"
        )
    
    def get_prompt_template(self) -> str:
        return "Let's think step by step and output the final answer after \"####\"."
    
    def get_system_prompt(self) -> str:
        return "You are a helpful assistant that solves math problems step by step."
