import os
from .base_config import BaseConfig, DataSourceConfig, ModelConfig, EvaluationConfig


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


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
            local_model_path=os.environ.get("LOCAL_MODEL_PATH", "./models/qwen3-0.6b"),
            enable_thinking=True,
            use_vllm=_env_bool("USE_VLLM", False),
            use_vllm_batch=_env_bool("USE_VLLM_BATCH", True),
            vllm_batch_size=_env_int("VLLM_BATCH_SIZE", 64),
        )
        self.evaluation = EvaluationConfig(
            num_samples=10,
            output_dir="./results/gsm8k/qwen3_0.6b_local"
        )
    
    def get_prompt_template(self) -> str:
        return "Let's think step by step and output the final answer after \"####\"."
    
    def get_system_prompt(self) -> str:
        return "You are a helpful assistant that solves math problems step by step."
