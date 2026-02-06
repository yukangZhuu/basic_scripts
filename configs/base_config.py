from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class DataSourceConfig:
    source_type: str = "huggingface"
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    split: str = "test"
    local_path: Optional[str] = None


@dataclass
class ModelConfig:
    model_type: str = "api"
    model_name: str = "qwen3-32b"
    api_key: Optional[str] = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    local_model_path: Optional[str] = None
    enable_thinking: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class EvaluationConfig:
    num_samples: Optional[int] = None
    output_dir: str = "./results"
    save_results: bool = True
    verbose: bool = True


@dataclass
class BaseConfig:
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_source": self.data_source.__dict__,
            "model": self.model.__dict__,
            "evaluation": self.evaluation.__dict__
        }
