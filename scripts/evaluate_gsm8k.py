import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from configs.gsm8k_config import GSM8KConfig
from core.data_loader import DataLoader
from core.model_client import ModelClient
from core.answer_extractor import AnswerExtractor
from core.evaluator import Evaluator


def evaluate_gsm8k(num_samples: int = None, model_type: str = "api", **kwargs):
    config = GSM8KConfig()
    
    if num_samples is not None:
        config.evaluation.num_samples = num_samples
    
    if kwargs.get("model_name"):
        config.model.model_name = kwargs["model_name"]
    
    if kwargs.get("api_key"):
        config.model.api_key = kwargs["api_key"]
    elif not config.model.api_key:
        config.model.api_key = os.getenv("DASHSCOPE_API_KEY")
    
    if kwargs.get("local_model_path"):
        config.model.model_type = "local"
        config.model.local_model_path = kwargs["local_model_path"]
    
    data_loader = DataLoader(
        source_type=config.data_source.source_type,
        dataset_name=config.data_source.dataset_name,
        split=config.data_source.split,
        local_path=config.data_source.local_path
    )
    
    model_client = ModelClient(
        model_type=config.model.model_type,
        model_name=config.model.model_name,
        api_key=config.model.api_key,
        base_url=config.model.base_url,
        enable_thinking=config.model.enable_thinking,
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
        local_model_path=config.model.local_model_path
    )
    
    answer_extractor = AnswerExtractor(extractor_type="gsm8k")
    
    evaluator = Evaluator(
        data_loader=data_loader,
        model_client=model_client,
        answer_extractor=answer_extractor,
        output_dir=config.evaluation.output_dir,
        save_results=config.evaluation.save_results,
        verbose=config.evaluation.verbose
    )
    
    results = evaluator.evaluate(
        prompt_template=config.get_prompt_template(),
        system_prompt=config.get_system_prompt(),
        num_samples=config.evaluation.num_samples
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K dataset")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--model-type", type=str, default="api", choices=["api", "local"], help="Model type")
    parser.add_argument("--model-name", type=str, default="qwen3-32b", help="Model name for API")
    parser.add_argument("--local-model-path", type=str, default=None, help="Path to local model")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    
    args = parser.parse_args()
    
    evaluate_gsm8k(
        num_samples=args.num_samples,
        model_type=args.model_type,
        model_name=args.model_name,
        local_model_path=args.local_model_path,
        api_key=args.api_key
    )
