import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from configs.gsm8k_qwen_local_config import GSM8KQwenLocalConfig
from core.data_loader import DataLoader
from core.model_client import ModelClient
from core.answer_extractor import AnswerExtractor
from core.evaluator import Evaluator


def evaluate_gsm8k_local(num_samples: int = None, **kwargs):
    config = GSM8KQwenLocalConfig()
    
    if num_samples is not None:
        config.evaluation.num_samples = num_samples
    
    if kwargs.get("local_model_path"):
        config.model.local_model_path = kwargs["local_model_path"]
    if "use_vllm" in kwargs:
        config.model.use_vllm = kwargs["use_vllm"]
    if "use_vllm_batch" in kwargs:
        config.model.use_vllm_batch = kwargs["use_vllm_batch"]
    if "vllm_batch_size" in kwargs:
        config.model.vllm_batch_size = kwargs["vllm_batch_size"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.evaluation.output_dir = f"./results/gsm8k/qwen3_0.6b_local/{timestamp}"
    
    print("=" * 60)
    print("GSM8K Evaluation Configuration")
    print("=" * 60)
    print(f"Model type: {config.model.model_type}")
    print(f"Model path: {config.model.local_model_path}")
    print(f"Use vLLM: {config.model.use_vllm}")
    if config.model.use_vllm:
        print(f"Use vLLM batch: {config.model.use_vllm_batch}")
        if config.model.use_vllm_batch:
            print(f"vLLM batch size: {config.model.vllm_batch_size}")
    print(f"Enable thinking: {config.model.enable_thinking}")
    print(f"Temperature: {config.model.temperature}")
    print(f"Max tokens: {config.model.max_tokens}")
    print(f"Dataset: {config.data_source.dataset_name}")
    print(f"Split: {config.data_source.split}")
    print(f"Number of samples: {config.evaluation.num_samples}")
    print(f"Output directory: {config.evaluation.output_dir}")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)
    
    data_loader = DataLoader(
        source_type=config.data_source.source_type,
        dataset_name=config.data_source.dataset_name,
        split=config.data_source.split,
        local_path=config.data_source.local_path
    )
    
    print(f"\nLoading model...")
    model_client = ModelClient(
        model_type=config.model.model_type,
        local_model_path=config.model.local_model_path,
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
        enable_thinking=config.model.enable_thinking,
        use_vllm=config.model.use_vllm,
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
    
    print(f"\nStarting evaluation...")
    print("=" * 60)
    start_time = time.time()
    
    results = evaluator.evaluate(
        prompt_template=config.get_prompt_template(),
        system_prompt=config.get_system_prompt(),
        num_samples=config.evaluation.num_samples,
        use_batch=config.model.use_vllm and config.model.use_vllm_batch,
        batch_size=config.model.vllm_batch_size if config.model.use_vllm_batch else None,
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=" * 60)
    print(f"\nEvaluation completed!")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Average time per question: {total_time/len(results['results']):.2f}s")
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct_count']}/{results['total_count']})")
    print("=" * 60)
    
    generate_evaluation_report(config, results, total_time)
    
    return results


def generate_evaluation_report(config, results, total_time):
    report = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "model_config": {
                "model_type": config.model.model_type,
                "local_model_path": config.model.local_model_path,
                "use_vllm": config.model.use_vllm,
                "use_vllm_batch": config.model.use_vllm_batch,
                "vllm_batch_size": config.model.vllm_batch_size,
                "enable_thinking": config.model.enable_thinking,
                "temperature": config.model.temperature,
                "max_tokens": config.model.max_tokens
            },
            "dataset_config": {
                "dataset_name": config.data_source.dataset_name,
                "split": config.data_source.split
            },
            "evaluation_config": {
                "num_samples": config.evaluation.num_samples,
                "output_dir": config.evaluation.output_dir
            }
        },
        "results": {
            "accuracy": results["accuracy"],
            "correct_count": results["correct_count"],
            "total_count": results["total_count"],
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "average_time_per_question": total_time / len(results["results"])
        },
        "sample_results": results["results"][:5]
    }
    
    report_path = os.path.join(config.evaluation.output_dir, "evaluation_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation report saved to: {report_path}")
    
    print_report_summary(report)


def print_report_summary(report):
    print("\n" + "=" * 60)
    print("EVALUATION REPORT SUMMARY")
    print("=" * 60)
    
    print(f"\nTimestamp: {report['evaluation_info']['timestamp']}")
    
    print("\nModel Configuration:")
    model_config = report["evaluation_info"]["model_config"]
    print(f"  - Model type: {model_config['model_type']}")
    print(f"  - Model path: {model_config['local_model_path']}")
    print(f"  - Use vLLM: {model_config.get('use_vllm', False)}")
    print(f"  - vLLM batch size: {model_config.get('vllm_batch_size', 'N/A')}")
    print(f"  - Enable thinking: {model_config['enable_thinking']}")
    print(f"  - Temperature: {model_config['temperature']}")
    print(f"  - Max tokens: {model_config['max_tokens']}")
    
    print("\nDataset Configuration:")
    dataset_config = report["evaluation_info"]["dataset_config"]
    print(f"  - Dataset: {dataset_config['dataset_name']}")
    print(f"  - Split: {dataset_config['split']}")
    
    print("\nEvaluation Configuration:")
    eval_config = report["evaluation_info"]["evaluation_config"]
    print(f"  - Number of samples: {eval_config['num_samples']}")
    print(f"  - Output directory: {eval_config['output_dir']}")
    
    print("\nResults:")
    results = report["results"]
    print(f"  - Accuracy: {results['accuracy']:.2%}")
    print(f"  - Correct: {results['correct_count']}")
    print(f"  - Total: {results['total_count']}")
    print(f"  - Total time: {results['total_time_seconds']:.2f}s ({results['total_time_minutes']:.2f} minutes)")
    print(f"  - Average time per question: {results['average_time_per_question']:.2f}s")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate local Qwen3-0.6B model on GSM8K dataset")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--local-model-path", type=str, default=None, help="Path to local model")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM for faster local inference")
    parser.add_argument("--no-vllm-batch", action="store_true", help="Disable vLLM batch generation (use when --use-vllm)")
    parser.add_argument("--vllm-batch-size", type=int, default=None, help="Max samples per vLLM batch (default: 64; tune down if OOM)")
    
    args = parser.parse_args()
    
    kwargs = dict(num_samples=args.num_samples, local_model_path=args.local_model_path)
    if args.use_vllm:
        kwargs["use_vllm"] = True
    if args.no_vllm_batch:
        kwargs["use_vllm_batch"] = False
    if args.vllm_batch_size is not None:
        kwargs["vllm_batch_size"] = args.vllm_batch_size
    
    evaluate_gsm8k_local(**kwargs)
