from typing import List, Dict, Optional
from .data_loader import DataLoader
from .model_client import ModelClient
from .answer_extractor import AnswerExtractor
import json
import os


class Evaluator:
    def __init__(
        self,
        data_loader: DataLoader,
        model_client: ModelClient,
        answer_extractor: AnswerExtractor,
        output_dir: str = "./results",
        save_results: bool = True,
        verbose: bool = True
    ):
        self.data_loader = data_loader
        self.model_client = model_client
        self.answer_extractor = answer_extractor
        self.output_dir = output_dir
        self.save_results = save_results
        self.verbose = verbose
        
        if self.save_results and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def evaluate(
        self,
        prompt_template: str,
        system_prompt: Optional[str] = None,
        num_samples: Optional[int] = None,
        use_batch: bool = False,
        batch_size: Optional[int] = None,
    ) -> Dict:
        data = self.data_loader.load()
        questions = self.data_loader.sample(data, num_samples)
        
        total_samples = len(questions)
        show_details = self.verbose and (num_samples is not None and num_samples <= 10)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting evaluation on {total_samples} questions")
            if show_details:
                print("Showing detailed output for each question")
            else:
                print("Detailed output disabled (sample count > 10)")
            print(f"{'='*60}")
        
        # Optional: batch generation (e.g. when using vLLM), chunked by batch_size
        batch_outputs = None
        if use_batch and total_samples > 0:
            prompts = [f"{item['question']} {prompt_template}" for item in questions]
            chunk_size = batch_size or len(prompts)
            batch_outputs = []
            for start in range(0, len(prompts), chunk_size):
                chunk = prompts[start : start + chunk_size]
                batch_outputs.extend(
                    self.model_client.generate_batch(chunk, system_prompt)
                )
            if self.verbose and chunk_size < len(prompts):
                num_chunks = (len(prompts) + chunk_size - 1) // chunk_size
                print(f"\nBatched inference: {len(prompts)} samples in {num_chunks} chunk(s) (size={chunk_size})")
        
        results = []
        for i, item in enumerate(questions, 1):
            if self.verbose:
                if show_details:
                    print(f"\n{'='*60}")
                    print(f"Processing question {i}/{total_samples}")
                    print(f"{'='*60}")
                    print(f"Question: {item['question'][:100]}...")
                else:
                    print(f"\rProgress: [{i}/{total_samples}] {i/total_samples:.1%}", end="", flush=True)
            
            if batch_outputs is not None and i <= len(batch_outputs):
                reasoning, answer = batch_outputs[i - 1]
            else:
                prompt = f"{item['question']} {prompt_template}"
                reasoning, answer = self.model_client.generate(prompt, system_prompt)
            
            if show_details:
                print(f"\nReasoning process:")
                print(reasoning if reasoning else "(No reasoning process)")
                print(f"\nModel answer:")
                print(answer)
            
            is_correct, model_num, gt_num = self.answer_extractor.compare(
                answer, item['answer']
            )
            
            if show_details:
                print(f"\nGround truth: {item['answer'][:100]}...")
                print(f"Extracted model answer: {model_num}")
                print(f"Extracted ground truth: {gt_num}")
                print(f"Result: {'✓ Correct' if is_correct else '✗ Incorrect'}")
            
            results.append({
                "question": item['question'],
                "ground_truth": item['answer'],
                "model_reasoning": reasoning,
                "model_answer": answer,
                "extracted_model_answer": model_num,
                "extracted_ground_truth": gt_num,
                "is_correct": is_correct
            })
        
        if self.verbose and not show_details:
            print()
        
        correct_count = sum(1 for r in results if r['is_correct'])
        accuracy = correct_count / len(results) if results else 0
        
        summary = {
            "results": results,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(results)
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluation completed!")
            print(f"Accuracy: {correct_count}/{len(results)} = {accuracy:.2%}")
            print(f"{'='*60}")
        
        if self.save_results:
            self._save_results(summary)
        
        return summary
    
    def _save_results(self, results: Dict):
        timestamp = results.get("timestamp", "")
        filename = f"evaluation_results_{timestamp}.json" if timestamp else "evaluation_results.json"
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"\nResults saved to {output_path}")
