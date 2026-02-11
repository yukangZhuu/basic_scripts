from typing import Tuple, Optional, List
from abc import ABC, abstractmethod
from openai import OpenAI
import os
import torch


def _patch_qwen3_extra_special_tokens():
    """Patch transformers so Qwen3 tokenizer config (extra_special_tokens as list) works."""
    try:
        from transformers import tokenization_utils_base
    except ImportError:
        return
    base = tokenization_utils_base.PreTrainedTokenizerBase
    if getattr(base, "_qwen3_extra_tokens_patched", False):
        return

    _orig = base._set_model_specific_special_tokens

    def _patched(self, special_tokens):
        if isinstance(special_tokens, list):
            # Qwen3 config can ship extra_special_tokens as a list; transformers expects a dict.
            special_tokens = {str(i): (t if isinstance(t, str) else str(t)) for i, t in enumerate(special_tokens)}
        return _orig(self, special_tokens=special_tokens)

    base._set_model_specific_special_tokens = _patched
    base._qwen3_extra_tokens_patched = True


class BaseModelClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        pass


class APIClient(BaseModelClient):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_thinking: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url,
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        reasoning_content = ""
        answer_content = ""
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body={"enable_thinking": self.enable_thinking},
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        for chunk in completion:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content
            if hasattr(delta, "content") and delta.content:
                answer_content += delta.content
        
        return reasoning_content, answer_content


class LocalModelClient(BaseModelClient):
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = False
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch are required for local model inference")
        
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"Loading local model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            device = self._get_device()
            print(f"Using device: {device}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=device
            )
            
            self.model.eval()
            print("Local model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")
    
    def _get_device(self):
        import torch
        
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                use_cache=True
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return "", response


class VLLMModelClient(BaseModelClient):
    """Local model client using vLLM for faster inference and optional batched generation."""

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        tensor_parallel_size: int = 1,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.tensor_parallel_size = tensor_parallel_size

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for VLLMModelClient. Install with: pip install vllm"
            )

        # Qwen3 tokenizer config uses extra_special_tokens as list; transformers expects dict.
        _patch_qwen3_extra_special_tokens()

        # Load LLM first so vLLM loads the tokenizer
        self._load_llm()
        self.tokenizer = self.llm.get_tokenizer()
        self._sampling_params = self._make_sampling_params()

    def _load_llm(self):
        from vllm import LLM

        print(f"Loading vLLM model from {self.model_path}...")
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        print("vLLM model loaded successfully.")

    def _make_sampling_params(self):
        from vllm import SamplingParams

        # Align with LocalModelClient: temperature=0.7, top_p=0.95 (from config).
        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.95 if self.temperature > 0 else 1.0,
        )

    def _messages_to_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        apply_chat = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat):
            return self._messages_to_prompt_fallback(prompt, system_prompt)

        try:
            return apply_chat(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            try:
                return apply_chat(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return self._messages_to_prompt_fallback(prompt, system_prompt)

    def _messages_to_prompt_fallback(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Simple Qwen-style chat prompt if tokenizer has no apply_chat_template."""
        parts = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        return "".join(parts)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        text = self._messages_to_prompt(prompt, system_prompt)
        outputs = self.llm.generate([text], self._sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return "", response

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Generate for multiple prompts in one batch. Much faster than calling generate in a loop."""
        if not prompts:
            return []
        texts = [
            self._messages_to_prompt(p, system_prompt) for p in prompts
        ]
        outputs = self.llm.generate(texts, self._sampling_params)
        results = []
        for out in outputs:
            text = out.outputs[0].text.strip() if out.outputs else ""
            results.append(("", text))
        return results


class ModelClient:
    def __init__(self, model_type: str = "api", **kwargs):
        self.model_type = model_type
        self.client = self._create_client(**kwargs)
    
    def _create_client(self, **kwargs) -> BaseModelClient:
        if self.model_type == "api":
            return APIClient(
                model_name=kwargs.get("model_name"),
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("base_url"),
                enable_thinking=kwargs.get("enable_thinking", False),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2048)
            )
        elif self.model_type == "local":
            if kwargs.get("use_vllm", False):
                return VLLMModelClient(
                    model_path=kwargs.get("local_model_path"),
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 2048),
                    enable_thinking=kwargs.get("enable_thinking", False),
                    tensor_parallel_size=kwargs.get("vllm_tensor_parallel_size", 1),
                )
            return LocalModelClient(
                model_path=kwargs.get("local_model_path"),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2048),
                enable_thinking=kwargs.get("enable_thinking", False)
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        return self.client.generate(prompt, system_prompt)

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
    ) -> Optional[List[Tuple[str, str]]]:
        """Batch generate if the underlying client supports it (e.g. VLLMModelClient)."""
        gen_batch = getattr(self.client, "generate_batch", None)
        if gen_batch is not None and callable(gen_batch):
            return gen_batch(prompts, system_prompt)
        return None
