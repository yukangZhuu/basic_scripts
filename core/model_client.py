from typing import Tuple, Optional
from abc import ABC, abstractmethod
from openai import OpenAI
import os
import torch


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
                do_sample=False,
                use_cache=True
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return "", response


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
