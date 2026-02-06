from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from datasets import load_dataset
import json
import os


class BaseDataLoader(ABC):
    @abstractmethod
    def load(self) -> List[Dict]:
        pass


class HuggingFaceDataLoader(BaseDataLoader):
    def __init__(self, dataset_name: str, split: str = "test"):
        self.dataset_name = dataset_name
        self.split = split
    
    def load(self) -> List[Dict]:
        try:
            dataset = load_dataset(self.dataset_name, "main")
            data = list(dataset[self.split])
            print(f"Successfully loaded {len(data)} samples from {self.dataset_name} ({self.split} split)")
            return data
        except Exception as e:
            print(f"Failed to load dataset from HuggingFace: {e}")
            raise


class LocalDataLoader(BaseDataLoader):
    def __init__(self, file_path: str, file_type: str = "jsonl"):
        self.file_path = file_path
        self.file_type = file_type
    
    def load(self) -> List[Dict]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        data = []
        if self.file_type == "jsonl":
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif self.file_type == "json":
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        
        print(f"Successfully loaded {len(data)} samples from {self.file_path}")
        return data


class DataLoader:
    def __init__(self, source_type: str = "huggingface", **kwargs):
        self.source_type = source_type
        self.loader = self._create_loader(**kwargs)
    
    def _create_loader(self, **kwargs) -> BaseDataLoader:
        if self.source_type == "huggingface":
            return HuggingFaceDataLoader(
                dataset_name=kwargs.get("dataset_name"),
                split=kwargs.get("split", "test")
            )
        elif self.source_type == "local":
            return LocalDataLoader(
                file_path=kwargs.get("local_path"),
                file_type=kwargs.get("file_type", "jsonl")
            )
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
    
    def load(self) -> List[Dict]:
        return self.loader.load()
    
    def sample(self, data: List[Dict], num_samples: Optional[int] = None) -> List[Dict]:
        if num_samples is None:
            return data
        
        num_samples = min(num_samples, len(data))
        sampled_data = data[:num_samples]
        print(f"Sampled {num_samples} questions")
        return sampled_data
