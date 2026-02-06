import re
from typing import Optional, List
from abc import ABC, abstractmethod


class BaseAnswerExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> Optional[float]:
        pass


class GSM8KAnswerExtractor(BaseAnswerExtractor):
    def extract(self, text: str) -> Optional[float]:
        match = re.search(r"####\s+(-?\d+\.?\d*)", text)
        if match:
            return float(match.group(1).replace(",", ""))
        
        match = re.search(r'The answer is\s+(-?\d+\.?\d*)', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[-1])
        
        return None


class GenericAnswerExtractor(BaseAnswerExtractor):
    def __init__(self, patterns: List[str] = None):
        self.patterns = patterns or [
            r'(?:answer|答案)[：:]\s*(-?\d+\.?\d*)',
            r'[-=]\s*(-?\d+\.?\d*)',
            r'(?:therefore|thus|因此|所以)[，,]?\s*(-?\d+\.?\d*)'
        ]
    
    def extract(self, text: str) -> Optional[float]:
        for pattern in self.patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[-1])
        
        return None


class AnswerExtractor:
    def __init__(self, extractor_type: str = "gsm8k", **kwargs):
        self.extractor_type = extractor_type
        self.extractor = self._create_extractor(**kwargs)
    
    def _create_extractor(self, **kwargs) -> BaseAnswerExtractor:
        if self.extractor_type == "gsm8k":
            return GSM8KAnswerExtractor()
        elif self.extractor_type == "generic":
            return GenericAnswerExtractor(patterns=kwargs.get("patterns"))
        else:
            raise ValueError(f"Unsupported extractor type: {self.extractor_type}")
    
    def extract(self, text: str) -> Optional[float]:
        return self.extractor.extract(text)
    
    def compare(self, model_answer: str, ground_truth: str, tolerance: float = 0.01) -> tuple[bool, Optional[float], Optional[float]]:
        model_num = self.extract(model_answer)
        gt_num = self.extract(ground_truth)
        
        if model_num is None:
            return False, None, gt_num
        
        if gt_num is None:
            return False, model_num, None
        
        is_correct = abs(model_num - gt_num) < tolerance
        
        return is_correct, model_num, gt_num
