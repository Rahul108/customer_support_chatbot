from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.settings import Config


class BaseModel(ABC):
    """Base class for all model implementations."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load base model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model,
            device_map=self.config.model.model_device,
            load_in_8bit=self.config.model.load_in_8bit,
        )

    @abstractmethod
    def prepare_model(self):
        """Prepare model for training or inference."""
        pass

    @abstractmethod
    def get_trainable_params(self):
        """Get the number of trainable parameters."""
        pass

    def generate(self, prompt: str, max_length: int = None) -> str:
        """Generate response from prompt."""
        if max_length is None:
            max_length = self.config.model.max_length

        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.config.model.model_device
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def save_model(self, path: str):
        """Save model checkpoint."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str):
        """Load model from checkpoint."""
        self.model = AutoModelForCausalLM.from_pretrained(
            path, device_map=self.config.model.model_device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path)
