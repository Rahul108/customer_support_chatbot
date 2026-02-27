import torch
from peft import get_peft_model, LoraConfig, TaskType
from config.settings import Config
from .base_model import BaseModel


class PEFTModel(BaseModel):
    """Unified PEFT (Parameter-Efficient Fine-Tuning) model interface."""

    def __init__(self, config: Config, technique: str = "lora"):
        self.technique = technique
        super().__init__(config)
        self.prepare_model()

    def prepare_model(self):
        """Prepare model using specified PEFT technique."""
        if self.technique == "lora":
            self._apply_lora()
        else:
            raise ValueError(f"Unsupported technique: {self.technique}")

    def _apply_lora(self):
        """Apply LoRA configuration."""
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

    def get_trainable_params(self) -> dict:
        """Get trainable and total parameters."""
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "trainable": trainable_params,
            "total": total_params,
            "percentage": 100 * trainable_params / total_params,
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass through model."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
