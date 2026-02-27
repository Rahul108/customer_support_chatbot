from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    r: int = 8
    lora_alpha: int = 16
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class AdapterConfig:
    """Adapter configuration."""
    adapter_size: int = 64
    activation: str = "relu"
    dropout: float = 0.1
    use_gating: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""
    base_model: str = "gpt2"
    max_length: int = 512
    model_device: str = "cuda"
    load_in_8bit: bool = False


@dataclass
class PromptConfig:
    """Prompt engineering configuration."""
    use_few_shot: bool = True
    few_shot_examples: int = 2
    use_cot: bool = True
    use_role_context: bool = True
    system_role: str = "customer_support_specialist"


@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 2e-4
    batch_size: int = 8
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    """Data configuration."""
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42


@dataclass
class Config:
    """Main configuration."""
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


# Default configuration instance
config = Config()
