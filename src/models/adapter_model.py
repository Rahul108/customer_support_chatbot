import torch
import torch.nn as nn
from config.settings import Config
from .base_model import BaseModel


class AdapterLayer(nn.Module):
    """Adapter layer with bottleneck architecture."""

    def __init__(self, hidden_size: int, adapter_size: int, dropout: float = 0.1):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class AdapterModel(BaseModel):
    """Adapter-based model implementation."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.adapters = nn.ModuleDict()
        self.prepare_model()

    def prepare_model(self):
        """Add adapters to transformer layers."""
        hidden_size = self.model.config.hidden_size
        adapter_size = self.config.adapter.adapter_size

        # Add adapters to each transformer layer
        for i, layer in enumerate(self.model.transformer.h):
            adapter = AdapterLayer(
                hidden_size=hidden_size,
                adapter_size=adapter_size,
                dropout=self.config.adapter.dropout,
            )
            self.adapters[f"layer_{i}"] = adapter
            # Insert adapter after layer's feed-forward
            self._insert_adapter_to_layer(layer, adapter)

    def _insert_adapter_to_layer(self, layer: nn.Module, adapter: AdapterLayer):
        """Insert adapter into transformer layer."""
        original_forward = layer.forward

        def new_forward(hidden_states, *args, **kwargs):
            output = original_forward(hidden_states, *args, **kwargs)
            if isinstance(output, tuple):
                hidden_states = output[0]
                hidden_states = adapter(hidden_states)
                return (hidden_states,) + output[1:]
            else:
                return adapter(output)

        layer.forward = new_forward

    def get_trainable_params(self) -> dict:
        """Get trainable and total parameters."""
        trainable_params = sum(
            p.numel() for p in self.adapters.parameters() if p.requires_grad
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
