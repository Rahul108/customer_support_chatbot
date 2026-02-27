# Setup Guide - Customer Support Chatbot

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- 4GB+ RAM (8GB+ recommended for model training)
- GPU (optional, but recommended for faster inference/training)

## Installation Steps

### 1. Clone/Navigate to Project

```bash
cd /home/pyro/projects/customer_support_chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import transformers; import peft; print('All dependencies installed successfully!')"
```

## Project Structure

```
customer_support_chatbot/
├── config/
│   └── settings.py              # Configuration classes
├── src/
│   ├── models/
│   │   ├── base_model.py        # Base model class
│   │   ├── lora_model.py        # LoRA implementation
│   │   ├── adapter_model.py     # Adapter implementation
│   │   └── peft_model.py        # Unified PEFT interface
│   ├── prompts/
│   │   ├── few_shot.py          # Few-shot prompting
│   │   ├── chain_of_thought.py  # Chain-of-thought prompting
│   │   └── role_context.py      # Role-based prompting
│   ├── pipelines/
│   │   ├── inference_pipeline.py # Inference pipeline
│   │   └── training_pipeline.py  # Training pipeline
│   └── utils/
│       ├── data_loader.py       # Data loading utilities
│       └── evaluation.py        # Evaluation metrics
├── tests/
│   └── test_models.py           # Unit tests
└── requirements.txt             # Package dependencies
```

## Quick Start

### 1. Basic Model Loading

```python
from config.settings import Config
from src.models.lora_model import LoRAModel

# Initialize configuration
config = Config()

# Load LoRA model
model = LoRAModel(config)
print(f"Model parameters: {model.get_trainable_params()}")
```

### 2. Generate Response with Inference Pipeline

```python
from src.pipelines.inference_pipeline import InferencePipeline

pipeline = InferencePipeline(model, config)

# Generate response using few-shot strategy
response = pipeline.generate_response(
    "How can I reset my password?",
    strategy="few_shot"
)
print(response)
```

### 3. Add Few-Shot Examples

```python
pipeline.add_few_shot_example(
    "How do I login?",
    "Click the login button on the home page and enter your credentials."
)
```

### 4. Use Role-Based Prompting

```python
pipeline.set_role("technical_support")
response = pipeline.generate_response(
    "My app keeps crashing",
    strategy="role_context"
)
```

### 5. Train Model

```python
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.data_loader import DataLoader

loader = DataLoader(config)
training_pipeline = TrainingPipeline(model, config)

# Create sample dataloaders (implement your own data loading)
# training_pipeline.train(train_dataloader, val_dataloader)
```

## Running Tests

```bash
python -m pytest tests/ -v
# OR
python -m unittest discover tests/ -v
```

## Configuration

Edit `config/settings.py` to customize:

- **Model settings**: Base model, max length, device
- **LoRA settings**: Rank (r), alpha, dropout
- **Adapter settings**: Size, activation function
- **Prompt settings**: Example count, roles
- **Training settings**: Learning rate, batch size, epochs

## Common Issues

### CUDA Out of Memory
- Reduce `batch_size` in `config/settings.py`
- Use `load_in_8bit=True` in `ModelConfig`
- Use smaller base model (e.g., `distilgpt2`)

### Model Download Timeout
- Set custom cache directory: `export TRANSFORMERS_CACHE=/path/to/cache`
- Download models manually beforehand

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, ensure CUDA/cuDNN are properly installed.

## Environment Variables

Create `.env` file (optional):
```bash
TRANSFORMERS_CACHE=/path/to/models
HF_TOKEN=your_hugging_face_token
```

## Troubleshooting

1. **Import errors**: Ensure virtual environment is activated
2. **CUDA errors**: Check NVIDIA driver: `nvidia-smi`
3. **Memory errors**: Reduce batch size or use CPU
4. **Model not found**: Check internet connection for model download

## Next Steps

- Read `README.md` for project overview
- Explore example notebooks in `notebooks/`
- Check individual module docstrings for detailed API
- Run tests to verify everything works
