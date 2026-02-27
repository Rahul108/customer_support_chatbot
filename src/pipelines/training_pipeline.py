import torch
from torch.optim import AdamW
from tqdm import tqdm
from config.settings import Config
from src.models.base_model import BaseModel
from src.utils.data_loader import DataLoader as DataLoaderUtil
from src.utils.evaluation import Evaluator


class TrainingPipeline:
    """Training pipeline for fine-tuning models."""

    def __init__(self, model: BaseModel, config: Config):
        self.model = model
        self.config = config
        self.optimizer = AdamW(
            model.model.parameters(), lr=config.training.learning_rate
        )
        self.evaluator = Evaluator()
        self.training_history = {"loss": [], "val_loss": []}

    def train_epoch(self, dataloader, device: str = "cuda"):
        """
        Train for one epoch.
        Args:
            dataloader: PyTorch DataLoader
            device: Device to train on
        Returns:
            Average loss for the epoch
        """
        self.model.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            self.optimizer.zero_grad()

            outputs = self.model.forward(input_ids, attention_mask)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(), self.config.training.max_grad_norm
            )
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.training_history["loss"].append(avg_loss)
        return avg_loss

    def evaluate(self, dataloader, device: str = "cuda") -> float:
        """
        Evaluate model on validation set.
        Args:
            dataloader: PyTorch DataLoader
            device: Device to evaluate on
        Returns:
            Average validation loss
        """
        self.model.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids, attention_mask = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                outputs = self.model.forward(input_ids, attention_mask)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.training_history["val_loss"].append(avg_loss)
        return avg_loss

    def train(self, train_dataloader, val_dataloader=None, device: str = "cuda"):
        """
        Train the model for multiple epochs.
        Args:
            train_dataloader: Training DataLoader
            val_dataloader: Validation DataLoader
            device: Device to train on
        """
        for epoch in range(self.config.training.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")

            train_loss = self.train_epoch(train_dataloader, device)
            print(f"Training Loss: {train_loss:.4f}")

            if val_dataloader:
                val_loss = self.evaluate(val_dataloader, device)
                print(f"Validation Loss: {val_loss:.4f}")

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save(
            {
                "model_state": self.model.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.training_history,
            },
            path,
        )

    def load_checkpoint(self, path: str, device: str = "cuda"):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        self.model.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.training_history = checkpoint["history"]
