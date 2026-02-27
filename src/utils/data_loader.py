import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from typing import Tuple, List
from config.settings import Config


class DataLoader:
    """Data loading and preprocessing utilities."""

    def __init__(self, config: Config):
        self.config = config

    def load_csv_data(self, filepath: str) -> Tuple[List[str], List[str]]:
        """
        Load data from CSV file.
        Args:
            filepath: Path to CSV file with 'input' and 'output' columns
        Returns:
            Tuple of (inputs, outputs) lists
        """
        df = pd.read_csv(filepath)
        inputs = df["input"].tolist()
        outputs = df["output"].tolist()
        return inputs, outputs

    def tokenize_data(
        self, texts: List[str], tokenizer, max_length: int = None
    ) -> dict:
        """
        Tokenize text data.
        Args:
            texts: List of text strings
            tokenizer: Hugging Face tokenizer
            max_length: Maximum token length
        Returns:
            Dictionary with tokenized data
        """
        if max_length is None:
            max_length = self.config.model.max_length

        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encodings

    def create_data_splits(
        self, inputs: List[str], outputs: List[str]
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Split data into train/val/test sets.
        Args:
            inputs: List of input texts
            outputs: List of output texts
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        import numpy as np

        total_samples = len(inputs)
        indices = np.arange(total_samples)
        np.random.seed(self.config.data.random_seed)
        np.random.shuffle(indices)

        train_idx = int(total_samples * self.config.data.train_split)
        val_idx = int(total_samples * (self.config.data.train_split + self.config.data.val_split))

        train_indices = indices[:train_idx]
        val_indices = indices[train_idx:val_idx]
        test_indices = indices[val_idx:]

        train_data = ([inputs[i] for i in train_indices], [outputs[i] for i in train_indices])
        val_data = ([inputs[i] for i in val_indices], [outputs[i] for i in val_indices])
        test_data = ([inputs[i] for i in test_indices], [outputs[i] for i in test_indices])

        return train_data, val_data, test_data

    def create_dataloader(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int = None
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader.
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention masks
            batch_size: Batch size
        Returns:
            DataLoader instance
        """
        if batch_size is None:
            batch_size = self.config.training.batch_size

        dataset = TensorDataset(input_ids, attention_mask)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
