import torch
from torch.utils.data import Dataset

from app.model.tokenizer import Tokenizer


class ShakespeareDataset(Dataset):
    """
    PyTorch Dataset for Tiny Shakespeare.
    """

    def __init__(self, data_path: str, tokenizer: Tokenizer, block_size: int) -> None:
        """
        Initializes the dataset.

        Args:
            data_path: Path to the input text file.
            tokenizer: The tokenizer to use.
            block_size: The sequence length.
        """
        with open(data_path, encoding="utf-8") as f:
            text = f.read()

        self.tokenizer = tokenizer
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a chunk of data of length block_size.
        """
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
