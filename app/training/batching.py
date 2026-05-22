import torch
from typing import Tuple

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a small batch of data of inputs x and targets y.
    
    Args:
        data: The full encoded data as a 1D tensor.
        batch_size: The number of sequences in the batch.
        block_size: The sequence length.
        device: The device to move the tensors to.
        
    Returns:
        A tuple (x, y) where x and y are tensors of shape (batch_size, block_size).
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
