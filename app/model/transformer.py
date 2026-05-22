import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional

from app.model.embeddings import MiniGPTEmbeddings
from app.model.transformer_block import Block

class MiniGPT(nn.Module):
    """
    The MiniGPT language model.
    
    A decoder-only transformer model designed for character-level language modeling.
    """

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int, dropout: float) -> None:
        """
        Initializes the MiniGPT model.
        
        Args:
            vocab_size: The size of the vocabulary.
            n_embd: The dimensionality of the embeddings.
            n_head: The number of attention heads.
            n_layer: The number of transformer blocks.
            block_size: The maximum sequence length.
            dropout: The dropout probability.
        """
        super().__init__()
        self.block_size = block_size
        
        self.embeddings = MiniGPTEmbeddings(vocab_size, n_embd, block_size)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initializes weights using a normal distribution.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the MiniGPT model.
        
        Args:
            idx: A tensor of shape (B, T) containing token IDs.
            targets: An optional tensor of shape (B, T) containing target token IDs.
            
        Returns:
            A tuple (logits, loss). Logits is (B, T, vocab_size). 
            Loss is a scalar tensor if targets are provided, else None.
        """
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        x = self.embeddings(idx) # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
