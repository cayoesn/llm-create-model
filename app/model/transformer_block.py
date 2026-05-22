import torch
import torch.nn as nn

from app.model.attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity.
    """

    def __init__(self, n_embd: int, dropout: float) -> None:
        """
        Initializes the feedforward network.

        Args:
            n_embd: The dimensionality of the embeddings.
            dropout: The dropout probability.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feedforward network.
        """
        return self.net(x)


class Block(nn.Module):
    """
    A Transformer decoder block.

    Contains one layer of causal self-attention and one layer of feedforward
    neural network, with layer normalization and residual connections.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        """
        Initializes the transformer block.

        Args:
            n_embd: The dimensionality of the embeddings.
            n_head: The number of attention heads.
            block_size: The maximum sequence length.
            dropout: The dropout probability.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer block.

        Applies layer normalization before attention and feedforward layers
        (pre-norm architecture) and adds residual connections.
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x
