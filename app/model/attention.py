import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention.
    
    This module implements multi-head attention with a causal mask to ensure 
    that each position can only attend to previous positions.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        """
        Initializes the multi-head attention module.
        
        Args:
            n_embd: The dimensionality of the embeddings.
            n_head: The number of attention heads.
            block_size: The maximum sequence length.
            dropout: The dropout probability.
        """
        super().__init__()
        assert n_embd % n_head == 0
        
        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for causal self-attention.
        
        Args:
            x: A tensor of shape (B, T, C) containing the input embeddings.
            
        Returns:
            A tensor of shape (B, T, C) containing the attended values.
        """
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
