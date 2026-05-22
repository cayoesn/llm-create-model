import torch
import torch.nn as nn

class MiniGPTEmbeddings(nn.Module):
    """
    Combined Token and Positional Embeddings for MiniGPT.
    
    This module maps token IDs to their corresponding token embeddings and 
    adds learnable positional embeddings.
    """

    def __init__(self, vocab_size: int, n_embd: int, block_size: int) -> None:
        """
        Initializes the embeddings.
        
        Args:
            vocab_size: The size of the vocabulary.
            n_embd: The dimensionality of the embeddings.
            block_size: The maximum sequence length (context window).
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embeddings.
        
        Args:
            idx: A tensor of shape (B, T) containing token IDs.
            
        Returns:
            A tensor of shape (B, T, C) containing the combined embeddings.
        """
        B, T = idx.shape
        
        # Token embeddings (B, T, C)
        tok_emb = self.token_embedding_table(idx)
        
        # Positional embeddings (T, C)
        # We use torch.arange to get positions 0, 1, ..., T-1
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding_table(pos) # (T, C)
        
        # Combine token and positional embeddings
        x = tok_emb + pos_emb
        
        return x
