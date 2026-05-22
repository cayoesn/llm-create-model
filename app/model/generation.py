import torch
import torch.nn.functional as f

from app.model.transformer import MiniGPT


@torch.no_grad()
def generate(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Generates new tokens from the model autoregressively.

    Args:
        model: The MiniGPT model.
        idx: Initial context as a tensor of shape (B, T).
        max_new_tokens: The maximum number of tokens to generate.
        temperature: Sampling temperature. Higher values mean more random.
        top_k: If provided, only sample from the top k most likely tokens.

    Returns:
        The sequence of tokens including the generated ones, shape (B, T + max_new_tokens).
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Crop idx to the last block_size tokens
        idx_cond = idx[:, -model.block_size :]

        # Get the predictions
        logits, _ = model(idx_cond)

        # Focus only on the last time step
        logits = logits[:, -1, :] / temperature  # (B, C)

        # Optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # Apply softmax to get probabilities
        probs = f.softmax(logits, dim=-1)  # (B, C)

        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

    return idx
