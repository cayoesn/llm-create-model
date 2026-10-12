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
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Generates new tokens from the model autoregressively with Temperature, Top-K,
    Top-P (Nucleus Sampling), and Repetition Penalty.

    Args:
        model: The MiniGPT model.
        idx: Initial context as a tensor of shape (B, T).
        max_new_tokens: The maximum number of tokens to generate.
        temperature: Sampling temperature. Higher values mean more diversity.
        top_k: If provided, only sample from the top k most likely tokens.
        top_p: If provided (0.0 < top_p < 1.0), filters tokens via Nucleus Sampling.
        repetition_penalty: Penalty applied to previously generated tokens (> 1.0 discourages repetition).

    Returns:
        The sequence of tokens including the generated ones, shape (B, T + max_new_tokens).
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Crop idx to the last block_size tokens
        idx_cond = idx[:, -model.block_size :]

        # Forward pass through MiniGPT
        logits, _ = model(idx_cond)

        # Focus only on the last time step
        logits = logits[:, -1, :]  # Shape: (B, C)

        # 1. Apply Repetition Penalty if specified
        if repetition_penalty != 1.0:
            for b in range(idx.shape[0]):
                unique_tokens = torch.unique(idx[b])
                for prev_token in unique_tokens:
                    token_id = prev_token.item()
                    if logits[b, token_id] > 0:
                        logits[b, token_id] /= repetition_penalty
                    else:
                        logits[b, token_id] *= repetition_penalty

        # 2. Temperature scaling
        temperature = max(1e-5, temperature)
        logits = logits / temperature

        # 3. Top-K filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        # 4. Top-P (Nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(f.softmax(sorted_logits, dim=-1), dim=-1)

            # Mask tokens where cumulative probability exceeds top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep at least the most probable token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float("Inf")

        # 5. Softmax to get probabilities
        probs = f.softmax(logits, dim=-1)  # (B, C)

        # 6. Sample next token
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # Append to running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
