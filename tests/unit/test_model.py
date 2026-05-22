import torch

from app.model.attention import CausalSelfAttention
from app.model.embeddings import MiniGPTEmbeddings
from app.model.generation import generate
from app.model.transformer import MiniGPT
from app.model.transformer_block import Block


def test_embeddings():
    vocab_size = 10
    n_embd = 16
    block_size = 32
    embeddings = MiniGPTEmbeddings(vocab_size, n_embd, block_size)
    idx = torch.randint(0, vocab_size, (2, 8))
    out = embeddings(idx)
    assert out.shape == (2, 8, n_embd)


def test_causal_self_attention():
    n_embd = 16
    n_head = 4
    block_size = 32
    dropout = 0.1
    attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
    x = torch.randn(2, 8, n_embd)
    out = attn(x)
    assert out.shape == (2, 8, n_embd)


def test_transformer_block():
    n_embd = 16
    n_head = 4
    block_size = 32
    dropout = 0.1
    block = Block(n_embd, n_head, block_size, dropout)
    x = torch.randn(2, 8, n_embd)
    out = block(x)
    assert out.shape == (2, 8, n_embd)


def test_mini_gpt():
    vocab_size = 10
    n_embd = 16
    n_head = 4
    n_layer = 2
    block_size = 32
    dropout = 0.1
    model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    idx = torch.randint(0, vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 8, vocab_size)
    assert loss is None

    targets = torch.randint(0, vocab_size, (2, 8))
    logits, loss = model(idx, targets)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)


def test_generation():
    vocab_size = 10
    n_embd = 16
    n_head = 4
    n_layer = 2
    block_size = 32
    dropout = 0.1
    model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    idx = torch.randint(0, vocab_size, (1, 5))
    out = generate(model, idx, max_new_tokens=5)
    assert out.shape == (1, 10)
