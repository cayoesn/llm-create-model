import torch
import os
from functools import lru_cache
from app.config.settings import settings
from app.model.tokenizer import Tokenizer
from app.model.transformer import MiniGPT
from app.training.checkpoints import load_checkpoint

@lru_cache()
def get_tokenizer() -> Tokenizer:
    """
    Loads and returns the tokenizer.
    """
    tokenizer = Tokenizer()
    if os.path.exists(settings.VOCAB_PATH):
        tokenizer.load_vocab(settings.VOCAB_PATH)
    else:
        print(f"Warning: Vocab file not found at {settings.VOCAB_PATH}. Tokenizer is uninitialized.")
    return tokenizer

@lru_cache()
def get_model() -> MiniGPT:
    """
    Loads and returns the MiniGPT model.
    """
    tokenizer = get_tokenizer()
    # If vocab is not loaded, vocab_size might be 0 or small. 
    # MiniGPT needs a valid vocab_size.
    vocab_size = tokenizer.vocab_size if tokenizer.vocab_size > 0 else 100 
    
    model = MiniGPT(
        vocab_size=vocab_size,
        n_embd=settings.N_EMBD,
        n_head=settings.N_HEAD,
        n_layer=settings.N_LAYER,
        block_size=settings.BLOCK_SIZE,
        dropout=settings.DROPOUT
    )
    
    if os.path.exists(settings.MODEL_PATH):
        load_checkpoint(settings.MODEL_PATH, model)
    else:
        print(f"Warning: Model checkpoint not found at {settings.MODEL_PATH}. Model is uninitialized.")
        
    model.to(settings.DEVICE)
    model.eval()
    return model
