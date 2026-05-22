import torch
import pytest
from app.model.transformer import MiniGPT
from app.training.trainer import Trainer
from unittest.mock import MagicMock, patch

def test_trainer_estimate_loss():
    vocab_size = 10
    n_embd = 16
    n_head = 4
    n_layer = 1
    block_size = 8
    dropout = 0.1
    
    model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_data = torch.randint(0, vocab_size, (100,))
    val_data = torch.randint(0, vocab_size, (100,))
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        device='cpu',
        batch_size=4,
        block_size=block_size,
        eval_interval=10,
        eval_iters=2,
        checkpoint_path='dummy.pt'
    )
    
    losses = trainer.estimate_loss()
    assert 'train' in losses
    assert 'val' in losses
    assert isinstance(losses['train'], float)

@patch('app.training.trainer.log_metrics')
@patch('app.training.trainer.save_checkpoint')
def test_trainer_train(mock_save, mock_log):
    vocab_size = 10
    n_embd = 16
    n_head = 4
    n_layer = 1
    block_size = 8
    dropout = 0.1
    
    model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_data = torch.randint(0, vocab_size, (100,))
    val_data = torch.randint(0, vocab_size, (100,))
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        device='cpu',
        batch_size=4,
        block_size=block_size,
        eval_interval=1,
        eval_iters=1,
        checkpoint_path='dummy.pt'
    )
    
    trainer.train(max_iters=2)
    assert mock_log.called
    assert mock_save.called
