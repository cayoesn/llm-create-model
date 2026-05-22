import os

import torch

from app.model.tokenizer import Tokenizer
from app.training.batching import get_batch
from app.training.checkpoints import load_checkpoint, save_checkpoint
from app.training.dataset import ShakespeareDataset


def test_get_batch():
    data = torch.arange(100)
    batch_size = 4
    block_size = 8
    x, y = get_batch(data, batch_size, block_size, "cpu")
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    # Check that y is x shifted by 1
    for i in range(batch_size):
        assert torch.equal(x[i, 1:], y[i, :-1])


def test_shakespeare_dataset(tmp_path):
    text = "hello world"
    data_path = tmp_path / "input.txt"
    data_path.write_text(text)

    tokenizer = Tokenizer()
    tokenizer.build_vocab(text)

    block_size = 4
    dataset = ShakespeareDataset(str(data_path), tokenizer, block_size)
    assert len(dataset) == len(text) - block_size

    x, y = dataset[0]
    assert x.shape == (block_size,)
    assert y.shape == (block_size,)


def test_checkpoints(tmp_path):
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(1, 1)

        def forward(self, x):
            return self.l(x)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(model, optimizer, 10, 0.5, str(checkpoint_path))
    assert os.path.exists(checkpoint_path)

    new_model = SimpleModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    checkpoint = load_checkpoint(str(checkpoint_path), new_model, new_optimizer)

    assert checkpoint["epoch"] == 10
    assert checkpoint["loss"] == 0.5
    # Check that weights are loaded
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
