import torch
import numpy as np
import random
from app.utils.seed import set_seed

def test_set_seed():
    set_seed(42)
    a = random.random()
    b = np.random.random()
    c = torch.rand(1)
    
    set_seed(42)
    assert a == random.random()
    assert b == np.random.random()
    assert torch.equal(c, torch.rand(1))
