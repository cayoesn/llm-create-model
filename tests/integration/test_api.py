import pytest
from fastapi.testclient import TestClient
from app.api.main import app
from app.api.dependencies import get_model, get_tokenizer
from app.model.tokenizer import Tokenizer
from app.model.transformer import MiniGPT
import torch

# Mock dependencies
class MockTokenizer:
    def __init__(self):
        self.vocab_size = 10
    def encode(self, s):
        return [1, 2, 3]
    def decode(self, l):
        return "mock text"

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = 64
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, idx, targets=None):
        return torch.randn(1, idx.shape[1], 10), None
    def eval(self):
        return self

def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_metrics_endpoint():
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200

def test_generate_endpoint():
    # Override dependencies
    app.dependency_overrides[get_model] = lambda: MockModel()
    app.dependency_overrides[get_tokenizer] = lambda: MockTokenizer()
    
    client = TestClient(app)
    response = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 5})
    
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert data["tokens_generated"] == 5
    
    app.dependency_overrides.clear()
