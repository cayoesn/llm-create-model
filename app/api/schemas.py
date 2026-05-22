from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., example="To be or not to be")
    max_new_tokens: int = Field(default=100, ge=1, le=500)
    temperature: float = Field(default=1.0, gt=0.0, le=2.0)
    top_k: int = Field(default=None, ge=1)


class GenerateResponse(BaseModel):
    generated_text: str
    tokens_generated: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
