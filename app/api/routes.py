import time

import torch
from fastapi import APIRouter, Depends, HTTPException
from opentelemetry import trace

from app.api.dependencies import get_model, get_tokenizer
from app.api.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.model.generation import generate
from app.observability.metrics import INFERENCE_DURATION, TOKENS_GENERATED

router = APIRouter()
tracer = trace.get_tracer(__name__)


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    """
    import os

    from app.config.settings import settings

    model_loaded = os.path.exists(settings.MODEL_PATH)
    return HealthResponse(status="ok", model_loaded=model_loaded)


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest, model=Depends(get_model), tokenizer=Depends(get_tokenizer)
):
    """
    Generates text based on the provided prompt.
    """
    start_time = time.time()

    with tracer.start_as_current_span("generate_text") as span:
        span.set_attribute("prompt", request.prompt)
        span.set_attribute("max_new_tokens", request.max_new_tokens)

        try:
            # Encode prompt
            idx = (
                torch.tensor(tokenizer.encode(request.prompt), dtype=torch.long)
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )

            # Generate
            inf_start = time.time()
            out = generate(
                model,
                idx,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
            )
            inf_duration = time.time() - inf_start

            # Decode
            generated_text = tokenizer.decode(out[0].tolist())

            # Update metrics
            TOKENS_GENERATED.inc(request.max_new_tokens)
            INFERENCE_DURATION.observe(inf_duration)

            latency = (time.time() - start_time) * 1000

            span.set_attribute("generated_text", generated_text)
            span.set_attribute("latency_ms", latency)

            return GenerateResponse(
                generated_text=generated_text,
                tokens_generated=request.max_new_tokens,
                latency_ms=latency,
            )

        except Exception as e:
            span.record_exception(e)
            raise HTTPException(status_code=500, detail=str(e))
