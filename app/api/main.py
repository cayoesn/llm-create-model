from fastapi import FastAPI

from app.api.routes import router
from app.config.settings import settings
from app.observability.metrics import metrics_endpoint
from app.observability.tracing import setup_tracing

app = FastAPI(
    title="mini-gpt API",
    description="Educational platform serving a mini GPT-style model.",
    version="1.0.0",
)

# Setup Tracing
setup_tracing(app)

# Include Routes
app.include_router(router)


# Metrics Endpoint
@app.get("/metrics")
async def metrics():
    return metrics_endpoint()


@app.on_event("startup")
async def startup_event():
    from app.api.dependencies import get_model, get_tokenizer

    # Preload model and tokenizer
    try:
        get_tokenizer()
        get_model()
    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
