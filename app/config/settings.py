try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings

    SettingsConfigDict = None


class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    # Model Settings
    DEVICE: str = "cpu"
    MODEL_PATH: str = "models/checkpoints/best_model.pt"
    VOCAB_PATH: str = "models/vocab.json"

    # Model Hyperparameters
    N_EMBD: int = 128
    N_HEAD: int = 4
    N_LAYER: int = 4
    BLOCK_SIZE: int = 64
    DROPOUT: float = 0.1

    # Training Settings
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 3e-4
    MAX_ITERS: int = 5000
    EVAL_INTERVAL: int = 500
    EVAL_ITERS: int = 200

    # Data Settings
    DATA_PATH: str = "data/input.txt"

    # MLflow Settings
    MLFLOW_TRACKING_URI: str | None = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "mini-llmops"

    # Observability
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = "http://localhost:4317"
    PROMETHEUS_METRICS_PATH: str = "/metrics"

    if SettingsConfigDict:
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    else:

        class Config:
            env_file = ".env"
            extra = "ignore"


settings = Settings()
