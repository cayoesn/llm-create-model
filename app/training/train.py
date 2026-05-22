import logging
import os

import mlflow
import torch

from app.config.settings import settings
from app.model.tokenizer import Tokenizer
from app.model.transformer import MiniGPT
from app.observability.mlflow_tracking import log_artifact, log_params, setup_mlflow
from app.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_training() -> None:
    """
    The main training script.
    """
    # Setup MLflow
    setup_mlflow(settings.MLFLOW_TRACKING_URI, settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        # Log parameters
        log_params(
            {
                "batch_size": settings.BATCH_SIZE,
                "learning_rate": settings.LEARNING_RATE,
                "max_iters": settings.MAX_ITERS,
                "n_embd": settings.N_EMBD,
                "n_head": settings.N_HEAD,
                "n_layer": settings.N_LAYER,
                "block_size": settings.BLOCK_SIZE,
                "dropout": settings.DROPOUT,
            }
        )

        # Load data
        if not os.path.exists(settings.DATA_PATH):
            logger.error(f"Data not found at {settings.DATA_PATH}.")
            return

        with open(settings.DATA_PATH, encoding="utf-8") as f:
            text = f.read()

        # Tokenizer
        tokenizer = Tokenizer()
        tokenizer.build_vocab(text)
        tokenizer.save_vocab(settings.VOCAB_PATH)
        log_artifact(settings.VOCAB_PATH)

        vocab_size = tokenizer.vocab_size
        logger.info(f"Vocab size: {vocab_size}")

        # Train/Val splits
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

        # Model
        model = MiniGPT(
            vocab_size=vocab_size,
            n_embd=settings.N_EMBD,
            n_head=settings.N_HEAD,
            n_layer=settings.N_LAYER,
            block_size=settings.BLOCK_SIZE,
            dropout=settings.DROPOUT,
        )
        model.to(settings.DEVICE)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE)

        # Trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            val_data=val_data,
            device=settings.DEVICE,
            batch_size=settings.BATCH_SIZE,
            block_size=settings.BLOCK_SIZE,
            eval_interval=settings.EVAL_INTERVAL,
            eval_iters=settings.EVAL_ITERS,
            checkpoint_path=settings.MODEL_PATH,
        )

        # Train
        trainer.train(settings.MAX_ITERS)

        # Log final model as artifact
        log_artifact(settings.MODEL_PATH)


if __name__ == "__main__":
    run_training()
