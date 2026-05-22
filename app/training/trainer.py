import torch
import torch.nn as nn
import time
import logging
import math
from typing import Dict, Optional, Tuple

from app.model.transformer import MiniGPT
from app.training.batching import get_batch
from app.training.checkpoints import save_checkpoint
from app.observability.mlflow_tracking import log_metrics

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer for MiniGPT.
    """

    def __init__(
        self,
        model: MiniGPT,
        optimizer: torch.optim.Optimizer,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        device: str,
        batch_size: int,
        block_size: int,
        eval_interval: int,
        eval_iters: int,
        checkpoint_path: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.batch_size = batch_size
        self.block_size = block_size
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.checkpoint_path = checkpoint_path

    @torch.no_grad()
    def estimate_loss(self) -> Dict[str, float]:
        """
        Estimates the loss on train and validation sets.
        """
        out = {}
        self.model.eval()
        for split, data in [('train', self.train_data), ('val', self.val_data)]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = get_batch(data, self.batch_size, self.block_size, self.device)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def train(self, max_iters: int) -> None:
        """
        Runs the training loop.
        """
        self.model.train()
        start_time = time.time()
        
        best_val_loss = float('inf')

        for iter in range(max_iters):
            # Evaluate the loss on train/val sets periodically
            if iter % self.eval_interval == 0 or iter == max_iters - 1:
                losses = self.estimate_loss()
                train_loss = losses['train']
                val_loss = losses['val']
                perplexity = math.exp(val_loss)
                
                logger.info(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, perplexity {perplexity:.4f}")
                
                # Log to MLflow
                log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "perplexity": perplexity,
                }, step=iter)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(self.model, self.optimizer, iter, val_loss, self.checkpoint_path)
                    logger.info(f"New best model saved to {self.checkpoint_path}")

            # Sample a batch of data
            xb, yb = get_batch(self.train_data, self.batch_size, self.block_size, self.device)

            # Evaluate the loss
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()

        total_time = time.time() - start_time
        logger.info(f"Training finished in {total_time:.2f}s")
        log_metrics({"total_training_time": total_time}, step=max_iters)
