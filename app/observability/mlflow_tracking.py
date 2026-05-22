import logging
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """
    Sets up MLflow tracking.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_params(params: dict[str, Any]) -> None:
    """
    Logs parameters to MLflow.
    """
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int) -> None:
    """
    Logs metrics to MLflow.
    """
    mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    """
    Logs an artifact to MLflow.
    """
    mlflow.log_artifact(local_path, artifact_path)
