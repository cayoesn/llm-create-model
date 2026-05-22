# MiniLLMOps

MiniLLMOps is a complete educational LLMOps platform that trains and serves a mini GPT-style language model from scratch using PyTorch. This project is designed to demonstrate the entire lifecycle of an LLM project, from data preparation and model architecture to training, serving, and production-grade observability.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Venv Setup](#venv-setup)
- [Installation](#installation)
- [Training Instructions](#training-instructions)
- [API Instructions](#api-instructions)
- [Docker Instructions](#docker-instructions)
- [Testing Instructions](#testing-instructions)
- [MLflow Usage](#mlflow-usage)
- [Grafana Usage](#grafana-usage)
- [Tokenizer Explanation](#tokenizer-explanation)
- [Transformer Explanation](#transformer-explanation)
- [Causal Masking Explanation](#causal-masking-explanation)
- [Autoregressive Generation Explanation](#autoregressive-generation-explanation)
- [Observability Explanation](#observability-explanation)
- [Tracing Explanation](#tracing-explanation)

## Project Overview

The goal of this project is to build a character-level language model trained on the Tiny Shakespeare dataset. The model is a decoder-only transformer, similar to the architecture used in GPT models.

## Architecture

The system consists of several components:
- **Model:** A custom transformer implementation using PyTorch.
- **Training Pipeline:** A manual training loop with MLflow integration for experiment tracking.
- **API:** A FastAPI application for serving predictions.
- **Observability:** Prometheus for metrics, OpenTelemetry for tracing, and Grafana for dashboards.
- **Infrastructure:** Docker and Docker Compose for easy deployment.

## Project Structure

```text
mini-llmops/
├── app/
│   ├── api/                # FastAPI application
│   ├── model/              # Transformer model components
│   ├── training/           # Training and dataset logic
│   ├── observability/      # Tracing, metrics, and tracking
│   ├── config/             # Configuration settings
│   └── utils/              # Utility functions
├── data/                   # Dataset storage (includes tiny_shakespeare.txt)
├── models/                 # Model checkpoints and vocabulary
├── tests/                  # Unit and integration tests
├── docker/                 # Dockerfiles and infrastructure config
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```

## Venv Setup

### Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

## Installation

```bash
pip install -r requirements.txt
```

The dataset is already included in the `data/` directory.

## Training Instructions

To train the model using Docker:
```bash
make train
```
This will start the training process in a container and save checkpoints to the `models/` directory (mapped via volume).

## API Instructions

To start the API using Docker:
```bash
make api
```
The API will be available at `http://localhost:8000`.

### Example Generation Request:
```bash
curl -X POST http://localhost:8000/generate \
-H "Content-Type: application/json" \
-d '{"prompt":"To be or not to be"}'
```

## Docker Instructions

To start the entire stack (API, MLflow, Prometheus, Grafana):
```bash
make docker-up
```

## Testing Instructions

To run tests locally:
```bash
make test
```

## MLflow Usage

MLflow is used to track experiments. You can view the dashboard at `http://localhost:5000`. It captures:
- Hyperparameters (learning rate, batch size, etc.)
- Metrics (loss, perplexity)
- Artifacts (checkpoints, vocabulary)

## Grafana Usage

Grafana is used for real-time monitoring. Access it at `http://localhost:3000`. You can create dashboards to monitor API latency, request counts, and model performance.

## Tokenizer Explanation

We use a custom character-level tokenizer. It maps every unique character in the training text to a unique integer. This is the simplest form of tokenization and is excellent for educational purposes.

## Transformer Explanation

The model is a decoder-only transformer. It uses:
- **Token Embeddings:** Maps token IDs to vectors.
- **Positional Embeddings:** Learns the position of each token in the sequence.
- **Multi-Head Attention:** Allows the model to attend to different parts of the sequence simultaneously.
- **FeedForward Network:** A point-wise MLP applied to each token.
- **Layer Normalization:** Stabilizes training.

## Causal Masking Explanation

In the self-attention mechanism, we use a causal mask (a lower triangular matrix of ones). This ensures that when predicting the next token, the model can only "see" tokens that came before it, preventing it from "cheating" by looking at the future.

## Autoregressive Generation Explanation

Generation is done one token at a time. The model predicts the next token, which is then appended to the sequence and used as input for the next prediction. We support temperature sampling and top-k filtering to control the randomness and quality of the generated text.

## Observability Explanation

- **Prometheus:** Scrapes the `/metrics` endpoint to collect API and model metrics.
- **Structured Logging:** Logs are emitted in JSON format for easier ingestion and analysis.

## Tracing Explanation

We use OpenTelemetry to trace requests through the API. This allows us to see the latency of different stages (encoding, inference, decoding) and track the flow of data through the system.
