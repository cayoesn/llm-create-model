.PHONY: help install run api train train-local test coverage lint format docker-up docker-down logs

DC = docker compose
VENV = .venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip

help:
	@echo "mini-gpt Makefile"
	@echo "  make install      - cria a virtualenv e instala dependencias"
	@echo "  make run          - sobe a API localmente com reload"
	@echo "  make api          - sobe apenas a API via Docker"
	@echo "  make train        - executa o treinamento em container"
	@echo "  make train-local  - executa o treinamento localmente"
	@echo "  make test         - executa os testes"
	@echo "  make coverage     - gera coverage HTML"
	@echo "  make lint         - roda ruff e mypy"
	@echo "  make format       - formata o codigo com black e ruff --fix"
	@echo "  make docker-up    - sobe a stack Docker completa"
	@echo "  make docker-down  - derruba a stack Docker"
	@echo "  make logs         - acompanha logs da stack"

install:
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

api:
	$(DC) up api

train:
	$(DC) run --rm training

train-local:
	$(PYTHON) app/training/train.py

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=app --cov-report=term-missing --cov-report=html:tests/htmlcov

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy .

format:
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check . --fix

docker-up:
	$(DC) up -d --build

docker-down:
	$(DC) down

logs:
	$(DC) logs -f
