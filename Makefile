.PHONY: test lint format coverage docker-up docker-down train api

test:
	python -m pytest

lint:
	ruff check .
	mypy .

format:
	black .
	ruff check . --fix

coverage:
	python -m pytest --cov=app --cov-report=html

train:
	docker-compose run --rm training

api:
	docker-compose up api

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down
