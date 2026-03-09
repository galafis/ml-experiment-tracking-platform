.PHONY: install test lint format run api docker-build docker-up docker-down clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ main.py
	ruff check --fix src/ tests/

run:
	python main.py

api:
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .coverage
	rm -rf demo_tracking.db demo_artifacts/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
