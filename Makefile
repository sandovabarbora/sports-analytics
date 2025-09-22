.PHONY: help install test run serve clean format lint benchmark

help:
    @echo "Veo Analytics with UV - Commands:"
    @echo "  make install  - Install dependencies with UV"
    @echo "  make test     - Run tests"
    @echo "  make run      - Run demo"
    @echo "  make serve    - Start API server"
    @echo "  make format   - Format code with black & ruff"
    @echo "  make lint     - Lint code"
    @echo "  make clean    - Clean cache files"

install:
    uv pip install -e ".[dev]"

test:
    uv run pytest tests/ -v --cov=src

run:
    uv run python demo_uv.py

serve:
    uv run uvicorn src.interfaces.api.app:app --reload --host 0.0.0.0 --port 8000

format:
    uv run black src/ tests/
    uv run ruff --fix src/ tests/

lint:
    uv run ruff src/ tests/
    uv run mypy src/

benchmark:
    uv run python -c "from src.infrastructure.monitoring.benchmarks import run_benchmark; print(run_benchmark('data/samples/soccer/synthetic_hd.mp4'))"

clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
    rm -rf .pytest_cache .ruff_cache .mypy_cache
    rm -rf outputs/*
