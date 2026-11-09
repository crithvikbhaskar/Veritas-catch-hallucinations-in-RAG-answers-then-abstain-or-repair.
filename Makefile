.PHONY: install test eval api web clean

install:
	uv sync --extra dev
	cd web && npm install

test:
	uv run pytest -v

eval:
	uv run python eval/run.py

api:
	uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

web:
	cd web && npm run dev

clean:
	rm -rf .pytest_cache docs/results/__pycache__ eval/__pycache__ src/groundcheck/__pycache__ tests/__pycache__

