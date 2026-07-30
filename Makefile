.PHONY: install lint format format-check security test check sonar run-api run-ui clean

install:
	uv sync --dev

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

security:
	uv run bandit -c pyproject.toml -r app

test:
	uv run pytest --cov=app --cov-report=term-missing --cov-report=xml

check: lint format-check security test

sonar:
	@test -n "$$SONAR_TOKEN" || (echo "SONAR_TOKEN is required" >&2; exit 1)
	$(MAKE) test
	uv run pysonar

run-api:
	LLM_BACKEND=dummy uv run uvicorn app.main:app --port 8000

run-ui:
	API_BASE_URL=http://127.0.0.1:8000 uv run streamlit run app/ui.py

clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache htmlcov .coverage coverage.xml logs
	find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
