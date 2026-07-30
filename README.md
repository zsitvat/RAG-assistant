# RAG-assistant

Agentic RAG prototype: corporate expense reimbursement and employee benefits assistant.
See [`.docs/plan/01-idea-plan.en.md`](.docs/plan/01-idea-plan.en.md) for the problem statement and
[`.docs/plan/02-technical-design.en.md`](.docs/plan/02-technical-design.en.md) for the implementation
reference. The full README (architecture, results, run instructions) lands at milestone M7; this
section only covers the current development quickstart.

## Development quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it automatically).

```bash
uv sync --dev
cp .env.example .env
```

`Settings` loads `.env` automatically. The committed `.env.example` documents every application
setting; the ignored local `.env` is the place for machine-specific URLs and credentials. For an
offline shell, set `LLM_BACKEND=dummy` and `LANGFUSE_ENABLED=false`.

Run the quality gates:

```bash
uv run ruff check .
uv run ruff format --check .
uv run bandit -c pyproject.toml -r app
uv run pytest --cov=app --cov-report=term-missing
```

Run the SonarQube Cloud analysis after creating a user token for the project:

```bash
export SONAR_TOKEN="<your-token>"
make sonar
```

The token is read only from the environment and must not be committed. The command generates
`coverage.xml`, uploads the analysis for project `zsitvat_RAG-assistant`, and waits for the configured
quality gate.

The optional SonarQube management CLI uses a separate, interactive browser login:

```bash
sonar auth login -o zsitvat
sonar auth status
```

This login must be completed manually and does not replace `SONAR_TOKEN` for `make sonar`.

Run the API in dummy-model mode (no Ollama/Redis required) and the Streamlit shell against it. The
repository's local `.env` already selects the dummy backend and loopback URLs:

```bash
uv run uvicorn app.main:app --port 8000
# in a second terminal
uv run streamlit run app/ui.py
```

`GET /health` reports liveness, `GET /ready` reports per-dependency readiness (`llm`, `redis`); with
`LLM_BACKEND=dummy` the LLM check is `ok` and no external service is contacted.
