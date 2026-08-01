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

Start Redis 8 and the Redis Insight UI:

```bash
docker compose up -d redis redisinsight
docker compose ps
```

Redis is available to host processes at `redis://127.0.0.1:6379/0`; Redis Insight is available at
<http://127.0.0.1:5540> and is preconfigured for the Redis service. Redis and UI state are persisted
in the `redis8_data` and `redisinsight_data` Docker volumes. Use `docker compose down` to stop the
services without deleting their data.

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

The same checks run in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for pull requests and
pushes to `main`. Before the first internal CI run, configure the `SONAR_TOKEN` repository secret
and the `SONAR_ORGANIZATION`/`SONAR_PROJECT_KEY` repository variables. Fork pull requests run lint,
formatting, security and tests without receiving the secret.

Run the SonarQube Cloud analysis locally:

```bash
cp sonar-project.properties.example sonar-project.properties
# Fill in your own organization/project key in sonar-project.properties (gitignored).
export SONAR_TOKEN="<your-token>"
make sonar
```

The token is read only from the environment and must not be committed; neither is
`sonar-project.properties` (see `sonar-project.properties.example`). The command generates
`coverage.xml`, uploads the analysis, and waits for the configured quality gate.

The optional SonarQube management CLI uses a separate, interactive browser login:

```bash
sonar auth login -o <your-sonarcloud-organization-key>
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
`POST /chat` returns the answer together with the deterministic eligibility `decision`, cited sources, and stable completed-step labels.
