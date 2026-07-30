# 2026-07-30 18:26 UTC — GitHub Actions quality gate

## What changed

- Added `.github/workflows/ci.yml` for pull requests, `main` pushes and manual runs.
- Added locked Python 3.12/uv setup with dependency caching.
- Added ordered Ruff, formatting, Bandit, pytest/coverage and SonarQube Cloud steps.
- Migrated the runtime HTTP client and ASGI API tests from deprecated `httpx`/`TestClient` usage to
  `httpx2`; async tests now exercise the FastAPI lifespan through `ASGITransport`.
- Made the FastAPI dependency providers async so state-only dependency resolution stays on the
  request event loop instead of entering AnyIO's worker threadpool.
- Restricted the workflow token to read-only repository contents and scoped `SONAR_TOKEN` to the
  Sonar step.
- Skipped only the secret-dependent Sonar step for fork pull requests while retaining all local
  quality checks.

## Why

The repository already exposed equivalent local commands, but lacked an automated clean-runner
quality gate. The workflow now verifies the lockfile and the complete shell on every proposed
change, while SonarQube Cloud receives the same `coverage.xml` produced by the CI test run.
