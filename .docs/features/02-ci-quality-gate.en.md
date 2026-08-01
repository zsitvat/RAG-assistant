# Feature: Continuous integration quality gate

## What it does

Runs the repository's complete static-analysis and test gate for every pull request, every push to
`main`, and manual workflow dispatch. Internal branches also submit coverage-backed analysis to
SonarQube Cloud and wait for its quality gate.

## How it works


The jobs execute these checks:

1. The `lint` job runs Ruff lint verification.
2. The `format` job runs Ruff format verification.
3. The `bandit` job runs the security scan.
4. The `pytest` job runs the test suite with terminal and XML coverage, then uploads `coverage.xml`
   as a short-lived artifact.
5. After all quality jobs pass, the `sonar` job downloads the coverage artifact and materialises
   `sonar-project.properties` from
   `sonar-project.properties.example`, substituting the
   `SONAR_ORGANIZATION`/`SONAR_PROJECT_KEY` repository variables.
6. The `sonar` job runs the locked `pysonar` scanner using that generated configuration.

`LLM_BACKEND=dummy` and `LANGFUSE_ENABLED=false` keep CI independent of Ollama, Redis and Langfuse.
The Sonar job receives `SONAR_TOKEN`, `SONAR_ORGANIZATION` and `SONAR_PROJECT_KEY` only through
its step environment; none of the three are committed to the repository. It runs for pushes,
workflow dispatches and pull requests whose source branch belongs to this repository, and is
skipped for fork pull requests because GitHub does not expose repository secrets/variables to them;
lint, security and tests still run.

The workflow-level concurrency group cancels an older run for the same ref when a newer commit
arrives.

## Repository setup

Add a GitHub Actions repository secret named `SONAR_TOKEN` (a SonarQube Cloud user or organization
token with analysis permission) and two repository variables, `SONAR_ORGANIZATION` and
`SONAR_PROJECT_KEY`, matching your SonarCloud project. A missing token fails the analysis step with
an explicit message; missing variables fail the configuration step before analysis runs.

## Key files

| File | Responsibility |
| --- | --- |
| `.github/workflows/ci.yml` | triggers, permissions, locked environment and ordered quality steps |
| `pyproject.toml`, `uv.lock` | tool configuration and reproducible dependencies |
| `sonar-project.properties.example` | committed Sonar template (org/project key as placeholders) |
| `sonar-project.properties` | gitignored, locally/CI-generated Sonar project configuration |
| `Makefile` | equivalent local quality and Sonar commands |
