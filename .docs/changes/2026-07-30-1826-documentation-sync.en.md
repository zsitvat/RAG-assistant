# 2026-07-30 18:26 UTC — Documentation synchronized with the current shell

## What changed

- Split the technical design's repository layout into the implemented runnable shell and the modules
  planned for later slices.
- Documented `.env.example`, the ignored local `.env`, the locked `pysonar` scanner,
  `sonar-project.properties`, and the `make sonar` quality-gate path.
- Updated the FastAPI lifespan description to match its
  `@asynccontextmanager`/`AsyncGenerator[None, None]` implementation.
- Recorded the current source conventions: no file-level or module-level docstrings, central
  dependency wiring, module-owned Pydantic models for future feature modules, and no ad hoc helper
  or boilerplate getter/setter functions.
- Updated the README, runnable-shell feature document, task 1 and development-flow quality-gate
  command to match the current repository.

## Why

The application shell evolved after its first implementation: environment templates and SonarQube
Cloud support were added, the lifespan annotation was corrected, and source conventions were
tightened. The documentation now distinguishes implemented behaviour from the target architecture
so readers do not mistake future Redis, RAG, graph, evaluation or container modules for files that
already exist.
