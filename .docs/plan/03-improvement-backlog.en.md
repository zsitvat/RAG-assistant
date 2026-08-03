# Improvement Backlog

Follow-up to [02-technical-design.en.md](02-technical-design.en.md). This backlog captures
concrete, code-grounded improvement opportunities identified after the initial submission, scoped
deliberately to exclude two topics already covered elsewhere:

- **Functional-eval accuracy** (extraction/retrieval/tool-selection scores) — tracked as a known
  small-model capability limit in README §8, not a software defect. Out of scope here; revisiting
  it would need a fresh eval run before any refactor of the extraction path is worth considering.
- **Already-documented, deliberate PoC boundaries** — README §10 ("PoC boundaries") and §11
  ("Recommendations and ideas") already enumerate auth, multi-tenancy, PII/content-safety, GDPR,
  EU AI Act, audit trail, rate limiting, etc. This backlog does not repeat those; every item below
  was newly identified by reading the current source.

Items are grouped by where a reviewer would notice them: production-adjacent robustness (B),
demo/UX quality (C), and code/engineering quality (D). Each numbered item keeps its working label
from triage for traceability. Every item below has since been resolved, skipped, or deliberately
reverted — see the outcome line under each.

---

## 1. Action items — resolved

### Production-adjacent robustness

**B1 — Bound `ChatRequest` field sizes**
`src/app/api/schemas.py`. `thread_id` and `message` were unconstrained `str` fields: an unbounded
`message` flowed straight into the LLM prompt, and an arbitrary `thread_id` flowed straight into a
Redis checkpoint key prefix with no length or format check.
**Resolved:** added pydantic `Field` constraints — `message` (`min_length=1, max_length=500`),
`thread_id` (`min_length=1, max_length=128`, pattern `^[A-Za-z0-9_.:-]+$`, chosen to admit the
UI's UUIDs, the eval/load-test harnesses' `eval-*`/`*-rep*` ids, and plain test ids). The Streamlit
input is additionally capped at 500 characters via `st.chat_input(max_chars=...)` so the limit is
enforced before the user can even type past it. Tests: `test_api.py`.

**B2 — Guard `/admin/ingest` against concurrent rebuilds**
`src/app/rag/ingest/pipeline.py`. `CorpusIngestor.run()` could race with itself: two overlapping
`/admin/ingest` calls could interleave an index drop with an upsert.
**Resolved, simpler than proposed:** a Redis-backed lock was considered but rejected in favour of a
module-level `threading.Lock` (`_INGEST_LOCK`) guarding `run()` — the app runs as a single
`uvicorn` worker (per the design's own constraint, §14), so an in-process lock is sufficient and
avoids adding Redis-coordination complexity for a single-process deployment. A concurrent call
raises `IngestionInProgressError`, mapped to **409** by `/admin/ingest`. Tests:
`rag/ingest/tests/test_run_ingest.py`, `tests/api/test_admin.py`.

**B3 — Resolve the `redis_index | None` contradiction**
Startup already aborted the process when Redis was unreachable, so `redis_index`/`vector_store`
were never actually `None` at runtime — yet routes and `/ready` still carried dead `is None`
branches implying a graceful-degradation behaviour that never existed.
**Resolved via option (a):** `redis_index`/`vector_store` are now non-Optional throughout
(`dependencies.py`, `routes/ingest.py`, `routes/stats.py`, `routes/health.py`, `readiness.py`).
`connect_and_ingest` raises `RuntimeError` directly on `redis.RedisError` instead of returning a
`(None, None)` sentinel. `/ready`'s live `ping()`-based check remains the sole signal for a
mid-flight Redis outage, unchanged.

**B4 — Verify or remove the CORS middleware**
`src/app/main.py` configured `CORSMiddleware` as if a browser called the API cross-origin.
**Resolved:** confirmed the only caller is Streamlit's server-to-server `httpx2` client (never
subject to browser CORS) and the backing `ui_origin` setting was configured nowhere. Removed the
middleware and the unused `Settings.ui_origin` field entirely, and updated the now-stale CORS
mention in `.docs/features/10-streaming-chat-experience-and-thread-controls.en.md`.

**B6 — Make `ollama-pull` resilient to transient failures**
`docker-compose.yml`. The one-shot model-pull job had `restart: "no"` and no retry.
**Resolved:** changed to `restart: on-failure:3` — Compose-native, no custom retry-loop script
needed for a one-shot job.

**B8 — Log exception detail in `langfuse.update_trace`**
**Resolved:** the broad `except Exception` now logs `type(e).__name__: {e}`, matching the pattern
used elsewhere in the codebase (e.g. `structured.py`).

### Demo/UX quality

**C1 — Fix the frozen status widget on mid-stream failure**
**Resolved:** `_consume_stream`'s streaming loop is wrapped in `try/except httpx2.HTTPError`, which
marks the `st.status` widget `state="error"` before re-raising, so it never spins forever.

**C2 — Replace raw exception text with a user-facing message**
**Resolved:** `_call_or_warn` now logs the full exception (`type(e).__name__: {e}`) via a plain
`logging.getLogger(__name__)` (captured by the `ui` container's existing `json-file` logging
driver) and shows only the friendly `error_message` to the user — no exception text or internal
hostnames reach the UI.

**C3 — Surface the existing `degraded` signal in the chat UI**
**Resolved:** added `degraded: bool` to `ChatResponse` (wired through `ResponseBuilder.build_chat`,
so both `/chat` and the streamed `result` event carry it), and the UI renders
`⚠️ This answer may be incomplete or less reliable.` under the answer when set. The
`test_evaluation_only_fields_never_leak_into_the_public_chat_contract` test's assumption that
`degraded` was evaluation-only was updated accordingly — this is a deliberate contract change, not
a regression.

### Code/engineering quality

**D1 — Extract the repeated "first matching rule" lookup helper**
Found 10 occurrences in `rule_checker.py` (more than the initially estimated 8), plus 2 in
`calculator.py`.
**Resolved:** added `first_matching(rules, predicate)` to a new `src/app/rules/lookup.py`; all 10
sites in `rule_checker.py` and both sites in `calculator.py` (`_first_rule_with`,
`_meal_limit_rule`) now call it. Public behaviour of both classes is unchanged. Tests:
`rules/tests/test_lookup.py`.

**D3 — Add unit tests for the Streamlit SSE client, or narrow the coverage exclusion**
**Resolved:** extracted the SSE `event:`/`data:` line-parsing into a pure `parse_sse_lines()`
function in `src/app/api/schemas.py`, placed next to `StreamEvent.to_sse()` since it is that
method's exact inverse (and `ui.py` already imported from this module). `ui.py`'s
`ChatApiClient.stream_turn` now just calls it; `ui.py` itself stays in the coverage exclusion
(Streamlit rendering needs a live session to test meaningfully), but `parse_sse_lines` is no longer
excluded — added a new co-located `src/app/api/tests/` (`test_schemas.py`), and registered it in
`pyproject.toml`'s coverage `omit` list and both `sonar-project.properties*` files' `sonar.tests`.

**D4 — Unify the two independent "which steps happened this turn" implementations**
**Skipped, by decision.** Investigation found this is not simple accidental duplication: the
streaming path (`streaming.py`) knows the identity of the LangGraph node that just finished, which
the blocking path (`responses.py`, working from message history alone) cannot recover, because
`classify_intent`/`extract_information` leave no `ToolMessage`/`AIMessage` trace. A real merge would
require adding a "which nodes ran" field to state — bigger and riskier than the original estimate,
and touching the most demo-visible code path. Left as two implementations sharing the same
`STEP_LABELS`/`step_label`/`ALWAYS_FIRST_STEPS`/`DECISION_FINAL_STEPS` vocabulary; an existing
parity test (`test_chat_stream.py::test_streamed_result_matches_the_blocking_endpoint`) guards one
scenario against drift.

**D5 — Give malformed `.env` values an actionable error**
**Resolved:** added `SettingsError(RuntimeError)`; `get_settings()` catches
`pydantic.ValidationError` and re-raises `SettingsError` with the wrapped message, following the
same pattern already used by `rules/loader.py`'s `RuleCatalogueError`. `lru_cache` behaviour
unchanged. Tests: `tests/test_settings.py`.

**D6 — Extract a shared "best-effort call with fallback" helper**
**Implemented, then reverted by decision.** A `best_effort`/`best_effort_async` helper
(`src/app/logging/best_effort.py`) was built and wired into all four call sites
(`langfuse_prompt_library.py`, `integrations/langfuse.py` ×2, `structured.py` ×2), with passing
tests throughout. Reverted at explicit request — the abstraction was judged not worth it. All four
sites are back to their original per-call `try/except`, with B8's exception-detail logging on
`update_trace` preserved.

**D8 — Add unit tests for `docx_converter.py`/`docx_loader.py`**
**Resolved:** added `rag/ingest/tests/test_docx_converter.py` (empty document, table-cell newline
collapsing, header-only table, malformed-`.docx` raising `PackageNotFoundError`) and
`test_docx_loader.py` (`DocxMarkdownLoader.lazy_load()`'s sorted multi-file ordering and metadata,
empty corpus directory). No behaviour changed — these document and lock in the existing (including
currently-unhandled-malformed-input) behaviour.

---

## 2. Noted, not actioned

These are real, but judged lower-value relative to their cost, or already sufficiently mitigated —
kept here as a record rather than expanded into action plans.

- **B7 — No request/thread correlation id in structured logs.** `src/app/logging/config.py`
  emits `timestamp/level/service/logger/event` but no per-request or per-thread id, so correlating
  the several log lines belonging to one `/chat` turn currently relies on timestamp proximity
  rather than a grep-able id.
- **C4 — No way to resume or browse a previous conversation thread.** `thread_id` lives only in
  `st.session_state` (`src/app/ui.py`), so refreshing the page or opening a new tab mints a
  fresh thread even though the server-side checkpoint survives 24h.
- **C5 — No stop/cancel control for a running turn.** Once submitted, there is no way to abort a
  slow generation from the UI short of waiting out `STREAM_TIMEOUT_SECONDS` (`src/app/ui.py`).

## 3. Considered, excluded

Identified during triage but deliberately left out of this backlog:

- **B5** — Docker healthcheck interval (5 min) on Redis/Ollama. Left as is.
- **C6** — Sidebar re-fetching `/admin/stats` on every Streamlit rerun.
- **C7** — No dedicated UI affordance for the two-branch distance-clarification answer.
- **C8** — Silent retry/backoff during model calls with no streamed feedback.
- **D2** — No static type checking (mypy/pyright) in CI.
- **D7** — `RuleChecker`'s `"-"` sentinel rule id is undocumented/unnamed.
