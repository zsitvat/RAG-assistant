# 2026-08-02 19:00 — Fix crash-free but silently-dropped Langfuse scores for not-applicable metrics

## What changed

- `eval/metrics.py`'s `slot_accuracy`, `retrieval_hit_at_4` and `citation_accuracy` now return `[]`
  instead of `Evaluation(value=None, ...)` when a case has nothing to score (no `expected_slots`, no
  `expected_doc_ids`).

## Why

Discovered while running the official functional evaluation against the real Langfuse Cloud project:
the SDK's `create_score()` validates a score's `value` as a required float/string/bool — `None`
fails that validation, so every "not applicable" evaluation logged a `ValidationError` (caught
internally by the SDK, so the experiment itself didn't fail, but the score was silently never
recorded in Langfuse). Returning `[]` is Langfuse's documented way for an evaluator to skip scoring
an item entirely, which is what "not applicable" actually means here — it also matches
`eval/report.py`'s own aggregation, which already excluded `None`-valued scores from a metric's
denominator, so the local report's numbers are unaffected.
