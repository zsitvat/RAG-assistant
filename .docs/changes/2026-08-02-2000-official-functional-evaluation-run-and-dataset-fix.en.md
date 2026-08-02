# 2026-08-02 20:00 — Official functional evaluation run; one dataset authoring bug found and fixed

## What changed

- Fixed `eval/dataset.json`'s `general-01` case: `expected_doc_ids` was `["00", "07"]`
  (doc `00` = *Document Index and Glossary*), changed to `["01", "06", "07"]` (doc `01` = *General
  Expense Reimbursement Policy*, the actual best-matching document; `06` = *Receipt and Approval
  Requirements*).
- Ran the official functional evaluation (`python -m eval.run_eval`) against the real
  `qwen2.5:7b-instruct-q4_K_M` model. Report: `.docs/eval/functional-20260802-102625.md`. Langfuse
  run: <https://cloud.langfuse.com/project/cms7txjr50008ad0jxqp7myo0/datasets/cmsbnbrox02a9ad0h25lj6w0b/runs/8ae3032e-758d-407b-8990-045ea29d1e19>.

## Why / what was found

`general-01`'s expected documents were assigned from `rules.yaml`'s `category: general` tag alone,
without checking the actual corpus file titles. Doc `00` (`00_Document_Index_and_Glossary.docx`) is
a glossary, not policy content, so it was never a plausible retrieval target — confirmed by
requesting the same question directly against `/admin/eval` and inspecting `retrieved_doc_ids`
(`["01", "06"]`, both legitimate). This was a dataset-authoring mistake, not a retrieval bug.

The completed run's real, honest numbers are lower than the design's aspirational framing might
suggest (`slot_accuracy` 11.8%, `retrieval_hit_at_4`/`citation_accuracy` 0%, `tool_selection_accuracy`
15%, `outcome_accuracy` 25%, `classification_accuracy` 90%). Spot-checking individual `/admin/eval`
calls against cases with low scores shows a consistent, genuine root cause, not a software defect:
the 7B model frequently fails to extract an exact canonical enum value the prompt asks for (e.g.
`expense_type: "transport"` instead of the required `"pass"`) or fails to infer an implied numeric
zero (e.g. "no alcohol" → `non_reimbursable_amount: 0`), so `route_after_extraction` correctly (and
deterministically) routes the turn to `ask_clarification` — before the agent ever reaches
`agent_step`/`search_policies`. Retrieval, citation and tool-selection scores are structurally zero
for any turn that never reaches the tool-calling loop, which is most of them. This is the evaluation
harness doing exactly its job: surfacing a genuine capability limit of a small local model under
this design, not a bug to route around by loosening the dataset. It is retained as real evidence
rather than re-run with an easier dataset.

The `general-01` fix was **not** re-verified with a second full 20-case run (each real run takes
~15–20 minutes against local CPU-served Ollama) — it was verified directly by re-issuing that one
case's question against the live `/admin/eval` endpoint and confirming `retrieved_doc_ids` now
satisfies the corrected `expected_doc_ids`. A future re-run would show `retrieval_hit_at_4` improve
by exactly this one case.
