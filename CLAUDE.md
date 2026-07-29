# RAG-assistant

Agentic RAG prototype: corporate expense reimbursement and employee benefits assistant
(LangGraph + local LLM + Streamlit). Assignment brief: `.docs/RAG- Feladatkiírás- Medior (1).pdf`.

## Documentation language

- Write all documentation in English — plans, README, code comments, docstrings, and doc file
  names — regardless of the conversation language.
- Produce a translated (e.g. Hungarian) version only when explicitly asked, as a sibling file with
  a `.hu.md` suffix, cross-linked with the English original.
- The RAG knowledge base lives in `.docs/sources/<lang>/` (`en` + `hu` mirrors, same `doc_id`s) with one
  Redis index per language; `ACTIVE_LANG` in the config selects which index and answer language is used.
  Rule ids, enum values and `rules.yaml` stay English and language-independent.

## Data provenance

The policy corpus under `.docs/sources/` and the hand-authored PoC `rules.yaml` describe a fictional
company. Never present them as a real company's policy or as tax or legal advice.

## Key documents

- `.docs/plan/01-idea-plan.en.md` — problem, topic justification, scope
- `.docs/plan/02-technical-design.en.md` — implementation reference
