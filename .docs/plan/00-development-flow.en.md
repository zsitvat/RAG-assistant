# Development Flow

This document defines the end-to-end workflow for planning, implementing and reviewing the
project. It is the process reference: each stage has an output and an exit gate, and development
does not start until the design and task breakdown have both been challenged.

```mermaid
flowchart LR
    A[1 Topic ideation] --> B[2 High-level plan]
    B --> C[3 Technical design]
    C --> D[4 Design grilling]
    D --> E[5 Task breakdown]
    E --> F[6 Task grilling]
    F --> G[7 Ordered development]
    G --> H[8 Automated quality gates]
    H --> I[9 AI review]
    I --> J[10 Human review]
```

## Working principles

- Inspect repository facts before asking a person to decide.
- Record decisions in the document that owns them; planning does not require a separate changelog.
- Keep the PoC proportional to the assignment and explicitly label production-only proposals.
- Give every implementation task an observable acceptance criterion and relevant verification.
- A later stage may send the work back to the stage that owns a discovered problem.
- Automated checks support review; they do not replace AI or human judgement.

## 1. Topic ideation and rationale

Define the real user problem before choosing the architecture:

- who has the problem and what they are trying to achieve;
- why the problem is relevant;
- what current friction the assistant removes;
- why document-grounded answers are necessary;
- why agentic RAG adds value over plain RAG or a deterministic form.

**Output:** the topic, rationale, user need and explicit scope in
[`01-idea-plan.en.md`](01-idea-plan.en.md).

**Exit gate:** the use case is relevant, narrow enough for a PoC, compatible with the assignment and
clearly benefits from retrieval plus multi-step decision-making.

## 2. High-level planning

Describe the solution without committing to implementation details:

- main user journeys, including clarification and out-of-scope handling;
- high-level agent workflow and RAG subgraph;
- expected tools, data sources, UI and runtime shape;
- scope boundaries;
- functional evaluation and load-test outline.

**Output:** the solution outline and expected outcome in
[`01-idea-plan.en.md`](01-idea-plan.en.md).

**Exit gate:** every assignment requirement has a credible high-level answer, and unnecessary
features have been removed.

## 3. Technical design

Turn the high-level plan into implementable contracts:

- repository structure and dependency boundaries;
- graph nodes, state, routing and turn isolation;
- tool schemas and deterministic rules;
- ingestion, retrieval and citation mapping;
- FastAPI, Streamlit and streaming contracts;
- configuration, logging, Langfuse and containerisation;
- functional evaluation, load testing, failure handling and PoC boundaries.

Every material choice includes its reason and trade-off. The design distinguishes implemented PoC
behaviour from production recommendations.

**Output:** [`02-technical-design.en.md`](02-technical-design.en.md).

**Exit gate:** another developer could implement the system without inventing missing contracts.

## 4. Technical-design grilling

Stress-test the design before converting it into tasks. The grilling is a decision interview, not a
batch checklist:

1. inspect facts available in the repository instead of asking about them;
2. ask one decision question at a time;
3. include a recommended answer and its trade-offs with every question;
4. resolve prerequisite decisions before dependent ones;
5. update the owning plan after each accepted decision;
6. continue until both sides explicitly agree that the design is understood and no material
   decision remains unresolved.

Challenge at least requirement coverage, unnecessary complexity, state transitions, failure modes,
data consistency, security boundaries, reproducibility, evaluation validity and hardware
feasibility.

**Output:** resolved decisions incorporated directly into the technical design.

**Exit gate:** shared understanding is explicitly confirmed; open implementation choices are either
resolved or clearly marked as empirical decisions with a planned measurement.

## 5. Task breakdown

Convert the accepted design into ordered, reviewable implementation tasks. Each task records:

- objective and owning module;
- dependencies on earlier tasks;
- concrete files or interfaces in scope;
- acceptance criteria;
- focused tests and verification commands;
- explicit exclusions.

Prefer vertical slices that leave the repository runnable. Infrastructure foundations may come
first, but large horizontal tasks such as “build the backend” must be split.

**Output:** `.docs/plan/03-implementation-tasks.en.md`, created when implementation planning begins.

**Exit gate:** every technical-design requirement maps to at least one task, every task maps back to
a requirement, and the dependency order is executable.

## 6. Task-breakdown grilling

Challenge the task list using the same one-question-at-a-time method as the design grilling. Focus
on:

- missing work and hidden dependencies;
- tasks that are too large or cannot be verified independently;
- incorrect ordering and unsafe parallel work;
- vague acceptance criteria;
- missing negative, integration or regression tests;
- scope that is not required by the accepted design.

For every question, propose the recommended correction. Apply accepted answers directly to the task
plan and re-check requirement coverage after structural changes.

**Output:** an implementation-ready, dependency-ordered task plan.

**Exit gate:** shared understanding is explicitly confirmed and the next task is unambiguous.

## 7. Ordered development

Implement tasks in dependency order. For each task:

1. re-read its acceptance criteria and the relevant technical-design section;
2. inspect the current workspace and preserve unrelated user changes;
3. implement the smallest complete slice;
4. add or update focused tests;
5. run the narrowest relevant checks, then the affected integration checks;
6. update the task status only when its acceptance criteria pass.
7. human review per task

If implementation reveals a faulty design assumption, return to stage 3 or 5 and correct the owning
document before continuing. Do not silently diverge from the agreed design.

**Output:** working source code and tests, completed one verified task at a time.

**Exit gate:** all planned tasks meet their acceptance criteria and the clean-clone runtime path is
ready for full validation.

## 8. Automated quality gates and tool-assisted review

Run the repository-defined checks over the complete implementation:

| Check | Command | Gate |
| --- | --- | --- |
| Ruff lint | `ruff check .` | no lint errors |
| Ruff formatting | `ruff format --check .` | no formatting drift |
| Bandit | `bandit -c pyproject.toml -r app` | no unaccepted security findings |
| Tests and coverage | `pytest --cov=app --cov-report=term-missing --cov-report=xml` | all tests pass and coverage is reported |
| Sonar | `make sonar` (`uv run pysonar`) | configured quality gate passes |

Also build the containers and exercise the clean-clone startup path. Tool findings must be fixed or
documented with a concrete reason; disabling a rule only to make the gate green is not resolution.

**Output:** reproducible check results and generated coverage/Sonar inputs.

**Exit gate:** every automated gate passes.

## 9. AI review

Ask an AI reviewer to inspect the complete change against two independent axes:

- **Specification:** assignment requirements, accepted plans, API contracts, evaluation method and
  PoC boundaries.
- **Engineering quality:** correctness, maintainability, security, failure behaviour, tests,
  observability and reproducibility.

The review should cite concrete files and lines, rank findings by severity, distinguish defects from
optional improvements and avoid changing code before findings are accepted. After fixes, rerun the
affected automated gates and request a focused re-review.

**Output:** resolved or explicitly rejected AI findings with reasons.

**Exit gate:** no unresolved high-severity finding and no known mismatch with the assignment or
technical design.

## 10. Human review

The human reviewer makes the final decision. The review includes:

- reading the README and architecture explanation;
- starting the system from a clean clone with the documented command;
- exercising normal, clarification, unsupported and failure paths in Streamlit;
- checking the streamed source/step summary and selected Langfuse traces;
- reviewing the functional-evaluation and load-test reports;
- confirming the identified bottleneck and optimisation proposals;
- inspecting the final automated and AI-review findings.

Human feedback returns to the stage that owns the issue, followed by the relevant checks and
re-review.

**Output:** explicit human approval.

**Exit gate:** the reviewer confirms that the project is understandable, reproducible, satisfies the
assignment and is ready to submit.
