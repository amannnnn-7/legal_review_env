## legal_review_env

`legal_review_env` is a stateful OpenEnv environment for contract abstraction, risk triage, and surgical redlining against CUAD.

It is designed around a "junior lawyer simulator" workflow:

- `SearchPlaybook` returns deterministic internal review rules.
- `ReadClause` retrieves exact CUAD-backed clause spans.
- `FlagRisk` marks risky spans for F1-based grading.
- `ApplyRedline` mutates the live contract document and rejects hallucinated edits when the original text is not found exactly.

## Dataset Notes

The published Hugging Face dataset `TheAtticusProject/cuad` currently exposes the raw CUAD asset bundle rather than a ready-made `test` split table. This project uses the canonical annotation file bundled in that dataset repo at `CUAD_v1/CUAD_v1.json`.

To preserve deterministic evaluation semantics:

- contracts are parsed from `CUAD_v1.json` into exact span annotations;
- task-specific held-out pools are derived deterministically by stable hashing;
- the environment defaults to those held-out pools on reset.

This preserves the handover intent even though the public HF dataset layout does not exactly match the prompt.

## Tasks

### Easy: Clause Abstraction

Extract the exact `Effective Date` and `Governing Law` spans from the active contract.

Metric: exact-match average across both labels.

### Medium: Risk Triage

Identify every `Non-Compete` span that violates the playbook rule:

- the restriction must be explicitly time-bounded; and
- the restriction cannot exceed 12 months.

Metric: deterministic span-level F1.

### Hard: Contract Redlining

CUAD does not natively label `Indemnification`, so the hard task uses `Non-Compete` clause redlining instead. The environment selects a multi-paragraph non-compete block, derives a deterministic house-style target by reducing overlong durations to `12 months`, and scores the applied edit with similarity minus edit-distance penalty.

Metric:

`score = sim(applied, target) - lambda * edit_distance(original, applied)`

## Quick Start

```bash
cd /home/t-apaliwal/legal_review_env
uv sync
uv run legal-review-env-prefetch
uv run legal-review-env
```

Development server:

```bash
uvicorn legal_review_env.server.app:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /tasks`
- `POST /grader`
- `POST /baseline`
- `WS /ws`

## Validation

```bash
uv run openenv validate -v
```

## Baseline

The baseline runner uses the `openai` client with `gpt-4o-mini` by default.

```bash
export OPENAI_API_KEY=...
uv run legal-review-env-baseline --base-url http://127.0.0.1:8000 --difficulty easy
```

## Docker

```bash
docker build -t legal-review-env:latest .
docker run -p 8000:8000 legal-review-env:latest
```

The included `Dockerfile` inherits from `openenv-base:latest` to stay aligned with the OpenEnv hackathon deployment model.
