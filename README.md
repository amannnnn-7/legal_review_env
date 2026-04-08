---
title: legal_review_env
emoji: scales
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
short_description: Stateful legal contract review, risk abstraction, and redlining environment built on CUAD.
tags:
	- openenv
	- legal
	- contracts
	- reinforcement-learning
datasets:
	- TheAtticusProject/cuad
suggested_hardware: cpu-basic
startup_duration_timeout: 30m
preload_from_hub:
	- TheAtticusProject/cuad CUAD_v1/CUAD_v1.json
---

## legal_review_env

`legal_review_env` is a stateful OpenEnv environment for contract abstraction, risk triage, and surgical redlining against CUAD.

It targets a real workflow junior commercial lawyers and legal operations teams actually perform: finding clause facts, identifying risky restrictions, and proposing narrow contract edits that preserve document integrity.

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

## Action Space

The environment accepts a typed `LegalReviewAction` with exactly one of these operations:

- `search_playbook(topic)`
- `read_clause(category)`
- `flag_risk(text_span, rationale)`
- `apply_redline(original_text, replacement_text)`

Unknown fields are rejected by Pydantic, and hallucinated spans are rejected if they are not exact substrings of the live document state.

## Observation Space

Each `LegalReviewObservation` returns:

- current task metadata and difficulty
- contract id and title
- retrieved clause matches
- loaded playbook details
- flagged risks and applied redlines so far
- validation errors for the previous action
- current document version and optional updated document text
- dense `reward`, `done`, and `score_preview` values

The internal `LegalReviewState` also tracks the mutable live contract text, extracted clauses, redline history, and playbook queries.

## Reward Design

Rewards are dense rather than terminal-only.

- `+0.15` for loading the correct playbook before editing
- `+0.15` for retrieving a task-relevant clause
- `+0.20` for correctly flagging a violating span
- positive shaping equal to score improvement after each action
- `+0.10` for recovering immediately after a validation error
- `-0.10` for invalid topics/categories
- `-0.20` for hallucinated risk flags or redlines

This gives useful training signal across the trajectory while penalizing loops and unsafe actions.

## Baseline Scores

Reference scores from the submission `inference.py` with fixed seed `7` and deterministic fallback guardrails are:

| Task | Reference Score |
|------|-----------------|
| Easy | 1.000 |
| Medium | 1.000 |
| Hard | 0.992 |

These are deterministic for the built-in fallback path. When a real model endpoint is available through `API_BASE_URL` and `HF_TOKEN`, the script still uses the OpenAI client for action selection, with the same validation guardrails.

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
./scripts/validate-submission.sh https://YOUR-SPACE.hf.space .
```

## Baseline

The repository contains two baseline entrypoints:

- `uv run legal-review-env-baseline` for local JSON output
- `python inference.py` for the hackathon submission protocol

The submission script uses the OpenAI client and the required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME` or `LEGAL_REVIEW_BASE_URL`

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="..."
export LOCAL_IMAGE_NAME="legal-review-env:latest"

python inference.py
```

The script emits strict single-line logs in the required format:

- `[START]`
- `[STEP]`
- `[END]`

One episode is run for each of the three tasks in order: easy, medium, hard.

## Docker

```bash
docker build -t legal-review-env:latest .
docker run -p 8000:8000 legal-review-env:latest
```

The included `Dockerfile` inherits from `openenv-base:latest` to stay aligned with the OpenEnv hackathon deployment model.

## Hugging Face Spaces Deployment

This repository is configured to be pushed as a Docker-based Hugging Face Space.

Typical flow:

```bash
openenv build
openenv push
```

After deployment, verify the Space with:

```bash
curl -X POST https://YOUR-SPACE.hf.space/reset -H "Content-Type: application/json" -d '{}'
```
