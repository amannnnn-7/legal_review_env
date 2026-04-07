# Technical Notes

## Architecture

The environment is built as a single OpenEnv-compatible FastAPI app with:

- Pydantic V2 models for action, observation, state, tasks, grading, and baseline responses.
- A stateful `LegalReviewEnvironment` wrapped by `openenv.core.env_server.http_server.HTTPEnvServer`.
- A shared singleton environment instance so `/reset`, `/state`, `/grader`, and `/baseline` all observe the same live contract state that `/ws` mutates.

## Dataset Handling

The public Hugging Face repo `TheAtticusProject/cuad` does not currently expose the prompt's claimed `test` table directly. Instead, the environment downloads the canonical annotation bundle `CUAD_v1/CUAD_v1.json` from the same dataset repo and parses it into:

- one contract-wide text context per contract;
- exact clause spans by label;
- deterministic task-specific held-out pools.

The held-out pools are stable-hash subsets of the candidate contracts for each difficulty.

## Deterministic Tasks

### Easy

- Labels: `Effective Date`, `Governing Law`
- Ground truth: exact CUAD answer spans
- Score: exact-match average across the two labels

### Medium

- Label: `Non-Compete`
- Ground truth: any non-compete span that is indefinite or longer than 12 months
- Score: deterministic span-level F1 with greedy exact-or-containing matching

### Hard

CUAD does not label indemnification, so the hard task uses a non-compete clause block instead.

- Label: `Non-Compete`
- Ground truth target: deterministic rewrite of the editable block where durations above 12 months become `12 months`
- Score: target similarity minus edit-distance penalty from the original block

## Reward Shaping

Per-step reward combines:

- score improvement from the deterministic grader preview;
- `+0.15` for loading the correct playbook before editing;
- `+0.15` for successful task-relevant clause reads;
- `+0.10` for recovering from the previous validation error;
- `-0.10` for environment-level validation errors;
- `-0.20` for hallucinated `FlagRisk` or `ApplyRedline` spans.

Transport-level JSON and Pydantic errors are still handled upstream by OpenEnv/FastAPI with HTTP 422 or WebSocket error messages.