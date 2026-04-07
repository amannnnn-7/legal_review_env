from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

import httpx
from openai import OpenAI

from .client import LegalReviewEnvClient
from .models import (
    BaselineRequest,
    BaselineResponse,
    BaselineTraceStep,
    GraderResponse,
    LegalReviewAction,
    LegalReviewObservation,
    LegalReviewState,
    TaskDifficulty,
)
from .scoring import rewrite_non_compete_block
from .server.environment import LegalReviewEnvironment


SYSTEM_PROMPT = """You are a careful junior lawyer agent.
Operate only through the provided action schema.
Return exactly one JSON object matching LegalReviewAction.
Never invent text spans. Only use exact text already returned by the environment.
When the task is complete, set metadata.finish to true on your final action.
"""


@dataclass
class _RunnerStep:
    observation: LegalReviewObservation
    reward: float | int | bool | None
    done: bool


class _RunnerAdapter(Protocol):
    def reset(self, **kwargs: object) -> _RunnerStep: ...

    def step(self, action: LegalReviewAction) -> _RunnerStep: ...

    def state(self) -> LegalReviewState: ...

    def grade(self) -> GraderResponse: ...


class _LocalAdapter:
    def __init__(self, env: LegalReviewEnvironment):
        self._env = env

    def reset(self, **kwargs: object) -> _RunnerStep:
        observation = self._env.reset(**kwargs)
        return _RunnerStep(observation=observation, reward=observation.reward, done=observation.done)

    def step(self, action: LegalReviewAction) -> _RunnerStep:
        observation = self._env.step(action)
        return _RunnerStep(observation=observation, reward=observation.reward, done=observation.done)

    def state(self) -> LegalReviewState:
        return self._env.state

    def grade(self) -> GraderResponse:
        report = self._env.grade()
        return GraderResponse(
            score=report.score,
            metric=report.metric,
            difficulty=self._env.state.difficulty,
            done=self._env.state.complete,
            details=report.details,
        )


class _RemoteAdapter:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client = LegalReviewEnvClient(base_url=base_url).sync()
        self._http = httpx.Client(base_url=self._base_url, timeout=60.0)

    def __enter__(self) -> _RemoteAdapter:
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._http.close()
        self._client.__exit__(exc_type, exc, tb)

    def reset(self, **kwargs: object) -> _RunnerStep:
        result = self._client.reset(**kwargs)
        return _RunnerStep(observation=result.observation, reward=result.reward, done=result.done)

    def step(self, action: LegalReviewAction) -> _RunnerStep:
        result = self._client.step(action)
        return _RunnerStep(observation=result.observation, reward=result.reward, done=result.done)

    def state(self) -> LegalReviewState:
        return self._client.state()

    def grade(self) -> GraderResponse:
        response = self._http.post("/grader", json={"include_details": True})
        response.raise_for_status()
        return GraderResponse.model_validate(response.json())


def _extract_json(text: str) -> dict[str, object]:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def _fallback_action(observation: LegalReviewObservation, state: LegalReviewState) -> LegalReviewAction:
    difficulty = observation.difficulty or TaskDifficulty.EASY

    if difficulty == TaskDifficulty.EASY:
        if "contract-basics" not in state.playbook_queries:
            return LegalReviewAction(action_type="search_playbook", topic="contract-basics")
        if "Effective Date" not in state.extracted_clauses:
            return LegalReviewAction(action_type="read_clause", category="Effective Date")
        return LegalReviewAction(
            action_type="read_clause",
            category="Governing Law",
            metadata={"finish": True},
        )

    if difficulty == TaskDifficulty.MEDIUM:
        if "non-compete" not in state.playbook_queries:
            return LegalReviewAction(action_type="search_playbook", topic="non-compete")
        if "Non-Compete" not in state.extracted_clauses:
            return LegalReviewAction(action_type="read_clause", category="Non-Compete")

        flagged = {risk.text_span for risk in state.flagged_risks}
        remaining = [span for span in state.extracted_clauses.get("Non-Compete", []) if span not in flagged]
        if remaining:
            return LegalReviewAction(
                action_type="flag_risk",
                text_span=remaining[0],
                rationale="The non-compete is missing an explicit end date or exceeds the 12 month playbook limit.",
                metadata={"finish": len(remaining) == 1},
            )

        return LegalReviewAction(action_type="search_playbook", topic="non-compete", metadata={"finish": True})

    if "non-compete" not in state.playbook_queries:
        return LegalReviewAction(action_type="search_playbook", topic="non-compete")
    if "Non-Compete" not in state.extracted_clauses:
        return LegalReviewAction(action_type="read_clause", category="Non-Compete")

    block = state.extracted_clauses.get("Non-Compete", [""])[0]
    target = rewrite_non_compete_block(block)
    return LegalReviewAction(
        action_type="apply_redline",
        original_text=block,
        replacement_text=target,
        metadata={"finish": True},
    )


def _prompt_for_observation(observation: LegalReviewObservation, state: LegalReviewState) -> str:
    payload = {
        "contract_id": observation.contract_id,
        "difficulty": observation.difficulty.value if observation.difficulty else None,
        "task": observation.task.model_dump() if observation.task else None,
        "message": observation.message,
        "clause_category": observation.clause_category,
        "clause_matches": observation.clause_matches,
        "validation_errors": observation.validation_errors,
        "flagged_risks": [item.model_dump() for item in observation.flagged_risks],
        "redlines": [item.model_dump() for item in observation.redlines],
        "score_preview": observation.score_preview,
        "state": state.model_dump(),
        "action_schema": LegalReviewAction.model_json_schema(),
    }
    return json.dumps(payload, indent=2)


def _choose_action(
    client: OpenAI,
    request: BaselineRequest,
    observation: LegalReviewObservation,
    state: LegalReviewState,
) -> LegalReviewAction:
    response = client.responses.create(
        model=request.model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt_for_observation(observation, state)},
        ],
        temperature=0,
    )
    try:
        payload = _extract_json(response.output_text)
        return LegalReviewAction.model_validate(payload)
    except Exception:
        return _fallback_action(observation, state)


def _run_with_adapter(adapter: _RunnerAdapter, request: BaselineRequest) -> BaselineResponse:
    api_key = request.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the baseline runner")

    client = OpenAI(api_key=api_key)
    result = adapter.reset(
        difficulty=request.difficulty.value,
        seed=request.seed,
        contract_id=request.contract_id,
    )
    trace: list[BaselineTraceStep] = []

    while not result.done and len(trace) < request.max_steps:
        state = adapter.state()
        action = _choose_action(client, request, result.observation, state)
        result = adapter.step(action)
        trace.append(
            BaselineTraceStep(
                step=len(trace) + 1,
                action=action.model_dump(exclude_none=True),
                reward=result.reward,
                done=result.done,
                message=result.observation.message,
                score_preview=result.observation.score_preview,
            )
        )

    final_state = adapter.state()
    grade = adapter.grade()
    return BaselineResponse(
        difficulty=request.difficulty,
        score=grade.score,
        metric=grade.metric,
        steps_taken=len(trace),
        contract_id=final_state.contract_id,
        contract_title=final_state.contract_title,
        trace=trace,
    )


def run_baseline_local(env: LegalReviewEnvironment, request: BaselineRequest) -> BaselineResponse:
    return _run_with_adapter(_LocalAdapter(env), request)


def run_baseline_remote(request: BaselineRequest) -> BaselineResponse:
    with _RemoteAdapter(request.base_url) as adapter:
        return _run_with_adapter(adapter, request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenAI baseline agent against legal_review_env.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--contract-id", default=None)
    args = parser.parse_args()

    request = BaselineRequest(
        base_url=args.base_url,
        difficulty=TaskDifficulty(args.difficulty),
        model=args.model,
        max_steps=args.max_steps,
        seed=args.seed,
        contract_id=args.contract_id,
    )
    response = run_baseline_remote(request)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()