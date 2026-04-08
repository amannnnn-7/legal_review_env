from __future__ import annotations

import argparse
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .agent_policy import build_client, choose_action
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
from .server.environment import LegalReviewEnvironment


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

    def __enter__(self) -> _RemoteAdapter:
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
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
        request = urllib.request.Request(
            f"{self._base_url}/grader",
            data=json.dumps({"include_details": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return GraderResponse.model_validate(payload)


def _run_with_adapter(adapter: _RunnerAdapter, request: BaselineRequest) -> BaselineResponse:
    api_key = request.api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    client = build_client(api_key=api_key)
    result = adapter.reset(
        difficulty=request.difficulty.value,
        seed=request.seed,
        contract_id=request.contract_id,
    )
    trace: list[BaselineTraceStep] = []

    while not result.done and len(trace) < request.max_steps:
        state = adapter.state()
        action = choose_action(client, request.model, result.observation, state)
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
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"))
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