from __future__ import annotations

import asyncio

import uvicorn
from fastapi import Body, FastAPI
from openenv.core.env_server.http_server import HTTPEnvServer

from ..baseline import run_baseline_local
from ..cuad import get_default_repository
from ..models import (
    ActionKind,
    ActionSchemaSummary,
    ApplyRedlineArgs,
    BaselineRequest,
    BaselineResponse,
    FlagRiskArgs,
    GraderRequest,
    GraderResponse,
    LegalReviewAction,
    LegalReviewObservation,
    ReadClauseArgs,
    SearchPlaybookArgs,
    TaskDifficulty,
    TaskInfo,
    TasksResponse,
)
from .environment import LegalReviewEnvironment


class _SharedEnvironmentFactory:
    def __init__(self) -> None:
        self._env = LegalReviewEnvironment(repository=get_default_repository())

    def __call__(self) -> LegalReviewEnvironment:
        return self._env

    @property
    def env(self) -> LegalReviewEnvironment:
        return self._env


shared_environment = _SharedEnvironmentFactory()
server = HTTPEnvServer(
    shared_environment,
    LegalReviewAction,
    LegalReviewObservation,
    max_concurrent_envs=1,
)
app = FastAPI(title="legal_review_env", version="0.1.0")
server.register_routes(app)


def _task_catalog() -> TasksResponse:
    actions = [
        ActionSchemaSummary(
            action_type=ActionKind.SEARCH_PLAYBOOK,
            description="Load a deterministic internal playbook for a clause topic.",
            json_schema=SearchPlaybookArgs.model_json_schema(),
        ),
        ActionSchemaSummary(
            action_type=ActionKind.READ_CLAUSE,
            description="Retrieve exact CUAD-backed spans for a clause category.",
            json_schema=ReadClauseArgs.model_json_schema(),
        ),
        ActionSchemaSummary(
            action_type=ActionKind.FLAG_RISK,
            description="Flag an exact contract span as non-compliant with the active playbook.",
            json_schema=FlagRiskArgs.model_json_schema(),
        ),
        ActionSchemaSummary(
            action_type=ActionKind.APPLY_REDLINE,
            description="Replace an exact live-document substring with new text.",
            json_schema=ApplyRedlineArgs.model_json_schema(),
        ),
    ]
    return TasksResponse(
        tasks=[
            TaskInfo(
                task_id="easy-clause-abstraction",
                difficulty=TaskDifficulty.EASY,
                objective="Extract the exact Effective Date and Governing Law spans.",
                metric="exact_match_average",
                playbook_topic="contract-basics",
                target_categories=["Effective Date", "Governing Law"],
                notes=[
                    "A perfect score requires exact extractive matches for both categories.",
                    "Use action.metadata.finish=true on the last action to terminate early.",
                ],
                actions=actions,
            ),
            TaskInfo(
                task_id="medium-risk-triage",
                difficulty=TaskDifficulty.MEDIUM,
                objective="Flag every Non-Compete span that is indefinite or longer than 12 months.",
                metric="span_f1",
                playbook_topic="non-compete",
                target_categories=["Non-Compete"],
                notes=[
                    "Ground truth is derived deterministically from the CUAD Non-Compete annotations.",
                    "Only exact substrings of the live document are accepted by FlagRisk.",
                ],
                actions=actions,
            ),
            TaskInfo(
                task_id="hard-contract-redlining",
                difficulty=TaskDifficulty.HARD,
                objective="Redline a non-compete clause block so the restriction lasts no longer than 12 months.",
                metric="similarity_minus_edit_penalty",
                playbook_topic="non-compete",
                target_categories=["Non-Compete"],
                notes=[
                    "CUAD does not expose an indemnification label, so the hard task uses non-compete redlining instead.",
                    "ReadClause returns the editable clause block for the hard task.",
                ],
                actions=actions,
            ),
        ]
    )


@app.get("/tasks", response_model=TasksResponse, tags=["Tasks"])
def get_tasks() -> TasksResponse:
    return _task_catalog()


@app.post("/grader", response_model=GraderResponse, tags=["Evaluation"])
def grade_current_episode(
    request: GraderRequest = Body(default_factory=GraderRequest),
) -> GraderResponse:
    report = shared_environment.env.grade()
    details = report.details if request.include_details else {}
    return GraderResponse(
        score=report.score,
        metric=report.metric,
        difficulty=shared_environment.env.state.difficulty,
        done=shared_environment.env.state.complete,
        details=details,
    )


@app.post("/baseline", response_model=BaselineResponse, tags=["Evaluation"])
async def run_baseline_endpoint(request: BaselineRequest) -> BaselineResponse:
    return await asyncio.to_thread(run_baseline_local, shared_environment.env, request)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()