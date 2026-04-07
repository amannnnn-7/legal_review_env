from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionKind(str, Enum):
    SEARCH_PLAYBOOK = "search_playbook"
    READ_CLAUSE = "read_clause"
    FLAG_RISK = "flag_risk"
    APPLY_REDLINE = "apply_redline"


class SearchPlaybookArgs(BaseModel):
    topic: str = Field(..., min_length=1, description="Playbook topic or clause family")


class ReadClauseArgs(BaseModel):
    category: str = Field(..., min_length=1, description="CUAD clause category to retrieve")


class FlagRiskArgs(BaseModel):
    text_span: str = Field(..., min_length=1, description="Exact text span being flagged")
    rationale: str = Field(..., min_length=1, description="Why the span violates the playbook")


class ApplyRedlineArgs(BaseModel):
    original_text: str = Field(..., min_length=1, description="Exact source text to replace")
    replacement_text: str = Field(..., min_length=1, description="Replacement text")


class LegalReviewAction(Action):
    action_type: ActionKind = Field(..., description="The environment action to execute")
    topic: str | None = Field(default=None, description="Used by search_playbook")
    category: str | None = Field(default=None, description="Used by read_clause")
    text_span: str | None = Field(default=None, description="Used by flag_risk")
    rationale: str | None = Field(default=None, description="Used by flag_risk")
    original_text: str | None = Field(default=None, description="Used by apply_redline")
    replacement_text: str | None = Field(default=None, description="Used by apply_redline")

    @model_validator(mode="after")
    def validate_payload(self) -> LegalReviewAction:
        validators = {
            ActionKind.SEARCH_PLAYBOOK: SearchPlaybookArgs,
            ActionKind.READ_CLAUSE: ReadClauseArgs,
            ActionKind.FLAG_RISK: FlagRiskArgs,
            ActionKind.APPLY_REDLINE: ApplyRedlineArgs,
        }
        validator = validators[self.action_type]
        validator.model_validate(self.model_dump())
        return self


class PlaybookResponse(BaseModel):
    topic: str
    summary: str
    rules: list[str] = Field(default_factory=list)
    preferred_edit: str | None = None
    examples: list[str] = Field(default_factory=list)


class FlaggedRisk(BaseModel):
    text_span: str
    rationale: str
    matched_ground_truth: bool | None = None


class RedlineRecord(BaseModel):
    original_text: str
    replacement_text: str
    applied: bool
    version: int
    match_in_task_block: bool = False


class TaskDescriptor(BaseModel):
    task_id: str
    difficulty: TaskDifficulty
    objective: str
    grader: str
    playbook_topic: str
    target_categories: list[str] = Field(default_factory=list)
    max_steps: int = 8
    notes: list[str] = Field(default_factory=list)


class GradeReport(BaseModel):
    difficulty: TaskDifficulty
    metric: str
    score: float
    complete: bool
    details: dict[str, Any] = Field(default_factory=dict)


class LegalReviewObservation(Observation):
    status: Literal["ready", "ok", "validation_error", "done"] = "ready"
    message: str = ""
    contract_title: str = ""
    contract_id: str = ""
    difficulty: TaskDifficulty | None = None
    task: TaskDescriptor | None = None
    clause_category: str | None = None
    clause_matches: list[str] = Field(default_factory=list)
    playbook: PlaybookResponse | None = None
    flagged_risks: list[FlaggedRisk] = Field(default_factory=list)
    redlines: list[RedlineRecord] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    document_text: str | None = None
    document_version: int = 0
    score_preview: float = 0.0
    available_action_hints: list[str] = Field(default_factory=list)


class LegalReviewState(State):
    contract_id: str | None = None
    contract_title: str = ""
    difficulty: TaskDifficulty | None = None
    task_id: str | None = None
    document_version: int = 0
    document_text: str = ""
    original_document_text: str = ""
    extracted_clauses: dict[str, list[str]] = Field(default_factory=dict)
    flagged_risks: list[FlaggedRisk] = Field(default_factory=list)
    redlines: list[RedlineRecord] = Field(default_factory=list)
    playbook_queries: list[str] = Field(default_factory=list)
    status: str = "idle"
    complete: bool = False


class ActionSchemaSummary(BaseModel):
    action_type: ActionKind
    description: str
    json_schema: dict[str, Any]


class TaskInfo(BaseModel):
    task_id: str
    difficulty: TaskDifficulty
    objective: str
    metric: str
    playbook_topic: str
    target_categories: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    actions: list[ActionSchemaSummary] = Field(default_factory=list)


class TasksResponse(BaseModel):
    tasks: list[TaskInfo]


class GraderRequest(BaseModel):
    include_details: bool = True


class GraderResponse(BaseModel):
    score: float
    metric: str
    difficulty: TaskDifficulty | None = None
    done: bool
    details: dict[str, Any] = Field(default_factory=dict)


class BaselineRequest(BaseModel):
    base_url: str = Field(default="http://127.0.0.1:8000")
    difficulty: TaskDifficulty = TaskDifficulty.EASY
    model: str = Field(default="gpt-4o-mini")
    max_steps: int = Field(default=8, ge=1, le=20)
    seed: int | None = Field(default=None, ge=0)
    contract_id: str | None = None
    api_key: str | None = None


class BaselineTraceStep(BaseModel):
    step: int
    action: dict[str, Any]
    reward: float | int | bool | None = None
    done: bool
    message: str
    score_preview: float


class BaselineResponse(BaseModel):
    difficulty: TaskDifficulty
    score: float
    metric: str
    steps_taken: int
    contract_id: str | None = None
    contract_title: str | None = None
    trace: list[BaselineTraceStep] = Field(default_factory=list)