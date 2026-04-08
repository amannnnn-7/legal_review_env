from __future__ import annotations

from pathlib import Path
from threading import RLock
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from ..cuad import EpisodeSpec, LegalReviewRepository, get_default_repository
from ..models import (
    FlaggedRisk,
    GradeReport,
    LegalReviewAction,
    LegalReviewObservation,
    LegalReviewState,
    PlaybookResponse,
    RedlineRecord,
    TaskDifficulty,
)
from ..playbooks import action_hints, canonicalize_category, get_playbook
from ..scoring import grade_easy, grade_hard, grade_medium, normalize_text


class LegalReviewEnvironment(
    Environment[LegalReviewAction, LegalReviewObservation, LegalReviewState]
):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, repository: LegalReviewRepository | None = None):
        self._repository = repository or get_default_repository()
        self._lock = RLock()
        self._episode: EpisodeSpec | None = None
        self._hard_current_block: str | None = None
        self._previous_validation_error = False
        self._state = LegalReviewState(episode_id=str(uuid4()), status="idle")

    @staticmethod
    def _normalize_reward(value: float) -> float:
        return round(min(max(value, 0.0), 1.0), 4)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        difficulty: TaskDifficulty | str = TaskDifficulty.EASY,
        contract_id: str | None = None,
        **_: object,
    ) -> LegalReviewObservation:
        with self._lock:
            difficulty_value = (
                difficulty
                if isinstance(difficulty, TaskDifficulty)
                else TaskDifficulty(str(difficulty).lower())
            )
            self._episode = self._repository.build_episode(
                difficulty=difficulty_value,
                seed=seed,
                contract_id=contract_id,
            )
            current_episode_id = episode_id or str(uuid4())
            contract = self._episode.contract
            self._hard_current_block = self._episode.editable_block
            self._previous_validation_error = False
            self._state = LegalReviewState(
                episode_id=current_episode_id,
                step_count=0,
                contract_id=contract.contract_id,
                contract_title=contract.title,
                difficulty=self._episode.difficulty,
                task_id=self._episode.task.task_id,
                document_version=1,
                document_text=contract.document_text,
                original_document_text=contract.document_text,
                extracted_clauses={},
                flagged_risks=[],
                redlines=[],
                playbook_queries=[],
                status="ready",
                complete=False,
            )
            return self._observation(
                status="ready",
                message=(
                    f"Loaded contract `{contract.title}` for {difficulty_value.value} task `{self._episode.task.task_id}`."
                ),
                reward=0.0,
                done=False,
                playbook=None,
                clause_category=None,
                clause_matches=[],
                document_text=None,
                validation_errors=[],
            )

    def step(
        self,
        action: LegalReviewAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> LegalReviewObservation:
        del timeout_s
        with self._lock:
            if self._episode is None:
                return self._observation(
                    status="validation_error",
                    message="Call reset before taking a step.",
                    reward=-0.2,
                    done=False,
                    validation_errors=["environment not initialized"],
                )

            self._state.step_count += 1
            grade_before = self.grade()

            reward = 0.0
            playbook: PlaybookResponse | None = None
            clause_category: str | None = None
            clause_matches: list[str] = []
            document_text: str | None = None
            validation_errors: list[str] = []
            message = ""

            if action.action_type.value == "search_playbook":
                playbook = get_playbook(action.topic or "")
                if playbook is None:
                    validation_errors.append(f"unknown playbook topic: {action.topic}")
                    reward -= 0.1
                    message = "Playbook lookup failed."
                else:
                    is_new_query = playbook.topic not in self._state.playbook_queries
                    if is_new_query:
                        self._state.playbook_queries.append(playbook.topic)
                    if (
                        playbook.topic == self._episode.task.playbook_topic
                        and is_new_query
                        and not self._state.redlines
                    ):
                        reward += 0.15
                    message = f"Loaded `{playbook.topic}` playbook."

            elif action.action_type.value == "read_clause":
                clause_category = canonicalize_category(action.category or "")
                if clause_category is None:
                    validation_errors.append(f"unknown clause category: {action.category}")
                    reward -= 0.1
                    message = "Clause lookup failed."
                else:
                    if (
                        self._episode.difficulty == TaskDifficulty.HARD
                        and clause_category == "Non-Compete"
                        and self._episode.editable_block is not None
                    ):
                        clause_matches = [self._episode.editable_block]
                    else:
                        spans = self._episode.contract.spans_for(clause_category)
                        clause_matches = [span.text for span in spans]

                    if clause_matches:
                        self._state.extracted_clauses[clause_category] = clause_matches
                        if (
                            clause_category in self._episode.task.target_categories
                            and len(clause_matches) > 0
                        ):
                            reward += 0.15
                        message = f"Retrieved {len(clause_matches)} span(s) for `{clause_category}`."
                    else:
                        message = f"No spans were annotated for `{clause_category}` in this contract."

            elif action.action_type.value == "flag_risk":
                assert action.text_span is not None
                assert action.rationale is not None
                if action.text_span not in self._state.document_text:
                    validation_errors.append("text_span must be an exact substring of the live document")
                    reward -= 0.2
                    message = "FlagRisk rejected a hallucinated span."
                else:
                    predicted_key = normalize_text(action.text_span)
                    existing = {
                        normalize_text(item.text_span): item for item in self._state.flagged_risks
                    }
                    if predicted_key not in existing:
                        matched = any(
                            predicted_key == normalize_text(span)
                            or predicted_key in normalize_text(span)
                            or normalize_text(span) in predicted_key
                            for span in self._episode.violating_spans
                        )
                        self._state.flagged_risks.append(
                            FlaggedRisk(
                                text_span=action.text_span,
                                rationale=action.rationale,
                                matched_ground_truth=matched,
                            )
                        )
                        reward += 0.2 if matched else -0.05
                    message = "Recorded flagged risk span."

            elif action.action_type.value == "apply_redline":
                assert action.original_text is not None
                assert action.replacement_text is not None
                if action.original_text not in self._state.document_text:
                    validation_errors.append("original_text must match the live document exactly")
                    reward -= 0.2
                    message = "ApplyRedline rejected a hallucinated source span."
                else:
                    self._state.document_text = self._state.document_text.replace(
                        action.original_text,
                        action.replacement_text,
                        1,
                    )
                    self._state.document_version += 1
                    block_match = False
                    if self._hard_current_block and action.original_text in self._hard_current_block:
                        self._hard_current_block = self._hard_current_block.replace(
                            action.original_text,
                            action.replacement_text,
                            1,
                        )
                        block_match = True
                    self._state.redlines.append(
                        RedlineRecord(
                            original_text=action.original_text,
                            replacement_text=action.replacement_text,
                            applied=True,
                            version=self._state.document_version,
                            match_in_task_block=block_match,
                        )
                    )
                    document_text = self._state.document_text
                    reward += 0.1 if block_match else 0.0
                    message = "Applied redline to the live contract state."

            grade_after = self.grade()
            reward += max(0.0, grade_after.score - grade_before.score)

            if validation_errors:
                status = "validation_error"
                self._state.status = "validation_error"
            else:
                status = "ok"
                self._state.status = "running"
                if self._previous_validation_error:
                    reward += 0.1

            finish_requested = bool(action.metadata.get("finish"))
            done = finish_requested or self._state.step_count >= self._episode.task.max_steps or grade_after.complete
            if done:
                status = "done"
                self._state.status = "done"
            self._state.complete = done
            self._previous_validation_error = bool(validation_errors)
            normalized_reward = self._normalize_reward(reward)

            return self._observation(
                status=status,
                message=message,
                reward=normalized_reward,
                done=done,
                playbook=playbook,
                clause_category=clause_category,
                clause_matches=clause_matches,
                document_text=document_text,
                validation_errors=validation_errors,
            )

    @property
    def state(self) -> LegalReviewState:
        return self._state

    def grade(self) -> GradeReport:
        if self._episode is None:
            return GradeReport(
                difficulty=TaskDifficulty.EASY,
                metric="not_ready",
                score=0.0,
                complete=False,
                details={},
            )

        if self._episode.difficulty == TaskDifficulty.EASY:
            return grade_easy(self._state.extracted_clauses, self._episode.expected_clauses)
        if self._episode.difficulty == TaskDifficulty.MEDIUM:
            return grade_medium(
                [risk.text_span for risk in self._state.flagged_risks],
                self._episode.violating_spans,
            )
        return grade_hard(
            current_block=self._hard_current_block or self._episode.editable_block or "",
            original_block=self._episode.editable_block or "",
            target_block=self._episode.target_block or "",
        )

    def get_metadata(self) -> EnvironmentMetadata:
        readme_path = Path(__file__).resolve().parents[3] / "README.md"
        readme_content = readme_path.read_text() if readme_path.exists() else None
        return EnvironmentMetadata(
            name="legal_review_env",
            description="Stateful legal contract review, risk abstraction, and redlining environment built on CUAD.",
            version="0.1.0",
            author="Aman Paliwal",
            readme_content=readme_content,
        )

    def close(self) -> None:
        return None

    def _observation(
        self,
        status: str,
        message: str,
        reward: float,
        done: bool,
        playbook: PlaybookResponse | None = None,
        clause_category: str | None = None,
        clause_matches: list[str] | None = None,
        document_text: str | None = None,
        validation_errors: list[str] | None = None,
    ) -> LegalReviewObservation:
        grade_report = self.grade() if self._episode is not None else None
        return LegalReviewObservation(
            status=status,
            message=message,
            contract_title=self._state.contract_title,
            contract_id=self._state.contract_id or "",
            difficulty=self._state.difficulty,
            task=self._episode.task if self._episode else None,
            clause_category=clause_category,
            clause_matches=clause_matches or [],
            playbook=playbook,
            flagged_risks=list(self._state.flagged_risks),
            redlines=list(self._state.redlines),
            validation_errors=validation_errors or [],
            document_text=document_text,
            document_version=self._state.document_version,
            score_preview=grade_report.score if grade_report else 0.0,
            available_action_hints=action_hints(self._state.difficulty),
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
            },
        )