from __future__ import annotations

from dataclasses import dataclass

from legal_review_env.cuad import ClauseSpan, ContractRecord, EpisodeSpec
from legal_review_env.models import LegalReviewAction, TaskDescriptor, TaskDifficulty
from legal_review_env.scoring import find_enclosing_block, rewrite_non_compete_block
from legal_review_env.server.environment import LegalReviewEnvironment


SAMPLE_DOCUMENT = """MASTER SERVICES AGREEMENT

This Agreement is effective as of January 2, 2024.

This Agreement is governed by the laws of Delaware.

Non-Compete. Vendor shall not compete with Buyer for a period of 24 months after termination.

The restriction applies throughout the Territory and survives termination.
"""


def _span(label: str, text: str) -> ClauseSpan:
    start = SAMPLE_DOCUMENT.index(text)
    return ClauseSpan(
        label=label,
        text=text,
        answer_start=start,
        answer_end=start + len(text),
        question=label,
        qa_id=f"sample__{label}",
    )


@dataclass
class StubRepository:
    contract: ContractRecord

    def build_episode(
        self,
        difficulty: TaskDifficulty,
        seed: int | None = None,
        contract_id: str | None = None,
    ) -> EpisodeSpec:
        del seed, contract_id
        if difficulty == TaskDifficulty.EASY:
            return EpisodeSpec(
                difficulty=difficulty,
                contract=self.contract,
                task=TaskDescriptor(
                    task_id="easy-clause-abstraction",
                    difficulty=difficulty,
                    objective="Extract Effective Date and Governing Law.",
                    grader="exact_match_average",
                    playbook_topic="contract-basics",
                    target_categories=["Effective Date", "Governing Law"],
                    max_steps=4,
                ),
                expected_clauses={
                    "Effective Date": ("January 2, 2024",),
                    "Governing Law": ("the laws of Delaware",),
                },
            )
        if difficulty == TaskDifficulty.MEDIUM:
            return EpisodeSpec(
                difficulty=difficulty,
                contract=self.contract,
                task=TaskDescriptor(
                    task_id="medium-risk-triage",
                    difficulty=difficulty,
                    objective="Flag overlong non-compete restrictions.",
                    grader="span_f1",
                    playbook_topic="non-compete",
                    target_categories=["Non-Compete"],
                    max_steps=6,
                ),
                violating_spans=(
                    "Vendor shall not compete with Buyer for a period of 24 months after termination.",
                ),
            )
        block = find_enclosing_block(
            self.contract.document_text,
            self.contract.spans_for("Non-Compete")[0].answer_start,
            self.contract.spans_for("Non-Compete")[0].text,
        )
        return EpisodeSpec(
            difficulty=difficulty,
            contract=self.contract,
            task=TaskDescriptor(
                task_id="hard-contract-redlining",
                difficulty=difficulty,
                objective="Redline the non-compete block.",
                grader="similarity_minus_edit_penalty",
                playbook_topic="non-compete",
                target_categories=["Non-Compete"],
                max_steps=8,
            ),
            editable_block=block,
            target_block=rewrite_non_compete_block(block),
        )


def _repository() -> StubRepository:
    contract = ContractRecord(
        contract_id="sample-contract",
        title="sample-contract",
        document_text=SAMPLE_DOCUMENT,
        labels={
            "Effective Date": (_span("Effective Date", "January 2, 2024"),),
            "Governing Law": (_span("Governing Law", "the laws of Delaware"),),
            "Non-Compete": (
                _span(
                    "Non-Compete",
                    "Vendor shall not compete with Buyer for a period of 24 months after termination.",
                ),
            ),
        },
    )
    return StubRepository(contract)


def test_easy_task_scores_exact_matches() -> None:
    env = LegalReviewEnvironment(repository=_repository())
    initial_observation = env.reset(difficulty="easy")
    assert 0.0 < initial_observation.score_preview < 1.0
    env.step(LegalReviewAction(action_type="read_clause", category="Effective Date"))
    final_observation = env.step(
        LegalReviewAction(
            action_type="read_clause",
            category="Governing Law",
            metadata={"finish": True},
        )
    )
    assert final_observation.done is True
    assert 0.99 < final_observation.score_preview < 1.0
    assert env.grade().complete is True


def test_medium_task_flags_exact_violation() -> None:
    env = LegalReviewEnvironment(repository=_repository())
    initial_observation = env.reset(difficulty="medium")
    assert 0.0 < initial_observation.score_preview < 1.0
    env.step(LegalReviewAction(action_type="search_playbook", topic="non-compete"))
    env.step(LegalReviewAction(action_type="read_clause", category="Non-Compete"))
    final_observation = env.step(
        LegalReviewAction(
            action_type="flag_risk",
            text_span="Vendor shall not compete with Buyer for a period of 24 months after termination.",
            rationale="The restriction lasts more than 12 months.",
            metadata={"finish": True},
        )
    )
    assert final_observation.done is True
    assert 0.0 <= final_observation.reward <= 1.0
    assert 0.99 < final_observation.score_preview < 1.0
    assert env.grade().complete is True


def test_hard_task_redline_updates_document() -> None:
    env = LegalReviewEnvironment(repository=_repository())
    initial_observation = env.reset(difficulty="hard")
    assert 0.0 < initial_observation.score_preview < 1.0
    read_observation = env.step(LegalReviewAction(action_type="read_clause", category="Non-Compete"))
    clause_block = read_observation.clause_matches[0]
    target_block = rewrite_non_compete_block(clause_block)
    final_observation = env.step(
        LegalReviewAction(
            action_type="apply_redline",
            original_text=clause_block,
            replacement_text=target_block,
            metadata={"finish": True},
        )
    )
    assert final_observation.done is True
    assert "12 months" in env.state.document_text
    assert env.state.document_version == 2
    assert 0.0 <= final_observation.reward <= 1.0
    assert 0.9 < final_observation.score_preview < 1.0