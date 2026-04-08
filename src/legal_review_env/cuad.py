from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download

from .models import TaskDescriptor, TaskDifficulty
from .scoring import find_enclosing_block, non_compete_is_violation, rewrite_non_compete_block


@dataclass(frozen=True)
class ClauseSpan:
    label: str
    text: str
    answer_start: int
    answer_end: int
    question: str
    qa_id: str


@dataclass(frozen=True)
class ContractRecord:
    contract_id: str
    title: str
    document_text: str
    labels: dict[str, tuple[ClauseSpan, ...]]

    def spans_for(self, label: str) -> tuple[ClauseSpan, ...]:
        return self.labels.get(label, ())


@dataclass(frozen=True)
class EpisodeSpec:
    difficulty: TaskDifficulty
    contract: ContractRecord
    task: TaskDescriptor
    expected_clauses: dict[str, tuple[str, ...]] = field(default_factory=dict)
    violating_spans: tuple[str, ...] = ()
    editable_block: str | None = None
    target_block: str | None = None


class LegalReviewRepository:
    def __init__(
        self,
        repo_id: str = "TheAtticusProject/cuad",
        filename: str = "CUAD_v1/CUAD_v1.json",
    ):
        self.repo_id = repo_id
        self.filename = filename

    def prefetch(self) -> Path:
        for candidate in self._local_candidates():
            if candidate.exists():
                return candidate

        path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=self.filename,
        )
        return Path(path)

    def _local_candidates(self) -> tuple[Path, ...]:
        candidates: list[Path] = []

        configured_path = os.getenv("LEGAL_REVIEW_DATA_PATH")
        if configured_path:
            candidates.append(Path(configured_path).expanduser())

        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / "data" / "CUAD_v1.json")

        return tuple(candidates)

    @cached_property
    def contracts(self) -> tuple[ContractRecord, ...]:
        path = self.prefetch()
        with path.open() as handle:
            payload = json.load(handle)

        records: list[ContractRecord] = []
        for raw_contract in payload["data"]:
            paragraph = raw_contract["paragraphs"][0]
            document_text = paragraph["context"]
            grouped: dict[str, list[ClauseSpan]] = defaultdict(list)

            for qa in paragraph["qas"]:
                label = qa["id"].split("__", 1)[-1]
                for answer in qa.get("answers", []):
                    text = answer["text"]
                    start = int(answer["answer_start"])
                    grouped[label].append(
                        ClauseSpan(
                            label=label,
                            text=text,
                            answer_start=start,
                            answer_end=start + len(text),
                            question=qa["question"],
                            qa_id=qa["id"],
                        )
                    )

            labels = {
                label: tuple(sorted(spans, key=lambda item: (item.answer_start, item.answer_end, item.text)))
                for label, spans in grouped.items()
                if spans
            }
            contract_id = raw_contract["title"]
            records.append(
                ContractRecord(
                    contract_id=contract_id,
                    title=contract_id,
                    document_text=document_text,
                    labels=labels,
                )
            )

        return tuple(records)

    @staticmethod
    def _stable_subset(contracts: list[ContractRecord], fraction: float = 0.2, minimum: int = 5) -> tuple[ContractRecord, ...]:
        ordered = sorted(
            contracts,
            key=lambda contract: hashlib.sha256(contract.contract_id.encode("utf-8")).hexdigest(),
        )
        if not ordered:
            return tuple()
        count = min(len(ordered), max(minimum, math.ceil(len(ordered) * fraction)))
        return tuple(ordered[:count])

    @cached_property
    def easy_pool(self) -> tuple[ContractRecord, ...]:
        candidates = [
            contract
            for contract in self.contracts
            if contract.spans_for("Effective Date") and contract.spans_for("Governing Law")
        ]
        return self._stable_subset(candidates)

    @cached_property
    def medium_pool(self) -> tuple[ContractRecord, ...]:
        candidates = [
            contract
            for contract in self.contracts
            if self._violating_non_compete_spans(contract)
        ]
        return self._stable_subset(candidates)

    @cached_property
    def hard_pool(self) -> tuple[ContractRecord, ...]:
        candidates = [
            contract
            for contract in self.contracts
            if self._hard_block_candidate(contract) is not None
        ]
        return self._stable_subset(candidates)

    def build_episode(
        self,
        difficulty: TaskDifficulty,
        seed: int | None = None,
        contract_id: str | None = None,
    ) -> EpisodeSpec:
        pool = self._pool_for(difficulty)
        if not pool:
            raise RuntimeError(f"No CUAD candidates are available for difficulty={difficulty.value}")

        if contract_id is None:
            contract = self._select_contract(pool, difficulty, seed)
        else:
            contract = next((item for item in pool if item.contract_id == contract_id), None)
            if contract is None:
                raise ValueError(f"Contract `{contract_id}` is not available for difficulty={difficulty.value}")

        if difficulty == TaskDifficulty.EASY:
            return self._build_easy_episode(contract)
        if difficulty == TaskDifficulty.MEDIUM:
            return self._build_medium_episode(contract)
        return self._build_hard_episode(contract)

    def _pool_for(self, difficulty: TaskDifficulty) -> tuple[ContractRecord, ...]:
        if difficulty == TaskDifficulty.EASY:
            return self.easy_pool
        if difficulty == TaskDifficulty.MEDIUM:
            return self.medium_pool
        return self.hard_pool

    @staticmethod
    def _select_contract(
        pool: tuple[ContractRecord, ...],
        difficulty: TaskDifficulty,
        seed: int | None,
    ) -> ContractRecord:
        stable_key = f"{difficulty.value}:{seed or 0}"
        index = int(hashlib.sha256(stable_key.encode("utf-8")).hexdigest(), 16) % len(pool)
        return pool[index]

    @staticmethod
    def _unique_texts(spans: tuple[ClauseSpan, ...]) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for span in spans:
            if span.text in seen:
                continue
            seen.add(span.text)
            ordered.append(span.text)
        return tuple(ordered)

    def _violating_non_compete_spans(self, contract: ContractRecord) -> tuple[str, ...]:
        spans = tuple(
            span.text
            for span in contract.spans_for("Non-Compete")
            if non_compete_is_violation(span.text)
        )
        return tuple(dict.fromkeys(spans))

    def _hard_block_candidate(self, contract: ContractRecord) -> tuple[str, str] | None:
        for span in contract.spans_for("Non-Compete"):
            if not non_compete_is_violation(span.text):
                continue
            block = find_enclosing_block(contract.document_text, span.answer_start, span.text)
            if block.count("\n\n") < 1:
                continue
            target = rewrite_non_compete_block(block)
            if target != block:
                return block, target
        return None

    def _build_easy_episode(self, contract: ContractRecord) -> EpisodeSpec:
        expected = {
            "Effective Date": self._unique_texts(contract.spans_for("Effective Date")),
            "Governing Law": self._unique_texts(contract.spans_for("Governing Law")),
        }
        task = TaskDescriptor(
            task_id="easy-clause-abstraction",
            difficulty=TaskDifficulty.EASY,
            objective="Extract the exact Effective Date and Governing Law spans from the contract.",
            grader="Exact-match average across both categories.",
            playbook_topic="contract-basics",
            target_categories=["Effective Date", "Governing Law"],
            max_steps=4,
            notes=[
                "Use read_clause to retrieve exact CUAD spans.",
                "A perfect score requires both extracted categories to match exactly.",
            ],
        )
        return EpisodeSpec(
            difficulty=TaskDifficulty.EASY,
            contract=contract,
            task=task,
            expected_clauses=expected,
        )

    def _build_medium_episode(self, contract: ContractRecord) -> EpisodeSpec:
        violating = self._violating_non_compete_spans(contract)
        task = TaskDescriptor(
            task_id="medium-risk-triage",
            difficulty=TaskDifficulty.MEDIUM,
            objective="Flag every Non-Compete span that is missing an explicit end date or exceeds 12 months.",
            grader="Deterministic span-level F1 against playbook-derived violations.",
            playbook_topic="non-compete",
            target_categories=["Non-Compete"],
            max_steps=6,
            notes=[
                "Flag only exact spans already present in the contract.",
                "Indefinite restrictions and durations above 12 months are violations.",
            ],
        )
        return EpisodeSpec(
            difficulty=TaskDifficulty.MEDIUM,
            contract=contract,
            task=task,
            violating_spans=violating,
        )

    def _build_hard_episode(self, contract: ContractRecord) -> EpisodeSpec:
        candidate = self._hard_block_candidate(contract)
        if candidate is None:
            raise RuntimeError(f"No hard-task block is available for contract `{contract.contract_id}`")
        editable_block, target_block = candidate
        task = TaskDescriptor(
            task_id="hard-contract-redlining",
            difficulty=TaskDifficulty.HARD,
            objective="Redline the non-compete clause block so the restriction lasts no longer than 12 months while preserving surrounding syntax.",
            grader="Similarity to target minus edit-distance penalty from the original block.",
            playbook_topic="non-compete",
            target_categories=["Non-Compete"],
            max_steps=8,
            notes=[
                "CUAD does not label indemnification; this environment uses non-compete redlining as the deterministic hard task.",
                "ReadClause returns the editable clause block for this task.",
            ],
        )
        return EpisodeSpec(
            difficulty=TaskDifficulty.HARD,
            contract=contract,
            task=task,
            editable_block=editable_block,
            target_block=target_block,
        )


@lru_cache(maxsize=1)
def get_default_repository() -> LegalReviewRepository:
    return LegalReviewRepository()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch the CUAD annotation bundle used by legal_review_env.")
    parser.parse_args()
    path = get_default_repository().prefetch()
    print(path)