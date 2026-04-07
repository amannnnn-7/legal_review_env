from __future__ import annotations

import re

from .models import PlaybookResponse, TaskDifficulty


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


PLAYBOOKS: dict[str, PlaybookResponse] = {
    "contract-basics": PlaybookResponse(
        topic="contract-basics",
        summary="Extract exactly what the contract says for core metadata clauses.",
        rules=[
            "Return the exact extractive CUAD span with original punctuation.",
            "Do not normalize dates, party names, or governing law language.",
            "If a clause has multiple answers, keep every exact span.",
        ],
        examples=[
            "Effective Date: keep the original wording from the contract.",
            "Governing Law: preserve the full jurisdiction phrase.",
        ],
    ),
    "non-compete": PlaybookResponse(
        topic="non-compete",
        summary="Non-compete restrictions must be explicit, time-bounded, and no longer than 12 months.",
        rules=[
            "Flag any non-compete restriction with no explicit end date.",
            "Flag any non-compete restriction longer than 12 months.",
            "When redlining, preserve surrounding syntax and reduce the duration to 12 months.",
        ],
        preferred_edit="Replace any duration above 12 months with `12 months` and avoid broader rewrites.",
        examples=[
            "`for a period of 2 years` -> `for a period of 12 months`",
            "`during the two year period following the Effective Date` -> `during the 12 month period following the Effective Date`",
        ],
    ),
    "liability-cap": PlaybookResponse(
        topic="liability-cap",
        summary="Aggregate liability should be capped to the trailing 12 months of fees while preserving standard carve-outs.",
        rules=[
            "Use a trailing-12-month fee cap unless the playbook says otherwise.",
            "Keep death, personal injury, fraud, and confidentiality misuse carve-outs intact.",
            "Prefer narrow edits over clause rewrites.",
        ],
        preferred_edit="Cap aggregate liability at fees paid or payable in the previous 12 months.",
    ),
}


PLAYBOOK_ALIASES = {
    "basics": "contract-basics",
    "contract-basics": "contract-basics",
    "contract-basics-playbook": "contract-basics",
    "effective-date": "contract-basics",
    "governing-law": "contract-basics",
    "non-compete": "non-compete",
    "noncompete": "non-compete",
    "restrictive-covenants": "non-compete",
    "liability-cap": "liability-cap",
    "cap-on-liability": "liability-cap",
}


CATEGORY_ALIASES = {
    "effective-date": "Effective Date",
    "agreement-date": "Agreement Date",
    "governing-law": "Governing Law",
    "non-compete": "Non-Compete",
    "noncompete": "Non-Compete",
    "cap-on-liability": "Cap On Liability",
    "uncapped-liability": "Uncapped Liability",
}


def get_playbook(topic: str) -> PlaybookResponse | None:
    key = PLAYBOOK_ALIASES.get(_normalize(topic), _normalize(topic))
    return PLAYBOOKS.get(key)


def canonicalize_category(category: str) -> str | None:
    normalized = _normalize(category)
    return CATEGORY_ALIASES.get(normalized)


def action_hints(difficulty: TaskDifficulty | None) -> list[str]:
    if difficulty == TaskDifficulty.EASY:
        return [
            "Use search_playbook(topic='contract-basics') before extraction for bonus reward.",
            "Read the Effective Date and Governing Law clauses exactly.",
            "Set metadata.finish=true on the final action when you are done.",
        ]
    if difficulty == TaskDifficulty.MEDIUM:
        return [
            "Load the non-compete playbook first for partial credit.",
            "Use read_clause(category='Non-Compete') to retrieve exact spans.",
            "Only flag exact spans already present in the contract.",
        ]
    if difficulty == TaskDifficulty.HARD:
        return [
            "Read the non-compete clause block first.",
            "Apply the smallest possible edit that reduces the restriction to 12 months.",
            "ApplyRedline fails if original_text is not an exact substring of the live document.",
        ]
    return []