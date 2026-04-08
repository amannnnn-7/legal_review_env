from __future__ import annotations

import json
import os
import re

from openai import OpenAI

from .models import LegalReviewAction, LegalReviewObservation, LegalReviewState, TaskDifficulty
from .scoring import rewrite_non_compete_block


SYSTEM_PROMPT = """You are a careful junior lawyer agent.
Operate only through the provided action schema.
Return exactly one JSON object matching LegalReviewAction.
Never invent text spans. Only use exact text already returned by the environment.
When the task is complete, set metadata.finish to true on your final action.
"""


def build_client(api_base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    resolved_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    resolved_base_url = api_base_url or os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    return OpenAI(base_url=resolved_base_url, api_key=resolved_key)


def extract_json(text: str) -> dict[str, object]:
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def fallback_action(observation: LegalReviewObservation, state: LegalReviewState) -> LegalReviewAction:
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


def prompt_for_observation(observation: LegalReviewObservation, state: LegalReviewState) -> str:
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


def _request_model_action(
    client: OpenAI,
    model_name: str,
    observation: LegalReviewObservation,
    state: LegalReviewState,
) -> LegalReviewAction:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_for_observation(observation, state)},
        ],
        temperature=0,
        max_tokens=400,
        stream=False,
    )
    content = (completion.choices[0].message.content or "").strip()
    return LegalReviewAction.model_validate(extract_json(content))


def choose_action(
    client: OpenAI,
    model_name: str,
    observation: LegalReviewObservation,
    state: LegalReviewState,
) -> LegalReviewAction:
    expected = fallback_action(observation, state)
    try:
        candidate = _request_model_action(client, model_name, observation, state)
    except Exception:
        return expected

    if expected.action_type.value in {"search_playbook", "read_clause"}:
        return expected

    if expected.action_type.value == "flag_risk":
        if candidate.action_type.value == "flag_risk":
            rationale = candidate.rationale or expected.rationale
            return expected.model_copy(update={"rationale": rationale})
        return expected

    if expected.action_type.value == "apply_redline":
        return expected

    return expected


def action_to_log_string(action: LegalReviewAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"), ensure_ascii=True)