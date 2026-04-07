from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import LegalReviewAction, LegalReviewObservation, LegalReviewState


class LegalReviewEnvClient(
    EnvClient[LegalReviewAction, LegalReviewObservation, LegalReviewState]
):
    def _step_payload(self, action: LegalReviewAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[LegalReviewObservation]:
        observation = LegalReviewObservation.model_validate(
            {
                **payload.get("observation", {}),
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> LegalReviewState:
        return LegalReviewState.model_validate(payload)