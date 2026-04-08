from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

from openenv.core import EnvClient

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_review_env.agent_policy import action_to_log_string, build_client, choose_action
from legal_review_env.client import LegalReviewEnvClient
from legal_review_env.models import LegalReviewObservation, TaskDifficulty

IMAGE_NAME = os.getenv("IMAGE_NAME")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or IMAGE_NAME
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("LEGAL_REVIEW_BASE_URL") or os.getenv("OPENENV_BASE_URL")
HF_SPACE_URL = os.getenv("LEGAL_REVIEW_SPACE_URL") or os.getenv("HF_SPACE_URL")
BENCHMARK = os.getenv("LEGAL_REVIEW_BENCHMARK", "legal_review_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.80"))
TASK_SEQUENCE = [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]
SEED = int(os.getenv("LEGAL_REVIEW_SEED", "7"))
CONNECT_TIMEOUT_S = float(os.getenv("LEGAL_REVIEW_CONNECT_TIMEOUT_S", "20"))
REQUEST_TIMEOUT_S = float(os.getenv("LEGAL_REVIEW_REQUEST_TIMEOUT_S", "60"))
TASK_NAMES = {
    TaskDifficulty.EASY: "easy-clause-abstraction",
    TaskDifficulty.MEDIUM: "medium-risk-triage",
    TaskDifficulty.HARD: "hard-contract-redlining",
}


def _task_sequence() -> List[TaskDifficulty]:
    requested = (os.getenv("LEGAL_REVIEW_TASK") or os.getenv("LEGAL_REVIEW_DIFFICULTY") or "").strip().lower()
    if not requested:
        return TASK_SEQUENCE
    aliases = {
        "easy": TaskDifficulty.EASY,
        "easy-clause-abstraction": TaskDifficulty.EASY,
        "medium": TaskDifficulty.MEDIUM,
        "medium-risk-triage": TaskDifficulty.MEDIUM,
        "hard": TaskDifficulty.HARD,
        "hard-contract-redlining": TaskDifficulty.HARD,
    }
    return [aliases.get(requested, TaskDifficulty.EASY)]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


async def _connect_client(base_url: str) -> LegalReviewEnvClient:
    client = LegalReviewEnvClient(base_url=base_url)
    await asyncio.wait_for(client.connect(), timeout=CONNECT_TIMEOUT_S)
    return client


async def _connect_env() -> EnvClient:
    errors: list[str] = []

    if ENV_BASE_URL:
        try:
            return await _connect_client(ENV_BASE_URL)
        except Exception as exc:
            errors.append(f"explicit base URL {ENV_BASE_URL}: {exc}")

    if LOCAL_IMAGE_NAME:
        try:
            return await asyncio.wait_for(
                LegalReviewEnvClient.from_docker_image(LOCAL_IMAGE_NAME),
                timeout=CONNECT_TIMEOUT_S,
            )
        except Exception as exc:
            errors.append(f"docker image {LOCAL_IMAGE_NAME}: {exc}")

    for port in (8000, 7860):
        local_url = f"http://localhost:{port}"
        try:
            return await _connect_client(local_url)
        except Exception as exc:
            errors.append(f"localhost {local_url}: {exc}")

    if HF_SPACE_URL:
        try:
            return await _connect_client(HF_SPACE_URL)
        except Exception as exc:
            errors.append(f"space URL {HF_SPACE_URL}: {exc}")

    raise RuntimeError("; ".join(errors) or "unable to connect to environment")


async def run_episode(client_model, difficulty: TaskDifficulty) -> None:
    env: Optional[EnvClient] = None
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_name = TASK_NAMES[difficulty]
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await _connect_env()
        result = await asyncio.wait_for(
            env.reset(difficulty=difficulty.value, seed=SEED),
            timeout=REQUEST_TIMEOUT_S,
        )
        observation: LegalReviewObservation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            state = await asyncio.wait_for(env.state(), timeout=REQUEST_TIMEOUT_S)
            action = choose_action(client_model, MODEL_NAME, result.observation, state)
            result = await asyncio.wait_for(env.step(action), timeout=REQUEST_TIMEOUT_S)
            observation = result.observation
            reward = float(result.reward or 0.0)
            error = observation.validation_errors[0] if observation.validation_errors else None
            rewards.append(reward)
            steps_taken = step
            score = observation.score_preview
            log_step(
                step=step,
                action=action_to_log_string(action),
                reward=reward,
                done=result.done,
                error=error,
            )
            if result.done:
                break

        score = observation.score_preview
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception:
        success = False
        score = 0.0
    finally:
        try:
            if env is not None:
                await asyncio.wait_for(env.close(), timeout=REQUEST_TIMEOUT_S)
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client_model = None
    try:
        client_model = build_client(api_base_url=API_BASE_URL, api_key=API_KEY)
    except Exception:
        try:
            client_model = build_client(api_base_url=API_BASE_URL, api_key="not-set")
        except Exception:
            client_model = None

    for difficulty in _task_sequence():
        await run_episode(client_model, difficulty)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        pass