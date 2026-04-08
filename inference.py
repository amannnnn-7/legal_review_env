from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from openenv.core import EnvClient

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_review_env.agent_policy import action_to_log_string, build_client, choose_action
from legal_review_env.client import LegalReviewEnvClient
from legal_review_env.models import LegalReviewObservation, TaskDifficulty


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("LEGAL_REVIEW_BASE_URL")
BENCHMARK = os.getenv("LEGAL_REVIEW_BENCHMARK", "legal_review_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.80"))
TASK_SEQUENCE = [TaskDifficulty.EASY, TaskDifficulty.MEDIUM, TaskDifficulty.HARD]
SEED = int(os.getenv("LEGAL_REVIEW_SEED", "7"))
TASK_NAMES = {
    TaskDifficulty.EASY: "easy-clause-abstraction",
    TaskDifficulty.MEDIUM: "medium-risk-triage",
    TaskDifficulty.HARD: "hard-contract-redlining",
}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{item:.2f}" for item in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


async def _connect_env() -> EnvClient:
    if ENV_BASE_URL:
        client = LegalReviewEnvClient(base_url=ENV_BASE_URL)
        await client.connect()
        return client
    if LOCAL_IMAGE_NAME:
        return await LegalReviewEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
    raise RuntimeError("Set LOCAL_IMAGE_NAME or LEGAL_REVIEW_BASE_URL before running inference.py")


async def run_episode(client_model, difficulty: TaskDifficulty) -> None:
    env = None
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_name = TASK_NAMES[difficulty]
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await _connect_env()
        result = await env.reset(difficulty=difficulty.value, seed=SEED)
        observation: LegalReviewObservation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            state = await env.state()
            action = choose_action(client_model, MODEL_NAME, result.observation, state)
            result = await env.step(action)
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
    finally:
        if env is not None:
            await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client_model = build_client(api_base_url=API_BASE_URL, api_key=API_KEY)
    for difficulty in TASK_SEQUENCE:
        await run_episode(client_model, difficulty)


if __name__ == "__main__":
    asyncio.run(main())