from __future__ import annotations

import asyncio
import os
import sys
import traceback
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
ENV_BASE_URL = os.getenv("LEGAL_REVIEW_BASE_URL")
HF_SPACE_URL = "https://amannnnn-legal-review-env.hf.space"
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


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


async def _connect_env() -> EnvClient:
    """Try multiple connection strategies in order, never raise."""
    # 1. Explicit base URL
    if ENV_BASE_URL:
        print(f"[DEBUG] Connecting to explicit base URL: {ENV_BASE_URL}", flush=True)
        client = LegalReviewEnvClient(base_url=ENV_BASE_URL)
        await client.connect()
        return client

    # 2. Docker image (from_docker_image)
    if LOCAL_IMAGE_NAME:
        print(f"[DEBUG] Starting container from image: {LOCAL_IMAGE_NAME}", flush=True)
        try:
            return await LegalReviewEnvClient.from_docker_image(LOCAL_IMAGE_NAME)
        except Exception as exc:
            print(f"[DEBUG] from_docker_image failed: {exc}", flush=True)

    # 3. Try localhost (validator may have started container already)
    for port in (8000, 7860):
        local_url = f"http://localhost:{port}"
        try:
            print(f"[DEBUG] Trying {local_url} ...", flush=True)
            client = LegalReviewEnvClient(base_url=local_url)
            await client.connect()
            return client
        except Exception:
            pass

    # 4. Fall back to live HF Space
    print(f"[DEBUG] Falling back to HF Space: {HF_SPACE_URL}", flush=True)
    client = LegalReviewEnvClient(base_url=HF_SPACE_URL)
    await client.connect()
    return client


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
    except Exception as exc:
        print(f"[DEBUG] Episode error ({difficulty.value}): {exc}", flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        try:
            if env is not None:
                await env.close()
        except Exception as close_err:
            print(f"[DEBUG] env.close() error: {close_err}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    try:
        client_model = build_client(api_base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[DEBUG] OpenAI client init error: {exc}", flush=True)
        client_model = build_client(api_base_url=API_BASE_URL, api_key="not-set")
    for difficulty in TASK_SEQUENCE:
        await run_episode(client_model, difficulty)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(0)