"""
Demo script — runs one episode per difficulty against a live legal_review_env server.

Usage:
    # Against the deployed HF Space:
    python demo.py

    # Against a local server:
    python demo.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

# Allow importing the rewrite helper from the local package
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from legal_review_env.scoring import rewrite_non_compete_block

DEFAULT_BASE_URL = "https://amannnnn-legal-review-env.hf.space"
DIFFICULTIES = ["easy", "medium", "hard"]


def run_episode(client: httpx.Client, difficulty: str) -> dict:
    # Reset
    resp = client.post("/reset", json={"difficulty": difficulty, "seed": 7})
    resp.raise_for_status()
    data = resp.json()
    obs = data["observation"]
    done = data.get("done", False)

    print(f"\n{'='*60}")
    print(f"Task: {obs['task']['task_id']}  (difficulty={difficulty})")
    print(f"Contract: {obs['contract_title']}")
    print(f"Objective: {obs['task']['objective']}")
    print(f"{'='*60}")

    steps = []
    step = 0

    # Deterministic action sequences per difficulty
    if difficulty == "easy":
        actions = [
            {"action_type": "search_playbook", "topic": "contract-basics"},
            {"action_type": "read_clause", "category": "Effective Date"},
            {"action_type": "read_clause", "category": "Governing Law", "metadata": {"finish": True}},
        ]
    elif difficulty == "medium":
        actions = [
            {"action_type": "search_playbook", "topic": "non-compete"},
            {"action_type": "read_clause", "category": "Non-Compete"},
        ]
        # After reading clauses, flag each violation
    else:
        actions = [
            {"action_type": "search_playbook", "topic": "non-compete"},
            {"action_type": "read_clause", "category": "Non-Compete"},
        ]
        # After reading, apply redline

    for action in actions:
        if done:
            break
        step += 1
        resp = client.post("/step", json={"action": action})
        resp.raise_for_status()
        data = resp.json()
        obs = data["observation"]
        reward = obs.get("reward", 0)
        done = obs.get("done", False)
        print(f"  Step {step}: {action['action_type']} → reward={reward}, score={obs['score_preview']:.3f}")
        steps.append({"step": step, "action": action, "reward": reward, "done": done})

    # Medium: flag each clause match as a risk
    if difficulty == "medium" and not done:
        for match in obs.get("clause_matches", []):
            if done:
                break
            step += 1
            remaining = [m for m in obs.get("clause_matches", []) if m not in {s.get("text_span") for s in steps if "text_span" in s}]
            is_last = step >= 5 or len(remaining) <= 1
            action = {
                "action_type": "flag_risk",
                "text_span": match,
                "rationale": "Non-compete restriction exceeds 12-month playbook limit or lacks explicit end date.",
                "metadata": {"finish": is_last},
            }
            resp = client.post("/step", json={"action": action})
            resp.raise_for_status()
            data = resp.json()
            obs = data["observation"]
            reward = obs.get("reward", 0)
            done = obs.get("done", False)
            print(f"  Step {step}: flag_risk → reward={reward}, score={obs['score_preview']:.3f}")
            steps.append({"step": step, "action": action, "reward": reward, "done": done})

    # Hard: read the clause block, rewrite durations to 12 months, apply redline
    if difficulty == "hard" and not done:
        clause_block = obs.get("clause_matches", [""])[0]
        if clause_block:
            replacement = rewrite_non_compete_block(clause_block)
            step += 1
            action = {
                "action_type": "apply_redline",
                "original_text": clause_block,
                "replacement_text": replacement,
                "metadata": {"finish": True},
            }
            resp = client.post("/step", json={"action": action})
            resp.raise_for_status()
            data = resp.json()
            obs = data["observation"]
            reward = obs.get("reward", 0)
            done = obs.get("done", False)
            print(f"  Step {step}: apply_redline → reward={reward}, score={obs['score_preview']:.3f}")
            steps.append({"step": step, "action": action, "reward": reward, "done": done})

    # Final grader score
    resp = client.post("/grader", json={"include_details": False})
    resp.raise_for_status()
    grade = resp.json()
    print(f"  Final score: {grade['score']:.3f} ({grade['metric']})")
    return {"difficulty": difficulty, "score": grade["score"], "steps": len(steps)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: run all 3 tasks against legal_review_env")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Environment API base URL")
    args = parser.parse_args()

    print(f"legal_review_env demo — targeting {args.base_url}")

    results = []
    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        # Verify connectivity
        resp = client.get("/tasks")
        resp.raise_for_status()
        tasks = resp.json()["tasks"]
        print(f"Available tasks: {', '.join(t['task_id'] for t in tasks)}")

        for difficulty in DIFFICULTIES:
            result = run_episode(client, difficulty)
            results.append(result)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['difficulty']:8s}  score={r['score']:.3f}  steps={r['steps']}")


if __name__ == "__main__":
    main()
