"""
APIContractEnv — Baseline Inference Script

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import os
import json
import sys
from typing import List, Optional
from openai import OpenAI
from client import APIContractEnv
from models import APIContractAction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = HF_TOKEN or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "api_contract"
TASKS = ["missing_fields", "breaking_changes", "full_audit"]
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an expert API architect and API contract reviewer.
You will be given an OpenAPI specification (or two specs to compare).
Your job is to identify issues and propose fixes in structured JSON format.

You MUST respond with ONLY valid JSON in this exact format:
{
  "issues": [
    "Issue description 1",
    "Issue description 2"
  ],
  "fixes": [
    "Fix for issue 1",
    "Fix for issue 2"
  ]
}

Rules:
- issues and fixes must have the same length (one fix per issue)
- Each issue should be specific and actionable
- Each fix should be concrete and implementable
- Do NOT include any text outside the JSON object
- Do NOT use markdown code blocks
"""

# ---------------------------------------------------------------------------
# Logging helpers (match sample format exactly)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, observation) -> APIContractAction:
    user_prompt = f"""Task: {observation.task_description}

{observation.spec_content}

Review the spec(s) above and return your findings as JSON with "issues" and "fixes" arrays."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        issues = parsed.get("issues", [])
        fixes = parsed.get("fixes", [])
        min_len = min(len(issues), len(fixes))
        return APIContractAction(
            issues=issues[:min_len],
            fixes=fixes[:min_len],
            raw_response=raw,
        )

    except Exception as e:
        return APIContractAction(
            issues=[f"Parse error: {str(e)}"],
            fixes=["Fix parse error"],
            raw_response=str(e),
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("ERROR: Set HF_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    scores = {}

    with APIContractEnv(base_url=ENV_BASE_URL).sync() as env:
        for task_name in TASKS:
            rewards: List[float] = []
            score = 0.0
            success = False
            steps_taken = 0

            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

            try:
                reset_result = env.reset()
                obs = reset_result.observation
                action = call_llm(client, obs)

                step_result = env.step(action)
                obs = step_result.observation
                reward = step_result.reward if step_result.reward is not None else 0.0
                done = step_result.done
                score = obs.score_so_far
                rewards.append(reward)
                steps_taken = 1
                success = score >= SUCCESS_SCORE_THRESHOLD

                action_str = f"issues={len(action.issues)},fixes={len(action.fixes)}"
                log_step(step=1, action=action_str, reward=reward, done=done, error=None)

            except Exception as e:
                err = str(e).replace("\n", " ")
                log_step(step=1, action="error", reward=0.00, done=True, error=err)
                rewards = [0.0]
                score = 0.0
                success = False
                steps_taken = 1

            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
            scores[task_name] = score

    print("", flush=True)
    print("=" * 60, flush=True)
    print("FINAL SCORES", flush=True)
    print("=" * 60, flush=True)
    for task, s in scores.items():
        print(f"  {task:<25}: {s:.3f}", flush=True)
    avg = sum(scores.values()) / len(scores)
    print(f"  {'AVERAGE':<25}: {avg:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()