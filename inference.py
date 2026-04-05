"""
APIContractEnv — Baseline Inference Script

Mandatory stdout format:
  [START] task=<task_name> env=api_contract model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import os
import json
import sys
from openai import OpenAI
from client import APIContractEnv
from models import APIContractAction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "api_contract"

TASKS = ["missing_fields", "breaking_changes", "full_audit"]

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


def call_llm(client: OpenAI, observation) -> APIContractAction:
    """Call the LLM and parse its response into an APIContractAction."""
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
        # print(f"DEBUG RAW:\n{raw[:500]}\n", flush=True)
        
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        issues = parsed.get("issues", [])
        fixes = parsed.get("fixes", [])

        # Ensure equal length
        min_len = min(len(issues), len(fixes))
        issues = issues[:min_len]
        fixes = fixes[:min_len]

        return APIContractAction(
            issues=issues,
            fixes=fixes,
            raw_response=raw,
        )

    except (json.JSONDecodeError, KeyError, Exception) as e:
        # Fallback: return empty action rather than crash
        return APIContractAction(
            issues=[f"Parse error: {str(e)}"],
            fixes=["Fix parse error"],
            raw_response=str(e),
        )


def run_task(env_url: str, task_name: str, llm_client: OpenAI) -> float:
    """Run one task episode and return the score."""
    rewards = []
    last_error = "null"
    success = False
    score = 0.0

    with APIContractEnv(base_url=env_url).sync() as env:
        try:
            # ---- RESET ----
            reset_result = env.reset()
            obs = reset_result.observation

            print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

            step_num = 0
            done = False

            while not done:
                step_num += 1

                # Get action from LLM
                action = call_llm(llm_client, obs)

                # Step the environment
                step_result = env.step(action)
                obs = step_result.observation
                reward = step_result.reward if step_result.reward is not None else 0.0
                done = step_result.done

                rewards.append(reward)
                score = obs.score_so_far

                action_str = f"issues={len(action.issues)},fixes={len(action.fixes)}"
                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward={reward:.2f} done={str(done).lower()} error={last_error}",
                    flush=True,
                )

            success = score >= 0.5

        except Exception as e:
            last_error = str(e).replace("\n", " ")
            print(f"[STEP] step=1 action=error reward=0.00 done=true error={last_error}", flush=True)
            rewards = [0.0]
            score = 0.0
            success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    steps = len(rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
    return score


def main():
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    print("=" * 60, flush=True)
    print(f"APIContractEnv — Baseline Inference", flush=True)
    print(f"Model : {MODEL_NAME}", flush=True)
    print(f"Env   : {ENV_BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    scores = {}

    with APIContractEnv(base_url=ENV_BASE_URL).sync() as env:
        for task_name in TASKS:
            rewards = []
            score = 0.0
            success = False

            try:
                reset_result = env.reset()
                obs = reset_result.observation

                print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

                action = call_llm(llm_client, obs)
                step_result = env.step(action)
                obs = step_result.observation
                reward = step_result.reward if step_result.reward is not None else 0.0
                rewards.append(reward)
                score = obs.score_so_far
                success = score >= 0.5

                action_str = f"issues={len(action.issues)},fixes={len(action.fixes)}"
                print(
                    f"[STEP] step=1 action={action_str} "
                    f"reward={reward:.2f} done=true error=null",
                    flush=True,
                )

            except Exception as e:
                err = str(e).replace("\n", " ")
                print(f"[STEP] step=1 action=error reward=0.00 done=true error={err}", flush=True)
                rewards = [0.0]
                score = 0.0
                success = False

            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[END] success={str(success).lower()} steps=1 "
                f"score={score:.2f} rewards={rewards_str}",
                flush=True,
            )
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
