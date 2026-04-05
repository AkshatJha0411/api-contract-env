---
title: API Contract Env
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 📋 APIContractEnv

An OpenEnv RL environment where an AI agent learns to review OpenAPI specifications and identify contract violations — just like a senior API architect would during a real code review.

## Overview

Poor API design causes integration failures, security breaches, and broken client contracts. **APIContractEnv** challenges an agent to inspect OpenAPI 3.0 specs and identify real issues: missing authentication, undocumented error responses, breaking changes between versions, inconsistent naming, and dangerous endpoint design.

This mirrors what every platform team, API gateway, and senior engineer does before shipping an API.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `missing_fields` | Easy | Find missing auth, 4xx/5xx responses, and required markers in a spec |
| `breaking_changes` | Medium | Identify all breaking changes between v1 and v2 of an API |
| `full_audit` | Hard | Full audit — REST violations, naming inconsistencies, security issues, dangerous endpoints |

## Action & Observation Space

### Action (`APIContractAction`)
```json
{
  "issues": [
    "GET /users missing 401 Unauthorized response",
    "POST /users/{id} path parameter missing required: true"
  ],
  "fixes": [
    "Add 401 response to GET /users with schema for error body",
    "Set required: true on the id path parameter"
  ]
}
```

### Observation (`APIContractObservation`)
```json
{
  "spec_content": "{ \"openapi\": \"3.0.0\", ... }",
  "task_name": "missing_fields",
  "task_description": "Review the OpenAPI spec and find missing required elements...",
  "step_feedback": "Auth issues found: yes | Error codes: 50% | Required markers: no",
  "score_so_far": 0.72
}
```

## Reward Function

```
score = 0.40 * issue_detection_score   # found the right problems
      + 0.30 * fix_correctness_score   # proposed valid, implementable fixes
      + 0.20 * precision_score         # no hallucinated or irrelevant issues
      + 0.10 * format_score            # parseable structured JSON response
```

All scores are in `[0.0, 1.0]` with partial credit throughout — never binary.

## Setup

### Install locally
```bash
pip install -r requirements.txt
```

### Run server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run with Docker
```bash
docker build -t api-contract-env .
docker run -p 7860:7860 api-contract-env
```

### Verify
```bash
curl http://localhost:7860/health
# → {"status": "healthy"}
```

## Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

## Baseline Scores

| Task | Score |
|------|-------|
| missing_fields | 0.850 |
| breaking_changes | 0.950 |
| full_audit | 0.637 |
| **Average** | **0.812** |

*(Scores obtained with Qwen/Qwen2.5-72B-Instruct)*

*(Run inference.py to generate baseline scores)*

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face / API key |
| `ENV_BASE_URL` | Environment server URL (default: http://localhost:7860) |
