"""
APIContractEnv — Server-side environment logic and graders.
"""

import os
import uuid
import json
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import APIContractAction, APIContractObservation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SPECS_DIR = os.path.join(os.path.dirname(__file__), "specs")

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS = {
    "missing_fields": {
        "description": (
            "Review the OpenAPI spec below and identify missing required elements. "
            "Look for: (1) missing authentication/security definitions, "
            "(2) endpoints missing 4xx/5xx error responses (especially 400, 401, 403, 404, 500), "
            "(3) request bodies missing 'required' markers, "
            "(4) path parameters missing 'required: true', "
            "(5) missing response schemas. "
            "For each issue found, propose a concrete fix."
        ),
        "spec_file": "easy.json",
        "spec_label": "SPEC (v1.0)",
    },
    "breaking_changes": {
        "description": (
            "You are given two versions of an API spec (V1 and V2). "
            "Identify ALL breaking changes introduced in V2 that would break existing clients. "
            "Breaking changes include: renamed parameters, changed parameter types, "
            "removed endpoints, changed required fields, removed response codes, "
            "changed authentication schemes, changed HTTP status codes for success responses. "
            "For each breaking change, explain the impact and propose a migration strategy."
        ),
        "spec_file": ["medium_v1.json", "medium_v2.json"],
        "spec_label": "TWO-VERSION COMPARISON",
    },
    "full_audit": {
        "description": (
            "Perform a complete API contract audit on the spec below. Identify ALL of the following: "
            "(1) REST design violations (wrong HTTP verbs, RPC-style paths like /getProducts or /deleteProduct), "
            "(2) Inconsistent naming conventions (mixed camelCase/snake_case/PascalCase in params or paths), "
            "(3) Security issues (sensitive data in request/response bodies, missing auth, credentials in specs), "
            "(4) Missing error responses (no 4xx/5xx defined), "
            "(5) Dangerous endpoint design (destructive actions via GET, overly broad data exposure). "
            "For every issue, propose a concrete fix."
        ),
        "spec_file": "hard.json",
        "spec_label": "SPEC (audit target)",
    },
}

TASK_ORDER = list(TASKS.keys())

# ---------------------------------------------------------------------------
# Ground truth for graders
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "missing_fields": {
        "no_auth": True,                  # no security schemes defined
        "missing_error_codes": ["400", "401", "404", "500"],
        "missing_required_params": True,   # path params missing required:true
        "missing_request_required": True,  # POST body missing required field list
    },
    "breaking_changes": {
        "renamed_params": ["user_id", "order_id", "customer_id"],  # user_id→customer_id
        "type_changes": ["user_id", "customer_id"],                 # int→string
        "removed_endpoints": ["/orders/{order_id}/status"],
        "removed_response_codes": ["401"],
        "auth_scheme_change": True,        # bearerAuth → apiKey
        "status_code_change": True,        # POST 201→200
        "newly_required_params": ["limit"],
        "renamed_field_in_body": ["quantity", "qty"],
        "removed_field_in_body": ["coupon_code"],
    },
    "full_audit": {
        "rest_violations": ["getProducts", "deleteProduct", "CreateUser"],
        "naming_inconsistencies": ["CategoryId", "maxprice", "Username", "Password", "SSN", "credit_card_number"],
        "security_issues": ["password", "SSN", "credit_card_number"],
        "missing_auth": True,
        "missing_error_responses": True,
        "dangerous_endpoints": ["deleteProduct GET", "/orders GET all"],
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_spec(filename) -> str:
    if isinstance(filename, list):
        parts = []
        for i, f in enumerate(filename):
            with open(os.path.join(SPECS_DIR, f)) as fh:
                label = f"=== V{i+1} SPEC ({f}) ===\n"
                parts.append(label + json.dumps(json.load(fh), indent=2))
        return "\n\n".join(parts)
    with open(os.path.join(SPECS_DIR, filename)) as fh:
        return json.dumps(json.load(fh), indent=2)


def _exact_hits(text: str, targets: list) -> float:
    """Score based on exact substring matches (case-insensitive). Strict."""
    t = text.lower()
    hits = sum(1 for target in targets if target.lower() in t)
    return hits / len(targets) if targets else 0.0


def _precision_penalty(found: int, expected: int) -> float:
    """
    Penalise over-reporting. If agent reports 3x more issues than expected,
    score is halved. Encourages precision over flooding.
    """
    if found == 0:
        return 0.0
    ratio = found / max(expected, 1)
    if ratio <= 2.0:
        return 1.0
    elif ratio <= 3.0:
        return 0.7
    else:
        return 0.4


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def grade_missing_fields(action: APIContractAction) -> tuple:
    """
    Strict grader for missing_fields task.
    Agent must identify:
      - No security/auth scheme defined anywhere in the spec
      - Specific missing HTTP error codes: 400, 401, 404, 500
      - Path parameter 'id' missing required:true (appears twice)
      - POST /users body missing 'required' field list
    Fixes must reference specific remediation (not generic advice).
    """
    all_text = " ".join(action.issues + action.fixes).lower()
    issues_text = " ".join(action.issues).lower()
    fixes_text = " ".join(action.fixes).lower()

    # 1. Must explicitly call out missing auth/security (not just mention "401")
    auth_terms = ["no security", "missing security", "no auth", "missing auth",
                  "security scheme", "securityschemes", "no authentication",
                  "unauthenticated", "bearer", "api key"]
    auth_score = min(1.0, sum(1 for t in auth_terms if t in issues_text) / 2)

    # 2. Must name specific missing error codes — partial credit per code
    error_codes = ["400", "401", "404", "500"]
    # Only count if mentioned in issues (not just fixes)
    error_score = _exact_hits(issues_text, error_codes)

    # 3. Must mention 'required' specifically for path params or request body
    required_in_issues = any(t in issues_text for t in
        ["required: true", "required:true", "required parameter", "path parameter",
         "missing required", "not marked required", "required field"])
    required_score = 1.0 if required_in_issues else 0.0

    # 4. Fix quality — fixes must name specific actions, not be generic
    specific_fix_terms = ["add 4", "add 5", "define security", "securityschemes",
                          "required: true", "bearer", "401", "404", "500"]
    fix_score = min(1.0, sum(1 for t in specific_fix_terms if t in fixes_text) / 3)

    # 5. Precision penalty — spec has ~6 real issues, penalise excessive over-reporting
    precision = _precision_penalty(len(action.issues), 6)

    total = round(
        0.30 * auth_score +
        0.30 * error_score +
        0.20 * required_score +
        0.10 * fix_score +
        0.10 * precision,
        3
    )
    feedback = (
        f"Auth identified: {auth_score:.0%} | "
        f"Error codes named: {error_score:.0%} | "
        f"Required markers: {'yes' if required_score else 'no'} | "
        f"Fix specificity: {fix_score:.0%} | "
        f"Precision: {precision:.0%}"
    )
    return total, feedback


def grade_breaking_changes(action: APIContractAction) -> tuple:
    """
    Strict grader for breaking_changes task.
    Agent must identify exact breaking changes between v1 and v2:
      - 'user_id' renamed to 'customer_id' (exact names required)
      - user_id type changed integer -> string
      - 'limit' changed from optional to required
      - endpoint /orders/{order_id}/status removed entirely
      - auth scheme changed bearerAuth -> apiKey (exact scheme names)
      - POST /orders success code changed 201 -> 200
      - field 'quantity' renamed to 'qty' in request body
      - field 'coupon_code' removed from request body
      - 401 response removed from all endpoints
    """
    issues_text = " ".join(action.issues).lower()
    fixes_text = " ".join(action.fixes).lower()
    all_text = issues_text + " " + fixes_text

    # Must name exact parameter that was renamed
    rename_score = 0.0
    if "user_id" in issues_text and "customer_id" in issues_text:
        rename_score = 1.0
    elif "user_id" in issues_text or "customer_id" in issues_text:
        rename_score = 0.4

    # Must name the removed endpoint exactly
    removed_endpoint_score = 1.0 if "/orders/{order_id}/status" in issues_text or "order_id}/status" in issues_text else 0.0

    # Must identify auth scheme change with both scheme names
    auth_score = 0.0
    if "bearerauth" in issues_text or "bearer" in issues_text:
        if "apikey" in issues_text or "api key" in issues_text or "x-api-key" in issues_text:
            auth_score = 1.0
        else:
            auth_score = 0.4

    # Must catch field-level breaking changes (qty/quantity, coupon_code, 201->200)
    field_changes = ["quantity", "qty", "coupon_code", "coupon", "201"]
    field_score = _exact_hits(issues_text, field_changes)

    # Must catch type change (integer to string)
    type_score = 1.0 if ("integer" in issues_text and "string" in issues_text) else 0.0

    # Fix quality — must suggest versioning or backward-compat strategy
    fix_terms = ["version", "deprecat", "backward compat", "alias", "maintain",
                 "migration guide", "keep", "v1", "v2"]
    fix_score = min(1.0, sum(1 for t in fix_terms if t in fixes_text) / 2)

    # Precision — spec has 8-9 real breaking changes
    precision = _precision_penalty(len(action.issues), 9)

    total = round(
        0.25 * rename_score +
        0.20 * removed_endpoint_score +
        0.20 * auth_score +
        0.15 * field_score +
        0.10 * type_score +
        0.05 * fix_score +
        0.05 * precision,
        3
    )
    feedback = (
        f"Param rename (user_id->customer_id): {rename_score:.0%} | "
        f"Removed endpoint: {removed_endpoint_score:.0%} | "
        f"Auth scheme change: {auth_score:.0%} | "
        f"Field changes: {field_score:.0%} | "
        f"Type change: {type_score:.0%}"
    )
    return total, feedback


def grade_full_audit(action: APIContractAction) -> tuple:
    """
    Strict grader for full_audit task.
    Agent must identify specific named issues:
      REST violations: /getProducts, /deleteProduct, /users/CreateUser (exact paths)
      Security: 'password' in request body, 'SSN' field, 'credit_card_number' field
      Naming: mix of camelCase (CategoryId, Username) and snake_case (credit_card_number, user_id)
      Dangerous: DELETE via GET (/deleteProduct), GET /orders returns ALL users' orders
      Missing auth: no securitySchemes defined anywhere
    """
    issues_text = " ".join(action.issues).lower()
    fixes_text = " ".join(action.fixes).lower()

    # 1. REST violations — must name the specific bad paths
    rest_paths = ["/getproducts", "/deleteproduct", "/users/createuser"]
    rest_hits = sum(1 for p in rest_paths if p in issues_text)
    rest_score = rest_hits / len(rest_paths)

    # 2. Security — must name specific sensitive fields
    security_fields = ["password", "ssn", "credit_card_number", "credit card"]
    sec_hits = sum(1 for f in security_fields if f in issues_text)
    security_score = sec_hits / len(security_fields)

    # 3. Naming — must identify the inconsistency (not just say "inconsistent")
    naming_specific = [
        ("categoryid", "camelcase"),       # CategoryId
        ("username", "password"),           # PascalCase fields
        ("credit_card_number", "snake"),    # snake_case mixed in
    ]
    naming_hits = sum(
        1 for (a, b) in naming_specific
        if a in issues_text and b in issues_text
    )
    naming_score = naming_hits / len(naming_specific)

    # 4. Dangerous endpoints — must specifically call out GET-as-DELETE
    danger_score = 0.0
    if "/deleteproduct" in issues_text and "get" in issues_text:
        danger_score += 0.5
    if "all orders" in issues_text or ("get" in issues_text and "all users" in issues_text):
        danger_score += 0.5

    # 5. Missing auth — must say no security/auth defined
    auth_score = 1.0 if any(t in issues_text for t in
        ["no security", "no auth", "missing auth", "no authentication",
         "securityschemes", "unauthenticated"]) else 0.0

    # 6. Fix quality — fixes must name specific HTTP methods or field removals
    fix_specifics = ["delete /deleteproduct", "use delete", "use put", "use patch",
                     "remove password", "remove ssn", "remove credit", "hash",
                     "encrypt", "snake_case", "camelcase", "consistent"]
    fix_score = min(1.0, sum(1 for t in fix_specifics if t in fixes_text) / 3)

    # Precision — spec has ~12 real issues, penalise flooding
    precision = _precision_penalty(len(action.issues), 12)

    total = round(
        0.25 * rest_score +
        0.25 * security_score +
        0.15 * naming_score +
        0.15 * danger_score +
        0.10 * auth_score +
        0.05 * fix_score +
        0.05 * precision,
        3
    )
    feedback = (
        f"REST violations: {rest_score:.0%} | "
        f"Security fields: {security_score:.0%} | "
        f"Naming specificity: {naming_score:.0%} | "
        f"Dangerous endpoints: {danger_score:.0%} | "
        f"Auth missing: {'yes' if auth_score else 'no'}"
    )
    return total, feedback


GRADERS = {
    "missing_fields": grade_missing_fields,
    "breaking_changes": grade_breaking_changes,
    "full_audit": grade_full_audit,
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class APIContractEnvironment(Environment):

    def __init__(self, transform=None, rubric=None):
        super().__init__(transform=transform, rubric=rubric)
        self._task_name: Optional[str] = None
        self._spec_content: str = ""
        self._episode_count: int = 0
        self._current_state = State(episode_id=str(uuid.uuid4()), step_count=0)

    def reset(self, seed=None, episode_id=None, **kwargs) -> APIContractObservation:
        self._episode_count += 1
        self._task_name = TASK_ORDER[(self._episode_count - 1) % len(TASK_ORDER)]
        task = TASKS[self._task_name]
        self._spec_content = load_spec(task["spec_file"])
        self._current_state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )
        return APIContractObservation(
            spec_content=self._spec_content,
            task_name=self._task_name,
            task_description=task["description"],
            step_feedback="",
            score_so_far=0.0,
            done=False,
            reward=None,
        )

    def step(self, action: APIContractAction, timeout_s=None, **kwargs) -> APIContractObservation:
        self._current_state.step_count += 1
        score, feedback = GRADERS[self._task_name](action)
        return APIContractObservation(
            spec_content=self._spec_content,
            task_name=self._task_name,
            task_description=TASKS[self._task_name]["description"],
            step_feedback=feedback,
            score_so_far=score,
            done=True,
            reward=score,
        )

    @property
    def state(self) -> State:
        return self._current_state
