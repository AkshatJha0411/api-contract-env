"""
APIContractEnv — Typed models for Action and Observation.
"""

from typing import Optional, List
from pydantic import Field, ConfigDict
from openenv.core.env_server.types import Action, Observation


class APIContractAction(Action):
    """
    The agent's response to an API spec review task.

    Fields:
        issues      : List of detected issues, each as a plain-English string.
                      e.g. ["GET /users missing 401 response",
                            "Parameter 'user_id' renamed to 'id' in v2 — breaking change"]
        fixes       : Corresponding list of proposed fixes (same length as issues).
                      e.g. ["Add 401 Unauthorized to GET /users responses",
                            "Keep 'user_id' as alias or version the endpoint"]
        raw_response: Full free-text response from the model (optional, for logging).
    """
    model_config = ConfigDict(extra="allow")

    issues: List[str] = Field(default_factory=list)
    fixes: List[str] = Field(default_factory=list)
    raw_response: Optional[str] = Field(default=None)


class APIContractObservation(Observation):
    """
    What the agent sees at each step.

    Fields:
        spec_content     : The API spec(s) as a JSON string.
        task_name        : One of 'missing_fields', 'breaking_changes', 'full_audit'.
        task_description : Plain-English description of what the agent must do.
        step_feedback    : Feedback from the previous step (empty on reset).
        score_so_far     : Running score (0.0–1.0) for the current episode.
    """
    model_config = ConfigDict(extra="allow")

    spec_content: str = Field(default="")
    task_name: str = Field(default="")
    task_description: str = Field(default="")
    step_feedback: str = Field(default="")
    score_so_far: float = Field(default=0.0)
