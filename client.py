"""
APIContractEnv — Client for connecting to the environment server.
"""

from typing import Dict
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import APIContractAction, APIContractObservation


class APIContractEnv(EnvClient[APIContractAction, APIContractObservation, State]):

    def _step_payload(self, action: APIContractAction) -> Dict:
        return {
            "issues": action.issues,
            "fixes": action.fixes,
            "raw_response": action.raw_response,
        }

    def _parse_result(self, payload: Dict) -> StepResult[APIContractObservation]:
        obs = payload.get("observation", {})
        return StepResult(
            observation=APIContractObservation(
                spec_content=obs.get("spec_content", ""),
                task_name=obs.get("task_name", ""),
                task_description=obs.get("task_description", ""),
                step_feedback=obs.get("step_feedback", ""),
                score_so_far=obs.get("score_so_far", 0.0),
                done=payload.get("done", False),
                reward=payload.get("reward"),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
