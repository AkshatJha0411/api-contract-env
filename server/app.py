"""
APIContractEnv — FastAPI server entry point.
"""

from openenv.core.env_server.http_server import create_app
from models import APIContractAction, APIContractObservation
from server.environment import APIContractEnvironment

app = create_app(APIContractEnvironment, APIContractAction, APIContractObservation)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
