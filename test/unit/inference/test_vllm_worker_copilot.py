import pytest
import json
from fastapi.testclient import TestClient
from your_application import app  # import your FastAPI app

# Mocks and fixtures
@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_environment(monkeypatch):
    monkeypatch.setenv("config_node", "test_config_node")

@pytest.fixture
def mock_acquire_worker_semaphore(mocker):
    mocker.patch('path.to.your.acquire_worker_semaphore')

@pytest.fixture
def mock_release_worker_semaphore(mocker):
    mocker.patch('path.to.your.release_worker_semaphore')

@pytest.fixture
def mock_worker_generate(mocker):
    async def mock_generate(payload):
        return {"result": "mocked_result"}
    mocker.patch('path.to.your.worker.generate', side_effect=mock_generate)

@pytest.fixture
def mock_engine_abort(mocker):
    mocker.patch('path.to.your.engine.abort')

# Test function
def test_generate_sql(client, mock_environment, mock_acquire_worker_semaphore, mock_release_worker_semaphore, mock_worker_generate, mock_engine_abort):
    request_payload = {"some": "data"}

    response = client.post("/custom_prompt", json=request_payload)
    assert response.status_code == 200
    assert response.json() == {"result": "mocked_result"}

