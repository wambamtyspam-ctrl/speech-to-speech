"""Tests for Hermes Voice Server FastAPI endpoints."""

from unittest.mock import MagicMock, patch


def test_health_endpoint():
    """Test health check endpoint returns status ok."""
    from fastapi.testclient import TestClient
    from hermes_voice_server import app
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_status_endpoint():
    """Test status endpoint returns service info."""
    from fastapi.testclient import TestClient
    from hermes_voice_server import app
    
    client = TestClient(app)
    response = client.get("/status")
    
    assert response.status_code == 200
    data = response.json()
    assert "tts" in data
    assert "stt" in data


def test_models_endpoint():
    """Test models listing endpoint."""
    from fastapi.testclient import TestClient
    from hermes_voice_server import app
    
    client = TestClient(app)
    response = client.get("/v1/models")
    
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 2


def test_voices_list_endpoint():
    """Test voices list endpoint."""
    from fastapi.testclient import TestClient
    from hermes_voice_server import app
    
    client = TestClient(app)
    response = client.get("/voices/list")
    
    # Returns 503 when TTS not loaded, which is expected in test environment
    assert response.status_code in [200, 503]


def test_device_status_endpoint():
    """Test device status endpoint."""
    from fastapi.testclient import TestClient
    from hermes_voice_server import app
    
    client = TestClient(app)
    response = client.get("/device/status")
    
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert "current" in data


def test_synthesize_speech_empty_text():
    """Test speech synthesis rejects empty text."""
    from fastapi.testclient import TestClient
    from hermes_voice_server import app
    
    client = TestClient(app)
    # Empty input should return 422 (validation error from Pydantic)
    response = client.post("/v1/audio/speech", data={"input": ""})
    
    assert response.status_code == 422


def test_voice_clone_endpoint_exists():
    """Test voice clone endpoint exists."""
    from fastapi.testclient import TestClient
    from hermes_voice_server import app
    
    client = TestClient(app)
    # Should return 503 (service unavailable) since TTS not loaded, not 404
    response = client.post("/voice/clone", data={"text": "test"})
    
    assert response.status_code in [200, 503]  # Either success or service not ready
