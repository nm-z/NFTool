from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_missing_api_key_rejected_before_validation():
    """Ensure missing X-API-Key is rejected before request body validation.

    Empty body requests should be rejected with 406 before body parsing/validation runs.
    """
    payload = {}  # empty body would normally trigger validation error 422
    res = client.post("/api/v1/training/train", json=payload)
    assert res.status_code == 406
    data = res.json()
    detail = data.get("detail", "")
    assert "Missing X-API-Key" in detail or "X-API-Key" in detail

