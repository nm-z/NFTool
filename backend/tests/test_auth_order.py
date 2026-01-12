import json

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_missing_api_key_rejected_before_validation():
    """
    Ensure that missing X-API-Key is rejected with 406 before request body
    validation runs (i.e., middleware runs prior to body parsing/validation).
    """
    payload = {}  # empty body would normally trigger validation error 422
    res = client.post("/api/v1/training/train", json=payload)
    assert res.status_code == 406, f"Expected 406, got {res.status_code}: {res.text}"
    data = res.json()
    assert "Missing X-API-Key" in data.get("detail", "") or "X-API-Key" in data.get("detail", "")

