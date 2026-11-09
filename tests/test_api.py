"""Integration tests for the FastAPI service."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_sample_payload(client):
    response = client.get("/sample")
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "chunks" in data
    assert "answer" in data
    assert len(data["chunks"]) > 0


def test_verify_empty_answer_fails(client):
    response = client.post(
        "/verify",
        json={"chunks": ["Some chunk"], "answer": "   "},
    )
    assert response.status_code == 400


def test_verify_endpoint_mocked(client, monkeypatch):
    from groundcheck.verify import CrossEncoderVerifier

    # Mock score_pairs to return deterministic result
    def mock_score_pairs(self, premises, hypotheses):
        return [
            {"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}
            for _ in premises
        ]

    monkeypatch.setattr(CrossEncoderVerifier, "score_pairs", mock_score_pairs)

    payload = {
        "question": "What is Python?",
        "chunks": ["Python is an interpreted high-level language."],
        "answer": "Python is an interpreted programming language.",
        "policy": {
            "supported_threshold": 0.70,
            "weak_threshold": 0.40,
            "action": "pass",
        },
    }
    response = client.post("/verify", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "ok"
    assert res["passed"] is True
    assert len(res["claims"]) == 1
    assert res["claims"][0]["verdict"] == "SUPPORTED"
