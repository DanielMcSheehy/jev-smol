"""API + engine tests. Mock-backend tests are torch-free; one real-model test
runs only when transformers/torch are installed AND the SmolVLM2 weights are
already in the local HF cache."""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from jev.api.app import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(backend="mock", model="jev-mock"))


BASE_REQUEST = {
    "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "team": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Payments", "technical": "Outages", "sales": "Plans"},
        },
        "severity": {
            "type": "score",
            "instructions": "How severe?",
            "criteria": ["No impact", "Minor", "Major", "Critical"],
        },
    },
}


def _png_data_url(size: int = 8) -> str:
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (size, size), (10, 20, 30)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Response shapes
# --------------------------------------------------------------------------- #


def test_noul_shape(client):
    resp = client.post("/v1/systemone", json={"state": "x", "questions": {"a": {"type": "noul"}}})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == "jev-mock"
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert isinstance(body["usage"]["input_tokens"], int) and body["usage"]["input_tokens"] >= 1
    assert body["usage"]["output_tokens"] == 0
    ans = body["answers"]["a"]
    assert set(ans) == {"type", "noul"}
    assert ans["type"] == "noul"
    assert 0.0 <= ans["noul"] <= 1.0
    assert round(ans["noul"], 4) == ans["noul"]


def test_choice_shape(client):
    resp = client.post("/v1/systemone", json={"state": "x", "questions": {"a": {
        "type": "choice", "criteria": {"b": "B desc", "c": "C desc"}}}})
    assert resp.status_code == 200
    ans = resp.json()["answers"]["a"]
    assert set(ans) == {"type", "choice", "probabilities", "confidence"}
    assert ans["type"] == "choice"
    assert set(ans["probabilities"]) == {"b", "c"}
    total = sum(ans["probabilities"].values())
    assert abs(total - 1.0) < 0.02
    assert ans["choice"] == max(ans["probabilities"], key=ans["probabilities"].get)
    assert ans["confidence"] == ans["probabilities"][ans["choice"]]
    for v in ans["probabilities"].values():
        assert round(v, 4) == v


def test_score_shape_and_legend_echo(client):
    resp = client.post("/v1/systemone", json={"state": "x", "questions": {"a": {
        "type": "score", "criteria": ["Calm", "Frustrated", "Angry"]}}})
    assert resp.status_code == 200
    ans = resp.json()["answers"]["a"]
    assert set(ans) == {"type", "score", "legend", "probabilities", "confidence"}
    assert ans["type"] == "score"
    assert ans["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Angry"}
    assert set(ans["probabilities"]) == {"0", "1", "2"}
    assert abs(sum(ans["probabilities"].values()) - 1.0) < 0.02
    expected = round(sum(int(k) * v for k, v in ans["probabilities"].items()), 4)
    assert ans["score"] == expected
    assert ans["confidence"] == max(ans["probabilities"].values())


def test_three_question_response_keys(client):
    resp = client.post("/v1/systemone", json=BASE_REQUEST)
    assert resp.status_code == 200
    answers = resp.json()["answers"]
    assert set(answers) == {"urgent", "team", "severity"}
    assert set(answers["urgent"]) == {"type", "noul"}
    assert set(answers["team"]) == {"type", "choice", "probabilities", "confidence"}
    assert set(answers["severity"]) == {"type", "score", "legend", "probabilities", "confidence"}


def test_mock_determinism(client):
    a = client.post("/v1/systemone", json=BASE_REQUEST).json()
    b = client.post("/v1/systemone", json=BASE_REQUEST).json()
    assert a == b


def test_probability_values_finite(client):
    body = client.post("/v1/systemone", json=BASE_REQUEST).json()
    for ans in body["answers"].values():
        for v in ans.get("probabilities", {}).values():
            assert v == v and 0.0 <= v <= 1.0  # finite, in range


# --------------------------------------------------------------------------- #
# Validation (422) and unsupported inputs (400)
# --------------------------------------------------------------------------- #


def test_422_choice_with_one_option(client):
    r = client.post("/v1/systemone", json={"state": "x", "questions": {"a": {
        "type": "choice", "criteria": {"only": "One option"}}}})
    assert r.status_code == 422


def test_422_bad_question_id(client):
    r = client.post("/v1/systemone", json={"state": "x", "questions": {
        "bad id!": {"type": "noul"}}})
    assert r.status_code == 422


def test_422_too_many_questions(client):
    questions = {f"q{i}": {"type": "noul"} for i in range(65)}
    r = client.post("/v1/systemone", json={"state": "x", "questions": questions})
    assert r.status_code == 422


def test_422_five_images(client):
    img = _png_data_url()
    r = client.post("/v1/systemone", json={
        "state": "x", "questions": {"a": {"type": "noul"}}, "images": [img] * 5})
    assert r.status_code == 422


def test_422_malformed_data_url(client):
    r = client.post("/v1/systemone", json={
        "state": "x", "questions": {"a": {"type": "noul"}}, "images": ["http://not/a/dataurl"]})
    assert r.status_code == 422


def test_422_bad_content_type(client):
    r = client.post("/v1/systemone", json={
        "state": "x", "questions": {"a": {"type": "noul"}},
        "images": [{"content_type": "image/gif", "base64": "AAAA"}]})
    assert r.status_code == 422


def test_videos_400(client):
    r = client.post("/v1/systemone", json={
        "state": "x", "questions": {"a": {"type": "noul"}}, "videos": ["clip.mp4"]})
    assert r.status_code == 400
    err = r.json()["error"]
    assert set(err) == {"type", "message"}
    assert err["type"] == "unsupported"


# --------------------------------------------------------------------------- #
# Meta routes
# --------------------------------------------------------------------------- #


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "mock"
    assert body["model"] == "jev-mock"


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == "jev-mock"
    assert data[0]["object"] == "model"


def test_alias_route(client):
    direct = client.post("/v1/systemone", json=BASE_REQUEST)
    alias = client.post("/v1/decision", json=BASE_REQUEST)
    assert alias.status_code == 200
    assert alias.json() == direct.json()


def test_413_body_too_large(client):
    big = b"x" * (13 * 1024 * 1024 + 1)
    r = client.post("/v1/systemone", content=big,
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.json()["error"]["type"] == "body_too_large"


def test_default_app_routes_exist():
    # The env-configured module-level app must expose the same routes.
    from jev.api.app import app as default_app

    paths = {route.path for route in default_app.routes}
    assert {"/v1/systemone", "/v1/decision", "/healthz", "/v1/models"} <= paths


# --------------------------------------------------------------------------- #
# Real model (skipped unless torch + transformers + cached weights exist)
# --------------------------------------------------------------------------- #


def _vlm_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    try:
        from huggingface_hub import snapshot_download

        snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", local_files_only=True)
        return True
    except Exception:
        return False


VLM_AVAILABLE = _vlm_available()


@pytest.mark.skipif(not VLM_AVAILABLE, reason="transformers/torch or SmolVLM2 weights not available locally")
def test_real_vlm_text_decision(capsys):
    """One real text-only decision: 2 questions, fp32 CPU, schema-valid output."""
    from jev.engine_vlm import VlmEngine
    from jev.schema import JevRequest

    engine = VlmEngine(device="cpu", model_id="jev-real-test")
    request = JevRequest(
        state="Our checkout started returning errors and orders are blocked.",
        questions={
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    )
    response = engine.decide(request)
    output = response.model_dump_json()
    print(output)

    assert response.model == "jev-real-test"
    assert response.usage.input_tokens > 0 and response.usage.output_tokens == 0
    score = response.answers["urgency"]
    assert score.type == "score"
    assert abs(sum(score.probabilities.values()) - 1.0) < 0.02
    assert all(v == v for v in score.probabilities.values())  # finite
    assert 0.0 <= score.score <= 2.0
    noul = response.answers["outage"]
    assert noul.type == "noul" and 0.0 <= noul.noul <= 1.0
    capsys.readouterr()  # output printed above via print()
