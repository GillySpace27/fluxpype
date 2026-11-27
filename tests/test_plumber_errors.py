"""One error shape (A3-08): every error body is {"detail": text, "code": stable code}; 404 for an unknown id, 409
only for a state conflict; the MCP server shows the code."""

import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_plumber_mcp import H, TOKEN, Rig  # noqa: E402,F401
from fastapi.testclient import TestClient  # noqa: E402

from fluxpype.plumber.server import create_app, error_code  # noqa: E402


@pytest.fixture
def rig(make_scheduler):
    sched = make_scheduler()
    app = create_app(sched, TOKEN, demo=True, tick_interval=0.1)
    with TestClient(app, base_url="http://localhost") as tc:
        yield Rig(tc, sched)


CASES = [
    ("get", "/api/jobs/nope", 404, "not_found"),
    ("post", "/api/jobs/nope/cancel", 404, "not_found"),
    ("post", "/api/jobs/nope/retry", 404, "not_found"),
    ("delete", "/api/jobs/nope", 404, "not_found"),
    ("get", "/api/tasks/nope", 404, "not_found"),
    ("get", "/api/tasks/nope/log", 404, "not_found"),
    ("get", "/api/nonsense", 404, "not_found"),
]


@pytest.mark.parametrize("method,url,status,code", CASES)
def test_unknown_ids_are_404_with_a_code(rig, method, url, status, code):
    r = getattr(rig.tc, method)(url, headers=H)
    assert r.status_code == status
    body = r.json()
    assert body["code"] == code and isinstance(body["detail"], str) and body["detail"]


def test_unauthorized_and_invalid_carry_codes(rig):
    r = rig.tc.get("/api/jobs/nope", headers={"X-Flux-Token": "wrong"})
    assert (r.status_code, r.json()["code"]) == (401, "unauthorized")
    r = rig.tc.post("/api/jobs", json={"profile": 5}, headers=H)
    assert r.status_code == 422 and r.json()["code"] == "invalid" and isinstance(r.json()["detail"], list)
    r = rig.tc.post("/api/jobs", json={"profile": "nope"}, headers=H)
    assert (r.status_code, r.json()["code"]) == (400, "invalid")


def test_deleting_a_live_job_is_a_409_conflict(rig):
    job = rig.tc.post(
        "/api/jobs", json={"profile": "t_one", "backend": "fake", "scenario": "ok", "speed": 1}, headers=H
    ).json()
    r = rig.tc.delete(f"/api/jobs/{job['id']}", headers=H)
    assert (r.status_code, r.json()["code"]) == (409, "conflict")
    assert "cancel the job" in r.json()["detail"]


def test_status_to_code_table():
    assert [error_code(s) for s in (400, 404, 409, 422, 429, 500, 503)] == [
        "invalid",
        "not_found",
        "conflict",
        "invalid",
        "busy",
        "internal",
        "unavailable",
    ]


def test_mcp_error_text_includes_the_code(rig):
    err, text = rig.call("get_job", job_id="nope")
    assert err and "unknown job" in text and "[not_found]" in text
    err, text = rig.call("retry_failed", job_id="nope")
    assert err and "no such job" in text and "[not_found]" in text
