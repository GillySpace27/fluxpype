"""The Ask bar's backend, with a stand-in for the Claude CLI (no network, no quota)."""

import json
import stat
import sys

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fluxpype.plumber import ask  # noqa: E402
from fluxpype.plumber.server import create_app  # noqa: E402

TOKEN = "t0k3n-for-tests"
H = {"X-Flux-Token": TOKEN}

FAKE = """#!{py}
import json, sys
prompt = sys.stdin.read()
def ev(**d): print(json.dumps(d), flush=True)
def delta(t): ev(type="stream_event", event={{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": t}}}})
ev(type="system", subtype="init")
ev(type="stream_event", event={{"type": "content_block_start", "content_block": {{"type": "tool_use", "name": "mcp__fluxpype__status"}}}})
delta("Hel"); delta("lo")
delta("|" + ("--tools" in sys.argv and "tools-off" or "") + "|" + ("submit_job" in " ".join(sys.argv) and "write-listed" or "") + "|" + prompt[-40:])
ev(type="result", subtype="success", is_error=False, total_cost_usd=0.01, duration_ms=5)
"""


@pytest.fixture
def client(make_scheduler, tmp_path, monkeypatch):
    cli = tmp_path / "fake-claude"
    cli.write_text(FAKE.format(py=sys.executable))
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FLUX_PLUMBER_CLAUDE", str(cli))
    with TestClient(create_app(make_scheduler(), TOKEN, demo=True, tick_interval=0.1), base_url="http://localhost") as c:
        yield c


def events(resp):
    return [json.loads(x[6:]) for x in resp.text.split("\n\n") if x.startswith("data: ")]


def test_parse_maps_cli_events_and_ignores_the_rest():
    delta = {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}}
    think = {"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "x"}}}
    assert ask.parse(json.dumps(delta)) == {"type": "text", "text": "hi"}
    assert ask.parse(json.dumps(think)) is None and ask.parse("not json") is None
    assert ask.parse(json.dumps({"type": "result", "is_error": True, "result": "Not logged in"}))["type"] == "error"
    assert ask.parse(json.dumps({"type": "result", "is_error": False, "total_cost_usd": 0.02, "duration_ms": 9}))["cost"] == 0.02


def test_the_cli_is_stripped_down_and_never_given_write_tools():
    cmd = ask.command("claude", "/home", True)
    assert cmd[cmd.index("--tools") + 1] == "Read,Grep,Glob" and "--strict-mcp-config" in cmd and "--setting-sources" in cmd
    assert "Bash" not in " ".join(cmd) and "Edit" not in " ".join(cmd) and "--add-dir" in cmd  # the repo, read-only
    allowed = cmd[cmd.index("--allowedTools") + 1]
    assert "mcp__fluxpype__status" in allowed and "submit_job" not in allowed and "cancel" not in allowed
    assert "submit_job" in cmd[cmd.index("--disallowedTools") + 1]
    bare = ask.command("claude", "/home", False)  # without the MCP SDK: the docs and the repo, no live state
    assert "mcp__fluxpype" not in " ".join(bare) and bare[bare.index("--allowedTools") + 1] == "Read,Grep,Glob"
    sp = cmd[cmd.index("--system-prompt") + 1]
    assert "VNEIGHBOR" in sp and "flux-action" in sp  # the reference and the guide are both in
    assert "flux-plumb-mcp" in sp and "relax_threshold" in sp  # and the MCP tools doc and the config.ini defaults


def test_ask_streams_text_tool_and_done(client):
    r = client.post("/api/ask", json={"question": "which end is positive?", "page": "/jobs"}, headers=H)
    ev = events(r)
    assert [e["type"] for e in ev] == ["tool", "text", "text", "text", "done"]
    assert ev[0]["name"] == "status" and "".join(e["text"] for e in ev if e["type"] == "text").startswith("Hello|tools-off|")
    assert "write-listed" in "".join(e.get("text", "") for e in ev)  # listed only in --disallowedTools
    assert ev[-1]["cost"] == 0.01


def test_ask_without_the_cli_says_why(client, monkeypatch):
    monkeypatch.setattr(ask, "find_cli", lambda: None)
    assert client.get("/api/ask", headers=H).json()["available"] is False
    ev = events(client.post("/api/ask", json={"question": "hi"}, headers=H))
    assert ev[0]["type"] == "error" and "Claude Code CLI" in ev[0]["message"]


def test_ask_refuses_a_second_question_while_one_runs(client):
    assert ask._busy.acquire(blocking=False)
    try:
        ev = events(client.post("/api/ask", json={"question": "hi"}, headers=H))
    finally:
        ask._busy.release()
    assert ev[0]["type"] == "error" and "still running" in ev[0]["message"]


def test_ask_needs_the_token_and_a_question(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 401
    assert client.post("/api/ask", json={"question": ""}, headers=H).status_code == 422
