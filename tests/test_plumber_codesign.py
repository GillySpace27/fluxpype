"""Co-design mode: the Ask token, conversations that resume a CLI session, modes, the daily cap and the wake turn.

The Claude CLI is a stand-in script that logs how it was called (no network, no quota)."""

import json
import os
import stat
import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fluxpype.plumber import ask, cli as plumber_cli, mcp_server, paths, proposals  # noqa: E402
from fluxpype.plumber.conversations import Conversations  # noqa: E402
from fluxpype.plumber.server import create_app  # noqa: E402

TOKEN = "t0k3n-for-tests"
ASK_TOKEN = "ask-t0k3n-for-tests-only"
H = {"X-Flux-Token": TOKEN}
A = {"X-Flux-Token": ASK_TOKEN}

FAKE = """#!{py}
import json, os, sys
prompt = sys.stdin.read()
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "prompt": prompt}}) + "\\n")
def ev(**d): print(json.dumps(d), flush=True)
lost = os.environ.get("FAKE_LOST") and "--resume" in sys.argv
if lost:
    ev(type="result", subtype="error_during_execution", is_error=True, errors=["No conversation found with session ID: x"])
    sys.exit(1)
ev(type="stream_event", event={{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": "answer to: " + prompt.split("Question: ")[-1][:40]}}}})
ev(type="result", subtype="success", is_error=False, total_cost_usd=float(os.environ.get("FAKE_COST", "0.5")), duration_ms=5,
   modelUsage={{"claude-opus-5": {{"inputTokens": 2, "outputTokens": 6, "cacheReadInputTokens": 1, "cacheCreationInputTokens": 5}}}})
"""


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    cli = tmp_path / "fake-claude"
    cli.write_text(FAKE.format(py=sys.executable))
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("FLUX_PLUMBER_CLAUDE", str(cli))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("FLUX_PLUMBER_FLUX_SOURCE", raising=False)

    def calls():
        return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []

    return calls


@pytest.fixture
def client(make_scheduler, fake_cli):
    with TestClient(
        create_app(make_scheduler(), TOKEN, demo=True, tick_interval=0.1, ask_token=ASK_TOKEN),
        base_url="http://localhost",
    ) as c:
        c.calls = fake_cli
        yield c


def events(resp):
    return [json.loads(x[6:]) for x in resp.text.split("\n\n") if x.startswith("data: ")]


def flag(argv, name):
    return argv[argv.index(name) + 1]


def say(c, question, conv, mode="codesign", **kw):
    return events(c.post("/api/ask", json={"question": question, "conversation": conv, "mode": mode, **kw}, headers=H))


# ------------------------------------------------------------------ the Ask token


def test_the_ask_token_is_read_only_whatever_header_it_sends(client):
    for hdr in ({}, {"X-Flux-Client": "mcp"}, {"X-Flux-Client": "web"}, {"X-Flux-Client": "ask"}):
        h = {**A, **hdr}
        assert client.get("/api/jobs", headers=h).status_code == 200  # reads work
        r = client.post("/api/jobs", json={"profile": "t_one"}, headers=h)
        assert r.status_code == 403 and "read-only" in r.json()["detail"], hdr
        assert client.patch("/api/queue", json={"paused": True}, headers=h).status_code == 403
        assert client.post("/api/profiles", json={"name": "x_y"}, headers=h).status_code == 403
        assert client.put("/api/ask/settings", json={"wake_on_results": False}, headers=h).status_code == 403
        assert client.post("/api/proposals/nope/approve", headers=h).status_code == 403
    assert client.get("/api/queue", headers=H).json()["paused"] is False
    assert (
        client.post("/api/jobs", json={"profile": "t_one"}, headers={**H, "X-Flux-Client": "mcp"}).status_code == 201
    )  # the main token works as before


def test_the_ask_token_is_not_a_login(client):
    assert client.get("/api/jobs", headers={"X-Flux-Token": "wrong"}).status_code == 401
    client.cookies.set("flux_plumber", ASK_TOKEN)  # the cookie only ever carries the person's token
    assert client.get("/api/jobs").status_code == 401
    client.cookies.clear()
    page = client.get("/", headers=A)
    assert page.status_code == 401 and "Sign in" in page.text  # the page wants the person's token
    with pytest.raises(Exception):
        with client.websocket_connect("/ws", headers=A):
            pass


def test_the_ask_token_alone_may_propose_and_a_different_one_may_not(client, monkeypatch):
    monkeypatch.setattr(ask, "_question", "q1")
    body = {"tool": "set_queue", "args": {"paused": True}, "question": "q1"}
    assert client.post("/api/proposals", json=body, headers=A).status_code == 201  # no X-Flux-Client needed
    assert client.post("/api/proposals", json=body, headers={**H, "X-Flux-Client": "mcp"}).status_code == 403
    pid = client.get("/api/proposals", headers=H).json()["proposals"][0]["id"]
    assert client.post(f"/api/proposals/{pid}/approve", headers=A).status_code == 403
    assert client.post(f"/api/proposals/{pid}/approve", headers=H).status_code == 200


def test_the_token_files_and_the_mcp_copy_that_reads_them(tmp_path, monkeypatch):
    t = plumber_cli.load_or_create_token(tmp_path, name="ask_token")
    assert len(t) >= 24 and t != plumber_cli.load_or_create_token(tmp_path)
    plumber_cli.write_atomic(paths.ask_token_path(tmp_path), t)
    assert oct(os.stat(paths.ask_token_path(tmp_path)).st_mode & 0o777) == "0o600"
    assert plumber_cli.load_or_create_token(tmp_path, name="ask_token") == t  # kept across restarts
    monkeypatch.setenv("FLUX_MCP_CLIENT", "ask")
    assert mcp_server.token_file_name() == "ask_token"
    monkeypatch.setenv("FLUX_MCP_CLIENT", "mcp")
    assert mcp_server.token_file_name() == "token"


# ------------------------------------------------------------------ conversations, modes, models


def test_codesign_names_a_cli_session_then_resumes_it_and_explain_stays_stateless(client):
    ev = say(client, "design a scene", "conv-aaaa1111")
    assert ev[-1]["type"] == "done" and ev[-1]["turn_cost"] == 0.5 and ev[-1]["conversation_cost"] == 0.5
    say(client, "now the lid", "conv-aaaa1111")
    one, two = (c["argv"] for c in client.calls())
    assert (
        flag(one, "--model") == "opus"
        and flag(one, "--max-budget-usd") == "3"
        and "--no-session-persistence" not in one
    )
    sid = flag(one, "--session-id")
    assert flag(two, "--resume") == sid and "--session-id" not in two
    assert "Earlier question" not in client.calls()[1]["prompt"]  # the CLI session holds the earlier turns
    rec = client.get("/api/ask/conversations/conv-aaaa1111", headers=H).json()
    assert [t["q"] for t in rec["turns"]] == ["design a scene", "now the lid"] and rec["cost"] == 1.0
    assert rec["mode"] == "codesign" and "session" not in rec  # the CLI session id is not sent to the browser
    say(client, "a new one", "conv-bbbb2222")
    assert flag(client.calls()[2]["argv"], "--session-id") != sid  # a new conversation, a new session

    ev = say(client, "what is a fluxon?", "conv-cccc3333", mode="explain")
    argv = client.calls()[3]["argv"]
    assert (
        flag(argv, "--model") == "sonnet"
        and flag(argv, "--max-budget-usd") == "1"
        and "--no-session-persistence" in argv
    )
    say(client, "and a vertex?", "conv-cccc3333", mode="explain")
    assert (
        "Earlier question: what is a fluxon?" in client.calls()[4]["prompt"]
    )  # the conversation's own turns are re-sent
    say(client, "unrelated", "conv-dddd4444", mode="explain")
    assert "Earlier question" not in client.calls()[5]["prompt"]  # a new conversation starts clean
    assert {c["id"]: c["mode"] for c in client.get("/api/ask/conversations", headers=H).json()["conversations"]}[
        "conv-cccc3333"
    ] == "explain"


def test_the_model_is_chosen_per_turn_and_a_conversation_keeps_its_mode(client):
    say(client, "q", "conv-eeee5555", model="sonnet")
    assert flag(client.calls()[0]["argv"], "--model") == "sonnet"
    say(client, "q2", "conv-eeee5555", mode="explain")  # the mode was fixed when the conversation began
    assert flag(client.calls()[1]["argv"], "--max-budget-usd") == "3"
    r = client.post("/api/ask", json={"question": "q", "model": "gpt"}, headers=H)
    assert r.status_code == 422
    r = client.post("/api/ask", json={"question": "q", "conversation": "../../etc"}, headers=H)
    assert r.status_code == 422


def test_a_lost_cli_session_falls_back_to_the_stored_turns(client, monkeypatch):
    say(client, "first question", "conv-ffff6666")
    monkeypatch.setenv("FAKE_LOST", "1")
    ev = say(client, "second question", "conv-ffff6666")
    assert [e["type"] for e in ev][-1] == "done" and not [e for e in ev if e["type"] == "error"]
    calls = client.calls()
    assert "--resume" in calls[1]["argv"] and "--session-id" in calls[2]["argv"]  # tried, then a fresh session
    assert flag(calls[2]["argv"], "--session-id") != flag(calls[0]["argv"], "--session-id")
    assert "Earlier question: first question" in calls[2]["prompt"]  # fed the stored turns instead
    monkeypatch.delenv("FAKE_LOST")
    say(client, "third", "conv-ffff6666")
    assert flag(client.calls()[3]["argv"], "--resume") == flag(
        calls[2]["argv"], "--session-id"
    )  # the new session is kept


def test_the_codesign_daily_cap_stops_a_turn_and_explain_is_not_capped(client, tmp_path):
    home = str(client.app.state.scheduler.home)
    row = {
        "ts": time.time(),
        "model": "m",
        "cost": 19.9,
        "ms": 1,
        "mode": "codesign",
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
    }
    with open(os.path.join(home, ask.USAGE_FILE), "w") as f:
        f.write(json.dumps(row) + "\n")
    assert (
        say(client, "one more", "conv-gggg7777")[-1]["type"] == "done"
    )  # $19.90 spent: under the cap, this turn runs ($0.50)
    ev = say(client, "and another", "conv-gggg7777")
    assert ev[0]["type"] == "error" and "daily cap" in ev[0]["message"] and len(client.calls()) == 1
    assert say(client, "explain still works", "conv-hhhh8888", mode="explain")[-1]["type"] == "done"
    a = client.get("/api/ask", headers=H).json()
    assert a["codesign"]["cap_usd"] == 20.0 and a["codesign"]["spent_today_usd"] == pytest.approx(20.4)
    assert (
        client.put("/api/ask/settings", json={"codesign_daily_cap_usd": 50}, headers=H).json()["codesign_daily_cap_usd"]
        == 50
    )
    assert say(client, "now allowed", "conv-gggg7777")[-1]["type"] == "done"


def test_settings_are_checked_and_stored(client):
    assert client.get("/api/ask/settings", headers=H).json() == {
        "wake_on_results": True,
        "codesign_daily_cap_usd": 20.0,
        "flux_source": None,
    }
    assert (
        client.put("/api/ask/settings", json={"wake_on_results": False}, headers=H).json()["wake_on_results"] is False
    )
    assert client.put("/api/ask/settings", json={"flux_source": "/no/such/dir"}, headers=H).status_code == 400
    assert client.put("/api/ask/settings", json={"codesign_daily_cap_usd": -1}, headers=H).status_code == 400
    assert client.get("/api/ask/settings", headers=H).json()["wake_on_results"] is False  # kept


def test_the_flux_source_is_added_read_only_in_codesign_only(client, tmp_path, monkeypatch):
    src = tmp_path / "fluxon-src"
    src.mkdir()
    say(client, "q", "conv-iiii9999")
    assert str(src) not in client.calls()[0]["argv"]  # nothing configured: nothing guessed
    monkeypatch.setenv("FLUX_PLUMBER_FLUX_SOURCE", str(src))
    say(client, "q", "conv-jjjj0000")
    say(client, "q", "conv-kkkk1111", mode="explain")
    codesign, explain = client.calls()[1]["argv"], client.calls()[2]["argv"]
    dirs = [codesign[i + 1] for i, a in enumerate(codesign) if a == "--add-dir"]
    assert str(src) in dirs and str(src) not in explain
    assert flag(codesign, "--tools") == "Read,Grep,Glob"  # no Edit, Write or Bash: nothing can write to it
    assert "Edit" not in codesign[codesign.index("--allowedTools") + 1] and str(src) in flag(
        codesign, "--system-prompt"
    )
    assert client.get("/api/ask", headers=H).json()["flux_source"] == str(src)


def test_codesign_only_proposals_are_not_offered_or_accepted_in_explain():
    for tool in ask.CODESIGN_ONLY:
        assert tool in proposals.TOOLS
        assert f"mcp__fluxpype__{tool}" not in flag(ask.command("claude", "/h", True, "q", "explain"), "--allowedTools")
        assert f"mcp__fluxpype__{tool}" in flag(ask.command("claude", "/h", True, "q", "codesign"), "--allowedTools")


# ------------------------------------------------------------------ the wake turn


def wait_for(cond, secs=20):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


def codesign_job(client, monkeypatch, conv="conv-wake0001", tool="submit_job", args=None):
    """A job that a click on a Co-design proposal started, as the card's Run does it."""
    say(client, "run a small job", conv)
    home = str(client.app.state.scheduler.home)
    qid = Conversations(home).load(conv)["turns"][-1]["id"]
    monkeypatch.setattr(ask, "_question", qid)  # the question that proposes it is "running"
    body = {"tool": tool, "args": args or {"profile": "t_one", "backend": "fake", "speed": 400}, "question": qid}
    r = client.post("/api/proposals", json=body, headers=A)
    assert r.status_code == 201, r.text
    monkeypatch.setattr(ask, "_question", "")
    return conv, r.json()["id"], home


def test_a_finished_job_from_a_codesign_proposal_wakes_her_conversation_once(client, monkeypatch):
    conv, pid, home = codesign_job(client, monkeypatch)
    jid = client.post(f"/api/proposals/{pid}/approve", headers=H).json()["result"]["job_id"]
    assert wait_for(lambda: any(t.get("system") for t in Conversations(home).load(conv)["turns"]))
    rec = Conversations(home).load(conv)
    woke = [t for t in rec["turns"] if t.get("system")]
    assert len(woke) == 1 and jid in woke[0]["q"] and "[Plumber, not the person]" in woke[0]["q"]
    assert woke[0]["answer"].startswith("answer to:") and rec["cost"] == pytest.approx(1.0)
    assert flag(client.calls()[-1]["argv"], "--resume") == flag(client.calls()[0]["argv"], "--session-id")
    time.sleep(1.0)
    assert len([t for t in Conversations(home).load(conv)["turns"] if t.get("system")]) == 1  # once per job


def test_the_wake_turn_respects_the_setting_the_cap_and_the_lock(client, monkeypatch):
    home = str(client.app.state.scheduler.home)
    client.put("/api/ask/settings", json={"wake_on_results": False}, headers=H)
    conv, pid, _ = codesign_job(client, monkeypatch, "conv-wake0002")
    client.post(f"/api/proposals/{pid}/approve", headers=H)
    time.sleep(2.5)
    assert (
        not any(t.get("system") for t in Conversations(home).load(conv)["turns"]) and len(client.calls()) == 1
    )  # off: no turn
    client.put("/api/ask/settings", json={"wake_on_results": True}, headers=H)
    conv, pid, _ = codesign_job(client, monkeypatch, "conv-wake0003")
    n = len(client.calls())
    client.put(
        "/api/ask/settings", json={"codesign_daily_cap_usd": 0.4}, headers=H
    )  # the turns so far cost more than that
    client.post(f"/api/proposals/{pid}/approve", headers=H)
    assert wait_for(lambda: any(t.get("system") for t in Conversations(home).load(conv)["turns"]))
    turn = [t for t in Conversations(home).load(conv)["turns"] if t.get("system")][0]
    assert "daily cap" in turn["error"] and len(client.calls()) == n  # said so in the conversation; the CLI never ran
    # the lock: a wake turn queues behind a running question instead of failing
    client.put("/api/ask/settings", json={"codesign_daily_cap_usd": 100}, headers=H)
    conv, pid, _ = codesign_job(client, monkeypatch, "conv-wake0004")
    n = len(client.calls())
    assert ask._busy.acquire(blocking=False)
    try:
        client.post(f"/api/proposals/{pid}/approve", headers=H)
        time.sleep(2.5)
        assert len(client.calls()) == n and not any(t.get("system") for t in Conversations(home).load(conv)["turns"])
    finally:
        ask._busy.release()
    assert wait_for(
        lambda: any(t.get("system") and not t.get("error") for t in Conversations(home).load(conv)["turns"])
    )
