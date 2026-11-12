"""The Ask bar's backend, with a stand-in for the Claude CLI (no network, no quota)."""

import json
import re
import stat
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fluxpype.plumber import ask, proposals  # noqa: E402
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
ev(type="rate_limit_event", rate_limit_info={{"unifiedWindows": {{"five_hour": {{"utilization": 0.23, "resetsAt": 4102444800}}, "seven_day": {{"utilization": 0.3, "resetsAt": 4102444900}}}}}})
ev(type="result", subtype="success", is_error=False, total_cost_usd=0.01, duration_ms=5, modelUsage={{"claude-sonnet-5-5": {{"inputTokens": 2, "outputTokens": 6, "cacheReadInputTokens": 100, "cacheCreationInputTokens": 855}}}})
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


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_the_cli_is_stripped_down_and_its_write_tools_only_propose():
    cmd = ask.command("claude", "/home", True, "q123")
    assert flag(cmd, "--tools") == "Read,Grep,Glob" and "--strict-mcp-config" in cmd and "--setting-sources" in cmd
    tools = " ".join(flag(cmd, f) for f in ("--tools", "--allowedTools", "--disallowedTools"))
    assert not any(t in tools for t in ("Bash", "Edit", "Write")) and "--add-dir" in cmd  # the repo, read-only
    allowed = {a.replace("mcp__fluxpype__", "") for a in flag(cmd, "--allowedTools").split(",")}
    assert allowed == set(ask.READ_TOOLS) | set(proposals.TOOLS)  # images and outcomes now; writes only propose
    assert {"get_task_image", "list_proposals", "submit_job", "create_preset"} <= allowed and "wait_for_task" not in allowed
    assert not {"Read", "Grep", "Glob"} & allowed  # a bare allow rule reaches every path; unlisted, only the repo
    denied = flag(cmd, "--disallowedTools").split(",")
    assert denied == ["Read(//home/**)", "mcp__fluxpype__wait_for_task"]  # never the Plumber home (its token)
    env = json.loads(flag(cmd, "--mcp-config"))["mcpServers"]["fluxpype"]["env"]
    assert env == {"FLUX_MCP_CLIENT": "ask", "FLUX_ASK_QUESTION": "q123"}  # "ask" is what makes its writes proposals
    bare = ask.command("claude", "/home", False)  # without the MCP SDK: the docs and the repo, no live state
    assert "mcp__fluxpype" not in " ".join(bare) and "--allowedTools" not in bare
    assert flag(bare, "--disallowedTools") == "Read(//home/**)"
    sp = cmd[cmd.index("--system-prompt") + 1]
    assert "VNEIGHBOR" in sp and "list_proposals" in sp and "flux-action" not in sp  # the reference and the guide
    assert "flux-plumb-mcp" in sp and "relax_threshold" in sp  # and the MCP tools doc and the config.ini defaults


def test_ask_streams_text_tool_and_done(client):
    r = client.post("/api/ask", json={"question": "which end is positive?", "page": "/jobs"}, headers=H)
    ev = events(r)
    assert [e["type"] for e in ev] == ["tool", "text", "text", "text", "done"]
    assert ev[0]["name"] == "status" and "".join(e["text"] for e in ev if e["type"] == "text").startswith("Hello|tools-off|")
    assert "write-listed" in "".join(e.get("text", "") for e in ev)  # listed only in --disallowedTools
    assert ev[-1]["cost"] == 0.01


def test_ask_keeps_model_tokens_and_plan_limits_for_the_agent_dialog(client):
    assert client.get("/api/agent", headers=H).json()["ask"] == {"model": None, "today": {"questions": 0, "input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cost": 0}, "limits": None}
    client.post("/api/ask", json={"question": "hi"}, headers=H)
    client.post("/api/ask", json={"question": "again"}, headers=H)
    a = client.get("/api/agent", headers=H).json()["ask"]
    assert a["model"] == "claude-sonnet-5-5" and a["today"]["questions"] == 2
    assert (a["today"]["input"], a["today"]["output"], a["today"]["cache_read"], a["today"]["cache_write"]) == (4, 12, 200, 1710)
    assert a["today"]["cost"] == pytest.approx(0.02)
    w = a["limits"]["windows"]
    assert w["five_hour"]["utilization"] == 0.23 and w["seven_day"]["resets_at"] == 4102444900


def test_token_rate_and_plan_share_left_feed_the_system_monitor(client, tmp_path):
    ev = {"type": "done", "cost": 0.01, "ms": 5, "model": "m", "tokens": {"input": 2, "output": 6, "cache_read": 9000, "cache_write": 52}}
    ask._note(str(tmp_path), ev)
    assert ask.token_rate() >= 60 / 60  # 2 + 52 + 6 new tokens in the last minute; cache reads are not counted
    ask._note(str(tmp_path), {"type": "limits", "windows": {"five_hour": {"utilization": 0.25, "resets_at": 4102444800}, "seven_day": {"utilization": 0.5, "resets_at": 1}}})
    p = ask.plan(str(tmp_path))
    assert p["five_hour"]["left"] == pytest.approx(0.75) and p["seven_day"]["left"] is None  # that window has rolled over
    client.post("/api/ask", json={"question": "hi"}, headers=H)
    d = client.get("/api/system", headers=H).json()
    assert d["plan"]["five_hour"]["left"] == pytest.approx(0.77) and d["now"]["tok"] > 0


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


def test_the_ask_bars_mcp_server_cannot_write(client):
    ask_h = {**H, "X-Flux-Client": "ask", "X-Flux-Tool": "submit_job"}
    r = client.post("/api/jobs", json={"profile": "quickstart"}, headers=ask_h)
    assert r.status_code == 403 and "read-only" in r.json()["detail"]
    assert client.get("/api/jobs", headers=ask_h).status_code == 200  # reads still work


def test_a_preset_takes_profile_only_keys_that_a_job_override_cannot(client):
    # the body the create_preset card builds: values stringified, lists as "[a, b]"
    body = {"name": "ask_light", "base": "quickstart", "values": {"light": "1", "rotations": "[2150]"}}
    assert client.post("/api/profiles", json=body, headers=H).status_code == 201
    assert client.post("/api/profiles", json=body, headers=H).status_code == 400  # the name is taken
    r = client.post("/api/jobs", json={"profile": "quickstart", "overrides": {"light": "1"}}, headers=H)
    assert r.status_code == 400 and "light" in r.text


def test_the_guide_the_cards_and_the_server_name_the_same_actions():
    tsx = (Path(ask.__file__).parent / "web" / "src" / "components" / "AskAnswer.tsx").read_text(encoding="utf-8")
    titles = set(re.findall(r"^  (\w+): \{", tsx.split("const TITLES", 1)[1].split("\n};", 1)[0], re.M))
    guide = (ask.DOCS / "plumber_guide.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"\w+", guide.split("Actions:", 1)[1].split(".", 1)[0]))
    assert titles == listed == set(proposals.TOOLS) == {"submit_job", "cancel", "retry_failed", "set_queue", "create_preset"}
    assert "flux-action" not in guide + tsx  # the old model-written cards are gone


# ------------------------------------------------------------------ proposals

ASK = {**H, "X-Flux-Client": "ask"}


def propose(c, tool, args, question="q1"):
    return c.post("/api/proposals", json={"tool": tool, "args": args, "question": question}, headers={**ASK, "X-Flux-Tool": tool})


def job_count(c):
    return len(c.get("/api/jobs", headers=H).json()["jobs"])


@pytest.fixture
def asking(client, monkeypatch):
    """A question with id q1 is running, and the queue is paused so queued tasks stay pending."""
    monkeypatch.setattr(ask, "_question", "q1")
    assert client.patch("/api/queue", json={"paused": True}, headers=H).status_code == 200
    return client


def test_only_the_ask_client_during_its_question_may_propose_and_nothing_else(client, monkeypatch):
    body = {"tool": "set_queue", "args": {"paused": True}, "question": "q1"}
    assert client.post("/api/proposals", json=body, headers=H).status_code == 403  # the browser and agents act directly
    assert client.post("/api/proposals", json=body, headers={**H, "X-Flux-Client": "mcp"}).status_code == 403
    assert client.post("/api/proposals", json=body, headers=ASK).status_code == 409  # no question is running
    monkeypatch.setattr(ask, "_question", "q1")
    assert client.post("/api/proposals", json={**body, "question": "q2"}, headers=ASK).status_code == 409
    assert client.post("/api/proposals", json={**body, "tool": "delete_job"}, headers=ASK).status_code == 422
    assert client.post("/api/proposals", json=body, headers=ASK).status_code == 201
    pid = client.get("/api/proposals", headers=ASK).json()["proposals"][0]["id"]
    for method, path in [("POST", "/api/jobs"), ("PATCH", "/api/queue"), ("POST", "/api/profiles"), ("POST", f"/api/proposals/{pid}/approve"), ("POST", f"/api/proposals/{pid}/reject")]:
        assert client.request(method, path, json={}, headers=ASK).status_code == 403, path
    assert client.get("/api/queue", headers=H).json()["paused"] is False  # still only a proposal
    assert client.get("/api/proposals", headers=H).json()["proposals"][0]["status"] == "pending"


def test_a_proposal_is_checked_stored_and_runs_once_on_a_click(asking):
    c, n = asking, job_count(asking)
    r = propose(c, "submit_job", {"profile": "t_two", "overrides": {"batch_name": "tb_new"}})
    assert r.status_code == 201, r.text
    p = r.json()
    assert p["status"] == "pending" and p["origin"] == "ask" and p["args"]["backend"] == "fake"  # the backend is fixed
    assert "4 task(s) from preset t_two" in p["summary"] and p["diff"] == [{"key": "batch_name", "from": "tb_two", "to": "tb_new"}]
    assert job_count(c) == n  # proposing ran nothing
    assert c.post(f"/api/proposals/{p['id']}/approve", headers={**H, "X-Flux-Client": "mcp"}).status_code == 403
    r = c.post(f"/api/proposals/{p['id']}/approve", headers=H)
    assert r.status_code == 200 and r.json()["status"] == "approved"
    jid = r.json()["result"]["job_id"]
    job = c.get(f"/api/jobs/{jid}", headers=H).json()
    assert job["profile"] == "t_two" and job["overrides"] == {"batch_name": "tb_new"} and len(job["tasks"]) == 4
    assert c.post(f"/api/proposals/{p['id']}/approve", headers=H).status_code == 409  # single use
    assert job_count(c) == n + 1
    seen = c.get("/api/proposals", headers=ASK).json()["proposals"][0]  # what the next question reads
    assert seen["id"] == p["id"] and seen["status"] == "approved" and seen["result"] == {"job_id": jid}
    calls = c.get("/api/agent", headers=H).json()["calls"]
    made = [x for x in calls if x["paths"] == ["POST /api/proposals"]]
    clicked = [x for x in calls if x["paths"] == [f"POST /api/proposals/{p['id']}/approve"]]
    assert made[0]["tool"] == "submit_job" and made[0]["via"] == "ask" and made[0]["status"] == 201
    assert clicked[0]["via"] == "ask" and json.loads(clicked[0]["args"]) == {"proposal": p["id"], "decision": "approved"}


def test_approve_runs_the_stored_arguments_and_never_the_requests(asking):
    c = asking
    p = propose(c, "submit_job", {"profile": "t_one"}).json()
    evil = {"tool": "submit_job", "args": {"profile": "t_two", "overrides": {"batch_name": "evil"}}, "profile": "t_two"}
    r = c.post(f"/api/proposals/{p['id']}/approve", json=evil, headers=H)
    job = c.get(f"/api/jobs/{r.json()['result']['job_id']}", headers=H).json()
    assert job["profile"] == "t_one" and job["overrides"] == {} and len(job["tasks"]) == 1
    q = propose(c, "create_preset", {"name": "ask_p", "base": "t_one", "values": {"fluxon_count": "[300]"}}).json()
    assert q["diff"] == [{"key": "fluxon_count", "from": "[100]", "to": "[300]"}]
    r = c.post(f"/api/proposals/{q['id']}/approve", json={"name": "other", "values": {"light": "1"}}, headers=H)
    assert r.json()["result"] == {"name": "ask_p"}
    own = {x["name"]: x["own"] for x in c.get("/api/profiles?cache=false", headers=H).json()["profiles"]}
    assert own["ask_p"]["fluxon_count"] == "[300]" and "light" not in own["ask_p"] and "other" not in own


def test_bad_arguments_are_refused_when_proposed(asking):
    c = asking
    cases = [
        ("submit_job", {"profile": "nope"}, "unknown profile"),
        ("submit_job", {"profile": "t_one", "overrides": {"light": "1"}}, "light"),  # a preset key, not this profile's
        ("submit_job", {"profile": "t_one", "overrides": {"run_script": "x.py"}}, "run_script"),
        ("submit_job", {"profile": "t_one", "speed": 5000}, "speed"),
        ("submit_job", {"profile": "t_one", "tasks": [["9999", "1", "0", "parker"]]}, "none of the requested tasks"),
        ("set_queue", {"max_parallel": 0}, "between 1 and 64"),
        ("set_queue", {}, "at least one"),
        ("cancel", {"job_id": "jnope"}, "unknown job"),
        ("cancel", {"job_id": "a", "task_id": "b"}, "exactly one"),
        ("retry_failed", {"job_id": "jnope"}, "unknown job"),
        ("create_preset", {"name": "1bad"}, "names start"),
        ("create_preset", {"name": "t_one"}, "already exists"),
        ("create_preset", {"name": "fine", "base": "t_one", "values": {"data_dir": "/tmp"}}, "data_dir"),
    ]
    for tool, args, why in cases:
        r = propose(c, tool, args)
        assert r.status_code == 400 and why in r.json()["detail"], (tool, args, r.text)
    assert c.get("/api/proposals", headers=H).json()["proposals"] == []


def test_cancel_and_retry_proposals_name_what_they_touch(asking):
    c = asking
    jid = c.post("/api/jobs", json={"profile": "t_two"}, headers=H).json()["id"]
    p = propose(c, "cancel", {"job_id": jid}).json()
    assert p["summary"] == f"Cancel job {jid}: stops its 4 pending or running task(s)"
    assert propose(c, "retry_failed", {"job_id": jid}).json()["detail"] == f"job {jid} has nothing to retry"
    c.post(f"/api/proposals/{p['id']}/approve", headers=H)
    assert c.get(f"/api/jobs/{jid}", headers=H).json()["status"] == "cancelled"
    r = propose(c, "retry_failed", {"job_id": jid}).json()
    assert "4 failed" in r["summary"] and c.post(f"/api/proposals/{r['id']}/approve", headers=H).json()["status"] == "approved"
    assert c.get(f"/api/jobs/{jid}", headers=H).json()["counts"]["pending"] == 4


def test_reject_expiry_and_caps(asking, monkeypatch, tmp_path):
    c = asking
    assert c.get("/api/queue", headers=H).json()["cpu_budget"] == 64
    ids = [propose(c, "set_queue", {"cpu_budget": i}).json()["id"] for i in range(1, 7)]
    r = propose(c, "set_queue", {"cpu_budget": 7})
    assert r.status_code == 429 and "per question" in r.json()["detail"]
    assert c.post(f"/api/proposals/{ids[0]}/reject", headers=H).json()["status"] == "rejected"
    assert c.post(f"/api/proposals/{ids[0]}/approve", headers=H).status_code == 409  # a rejected one never runs
    assert c.post(f"/api/proposals/{ids[0]}/reject", headers=H).status_code == 409
    assert c.post("/api/proposals/nope/approve", headers=H).status_code == 404
    assert c.get("/api/queue", headers=H).json()["cpu_budget"] == 64  # six proposals, no change
    monkeypatch.setattr(proposals, "PER_QUESTION", 100)
    while len([p for p in c.get("/api/proposals", headers=H).json()["proposals"] if p["status"] == "pending"]) < proposals.MAX_PENDING:
        assert propose(c, "set_queue", {"max_parallel": 8}).status_code == 201
    r = propose(c, "set_queue", {"max_parallel": 8})
    assert r.status_code == 429 and "waiting" in r.json()["detail"]
    monkeypatch.setattr(proposals, "EXPIRY_S", -1)  # everything pending is now past its expiry
    statuses = {p["status"] for p in c.get("/api/proposals", headers=H).json()["proposals"]}
    assert statuses == {"expired", "rejected"}
    assert c.post(f"/api/proposals/{ids[1]}/approve", headers=H).status_code == 409
    store = proposals.ProposalStore(str(tmp_path / "p.json"))
    with pytest.raises(proposals.ProposalError, match="over"):
        store.add("set_queue", {"paused": "x" * proposals.ARGS_CAP}, "s", "q")
    assert store.add("set_queue", {"paused": True}, "s", "q")["status"] == "pending"
    assert proposals.ProposalStore(str(tmp_path / "p.json")).list()[0]["args"] == {"paused": True}  # kept on disk


def test_the_answer_ends_with_plumbers_list_of_its_proposals(client, monkeypatch):
    seen = []
    monkeypatch.setattr(proposals, "for_question", lambda path, q: seen.append(q) or ["abc123def456"])
    ev = events(client.post("/api/ask", json={"question": "queue t_one"}, headers=H))
    assert ev[-2] == {"type": "text", "text": '\n\n```flux-proposals\n["abc123def456"]\n```\n'} and ev[-1]["type"] == "done"
    assert len(seen[0]) == 12 and ask.current_question() == ""  # the id lived only while the question ran
