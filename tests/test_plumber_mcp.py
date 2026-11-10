"""The MCP server, driven through the SDK client against an in-process Plumber (fake backend, temp home)."""

import json
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("mcp")

import anyio  # noqa: E402
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from mcp import Client  # noqa: E402

from fluxpype.plumber import mcp_server  # noqa: E402
from fluxpype.plumber.mcp_server import PlumberClient, build_server  # noqa: E402
from fluxpype.plumber.server import create_app  # noqa: E402

TOKEN = "t0k3n-for-tests"
H = {"X-Flux-Token": TOKEN}


class BridgeTransport(httpx.BaseTransport):
    """Sends requests from the MCP server's httpx client into the app through FastAPI's TestClient.

    (Starlette's TestClient may be built on httpx2, whose transport httpx cannot use directly.)
    """

    def __init__(self, tc):
        self.tc = tc

    def handle_request(self, request):
        r = self.tc.request(request.method, str(request.url), content=request.content, headers=dict(request.headers))
        skip = {"content-encoding", "content-length", "transfer-encoding"}
        headers = [(k, v) for k, v in r.headers.items() if k.lower() not in skip]
        return httpx.Response(r.status_code, headers=headers, content=r.content, request=request)


class Rig:
    """An in-process Plumber plus an MCP server wired to it through the test client's transport."""

    def __init__(self, tc, sched, token=TOKEN):
        self.tc, self.sched = tc, sched
        self.client = PlumberClient(base_url="http://localhost", token=token, transport=BridgeTransport(tc))
        self.server = build_server(self.client)

    def call(self, tool, /, **args):
        """Call a tool; returns (is_error, parsed JSON or text)."""

        async def go():
            async with Client(self.server) as c:
                return await c.call_tool(tool, args)

        res = anyio.run(go)
        text = res.content[0].text
        if res.is_error:
            return True, text
        try:
            return False, json.loads(text)
        except ValueError:
            return False, text

    def ok(self, tool, /, **args):
        err, out = self.call(tool, **args)
        assert not err, out
        return out

    def run_job(self, **body):
        body = {"profile": "t_one", "backend": "fake", "scenario": "ok", "speed": 300, **body}
        job = self.tc.post("/api/jobs", json=body, headers=H).json()
        end = time.time() + 60
        while time.time() < end:
            view = self.tc.get(f"/api/jobs/{job['id']}", headers=H).json()
            if view["status"] in ("ok", "failed", "cancelled", "partial"):
                return view
            time.sleep(0.1)
        raise AssertionError("job did not finish")


@pytest.fixture
def rig(make_scheduler):
    sched = make_scheduler()
    app = create_app(sched, TOKEN, demo=True, tick_interval=0.1)
    with TestClient(app, base_url="http://localhost") as tc:
        yield Rig(tc, sched)


@pytest.fixture
def done(rig):
    """A finished ok job and its task id."""
    view = rig.run_job()
    return view, view["tasks"][0]["id"]


# ------------------------------------------------------------------ pure helpers


def test_cap_and_dumps_are_compact_and_bounded():
    out = mcp_server.dumps({"a": "x" * 5000, "b": [1.5, None]})
    assert " " not in out.replace("more chars", "")[:20] and "...[3000 more chars]" in out
    assert len(out) < 2200


def test_downsample_keeps_endpoints_and_cycle_boundaries():
    pts = [{"step": i, "t": float(i)} for i in range(1000)]
    cycles = [{"t": 130.2}, {"t": 477.0}, {"t": 912.9}]
    got = mcp_server.downsample(pts, cycles, 50)
    steps = [p["step"] for p in got]
    assert len(got) <= 50 and steps[0] == 0 and steps[-1] == 999 and steps == sorted(set(steps))
    assert {130, 477, 913} <= set(steps)
    assert mcp_server.downsample(pts[:10], cycles, 50) == pts[:10]
    assert len(mcp_server.downsample(pts, [{"t": float(i)} for i in range(0, 1000, 5)], 20)) <= 20


def test_stringify_matches_the_config_syntax():
    assert mcp_server.stringify([2225, "2226"]) == "[2225, 2226]"
    assert mcp_server.stringify(True) == "1" and mcp_server.stringify(0.5) == "0.5" and mcp_server.stringify("x") == "x"


# ------------------------------------------------------------------ read-only tools


def test_status_on_an_idle_and_a_finished_plumber(rig):
    idle = rig.ok("status")
    assert idle["demo"] is True and idle["queue"]["paused"] is False and idle["running"] == [] and idle["recent"] == []
    assert idle["backends"][0]["name"] == "fake" and "ok" in idle["scenarios"]
    rig.run_job()
    after = rig.ok("status")
    assert after["recent"][0]["status"] == "ok" and after["recent"][0]["profile"] == "t_one"


def test_status_shows_failure_reasons(rig):
    rig.run_job(scenario="fail_in_pfss")
    recent = rig.ok("status")["recent"][0]
    assert recent["status"] == "failed" and recent["failure"]


def test_list_profiles(rig):
    profs = {p["name"]: p for p in rig.ok("list_profiles")["profiles"]}
    assert profs["t_two"]["task_count"] == 4 and profs["Real_Run"]["errors"] and profs["t_one"]["errors"] == []


def test_get_profile_strips_comments_and_lists_editable_keys(rig):
    p = rig.ok("get_profile", name="t_one")
    text = json.dumps(p)
    assert "comment" not in p and '"comment"' not in text
    assert p["settings"]["batch_name"] == "tb_one" and p["task_count"] == 1
    assert "relax_threshold" in p["editable_keys"] and "run_script" not in p["editable_keys"]
    assert p["tasks"][0]["cache"]["pfss"] is False and p["tasks"][0]["method"] == "parker"
    err, msg = rig.call("get_profile", name="nope")
    assert err and "unknown profile" in msg


def test_jobs_and_tasks(rig, done):
    view, tid = done
    jobs = rig.ok("list_jobs")["jobs"]
    assert jobs[0]["id"] == view["id"] and jobs[0]["n_tasks"] == 1
    assert rig.ok("list_jobs", status="failed")["jobs"] == []
    assert rig.ok("list_jobs", profile="t_one", status="ok")["jobs"][0]["id"] == view["id"]
    assert rig.ok("list_jobs", profile="t_two")["jobs"] == []
    job = rig.ok("get_job", job_id=view["id"])
    row = job["tasks"][0]
    assert row["id"] == tid and row["stages"]["relax"] == "ok" and row["stages"]["fluxlight"] == "skipped"
    task = rig.ok("get_task", task_id=tid)
    by_id = {s["id"]: s for s in task["stages"]}
    assert task["status"] == "ok" and by_id["relax"]["verified"] is True and by_id["relax"]["source"] == "event"
    assert task["relax"]["cycles"] == 6 and task["relax"]["stiffness"] > 0
    assert [a["index"] for a in task["artifacts"]] == [0, 1, 2] and task["artifacts"][0]["name"].endswith(".png")
    assert "failure" not in task
    err, msg = rig.call("get_job", job_id="nope")
    assert err and "unknown job" in msg
    err, msg = rig.call("get_task", task_id="nope")
    assert err and "no such task" in msg


def test_relax_series_is_downsampled_with_endpoints(rig, done):
    _, tid = done
    full = rig.tc.get(f"/api/tasks/{tid}/relax", headers=H).json()
    out = rig.ok("get_relax_series", task_id=tid, max_points=12)
    assert out["total_points"] == full["n"] > 12 and out["returned"] <= 12
    assert out["rows"][0][0] == full["points"][0]["step"] and out["rows"][-1][0] == full["points"][-1]["step"]
    assert out["columns"][:2] == ["step", "stiffness"] and len(out["cycles"]) == len(full["cycles"])
    everything = rig.ok("get_relax_series", task_id=tid, max_points=1000)
    assert everything["returned"] == full["n"]


def test_log_tail_and_grep_caps(rig, done):
    _, tid = done
    task = rig.sched.store.get_task(tid)
    lines = [f"line {i}" for i in range(1, 1001)]
    lines[499] = "the NEEDLE"
    lines[600] = "x" * 5000
    with open(f"{task['run_dir']}/run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    tail = rig.ok("read_task_log", task_id=tid, tail_lines=5)
    assert tail["lines"] == 5 and tail["text"].split("\n")[-1] == "line 1000"
    many = rig.ok("read_task_log", task_id=tid, tail_lines=100000)
    assert many["lines"] == 400 and many["text"].split("\n")[-1] == "line 1000"
    longest = max(len(ln) for ln in many["text"].split("\n"))
    assert longest <= 2000
    hit = rig.ok("read_task_log", task_id=tid, grep="needle", context=1)
    assert hit["matches"] == 1 and hit["text"].split("\n") == ["499-line 499", "500:the NEEDLE", "501-line 501"]
    flood = rig.ok("read_task_log", task_id=tid, grep="line", context=20)
    assert flood["truncated"] and flood["lines"] == 400
    err, msg = rig.call("read_task_log", task_id="nope")
    assert err and "unknown task" in msg


def test_cache_summary_filters_and_pages(rig, done):
    summary = rig.ok("cache_summary", rescan=True)
    assert summary["total_count"] >= 6 and {k["kind"] for k in summary["by_kind"]} >= {"magnetogram", "world"}
    only = rig.ok("cache_summary", kind="plots")
    assert only["matched"] == 3 and {f["kind"] for f in only["files"]} == {"plots"}
    assert only["by_kind"]  # totals are always the whole cache, not the filter
    page = rig.ok("cache_summary", limit=2, offset=1)
    assert len(page["files"]) == 2 and page["matched"] == summary["matched"]
    assert rig.ok("cache_summary", cr="2225", nflux="100")["matched"] >= 3
    assert rig.ok("cache_summary", cr="9999")["files"] == []


# ------------------------------------------------------------------ errors


def test_wrong_token_is_a_clear_tool_error(rig):
    bad = Rig(rig.tc, rig.sched, token="wrong")
    err, msg = bad.call("status")
    assert err and "token" in msg and "401" in msg


def test_unreachable_plumber_is_a_clear_tool_error(tmp_path):
    client = PlumberClient(base_url="http://127.0.0.1:1", token="x", timeout=2)
    err, msg = Rig.call(type("R", (), {"server": build_server(client)})(), "status")
    assert err and "cannot reach Plumber" in msg
