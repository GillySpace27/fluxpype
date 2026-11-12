"""The MCP server, driven through the SDK client against an in-process Plumber (fake backend, temp home)."""

import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("mcp")

import anyio  # noqa: E402
import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from mcp import Client, StdioServerParameters  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from fluxpype.plumber import cli, mcp_server  # noqa: E402
from fluxpype.plumber.launcher import process_alive  # noqa: E402
from fluxpype.plumber.mcp_server import PlumberClient, PlumberError, build_server  # noqa: E402
from fluxpype.plumber.server import create_app  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
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


# ------------------------------------------------------------------ write tools


def wait_status(rig, job_id, wanted=("ok", "failed", "cancelled", "partial"), timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        view = rig.tc.get(f"/api/jobs/{job_id}", headers=H).json()
        if view["status"] in wanted:
            return view
        time.sleep(0.1)
    raise AssertionError("job did not reach the wanted status")


def test_submit_job_stringifies_overrides_and_runs(rig):
    job = rig.ok(
        "submit_job",
        profile="t_one",
        overrides={"concurrency": 3, "relax_threshold": 0.5},
        backend="fake",
        scenario="ok",
        speed=300,
    )
    assert job["profile"] == "t_one" and job["overrides"] == {"concurrency": "3", "relax_threshold": "0.5"}
    assert wait_status(rig, job["id"])["status"] == "ok"
    sub = rig.ok("submit_job", profile="t_two", tasks=[[2225, 100, 0, "wsa"]], backend="fake", speed=300)
    assert [(t["cr"], t["nflux"], t["adapt"], t["method"]) for t in sub["tasks"]] == [("2225", "100", "0", "wsa")]
    wait_status(rig, sub["id"])


def test_submit_job_lists_become_config_lists(rig):
    sched_calls = []
    real = rig.sched.create_job

    def spy(profile, overrides=None, *a, **kw):
        sched_calls.append(overrides)
        return real(profile, overrides, *a, **kw)

    rig.sched.create_job = spy
    job = rig.ok("submit_job", profile="t_one", overrides={"rotations": [2225, 2226]}, backend="fake", speed=300)
    assert sched_calls == [{"rotations": "[2225, 2226]"}] and len(job["tasks"]) == 2
    wait_status(rig, job["id"])


def test_submit_job_surfaces_the_servers_rejection(rig):
    err, msg = rig.call("submit_job", profile="t_one", overrides={"run_script": "x"})
    assert err and "run_script" in msg
    err, msg = rig.call("submit_job", profile="nope")
    assert err and msg
    err, msg = rig.call("submit_job", profile="t_one", scenario="bogus", backend="fake")
    assert err and "unknown scenario" in msg
    err, msg = rig.call("submit_job", profile="t_one", tasks=[[1, 2, 3]])
    assert err and "[cr, nflux, adapt, method]" in msg
    err, msg = rig.call("submit_job", profile="t_one", overrides={"k": {"nested": 1}})
    assert err and "validation" in msg
    assert rig.sched.store.list_jobs() == []


def test_cancel_by_job_and_by_task_with_queue_paused(rig):
    q = rig.ok("set_queue", paused=True)
    assert q["paused"] is True
    first = rig.ok("submit_job", profile="t_one", backend="fake", speed=300)
    second = rig.ok("submit_job", profile="t_one", backend="fake", speed=300)
    assert rig.ok("status")["pending"] and first["status"] in ("pending", "queued")
    cancelled = rig.ok("cancel", job_id=first["id"])
    assert cancelled["status"] == "cancelled"
    task = rig.ok("cancel", task_id=second["tasks"][0]["id"])
    assert task["status"] == "cancelled" and task["id"] == second["tasks"][0]["id"]
    for args in ({}, {"job_id": first["id"], "task_id": "x"}):
        err, msg = rig.call("cancel", **args)
        assert err and "exactly one" in msg
    err, msg = rig.call("cancel", job_id="nope")
    assert err and "unknown job" in msg


def test_retry_failed(rig):
    view = rig.run_job(scenario="fail_in_pfss")
    assert view["status"] == "failed"
    retried = rig.ok("retry_failed", job_id=view["id"])
    assert retried["id"] == view["id"]
    wait_status(rig, view["id"])
    ok_job = rig.run_job()
    err, msg = rig.call("retry_failed", job_id=ok_job["id"])
    assert err and msg


def test_set_queue(rig):
    out = rig.ok("set_queue", max_parallel=3, cpu_budget=12, paused=False)
    assert (out["max_parallel"], out["cpu_budget"], out["paused"]) == (3, 12, False)
    assert rig.ok("status")["queue"]["max_parallel"] == 3
    err, msg = rig.call("set_queue")
    assert err and "at least one" in msg
    err, msg = rig.call("set_queue", max_parallel=0)
    assert err and msg


# ------------------------------------------------------------------ wait_for_task


class Progress:
    def __init__(self):
        self.events = []

    async def __call__(self, progress, total=None, message=None):
        self.events.append((progress, total, message))


def wait_call(rig, progress=None, **args):
    async def go():
        async with Client(rig.server) as c:
            return await c.call_tool("wait_for_task", args, progress_callback=progress)

    res = anyio.run(go)
    return res.is_error, (json.loads(res.content[0].text) if not res.is_error else res.content[0].text)


def test_wait_for_task_reports_progress_and_returns_the_final_state(rig, monkeypatch):
    monkeypatch.setattr(mcp_server, "POLL_S", 0.1)
    job = rig.ok("submit_job", profile="t_one", backend="fake", scenario="ok", speed=40)
    tid = job["tasks"][0]["id"]
    progress = Progress()
    err, out = wait_call(rig, progress, task_id=tid, timeout_s=60)
    assert not err and out["still_running"] is False and out["reached"] is True and out["status"] == "ok"
    assert out["id"] == tid and any(s["id"] == "relax" and s["status"] == "ok" for s in out["stages"])
    seen = [e[0] for e in progress.events]
    assert len(seen) >= 3 and seen == sorted(seen) and progress.events[-1][:2] == (10, 10)
    assert all(e[2] for e in progress.events)
    wait_status(rig, job["id"])


def test_wait_for_task_until_a_stage(rig, monkeypatch):
    monkeypatch.setattr(mcp_server, "POLL_S", 0.1)
    job = rig.ok("submit_job", profile="t_one", backend="fake", scenario="ok", speed=40)
    tid = job["tasks"][0]["id"]
    err, out = wait_call(rig, task_id=tid, until="pfss", timeout_s=60)
    assert not err and out["reached"] is True
    assert {s["id"]: s["status"] for s in out["stages"]}["pfss"] == "ok"
    err, msg = wait_call(rig, task_id=tid, until="nonsense")
    assert err and "unknown stage" in msg and "relax" in msg
    wait_status(rig, job["id"])


def test_wait_for_task_times_out_with_still_running(rig, monkeypatch):
    monkeypatch.setattr(mcp_server, "POLL_S", 0.1)
    job = rig.ok("submit_job", profile="t_one", backend="fake", scenario="ok", speed=1)
    tid = job["tasks"][0]["id"]
    try:
        err, out = wait_call(rig, task_id=tid, timeout_s=1)
        assert not err and out["still_running"] is True and out["status"] == "running" and out["stage"]
        assert 0.9 <= out["waited_s"] < 10
    finally:
        rig.ok("cancel", job_id=job["id"])
        wait_status(rig, job["id"], wanted=("cancelled",))


def test_wait_for_task_on_a_missing_task_is_an_error(rig):
    err, msg = wait_call(rig, task_id="nope")
    assert err and "no such task" in msg


# ------------------------------------------------------------------ annotations


def test_every_tool_is_in_the_agent_capability_table_and_calls_are_recorded(rig):
    from fluxpype.plumber.agent_log import TOOLS

    async def go():
        async with Client(rig.server) as c:
            return {t.name for t in (await c.list_tools()).tools}

    assert anyio.run(go) == {name for name, _, _ in TOOLS}
    rig.ok("list_jobs", limit=3)
    assert rig.call("get_job", job_id="nope")[0]
    calls = rig.tc.get("/api/agent", headers=H).json()["calls"]
    assert [(c["tool"], c["status"]) for c in calls] == [("get_job", 404), ("list_jobs", 200)]
    assert calls[1]["args"] == '{"limit": 3}' and len({c["call"] for c in calls}) == 2


def test_tool_annotations(rig):
    async def go():
        async with Client(rig.server) as c:
            return {t.name: t for t in (await c.list_tools()).tools}

    tools = anyio.run(go)
    assert len(tools) == 15 and "trash" not in " ".join(tools) and "delete" not in " ".join(tools)
    for name in ("status", "list_profiles", "get_profile", "list_jobs", "get_job", "get_task", "get_relax_series"):
        assert tools[name].annotations.read_only_hint is True, name
    for name in ("read_task_log", "wait_for_task", "cache_summary", "get_task_image"):
        assert tools[name].annotations.read_only_hint is True, name
    assert tools["cancel"].annotations.destructive_hint is True
    for name in ("submit_job", "retry_failed", "set_queue"):
        a = tools[name].annotations
        assert a.read_only_hint is False and a.destructive_hint is False, name
    assert tools["submit_job"].annotations.idempotent_hint is False
    assert tools["retry_failed"].annotations.idempotent_hint is False
    for name in ("read_task_log", "get_task", "wait_for_task", "get_job", "status"):
        assert "untrusted" in tools[name].description, name


# ------------------------------------------------------------------ images


def call_raw(rig, tool, **args):
    async def go():
        async with Client(rig.server) as c:
            return await c.call_tool(tool, args)

    return anyio.run(go)


def png_bytes(w, h):
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.effect_noise((w, h), 80).save(buf, format="PNG")
    return buf.getvalue()


def test_get_task_image_returns_the_image_and_a_caption(rig, done):
    _, tid = done
    arts = rig.ok("get_task", task_id=tid)["artifacts"]
    res = call_raw(rig, "get_task_image", task_id=tid)
    assert not res.is_error
    kinds = [c.type for c in res.content]
    assert kinds == ["image", "text"] and res.content[0].mime_type == "image/png"
    assert base64.b64decode(res.content[0].data)[:4] == b"\x89PNG"
    assert arts[-1]["name"] in res.content[1].text and f"artifact {arts[-1]['index']} of 3" in res.content[1].text
    first = call_raw(rig, "get_task_image", task_id=tid, index=0)
    assert arts[0]["name"] in first.content[1].text
    for bad in (3, -4):
        res = call_raw(rig, "get_task_image", task_id=tid, index=bad)
        assert res.is_error and "out of range" in res.content[0].text


def test_get_task_image_without_artifacts_is_an_error(rig):
    rig.ok("set_queue", paused=True)
    job = rig.ok("submit_job", profile="t_one", backend="fake", speed=300)
    res = call_raw(rig, "get_task_image", task_id=job["tasks"][0]["id"])
    assert res.is_error and "no image artifacts" in res.content[0].text
    rig.ok("cancel", job_id=job["id"])


def test_prepare_image_without_pillow_returns_small_and_rejects_large(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)  # makes `from PIL import ...` raise ImportError
    data = b"\x89PNG" + b"0" * 1000
    assert mcp_server.prepare_image(data, ".png", 512) == (data, "png", "1 KB")
    big = b"\x89PNG" + b"0" * (mcp_server.MAX_RAW_IMAGE + 1)
    with pytest.raises(ToolError, match=r"1\.5 MB.*Pillow"):
        mcp_server.prepare_image(big, ".png", 512)


def test_prepare_image_downscales_with_pillow():
    PILImage = pytest.importorskip("PIL.Image")
    data = png_bytes(3000, 2000)
    out, fmt, note = mcp_server.prepare_image(data, ".png", 500)
    assert fmt == "png" and "3000x2000 scaled to 500x333" in note and len(out) < len(data)
    assert PILImage.open(io.BytesIO(out)).size == (500, 333)
    small = png_bytes(100, 80)
    assert mcp_server.prepare_image(small, ".png", 500)[0] == small  # already small: untouched
    jpg = io.BytesIO()
    PILImage.new("RGBA", (800, 600), (255, 0, 0, 255)).convert("RGB").save(jpg, format="JPEG")
    out, fmt, _ = mcp_server.prepare_image(jpg.getvalue(), ".jpg", 200)
    assert fmt == "jpeg" and PILImage.open(io.BytesIO(out)).size == (200, 150)
    with pytest.raises(ToolError, match="could not decode"):
        mcp_server.prepare_image(b"not an image", ".png", 200)


def test_get_task_image_downscales_a_big_artifact_end_to_end(rig, done):
    PILImage = pytest.importorskip("PIL.Image")
    _, tid = done
    art = rig.ok("get_task", task_id=tid)["artifacts"][-1]
    with open(art["path"], "wb") as f:
        f.write(png_bytes(2400, 1800))
    res = call_raw(rig, "get_task_image", task_id=tid, max_edge=300)
    assert not res.is_error
    img = PILImage.open(io.BytesIO(base64.b64decode(res.content[0].data)))
    assert img.size == (300, 225) and "scaled to 300x225" in res.content[1].text


# ------------------------------------------------------------------ connection and CLI


def test_main_rejects_non_local_urls_and_missing_tokens(capsys, monkeypatch):
    monkeypatch.delenv("FLUX_PLUMBER_TOKEN", raising=False)
    assert mcp_server.main(["--url", "http://plumber.example.com:8765", "--token", "x"]) == 2
    assert "SSH tunnel" in capsys.readouterr().err
    assert mcp_server.main(["--url", "http://127.0.0.1:8765"]) == 2
    assert "--token" in capsys.readouterr().err
    assert mcp_server.main(["--url", "http://127.0.0.1:8765", "--token-file", "/nonexistent/token"]) == 2
    assert mcp_server.main(["--demo"]) == 2
    assert "--home" in capsys.readouterr().err


def test_connect_home_without_autostart_or_with_a_stuck_lock(plumber_home, monkeypatch):
    with pytest.raises(PlumberError, match="no Plumber is running"):
        mcp_server.connect_home(plumber_home, autostart=False)
    monkeypatch.setattr(mcp_server, "LOCK_WAIT_S", 0.3)
    assert cli.acquire_lock(plumber_home)
    try:
        for autostart in (False, True):  # never starts a second Plumber while the lock is held
            with pytest.raises(PlumberError, match=r"flux-plumb --stop --home .*server\.log"):
                mcp_server.connect_home(plumber_home, autostart=autostart)
    finally:
        os.close(cli._LOCK_FD)
        cli._LOCK_FD = None


def test_discovery_ignores_stale_and_foreign_server_json(plumber_home, tmp_path):
    (plumber_home / "token").write_text("tok")
    stale = {"pid": 2**22 + 777, "pid_start": "1", "port": 9, "host": "127.0.0.1", "home": str(plumber_home)}
    (plumber_home / "server.json").write_text(json.dumps(stale))
    assert mcp_server.discover_existing(plumber_home) is None
    (plumber_home / "server.json").write_text("{not json")
    assert mcp_server.discover_existing(plumber_home) is None


# ------------------------------------------------------------------ real process: autostart over stdio


def test_autostart_reuse_and_stop_over_stdio(tmp_path):
    home = tmp_path / "mcp-home"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "fluxpype.plumber.mcp_server", "--home", str(home), "--demo"],
        cwd=str(REPO),
    )

    async def status_and_pid():
        async with Client(params) as c:
            res = await c.call_tool("status", {})
            assert not res.is_error, res.content[0].text
            return json.loads(res.content[0].text), cli.read_server_json(home)

    async def shut_down_work():
        async with Client(params) as c:
            call = lambda name, args=None: c.call_tool(name, args or {})  # noqa: E731
            await call("set_queue", {"paused": True})
            for _ in range(100):  # the demo seeds jobs for a moment; stop them all so nothing outlives the test
                jobs = json.loads((await call("list_jobs", {"limit": 100})).content[0].text)["jobs"]
                for j in jobs:
                    if j["status"] in ("pending", "running"):
                        await call("cancel", {"job_id": j["id"]})
                st = json.loads((await call("status")).content[0].text)
                if len(jobs) >= 7 and not st["running"] and not st["pending"]:
                    return
                await anyio.sleep(0.3)
            raise AssertionError("demo jobs did not stop")

    async def bounded(fn):
        with anyio.fail_after(120):
            return await fn()

    info = None
    try:
        status, info = anyio.run(bounded, status_and_pid)
        assert info and status["demo"] is True and status["plumber_home"] == str(home)
        assert (home / "token").exists() and (home / "server.log").exists()
        status2, info2 = anyio.run(bounded, status_and_pid)  # a second MCP server reuses the running Plumber
        assert info2["pid"] == info["pid"] and info2["started"] == info["started"]
        anyio.run(bounded, shut_down_work)
        stop = subprocess.run(
            [sys.executable, "-m", "fluxpype.plumber.cli", "--stop", "--home", str(home)],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert stop.returncode == 0, stop.stderr
        assert cli.read_server_json(home) is None and not process_alive(info["pid"], info["pid_start"])
    finally:
        if info and process_alive(info["pid"], info["pid_start"]):
            os.kill(info["pid"], signal.SIGKILL)
