"""The event log (notify.py): outcomes from store transitions, telemetry, storage, the API and job origins.

Every scenario runs on the fake backend with no browser client connected: the Notifier is a loop step, not a
WebSocket side effect.
"""

import sqlite3
import threading
import time

import pytest

from fluxpype.plumber import paths
from fluxpype.plumber.backends.base import Handle, TaskStatus
from fluxpype.plumber.backends.fake import FakeBackend
from fluxpype.plumber.notify import Notifier

FAST = {"speed": "300"}
END = ("ok", "failed", "cancelled", "partial")


def run(sched, notifier, job_id, timeout=60, telemetry=False):
    end = time.time() + timeout
    while time.time() < end:
        sched.tick()
        notifier.observe()
        if telemetry:
            notifier.telemetry()
        if sched.job_view(job_id)["status"] in END:
            notifier.observe()
            return
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def rows(store, job_id=None):
    return [(e["kind"], e["key"], e["f"]) for e in reversed(store.events(limit=500, job=job_id))]


def keys(store, job_id):
    return {f"{k}|{key}" for k, key, _ in rows(store, job_id)}


def expected(job, kinds):
    jid, tid = job["id"], job["tasks"][0]["id"]
    out = {f"job_queued|job_queued|{jid}"}
    for k in kinds:
        if k == "task_started":
            out.add(f"task_started|task_started|{tid}|1")
        elif k.startswith("task_"):
            out.add(f"{k}|task_end|{tid}|1")
        else:
            out.add(f"{k}|job_end|{jid}|1")
    return out


@pytest.mark.parametrize(
    "scenario, kinds, f",
    [
        ("ok", ["task_started", "task_ok", "job_ok"], {"attempt": 1, "exit": 0}),
        ("fail_in_pfss", ["task_started", "task_failed", "job_failed"], {"exit": 1, "stage": "pfss", "reason": "exit"}),
        (
            "failure_swallowed",
            ["task_started", "task_failed", "job_failed"],
            {"exit": 0, "stage": "pfss", "reason": "stage"},
        ),
        ("timeout", ["task_started", "task_timeout", "job_failed"], {"exit": 124}),
        ("crash", ["task_started", "task_crashed", "job_failed"], {"attempt": 1}),
        ("launcher_lost", ["task_started", "task_ok", "job_ok"], {"exit": 0, "launcher_lost": 1}),
    ],
)
def test_each_scenario_gives_exactly_its_events(make_scheduler, scenario, kinds, f):
    sched = make_scheduler()
    n = Notifier(sched)
    job = sched.create_job("t_one", backend="fake", extra={**FAST, "scenario": scenario})
    run(sched, n, job["id"])
    assert keys(sched.store, job["id"]) == expected(job, kinds)
    outcome = next(ff for k, key, ff in rows(sched.store, job["id"]) if key.startswith("task_end|"))
    assert f.items() <= outcome.items()
    if scenario == "crash":
        assert "exit" not in outcome and "launcher_lost" not in outcome


def test_cancel_while_pending_and_while_running(make_scheduler):
    sched = make_scheduler()
    n = Notifier(sched)
    n.observe()  # the first pass only records the queue setting
    sched.set_queue(paused=True)
    job = sched.create_job("t_one", backend="fake", extra=FAST)
    sched.cancel_job(job["id"])
    run(sched, n, job["id"])
    assert keys(sched.store, job["id"]) == expected(job, ["task_cancelled", "job_cancelled"])
    sched.set_queue(paused=False)
    job = sched.create_job("t_one", backend="fake", extra={"speed": "1"})  # about a minute: still running
    end = time.time() + 30
    while sched.store.get_task(job["tasks"][0]["id"])["status"] != "running" and time.time() < end:
        sched.tick()
        time.sleep(0.05)
    n.observe()
    sched.cancel_job(job["id"])
    run(sched, n, job["id"])
    assert keys(sched.store, job["id"]) == expected(job, ["task_started", "task_cancelled", "job_cancelled"])
    assert [k for k, _, _ in rows(sched.store) if k.startswith("queue_")] == ["queue_paused", "queue_resumed"]


def test_submit_failure_is_a_failed_task_with_reason_submit(make_scheduler):
    class Refuses(FakeBackend):
        def submit(self, task):
            raise RuntimeError("no slots on /secret/path")

    sched = make_scheduler(backends={"fake": Refuses()})
    n = Notifier(sched)
    job = sched.create_job("t_one", backend="fake", extra=FAST)
    run(sched, n, job["id"])
    assert keys(sched.store, job["id"]) == expected(job, ["task_failed", "job_failed"])
    f = next(ff for k, _, ff in rows(sched.store, job["id"]) if k == "task_failed")
    assert f == {"attempt": 1, "reason": "submit"}


def test_stall_then_resume_and_sleep_suppression(make_scheduler):
    sched = make_scheduler()
    n = Notifier(sched, stall_after=2)
    job = sched.create_job("t_one", backend="fake", extra={"speed": "10", "scenario": "stall"})
    run(sched, n, job["id"], telemetry=True)
    kinds = [k for k, _, _ in rows(sched.store, job["id"])]
    assert kinds.count("task_stalled") == 1 and kinds.count("task_resumed") == 1, kinds
    assert kinds.index("task_stalled") < kinds.index("task_resumed") < kinds.index("task_ok")
    stalled = next(f for k, _, f in rows(sched.store, job["id"]) if k == "task_stalled")
    assert stalled["stage"] == "relax" and stalled["limit_s"] == 2 and stalled["quiet_s"] >= 2

    # the same quiet spell right after a 60 s gap between telemetry passes (the laptop slept) is not a stall
    job = sched.create_job("t_one", backend="fake", extra={"speed": "10", "scenario": "stall"})
    tid = job["tasks"][0]["id"]
    end = time.time() + 30
    while time.time() < end:
        sched.tick()
        n.observe()
        t = sched.store.get_task(tid)
        if t["status"] == "running" and time.time() - n.last_output(t) > 3:
            break
        time.sleep(0.05)
    n._last_tel = time.time() - 60
    n.telemetry()
    time.sleep(0.5)
    n.telemetry()
    assert not [k for k, _, _ in rows(sched.store, job["id"]) if k == "task_stalled"]
    run(sched, n, job["id"])


def test_a_second_app_on_the_same_home_writes_no_duplicates(make_scheduler):
    a, b = make_scheduler(), make_scheduler()
    na, nb = Notifier(a), Notifier(b)
    job = a.create_job("t_two", backend="fake", extra=FAST)
    end = time.time() + 60
    while a.job_view(job["id"])["status"] not in END and time.time() < end:
        a.tick()
        na.observe()
        nb.observe()
        time.sleep(0.05)
    na.observe()
    nb.observe()
    got = [key for _, key, _ in rows(a.store, job["id"])]
    assert len(got) == len(set(got)) == 1 + 2 * 4 + 1  # queued, started and end per task, job end


def test_an_upgraded_database_announces_nothing_old(plumber_home, make_scheduler):
    db = paths.db_path(plumber_home)
    con = sqlite3.connect(str(db))
    con.executescript("""
        CREATE TABLE jobs (id TEXT PRIMARY KEY, profile TEXT NOT NULL, overrides_json TEXT NOT NULL DEFAULT '{}',
            backend TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, started REAL, ended REAL,
            snapshot_path TEXT NOT NULL, extra_json TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE tasks (id TEXT PRIMARY KEY, job_id TEXT NOT NULL, idx INTEGER NOT NULL, cr TEXT NOT NULL,
            nflux TEXT NOT NULL, adapt TEXT NOT NULL, method TEXT NOT NULL, status TEXT NOT NULL, pid INTEGER,
            pid_start TEXT, exit_code INTEGER, created REAL NOT NULL, started REAL, ended REAL, run_dir TEXT NOT NULL,
            lock_key TEXT NOT NULL DEFAULT '', slots INTEGER NOT NULL DEFAULT 1, attempt INTEGER NOT NULL DEFAULT 1,
            summary_json TEXT);
        INSERT INTO jobs VALUES ('jaaaaaaaa', 't_one', '{}', 'fake', 'failed', 1, 1, 2, '/x', '{}');
        INSERT INTO tasks VALUES ('jaaaaaaaa-01', 'jaaaaaaaa', 1, '2225', '100', '0', 'parker', 'failed', NULL, NULL, 1,
            1, 1, 2, '/x/r', '', 1, 1, NULL);
        """)  # the schema before the event log
    con.commit()
    con.close()
    sched = make_scheduler()
    n = Notifier(sched)
    n.observe()
    assert rows(sched.store) == []
    assert sched.store.get_job("jaaaaaaaa")["origin"] == {}


class Stub:
    """A backend whose next poll answer the test sets."""

    name = kind = "stub"

    def __init__(self):
        self.state = TaskStatus(state="queued")

    def capabilities(self):
        return FakeBackend().capabilities()

    def probe(self):
        return FakeBackend().probe()

    def submit(self, task):
        return Handle(task.task_id, task.run_dir)

    def poll(self, handle):
        return self.state

    def cancel(self, handle, force=False):
        pass

    def read_file(self, handle, name, offset=0):
        return b""


def test_queued_and_unknown_never_finalize_and_flag_the_backend(make_scheduler):
    stub = Stub()
    sched = make_scheduler(backends={"stub": stub})
    n = Notifier(sched)
    job = sched.create_job("t_one", backend="stub")
    tid = job["tasks"][0]["id"]
    for state in ("queued", "unknown", "unknown"):
        stub.state = TaskStatus(state=state)
        sched.tick()
        n.observe()
        n.telemetry()
        assert sched.store.get_task(tid)["status"] == "running"
    assert sched.backend_unknown and sched.poll_state[tid] == "unknown"
    stub.state = TaskStatus(state="exited", exit_code=0, ended=time.time())
    sched.tick()
    n.observe()
    kinds = [k for k, _, _ in rows(sched.store)]
    assert kinds.count("backend_down") == 1 and kinds.count("backend_up") == 1
    assert "task_crashed" not in kinds and "task_ok" in kinds and "job_ok" in kinds
    assert sched.backend_unknown == {}


def test_retention_and_delete_acknowledges(make_scheduler):
    sched = make_scheduler()
    s = sched.store
    now = time.time()
    for i, (sev, age_days) in enumerate([("info", 8), ("success", 8), ("error", 8), ("warn", 31), ("info", 1)]):
        s.add_event({"t": now - age_days * 86400, "kind": "task_ok", "sev": sev, "key": f"k{i}"})
    s.trim_events(now)
    assert sorted(e["key"] for e in s.events()) == ["k2", "k4"]
    for i in range(5):
        s.add_event({"kind": "task_ok", "sev": "info", "key": f"m{i}"})
    s.trim_events(now, keep=3)
    assert [e["key"] for e in s.events()] == ["m4", "m3", "m2"]

    n = Notifier(sched)
    job = sched.create_job("t_one", backend="fake", extra={**FAST, "scenario": "fail_in_pfss"})
    run(sched, n, job["id"])
    assert s.event_counts()["attention"] == 2
    sched.delete_job(job["id"])
    assert s.event_counts()["attention"] == 0


# ------------------------------------------------------------------ API and origins

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fluxpype.plumber.server import create_app  # noqa: E402

TOKEN = "t0k3n-for-tests"
H = {"X-Flux-Token": TOKEN}


@pytest.fixture
def client(make_scheduler):
    sched = make_scheduler()
    app = create_app(sched, TOKEN, demo=True, tick_interval=0.1, telemetry_interval=0.2)
    with TestClient(app, base_url="http://localhost") as c:
        c.sched = sched
        end = time.time() + 10
        while not sched.store.last_seq() and time.time() < end:  # the loop's plumber_started event
            time.sleep(0.05)
        yield c


def wait_job(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}", headers=H).json()
        if job["status"] in END:
            time.sleep(0.3)  # one more loop pass writes the job's end event
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_events_api_orders_and_pages(client):
    s = client.sched.store
    base = s.last_seq()
    for i in range(5):
        s.add_event({"kind": "task_ok", "sev": "success", "key": f"x{i}"})
    seqs = [base + i for i in range(1, 6)]
    r = client.get(f"/api/events?since={base}&limit=3", headers=H).json()
    assert [e["seq"] for e in r["events"]] == seqs[:3] and r["last_seq"] == seqs[-1] and len(r["log_id"]) == 12
    r = client.get(f"/api/events?before={seqs[-1]}&limit=2", headers=H).json()
    assert [e["seq"] for e in r["events"]] == [seqs[3], seqs[2]]
    r = client.get("/api/events?limit=2", headers=H).json()
    assert [e["seq"] for e in r["events"]] == [seqs[4], seqs[3]]
    assert client.get("/api/events?min_sev=error", headers=H).json()["events"] == []


def test_long_poll_returns_on_insert_and_empty_at_timeout(client):
    s = client.sched.store
    last = s.last_seq()
    t0 = time.monotonic()
    r = client.get(f"/api/events?since={last}&wait=1", headers=H).json()
    assert r["events"] == [] and time.monotonic() - t0 >= 0.9
    threading.Timer(0.5, lambda: s.add_event({"kind": "task_ok", "sev": "success", "key": "late"})).start()
    t0 = time.monotonic()
    r = client.get(f"/api/events?since={last}&wait=10", headers=H).json()
    assert [e["key"] for e in r["events"]] == ["late"] and time.monotonic() - t0 < 2.0


def test_the_fifth_concurrent_wait_of_a_session_gets_429(client):
    last = client.sched.store.last_seq() + 1000
    hdr = {**H, "X-Flux-Session": "abcdef123456"}
    codes = []
    threads = [
        threading.Thread(
            target=lambda: codes.append(client.get(f"/api/events?since={last}&wait=3", headers=hdr).status_code)
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    time.sleep(1.0)
    r = client.get(f"/api/events?since={last}&wait=3", headers=hdr)
    assert r.status_code == 429 and r.headers["retry-after"] == "5"
    other = {**H, "X-Flux-Session": "zzzzzz999999"}
    assert client.get(f"/api/events?since={last}&wait=1", headers=other).status_code == 200
    for t in threads:
        t.join()
    assert codes == [200] * 4


def test_read_ack_and_job_events_without_a_browser(client):
    job = client.post(
        "/api/jobs", json={"profile": "t_one", "scenario": "fail_in_pfss", "speed": 300}, headers=H
    ).json()
    wait_job(client, job["id"])
    r = client.get(f"/api/events?job={job['id']}", headers=H).json()
    assert {e["kind"] for e in r["events"]} == {"job_queued", "task_started", "task_failed", "job_failed"}
    assert r["unread"] == 2 and r["attention"] == 2
    r = client.post("/api/events/read", json={"upto": r["last_seq"]}, headers=H).json()
    assert r["unread"] == 0 and r["attention"] == 2
    one = client.get(f"/api/events?job={job['id']}&min_sev=error", headers=H).json()["events"][0]
    assert client.post("/api/events/ack", json={"seqs": [one["seq"]]}, headers=H).json()["attention"] == 1
    assert client.post("/api/events/ack", json={"all": True}, headers=H).json()["attention"] == 0
    assert client.post("/api/events/ack", json={"all": True}, headers={**H, "X-Flux-Client": "ask"}).status_code == 403


def test_origins_web_mcp_ask_and_retry(client, monkeypatch):
    body = {"profile": "t_one", "scenario": "fail_in_pfss", "speed": 300}
    web = client.post("/api/jobs", json=body, headers={**H, "X-Flux-Tab": "a1b2c3d4e5f6"}).json()
    assert web["origin"] == {"kind": "web", "tab": "a1b2c3d4e5f6"}
    assert client.post("/api/jobs", json=body, headers={**H, "X-Flux-Tab": "NOT/valid"}).json()["origin"] == {
        "kind": "web"
    }
    mcp = client.post(
        "/api/jobs",
        json=body,
        headers={**H, "X-Flux-Client": "mcp", "X-Flux-Session": "../x", "X-Flux-Call": "c0ffee123456"},
    ).json()
    assert mcp["origin"] == {"kind": "mcp", "call": "c0ffee123456"}
    wait_job(client, web["id"])
    retried = client.post(f"/api/jobs/{web['id']}/retry", json={}, headers={**H, "X-Flux-Client": "mcp"}).json()
    assert retried["origin"] == {"kind": "web", "tab": "a1b2c3d4e5f6", "retried_by": "mcp"}
    ev = client.get(f"/api/events?job={web['id']}", headers=H).json()["events"]
    assert all(e["origin"] == {"kind": "web", "tab": "a1b2c3d4e5f6"} for e in ev)

    from fluxpype.plumber import ask

    monkeypatch.setattr(ask, "_question", "q0a1b2c3d4e5")
    client.patch("/api/queue", json={"paused": True}, headers=H)
    p = client.post(
        "/api/proposals",
        json={"tool": "submit_job", "args": {"profile": "t_one"}, "question": "q0a1b2c3d4e5"},
        headers={**H, "X-Flux-Client": "ask"},
    ).json()
    out = client.post(f"/api/proposals/{p['id']}/approve", headers={**H, "X-Flux-Tab": "f00d00f00d00"}).json()
    job = client.get(f"/api/jobs/{out['result']['job_id']}", headers=H).json()
    assert job["origin"] == {"kind": "ask", "question": "q0a1b2c3d4e5", "proposal": p["id"], "tab": "f00d00f00d00"}
