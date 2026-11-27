"""A3-09 / A3-01 / A3-02: the tick owns running-task state; readers copy one cached state.

Covers the measured race (relax points duplicated or lost when several threads refresh one monitor)
and the cost of a slow remote host (readers must not wait on a poll in flight).
"""

import json
import threading
import time

import pytest

from fluxpype.plumber.backends.fake import FakeBackend
from fluxpype.plumber.monitor import ProbeContext, TaskMonitor
from tests.conftest import wait_for


def step_line(n):
    return json.dumps({"v": 1, "t": float(n), "type": "relax.step", "pid": 1, "task": "x", "step": n}) + "\n"


@pytest.mark.parametrize("readers", [2, 3])
def test_concurrent_refresh_neither_duplicates_nor_loses_relax_points(tmp_path, readers):
    run = tmp_path / "run"
    run.mkdir()
    mon = TaskMonitor(str(run), ProbeContext(str(tmp_path), "b", "1", "1", "0", "hmi"))
    total = 4000
    done = threading.Event()
    overlap = threading.Barrier(readers + 1)
    first = [0] * readers  # refreshes each reader completed before the writer had finished

    def writer():
        overlap.wait()
        with open(run / "events.jsonl", "a", buffering=1) as f:
            for n in range(total):
                f.write(step_line(n))
        done.set()

    def reader(i):
        overlap.wait()
        while not done.is_set():
            mon.refresh(probe=False, alive=True)
            first[i] += 1

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(readers)]
    for th in threads:
        th.start()
    w = threading.Thread(target=writer)
    w.start()
    w.join()
    for th in threads:
        th.join()
    mon.refresh(probe=False, alive=True)
    steps = [e["step"] for e in mon.ev_steps]
    assert all(n > 0 for n in first)  # every reader refreshed while the writer was appending
    assert len(steps) == total and len(set(steps)) == total and mon.ev_bad == 0


class CountingBackend(FakeBackend):
    """Counts polls; an optional delay stands in for a slow ssh host."""

    def __init__(self, delay=0.0):
        super().__init__()
        self.polls = 0
        self.delay = delay

    def poll(self, handle):
        self.polls += 1
        if self.delay:
            time.sleep(self.delay)
        return super().poll(handle)


def running_task(sched, name="t_one"):
    job = sched.create_job(name, backend="fake", extra={"speed": "1", "scenario": "ok"})
    wait_for(sched, lambda: sched.store.tasks_with_status("running"))
    return job, job["tasks"][0]["id"]


def test_parallel_readers_cause_no_extra_polls_or_refreshes(make_scheduler):
    backend = CountingBackend()
    sched = make_scheduler(backends={"fake": backend})
    job, tid = running_task(sched)
    sched.tick()  # the tick caches the state
    refreshes = [0]
    mon = sched.monitor_for(sched.store.get_task(tid))
    real = mon._refresh

    def counting(*a, **kw):
        refreshes[0] += 1
        return real(*a, **kw)

    mon._refresh = counting
    polls_before = backend.polls
    out = []
    seeded = {"step": 1, "stiffness": 42.0}
    mon.ev_steps.append(seeded)  # known points the readers must return from the cache
    sched.tick()
    expected_n = sched.relax_series(tid)["n"]
    assert expected_n >= 1
    polls_before = backend.polls
    refreshes[0] = 0
    threads = [
        threading.Thread(target=lambda: out.append((sched.task_state(tid), sched.relax_series(tid)))) for _ in range(10)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(out) == 10
    assert backend.polls - polls_before == 0 and refreshes[0] == 0
    assert all(s["status"] == "running" and s["id"] == tid for s, _ in out)
    assert all(r["n"] == expected_n and seeded in [{k: p[k] for k in seeded} for p in r["points"]] for _, r in out)
    sched.cancel_job(job["id"])
    wait_for(sched, lambda: not sched.store.tasks_with_status("running"))


def test_a_slow_host_does_not_block_readers(make_scheduler):
    backend = CountingBackend(delay=0.4)
    sched = make_scheduler(backends={"fake": backend})
    job, tid = running_task(sched)
    sched.tick()
    backend.polls = 0
    ticker = threading.Thread(target=sched.tick)
    ticker.start()
    time.sleep(0.1)  # the tick is now inside a 0.4 s poll
    assert ticker.is_alive() and backend.polls == 1
    t0 = time.time()
    sched.monitor_for(sched.store.get_task(tid)).ev_steps.append({"step": 7, "stiffness": 9.0})
    state = sched.task_state(tid)
    explain = sched.queue_explain()
    series = sched.relax_series(tid)
    waited = time.time() - t0
    ticker.join()
    assert (
        state["status"] == "running"
        and isinstance(explain, dict)
        and any(p["step"] == 7 for p in series["points"])
        and state["id"] == tid
    )
    assert waited < 0.2, f"readers waited {waited:.3f}s behind a 0.4 s poll"
    assert backend.polls == 1  # the readers did not poll
    sched.cancel_job(job["id"])
    wait_for(sched, lambda: not sched.store.tasks_with_status("running"))
