"""A4-07 / A4-08: a task_state push means something changed; /api/system?since sends only new samples."""

import copy

import pytest

pytest.importorskip("fastapi")

from fluxpype.plumber.server import state_change_key  # noqa: E402
from fluxpype.plumber.sysmon import SysMon  # noqa: E402
from tests.test_plumber_monitor import Run, ev  # noqa: E402


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


def running_events(run):
    run.events(
        ev("run.start", 100, cr="2225", profile="p"),
        ev("stage.start", 101, stage="magnetogram"),
        ev("stage.end", 103, stage="magnetogram", status="ok", elapsed_s=2, detail=None),
        ev("stage.start", 103, stage="footpoints"),
    )


def test_clock_only_changes_do_not_change_the_key(run):
    running_events(run)
    mon = run.mon()
    a = mon.refresh(now=110, alive=True)
    b = mon.refresh(now=112.5, alive=True)
    # the fields that tick are really different between the two reads ...
    assert a["observed_at"] != b["observed_at"]
    assert a["sources"]["events"]["age_s"] != b["sources"]["events"]["age_s"]
    # ... and elapsed_s, which the scheduler adds, moves too
    a["elapsed_s"], b["elapsed_s"] = 5.0, 7.5
    assert state_change_key(a) == state_change_key(b)


def test_a_real_change_changes_the_key(run):
    running_events(run)
    mon = run.mon()
    a = mon.refresh(now=110, alive=True)
    run.events(ev("metric", 111, name="n_actual", value=97, unit="fluxons"))
    b = mon.refresh(now=112, alive=True)
    assert state_change_key(a) != state_change_key(b)
    c = copy.deepcopy(a)
    c["status"] = "ok"
    assert state_change_key(a) != state_change_key(c)


def test_key_ignores_age_s_on_both_sources_and_survives_odd_shapes():
    s = {
        "id": "t",
        "elapsed_s": 1.0,
        "observed_at": 5,
        "sources": {"events": {"age_s": 1.0, "lines": 3}, "log": {"age_s": 2.0}, "x": 7},
    }
    t = copy.deepcopy(s)
    t["elapsed_s"], t["observed_at"] = 9.0, 6
    t["sources"]["events"]["age_s"], t["sources"]["log"]["age_s"] = 4.0, 5.0
    assert state_change_key(s) == state_change_key(t)
    t["sources"]["events"]["lines"] = 4
    assert state_change_key(s) != state_change_key(t)
    assert state_change_key({"id": "t", "float": float("nan")})  # NaN is cleaned, not a crash


def test_system_since_returns_only_newer_samples():
    mon = SysMon(lambda: [])
    first = mon.snapshot()
    assert len(first["history"]) == 1
    last = first["history"][-1]["t"]
    assert mon.snapshot(since=last)["history"] == []  # nothing newer inside the sample interval
    mon._t -= 2.0  # let the next call take a new sample
    nxt = mon.snapshot(since=last)
    assert len(nxt["history"]) == 1 and nxt["history"][0]["t"] > last
    assert nxt["now"] == nxt["history"][-1]
    assert len(mon.snapshot()["history"]) == 2  # without `since` the whole history is still sent
