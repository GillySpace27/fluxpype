"""Scheduler against the FakeBackend: real detached launcher + fake_flux processes."""

import json
import os
import signal
import time
from pathlib import Path

import pytest

from fluxpype.studio.scheduler import SchedulerError
from tests.conftest import wait_for

FAST = {"speed": "300", "scenario": "ok"}


def statuses(sched, job_id):
    return [t["status"] for t in sched.store.tasks_for_job(job_id)]


def finish(sched, job_id, timeout=60):
    wait_for(sched, lambda: all(s not in ("pending", "running") for s in statuses(sched, job_id)), timeout)


def test_single_job_lifecycle(make_scheduler):
    sched = make_scheduler()
    seen = []
    sched.on_change = lambda kind, payload: seen.append(kind)
    job = sched.create_job("t_one", backend="fake", extra=FAST)
    assert job["tasks"][0]["status"] in ("pending", "running") and len(job["tasks"]) == 1
    finish(sched, job["id"])
    view = sched.job_view(job["id"])
    assert view["status"] == "ok" and view["ended"] and view["counts"]["ok"] == 1
    t = view["tasks"][0]
    assert t["exit_code"] == 0 and t["started"] <= t["ended"]
    state = sched.task_state(t["id"])
    assert state["status"] == "ok" and state["mode"] == "events"
    assert [s["status"] for s in state["stages"]] == ["ok"] * 9 + ["skipped"]
    assert all(s["verified"] in (True, None) for s in state["stages"])
    assert state["relax"]["n_points"] > 10 and state["relax"]["source"] == "event"
    assert {"job_state", "task_state", "queue_state"} <= set(seen)
    # finished tasks are summarized once and served from the store afterwards
    row = sched.store.get_task(t["id"])
    assert json.loads(row["summary_json"])["status"] == "ok"


def test_snapshot_is_frozen_and_used_by_the_run(make_scheduler, studio_config):
    sched = make_scheduler()
    job = sched.create_job("t_one", overrides={"relax_threshold": "80", "max_cycles": "3"}, backend="fake", extra=FAST)
    snap = Path(job["snapshot_path"])
    before = snap.read_text()
    assert "relax_threshold = 80" in before
    studio_config.write_text(studio_config.read_text().replace("relax_threshold = 1.5", "relax_threshold = 9.9"))
    finish(sched, job["id"])
    assert snap.read_text() == before  # editing config.ini mid-run never reaches the job
    state = sched.task_state(job["tasks"][0]["id"])
    # a target of 80 percent is met by the end of the first cycle
    assert state["relax"]["cycles"] == 1


def test_bad_overrides_and_profiles_are_rejected(make_scheduler):
    sched = make_scheduler()
    with pytest.raises(SchedulerError):
        sched.create_job("t_one", overrides={"run_script": "x"}, backend="fake")
    with pytest.raises(SchedulerError):
        sched.create_job("nope", backend="fake")
    with pytest.raises(SchedulerError, match="flow_method"):
        sched.create_job("Real_Run", backend="fake")
    assert sched.store.list_jobs() == []


def test_old_install_is_refused(make_scheduler):
    from fluxpype.studio.backends.fake import FakeBackend

    class Old(FakeBackend):
        def probe(self):
            return []

    sched = make_scheduler(backends={"fake": Old()})
    with pytest.raises(SchedulerError, match="too old"):
        sched.create_job("t_one", backend="fake")


def test_task_subset(make_scheduler):
    sched = make_scheduler()
    job = sched.create_job("t_two", backend="fake", tasks=[["2226", "100", "0", "wsa"]], extra=FAST)
    assert [(t["cr"], t["method"]) for t in job["tasks"]] == [("2226", "wsa")]
    sched.cancel_job(job["id"])
    with pytest.raises(SchedulerError):
        sched.create_job("t_two", backend="fake", tasks=[["9", "9", "9", "x"]])


def test_conflicting_outputs_never_overlap(make_scheduler):
    """Two jobs writing the same (data_dir, batch, cr) must serialize even when slots allow parallel runs."""
    sched = make_scheduler()
    sched.set_queue(max_parallel=4)
    a = sched.create_job("t_one", backend="fake", extra={"speed": "60"})
    b = sched.create_job("t_one", backend="fake", extra={"speed": "60"})
    peak = {"n": 0}

    def check():
        running = sched.store.tasks_with_status("running")
        peak["n"] = max(peak["n"], len(running))
        keys = [k for t in running for k in t["lock_key"].split("\n")]
        assert len(keys) == len(set(keys)), "two running tasks share a lock key"
        return all(s == "ok" for s in statuses(sched, a["id"]) + statuses(sched, b["id"]))

    wait_for(sched, check, 90)
    assert peak["n"] == 1


def test_independent_tasks_run_in_parallel_up_to_max_parallel(make_scheduler):
    sched = make_scheduler()
    sched.set_queue(max_parallel=3)
    job = sched.create_job("t_big", backend="fake", extra={"speed": "25"})  # 3 CRs, no shared keys, 6 slots each
    peak = {"n": 0}

    def check():
        peak["n"] = max(peak["n"], len(sched.store.tasks_with_status("running")))
        return all(s == "ok" for s in statuses(sched, job["id"]))

    wait_for(sched, check, 90)
    assert peak["n"] == 3


def test_cpu_budget_limits_parallelism(make_scheduler):
    sched = make_scheduler(cpu_budget=8)  # each t_big task needs 6 slots, so only one fits
    sched.set_queue(max_parallel=3)
    job = sched.create_job("t_big", backend="fake", extra={"speed": "60"})
    peak = {"n": 0}

    def check():
        peak["n"] = max(peak["n"], len(sched.store.tasks_with_status("running")))
        return all(s == "ok" for s in statuses(sched, job["id"]))

    wait_for(sched, check, 90)
    assert peak["n"] == 1


def test_a_task_bigger_than_the_budget_still_runs_alone(make_scheduler):
    sched = make_scheduler(cpu_budget=2)
    job = sched.create_job("t_big", backend="fake", extra=FAST)  # 6 slots > budget 2
    finish(sched, job["id"], 90)
    assert sched.job_view(job["id"])["status"] == "ok"


def test_pause_and_resume(make_scheduler):
    sched = make_scheduler()
    sched.set_queue(paused=True)
    job = sched.create_job("t_one", backend="fake", extra=FAST)
    time.sleep(0.3)
    sched.tick()
    assert statuses(sched, job["id"]) == ["pending"] and sched.queue_state()["paused"]
    sched.set_queue(paused=False)
    finish(sched, job["id"])
    assert statuses(sched, job["id"]) == ["ok"]


def test_queue_settings_are_validated(make_scheduler):
    sched = make_scheduler()
    with pytest.raises(SchedulerError):
        sched.set_queue(max_parallel=0)
    with pytest.raises(SchedulerError):
        sched.set_queue(cpu_budget=0)


def test_cancel_a_running_task(make_scheduler):
    sched = make_scheduler()
    job = sched.create_job("t_one", backend="fake", extra={"speed": "1", "scenario": "ok"})
    tid = job["tasks"][0]["id"]
    wait_for(sched, lambda: (Path(sched.store.get_task(tid)["run_dir"]) / "pid").exists())
    time.sleep(0.5)
    pid = json.loads((Path(sched.store.get_task(tid)["run_dir"]) / "pid").read_text())["pid"]
    sched.cancel_task(tid)
    finish(sched, job["id"], 30)
    assert statuses(sched, job["id"]) == ["cancelled"]
    assert sched.job_view(job["id"])["status"] == "cancelled"
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the whole process group is gone
    state = sched.task_state(tid)
    assert any(s["status"] == "failed" and s["detail"] for s in state["stages"])  # the open stage is closed, not left "running"


def test_cancel_pending_tasks_and_whole_job(make_scheduler):
    sched = make_scheduler()
    sched.set_queue(paused=True)
    job = sched.create_job("t_two", backend="fake", extra=FAST)
    assert len(job["tasks"]) == 4
    sched.cancel_task(job["tasks"][0]["id"])
    assert statuses(sched, job["id"]).count("cancelled") == 1
    sched.cancel_job(job["id"])
    assert set(statuses(sched, job["id"])) == {"cancelled"}


def test_retry_requeues_failed_tasks_and_keeps_the_old_run(make_scheduler):
    sched = make_scheduler()
    job = sched.create_job("t_one", backend="fake", extra={"speed": "300", "scenario": "fail_in_pfss"})
    finish(sched, job["id"])
    t = sched.store.get_task(job["tasks"][0]["id"])
    assert t["status"] == "failed" and t["exit_code"] == 1
    state = sched.task_state(t["id"])
    pfss = next(s for s in state["stages"] if s["id"] == "pfss")
    assert pfss["status"] == "failed" and "get_pfss.py failed" in pfss["detail"]
    sched.retry_job(job["id"])
    row = sched.store.get_task(t["id"])
    assert row["attempt"] == 2 and Path(t["run_dir"] + ".attempt1").is_dir()
    finish(sched, job["id"])
    assert sched.store.get_task(t["id"])["status"] == "failed"  # same scenario, fails again
    with pytest.raises(SchedulerError):
        # nothing failed in a fresh ok job
        j2 = sched.create_job("t_light", backend="fake", extra=FAST)
        finish(sched, j2["id"])
        sched.retry_job(j2["id"])


def test_delete_job_requires_it_to_be_finished(make_scheduler):
    sched = make_scheduler()
    sched.set_queue(paused=True)
    job = sched.create_job("t_one", backend="fake", extra=FAST)
    with pytest.raises(SchedulerError):
        sched.delete_job(job["id"])
    sched.cancel_job(job["id"])
    sched.delete_job(job["id"])
    assert sched.job_view(job["id"]) is None and not Path(job["snapshot_path"]).exists()


def test_reattach_after_a_server_restart(make_scheduler):
    first = make_scheduler()
    job = first.create_job("t_one", backend="fake", extra={"speed": "20", "scenario": "ok"})
    tid = job["tasks"][0]["id"]
    wait_for(first, lambda: first.store.get_task(tid)["status"] == "running")
    wait_for(first, lambda: (Path(first.store.get_task(tid)["run_dir"]) / "pid").exists())
    pid = json.loads((Path(first.store.get_task(tid)["run_dir"]) / "pid").read_text())["pid"]
    # the "server" goes away: nothing ticks while the detached task keeps running
    second = make_scheduler()
    assert second.store.get_task(tid)["status"] == "running"
    finish(second, job["id"], 60)
    t = second.store.get_task(tid)
    assert t["status"] == "ok" and t["exit_code"] == 0
    # it was adopted, not relaunched: the pid file still names the original process
    assert json.loads((Path(t["run_dir"]) / "pid").read_text())["pid"] == pid
    assert second.task_state(tid)["relax"]["n_points"] > 5


def test_a_vanished_task_is_reported_crashed(make_scheduler):
    sched = make_scheduler()
    job = sched.create_job("t_one", backend="fake", extra={"speed": "1", "scenario": "ok"})
    tid = job["tasks"][0]["id"]
    run_dir = Path(sched.store.get_task(tid)["run_dir"])
    wait_for(sched, lambda: (run_dir / "pid").exists())
    info = json.loads((run_dir / "pid").read_text())
    time.sleep(0.5)
    os.kill(info["launcher_pid"], signal.SIGKILL)  # launcher first, so nobody records an exit code
    os.killpg(info["pid"], signal.SIGKILL)
    finish(sched, job["id"], 30)
    t = sched.store.get_task(tid)
    assert t["status"] == "crashed" and t["exit_code"] is None
    state = sched.task_state(tid)
    assert state["outcome"] == "crashed" and state["current_stage"] is None
    assert any(s["status"] == "failed" for s in state["stages"])


def test_no_events_scenario_runs_on_log_and_files_alone(make_scheduler):
    sched = make_scheduler()
    job = sched.create_job("t_one", backend="fake", extra={"speed": "300", "scenario": "no_events"})
    finish(sched, job["id"])
    state = sched.task_state(job["tasks"][0]["id"])
    assert state["status"] == "ok" and state["mode"] == "log" and state["sources"]["events"]["present"] is False
    assert [s["status"] for s in state["stages"]][:9] == ["ok"] * 9
    assert all(s["source"] in ("log", "file") for s in state["stages"][:9])
    assert state["relax"]["source"] == "log" and state["relax"]["n_points"] > 10


def test_cached_relax_scenario_reports_reuse(make_scheduler):
    sched = make_scheduler()
    job = sched.create_job("t_one", backend="fake", extra={"speed": "300", "scenario": "cached_relax"})
    finish(sched, job["id"])
    relax = next(s for s in sched.task_state(job["tasks"][0]["id"])["stages"] if s["id"] == "relax")
    assert relax["detail"] == "reused relaxed world" and relax["status"] == "ok"


def test_wind_only_profile_skips_the_first_seven_stages(make_scheduler, studio_config):
    studio_config.write_text(
        studio_config.read_text()
        + "\n[t_wind]\nbatch_name = tb_wind\nrotations = [2400]\nfluxon_count = [100]\nadapts = [0]\nflow_method = [parker]\nwind_only = 1\n"
    )
    sched = make_scheduler()
    job = sched.create_job("t_wind", backend="fake", extra=FAST)
    finish(sched, job["id"])
    state = sched.task_state(job["tasks"][0]["id"])
    assert [s["status"] for s in state["stages"]] == ["skipped"] * 7 + ["ok", "ok", "skipped"]


def test_fluxlight_profile_reports_its_artifact(make_scheduler):
    sched = make_scheduler()
    job = sched.create_job("t_light", backend="fake", extra=FAST)
    finish(sched, job["id"])
    state = sched.task_state(job["tasks"][0]["id"])
    assert state["stages"][-1]["status"] == "ok"
    assert any(a["source"] == "event" and "fluxlight" in a["path"] for a in state["artifacts"])


def test_events_dropout_degrades_then_log_takes_over(make_scheduler):
    sched = make_scheduler(stale_after=1.5)
    job = sched.create_job("t_one", backend="fake", extra={"speed": "6", "scenario": "events_dropout"})
    tid = job["tasks"][0]["id"]
    seen = {"degraded": False}

    def check():
        if sched.store.get_task(tid)["status"] == "running":
            seen["degraded"] |= bool(sched.task_state(tid)["degraded"])
        return sched.store.get_task(tid)["status"] not in ("pending", "running")

    wait_for(sched, check, 90)
    assert seen["degraded"], "the monitor never noticed that events stopped"
    final = sched.task_state(tid)
    assert final["status"] == "ok" and final["relax"]["n_points"] > 10
