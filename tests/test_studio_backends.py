"""Launcher helpers and the LocalBackend command line (the real runner is probed, not run)."""

import os
import subprocess
import sys
import time

from fluxpype.studio import launcher
from fluxpype.studio.backends.base import TaskSpec
from fluxpype.studio.backends.fake import FakeBackend
from fluxpype.studio.backends.local import LocalBackend


def spec(tmp_path, **kw):
    return TaskSpec(
        task_id="j1-01",
        job_id="j1",
        run_dir=str(tmp_path / "r"),
        snapshot_path=str(tmp_path / "snap.ini"),
        profile="quickstart",
        cr="2225",
        nflux="100",
        adapt="0",
        method="parker",
        timeout_s=77,
        **kw,
    )


def test_start_token_identifies_a_process_instance():
    token = launcher.process_start_token(os.getpid())
    assert token and token == launcher.process_start_token(os.getpid())
    assert launcher.process_alive(os.getpid(), token)
    assert not launcher.process_alive(os.getpid(), token + "x")  # same pid, different start time: a reused pid
    assert not launcher.process_alive(None) and not launcher.process_alive(0)


def test_exited_and_zombie_processes_are_not_alive():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    token = launcher.process_start_token(p.pid)
    time.sleep(0.5)  # exited, not yet reaped: a zombie
    assert not launcher.process_alive(p.pid, token)
    p.wait()
    assert not launcher.process_alive(p.pid)


def test_local_command_and_environment(tmp_path):
    b = LocalBackend()
    t = spec(tmp_path)
    cmd = b.command(t)
    assert cmd[1:] == ["-m", "fluxpype.config_runner", "--config", "quickstart", "--only", "2225,100,0,parker"]
    env = b.environment(t)
    assert env["FLUXPYPE_CONFIG_FILE"] == t.snapshot_path
    assert env["FLUXPYPE_EVENTS"].endswith("events.jsonl") and env["FLUXPYPE_LOG_FILE"].endswith("run.log")
    assert env["FLUXPYPE_TASK_ID"] == "j1-01" and env["FLUXPYPE_TASK_TIMEOUT"] == "77"
    assert b.capabilities().kind == "local" and b.capabilities().slots >= 1


def test_local_backend_probes_the_real_runner():
    features = LocalBackend().probe()
    assert {"only", "events", "config_file", "log_file", "task_timeout"} <= set(features)


def test_a_missing_python_probes_as_old_install():
    assert LocalBackend(python="/nonexistent/python").probe() == []


def test_read_file_is_confined_to_the_run_dir(tmp_path):
    from fluxpype.studio.backends.base import Handle

    b = LocalBackend()
    run = tmp_path / "r"
    run.mkdir()
    (run / "events.jsonl").write_text("abcdef")
    h = Handle("t", str(run))
    assert b.read_file(h, "events.jsonl", 2) == b"cdef"
    assert b.read_file(h, "missing") == b""
    for bad in ("../x", "a/b", ".hidden"):
        try:
            b.read_file(h, bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_fake_backend_rejects_unknown_scenarios(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        FakeBackend().command(spec(tmp_path, extra={"scenario": "nope"}))
