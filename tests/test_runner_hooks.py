"""
Runner hooks: --only, --capabilities, FLUXPYPE_* env propagation, log file
override, per-task timeout kill, exit codes.

The pipeline itself cannot run here, so a shim ``perl`` (tests/perl_shim.py)
stands in for it. The science stack is replaced by tests/fake_science.py.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent

CONFIG = """\
[DEFAULT]
config_name = t
base_dir = auto
run_script = {script}
rc_path = ~/.zshrc
fl_mhdlib = ${{base_dir}}/
data_dir = {data}
nwant = 1
n_jobs = 1
batch_name = tb
rotations = [2225, 2226]
fluxon_count = [100]
adapts = [0]
flow_method = [parker, wsa]

[t]
"""


@pytest.fixture
def env(tmp_path):
    """A runnable sandbox: shim perl first on PATH, a config snapshot, fresh output files."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    perl = bindir / "perl"
    perl.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{TESTS / "perl_shim.py"}" "$@"\n')
    perl.chmod(0o755)
    script = tmp_path / "fake.pdl"
    script.write_text("# not a real pipeline\n")
    cfg = tmp_path / "config.snapshot.ini"
    cfg.write_text(CONFIG.format(script=script, data=tmp_path / "data"))
    e = {k: v for k, v in os.environ.items() if not k.startswith("FLUXPYPE_")}
    e.update(
        PATH=f"{bindir}{os.pathsep}{e['PATH']}",
        PYTHONPATH=str(REPO),
        FLUXPYPE_CONFIG_FILE=str(cfg),
        FLUXPYPE_LOG_FILE=str(tmp_path / "run.log"),
        FAKE_PERL_LOG=str(tmp_path / "invocations.jsonl"),
        FAKE_PERL_DUMP=str(tmp_path / "env.json"),
        FAKE_PERL_PIDFILE=str(tmp_path / "pids.json"),
        FAKE_PERL_REPO=str(REPO),
    )
    e["_tmp"] = str(tmp_path)  # convenience for tests; harmless to the child
    return e


def run_runner(env, *args, timeout=60):
    return subprocess.run(
        [sys.executable, str(TESTS / "_run_runner.py"), *args],
        env=env,
        cwd=env["_tmp"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def invocations(env):
    path = Path(env["FAKE_PERL_LOG"])
    if not path.exists():
        return []
    return [json.loads(line)["args"] for line in path.read_text().splitlines()]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        # a zombie still answers kill(0); treat it as gone
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def test_capabilities_lists_features(env):
    r = run_runner(env, "--capabilities")
    assert r.returncode == 0
    caps = json.loads(r.stdout)
    assert {"only", "events", "config_file", "log_file", "task_timeout"} <= set(caps["features"])
    assert "version" in caps
    assert invocations(env) == []


def test_full_profile_runs_every_task_in_order(env):
    r = run_runner(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert invocations(env) == [
        ["2225", "100", "0", "parker"],
        ["2225", "100", "0", "wsa"],
        ["2226", "100", "0", "parker"],
        ["2226", "100", "0", "wsa"],
    ]
    assert "All 4 jobs succeeded." in r.stdout


def test_only_runs_exactly_one_task(env):
    r = run_runner(env, "--only", "2226,100,0,wsa")
    assert r.returncode == 0, r.stdout + r.stderr
    assert invocations(env) == [["2226", "100", "0", "wsa"]]
    assert "All 1 jobs succeeded." in r.stdout


def test_only_is_case_insensitive_for_method(env):
    r = run_runner(env, "--only", "2225,100,0,PARKER")
    assert r.returncode == 0, r.stdout + r.stderr
    assert invocations(env) == [["2225", "100", "0", "parker"]]


@pytest.mark.parametrize("only", ["9999,100,0,wsa", "2225,100,0,ghosts", "2225,100", "a,b,c,d"])
def test_only_outside_the_product_is_rejected(env, only):
    r = run_runner(env, "--only", only)
    assert r.returncode == 2
    assert invocations(env) == []


def test_env_vars_reach_the_pipeline(env):
    env["FLUXPYPE_TASK_ID"] = "task-42"
    env["FLUXPYPE_EVENTS"] = str(Path(env["_tmp"]) / "events.jsonl")
    env["FLUXPYPE_TASK_TIMEOUT"] = "123"
    r = run_runner(env, "--only", "2225,100,0,parker")
    assert r.returncode == 0, r.stdout + r.stderr
    seen = json.loads(Path(env["FAKE_PERL_DUMP"]).read_text())
    for key in (
        "FLUXPYPE_CONFIG_FILE",
        "FLUXPYPE_EVENTS",
        "FLUXPYPE_TASK_ID",
        "FLUXPYPE_LOG_FILE",
        "FLUXPYPE_TASK_TIMEOUT",
    ):
        assert seen[key] == env[key]


def test_log_file_override_appends(env):
    log = Path(env["FLUXPYPE_LOG_FILE"])
    log.write_text("earlier run\n")
    r = run_runner(env, "--only", "2225,100,0,parker")
    assert r.returncode == 0, r.stdout + r.stderr
    text = log.read_text()
    assert text.startswith("earlier run\n")
    assert "fake pipeline for CR2225" in text
    assert "Program Complete for CR2225" in text


def test_runner_emits_run_start_and_end(env):
    events = Path(env["_tmp"]) / "events.jsonl"
    env["FLUXPYPE_EVENTS"] = str(events)
    env["FLUXPYPE_TASK_ID"] = "t-1"
    r = run_runner(env, "--only", "2225,100,0,parker")
    assert r.returncode == 0, r.stdout + r.stderr
    recs = [json.loads(line) for line in events.read_text().splitlines()]
    kinds = [x["type"] for x in recs]
    assert kinds[0] == "run.start" and kinds[-1] == "run.end"
    start = recs[0]
    assert (start["cr"], start["nflux"], start["adapt"], start["method"]) == ("2225", "100", "0", "parker")
    assert start["profile"] == "t"
    assert all(x["task"] == "t-1" and x["v"] == 1 for x in recs)
    assert recs[-1]["status"] == "ok" and recs[-1]["exit_code"] == 0
    # the shim's own (Perl-written) event lands between them
    assert "stage.start" in kinds


def test_no_events_file_without_the_env_var(env):
    r = run_runner(env, "--only", "2225,100,0,parker")
    assert r.returncode == 0
    assert not list(Path(env["_tmp"]).glob("events*.jsonl"))


def test_failure_exits_1(env):
    env["FAKE_PERL_MODES"] = "default=ok,2226=fail"
    events = Path(env["_tmp"]) / "events.jsonl"
    env["FLUXPYPE_EVENTS"] = str(events)
    r = run_runner(env)
    assert r.returncode == 1
    assert "FAILED 2 of 4 jobs" in r.stdout
    ends = [json.loads(x) for x in events.read_text().splitlines() if '"run.end"' in x]
    assert [e["status"] for e in ends] == ["ok", "ok", "failed", "failed"]
    assert ends[2]["exit_code"] == 3


def test_timeout_kills_the_whole_group_and_exits_124(env):
    env["FAKE_PERL_MODES"] = "default=hang"
    env["FLUXPYPE_TASK_TIMEOUT"] = "2"
    events = Path(env["_tmp"]) / "events.jsonl"
    env["FLUXPYPE_EVENTS"] = str(events)
    t0 = time.time()
    r = run_runner(env, "--only", "2225,100,0,parker")
    assert r.returncode == 124, r.stdout + r.stderr
    assert time.time() - t0 < 40
    pids = json.loads(Path(env["FAKE_PERL_PIDFILE"]).read_text())
    time.sleep(0.3)
    assert not pid_alive(pids["perl"])
    assert not pid_alive(pids["child"])
    end = [json.loads(x) for x in events.read_text().splitlines() if '"run.end"' in x][-1]
    assert end["status"] == "timeout" and end["exit_code"] == 124


def test_timeout_mixed_with_failure_exits_1(env):
    env["FAKE_PERL_MODES"] = "2225=hang,2226=fail"
    env["FLUXPYPE_TASK_TIMEOUT"] = "1"
    r = run_runner(env, "--only", "2225,100,0,parker")
    assert r.returncode == 124  # only a timeout in this selection
    r = run_runner(env)
    assert r.returncode == 1  # timeouts plus failures: plain failure


def test_sigterm_to_the_runner_kills_the_pipeline_group(env):
    env["FAKE_PERL_MODES"] = "default=hang"
    p = subprocess.Popen(
        [sys.executable, str(TESTS / "_run_runner.py"), "--only", "2225,100,0,parker"],
        env=env,
        cwd=env["_tmp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pidfile = Path(env["FAKE_PERL_PIDFILE"])
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip().endswith("}"):
            break
        time.sleep(0.1)
    pids = json.loads(pidfile.read_text())
    os.kill(p.pid, signal.SIGTERM)
    p.wait(timeout=30)
    time.sleep(0.3)
    assert not pid_alive(pids["perl"])
    assert not pid_alive(pids["child"])
