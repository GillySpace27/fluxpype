"""flux-plumb process behavior: single-instance lock, server.json, demo with --home, headless, --stop."""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fluxpype.plumber import cli, demo, paths  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn(home, config, *extra, port=None):
    argv = [sys.executable, "-m", "fluxpype.plumber.cli", "--home", str(home), "--config", str(config)]
    argv += ["--port", str(port or free_port()), *extra]
    return subprocess.Popen(argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_server_json(home, proc, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        info = cli.read_server_json(home)
        if info:
            return info
        if proc.poll() is not None:
            raise AssertionError(f"plumber exited early: {proc.communicate()}")
        time.sleep(0.1)
    raise AssertionError("server.json never appeared")


@pytest.fixture
def running(plumber_home, plumber_config):
    procs = []

    def start(*extra, **kw):
        proc = spawn(plumber_home, plumber_config, "--headless", *extra, **kw)
        procs.append(proc)
        return proc, wait_server_json(plumber_home, proc)

    yield start
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()


def test_server_json_token_and_headless_log(running, plumber_home):
    proc, info = running()
    assert set(info) == {"host", "port", "pid", "pid_start", "version", "home", "started", "demo"}
    assert info["pid"] == proc.pid and info["home"] == str(plumber_home) and info["demo"] is False
    assert info["host"] == "127.0.0.1" and isinstance(info["port"], int) and info["pid_start"]
    for name in ("server.json", "token"):
        assert (plumber_home / name).stat().st_mode & 0o777 == 0o600
    assert (plumber_home / "server.log").exists()
    assert not list(plumber_home.glob("*.tmp*"))  # atomic writes leave no temp files behind


def test_second_start_exits_3_and_leaves_the_token_alone(running, plumber_home, plumber_config):
    proc, info = running()
    token = paths.token_path(plumber_home).read_text()
    second = spawn(plumber_home, plumber_config, "--no-browser")
    out, err = second.communicate(timeout=60)
    assert second.returncode == 3, (out, err)
    assert f"another Plumber is already running for {plumber_home}" in err and str(info["pid"]) in err
    assert out == ""
    assert paths.token_path(plumber_home).read_text() == token
    assert cli.read_server_json(plumber_home) == info
    assert proc.poll() is None


def test_lock_probe_and_in_process_contention(plumber_home):
    assert not cli.lock_is_held(plumber_home)
    assert cli.acquire_lock(plumber_home)
    try:
        assert cli.lock_is_held(plumber_home)
        # a second open file description conflicts even inside one process, so main() refuses to start
        before = paths.token_path(plumber_home).exists()
        assert cli.main(["--demo", "--home", str(plumber_home), "--headless", "--port", str(free_port())]) == 3
        assert paths.token_path(plumber_home).exists() == before
        assert not (plumber_home / "config.ini").exists()  # the demo did not rewrite the held home
    finally:
        os.close(cli._LOCK_FD)
        cli._LOCK_FD = None
    assert not cli.lock_is_held(plumber_home)


def test_stop_ends_the_server_and_removes_server_json(running, plumber_home):
    proc, info = running()
    assert cli.main(["--stop", "--home", str(plumber_home)]) == 0
    assert proc.wait(timeout=15) == 0
    assert cli.read_server_json(plumber_home) is None
    assert not cli.lock_is_held(plumber_home)
    assert cli.main(["--stop", "--home", str(plumber_home)]) == 1  # nothing left to stop


def test_stop_ignores_a_stale_server_json(plumber_home):
    stale = {"pid": 2**22 + 12345, "pid_start": "1", "port": 1, "host": "127.0.0.1", "home": str(plumber_home)}
    (plumber_home / "server.json").write_text(json.dumps(stale))
    assert cli.main(["--stop", "--home", str(plumber_home)]) == 1


def test_demo_with_home_seeds_that_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("FLUX_PLUMBER_HOME", raising=False)
    home = tmp_path / "chosen"
    assert demo.prepare_home(home) == home
    text = (home / "config.ini").read_text()
    assert "[Demo_Sweep]" in text and str(home / "data") in text and (home / "data").is_dir()
    assert os.environ["FLUX_PLUMBER_HOME"] == str(home)
    random_home = demo.prepare_home()  # unchanged behavior without a home
    assert random_home.name.startswith("flux-plumb-demo-") and random_home != home


def test_python_flag_sets_the_job_interpreter(running, plumber_home):
    import urllib.request

    proc, info = running("--python", sys.executable)
    token = (plumber_home / "token").read_text().strip()
    req = urllib.request.Request(f"http://127.0.0.1:{info['port']}/api/status", headers={"X-Flux-Token": token})
    with urllib.request.urlopen(req, timeout=10) as r:
        st = json.load(r)
    local = [b for b in st["backends"] if b["name"] == "local"][0]
    assert sys.executable in local["description"]


def test_python_flag_rejects_a_missing_interpreter(plumber_home, plumber_config, tmp_path):
    rc = cli.main(["--home", str(plumber_home), "--config", str(plumber_config), "--python", str(tmp_path / "nope")])
    assert rc == 2
    assert not (plumber_home / "server.json").exists()
