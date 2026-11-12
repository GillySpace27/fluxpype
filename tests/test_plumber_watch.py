"""`python -m fluxpype.plumber.watch` against a real headless Plumber process (fake backend, temp home)."""

import json
import re
import signal
import subprocess
import sys
import time
import urllib.request

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from tests.test_plumber_cli import REPO, free_port, spawn, wait_server_json  # noqa: E402

LINE = re.compile(r"^flux (state|done|seq=\d+) [ -~]*$")


class Plumber:
    def __init__(self, home, config):
        self.home, self.config, self.proc, self.info = home, config, None, None

    def start(self, port=None):
        self.proc = spawn(self.home, self.config, "--headless", port=port)
        self.info = wait_server_json(self.home, self.proc)
        end = time.time() + 15
        while time.time() < end:  # server.json is written before uvicorn listens
            try:
                self.get("/api/status")
                return self
            except OSError:
                time.sleep(0.1)
        raise AssertionError("Plumber never answered")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.communicate(timeout=20)

    @property
    def token(self):
        return (self.home / "token").read_text().strip()

    def call(self, method, path, body=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.info['port']}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
            headers={"X-Flux-Token": self.token, "Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=10) as r:
            return json.loads(r.read())

    def get(self, path):
        return self.call("GET", path)

    def submit(self, scenario="ok", speed=300):
        return self.call(
            "POST", "/api/jobs", {"profile": "t_one", "backend": "fake", "scenario": scenario, "speed": speed}
        )


@pytest.fixture
def plumber(plumber_home, plumber_config):
    p = Plumber(plumber_home, plumber_config).start()
    yield p
    p.stop()


def watch(home, *args, timeout=90):
    argv = [sys.executable, "-m", "fluxpype.plumber.watch", "--home", str(home), *args]
    return subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=timeout)


def check(out, token):
    lines = out.stdout.splitlines()
    assert all(LINE.match(ln) for ln in lines), lines
    assert token not in out.stdout and token not in out.stderr
    seqs = [int(m) for ln in lines for m in re.findall(r"^flux seq=(\d+)", ln)]
    assert len(seqs) == len(set(seqs)), "a seq was printed twice"
    return lines


def test_ok_job_prints_state_then_outcomes_and_exits_0(plumber):
    job = plumber.submit("ok")
    out = watch(plumber.home, "--job", job["id"])
    lines = check(out, plumber.token)
    assert out.returncode == 0, out.stderr
    assert lines[0].startswith(f"flux state job={job['id']} status=")
    kinds = [re.search(r"kind=(\w+)", ln).group(1) for ln in lines if " kind=" in ln]
    if "status=ok" not in lines[0]:
        assert kinds[-2:] == ["task_ok", "job_ok"]
    assert lines[-1].startswith("flux done exit=0 ok=1 other=0") and lines[-1].endswith(f"next=get_job:{job['id']}")


def test_failing_job_exits_1_and_points_at_the_task(plumber):
    job = plumber.submit("fail_in_pfss", speed=20)
    out = watch(plumber.home, "--job", job["id"])
    lines = check(out, plumber.token)
    assert out.returncode == 1
    tid = job["tasks"][0]["id"]
    assert any(f"kind=task_failed sev=error job={job['id']} task={tid}" in ln and "stage=pfss" in ln for ln in lines)
    assert any(ln.endswith(f"next=get_task:{tid}") for ln in lines)
    assert lines[-1].startswith("flux done exit=1")


def test_finished_job_prints_only_state_and_unknown_job_exits_3(plumber):
    job = plumber.submit("ok")
    end = time.time() + 60
    while plumber.get(f"/api/jobs/{job['id']}")["status"] != "ok":
        assert time.time() < end
        time.sleep(0.1)
    t0 = time.time()
    out = watch(plumber.home, "--job", job["id"])
    lines = check(out, plumber.token)
    assert out.returncode == 0 and time.time() - t0 < 10
    assert lines[0].startswith(f"flux state job={job['id']} status=ok") and lines[1].startswith("flux done exit=0")
    assert len(lines) == 2
    quiet = watch(plumber.home, "--job", job["id"], "--quiet")
    assert quiet.stdout.splitlines() == [lines[1]]
    assert watch(plumber.home, "--job", "jdeadbeef").returncode == 3
    assert watch(plumber.home, "--job", "../../etc").returncode == 3
    assert watch(plumber.home).returncode == 2


def test_watch_survives_a_plumber_restart_on_another_port(plumber):
    job = plumber.submit("ok", speed=8)  # long enough to restart Plumber mid-run
    argv = [sys.executable, "-m", "fluxpype.plumber.watch", "--home", str(plumber.home), "--job", job["id"]]
    proc = subprocess.Popen(argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        first = proc.stdout.readline()
        assert first.startswith(f"flux state job={job['id']}")
        old_port = plumber.info["port"]
        plumber.stop()
        time.sleep(1)
        plumber.start(port=free_port())
        assert plumber.info["port"] != old_port
        rest, err = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
    out = subprocess.CompletedProcess(argv, proc.returncode, first + rest, err)
    lines = check(out, plumber.token)
    assert proc.returncode == 0, (lines, err)
    assert any("kind=job_ok" in ln for ln in lines) and lines[-1].startswith("flux done exit=0")
