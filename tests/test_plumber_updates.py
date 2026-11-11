"""Check for updates: behind/ahead against the upstream, fast-forward only, refusals, restart command."""

import subprocess
import sys

import pytest

from fluxpype.plumber import updates


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(origin), str(seed)], check=True, capture_output=True)
    for d in (seed,):
        git(d, "config", "user.email", "t@example.com")
        git(d, "config", "user.name", "T")
    (seed / "a.txt").write_text("1\n")
    git(seed, "add", ".")
    git(seed, "commit", "-qm", "first")
    git(seed, "push", "-q", "origin", "HEAD:main")
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True, capture_output=True)
    git(work, "config", "user.email", "t@example.com")
    git(work, "config", "user.name", "T")
    return seed, work


def push_change(seed, name="b.txt", msg="second"):
    (seed / name).write_text("x\n")
    git(seed, "add", ".")
    git(seed, "commit", "-qm", msg)
    git(seed, "push", "-q", "origin", "HEAD:main")


def test_up_to_date(repos):
    _, work = repos
    st = updates.Updater(str(work)).status(fetch=True, max_age=0)
    assert st["supported"] and st["behind"] == 0 and st["ahead"] == 0 and not st["can_update"]
    assert st["upstream"] == "origin/main" and st["reason"] == "already up to date"


def test_behind_then_fast_forward(repos):
    seed, work = repos
    push_change(seed, "pyproject.toml", "deps")
    u = updates.Updater(str(work))
    st = u.status(fetch=True, max_age=0)
    assert st["behind"] == 1 and st["can_update"] and st["commits"][0]["subject"] == "deps"
    assert st["deps_changed"] == ["pyproject.toml"]
    out = u.apply()
    assert out["to"] != out["from"] and (work / "pyproject.toml").exists()
    assert u.status(fetch=False, max_age=0)["behind"] == 0


def test_refuses_local_edits_and_local_commits(repos):
    seed, work = repos
    push_change(seed)
    (work / "a.txt").write_text("edited\n")
    u = updates.Updater(str(work))
    st = u.status(fetch=True, max_age=0)
    assert not st["can_update"] and "local edits" in st["reason"] and st["dirty"] == ["a.txt"]
    with pytest.raises(updates.UpdateError):
        u.apply()
    git(work, "commit", "-qam", "mine")
    st = u.status(fetch=True, max_age=0)
    assert st["ahead"] == 1 and st["behind"] == 1 and not st["can_update"]
    with pytest.raises(updates.UpdateError):
        u.apply()


def test_not_a_checkout_and_no_upstream(tmp_path, repos):
    assert updates.Updater(str(tmp_path)).status(fetch=False)["supported"] is False
    _, work = repos
    git(work, "switch", "-qc", "local-only")
    st = updates.Updater(str(work)).status(fetch=True, max_age=0)
    assert st["upstream"] is None and not st["can_update"]


def test_restart_argv_keeps_flags_but_not_token_rotation():
    argv = updates.restart_argv(["/x/cli.py", "--headless", "--port", "8765", "--rotate-token"])
    assert argv[:3] == [sys.executable, "-m", "fluxpype.plumber.cli"]
    assert argv[3:] == ["--headless", "--port", "8765"]
    assert "--no-browser" in updates.restart_argv(["cli", "--port", "1"])


def test_api_update(make_scheduler, repos):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fluxpype.plumber.server import create_app

    seed, work = repos
    push_change(seed)
    app = create_app(make_scheduler(), "tok", demo=True, updater=updates.Updater(str(work)), allow_restart=False)
    h = {"X-Flux-Token": "tok"}
    with TestClient(app, base_url="http://localhost") as c:
        st = c.get("/api/update", params={"fetch": 1}, headers=h).json()
        assert st["behind"] == 1 and st["can_update"]
        r = c.post("/api/update", json={"restart": True}, headers=h).json()
        assert r["restarting"] is False and r["commits"][0]["subject"] == "second"
        assert c.post("/api/update", json={}, headers=h).status_code == 409  # nothing left to update
        assert c.get("/api/status", headers=h).json()["commit"] == git(work, "rev-parse", "--short=10", "HEAD")
        assert c.post("/api/restart", headers=h).status_code == 409
        checks = {x["id"]: x for x in c.get("/api/health", headers=h).json()["system"]}
        assert checks["updates"]["status"] == "ok"
