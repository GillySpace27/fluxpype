"""The data folder real runs use: chosen in Plumber, pinned into job snapshots, discovered on disk."""

import os

import pytest

from fluxpype.plumber import config_io, datadirs
from fluxpype.plumber.backends.fake import FakeBackend
from fluxpype.plumber.scheduler import SchedulerError


def real_backend():
    b = FakeBackend()
    b.kind = "local"  # behaves like a real backend for data_dir purposes: not sandboxed
    return b


def fill(folder, mags=2, pkls=1, batches=("quickstart",)):
    os.makedirs(os.path.join(folder, "magnetograms"), exist_ok=True)
    os.makedirs(os.path.join(folder, "pfss"), exist_ok=True)
    for i in range(mags):
        open(os.path.join(folder, "magnetograms", f"CR22{i:02d}_r4_hmi.fits"), "w").close()
    for i in range(pkls):
        open(os.path.join(folder, "pfss", f"pfss_cr22{i:02d}_r4_hmi.pkl"), "w").close()
    for b in batches:
        os.makedirs(os.path.join(folder, "batches", b), exist_ok=True)


def test_default_is_config_data_dir_and_a_chosen_folder_wins(make_scheduler, plumber_config, tmp_path):
    sched = make_scheduler(backends={"local": real_backend()})
    cfg_dir = str(tmp_path / "data")
    assert sched.data_dir() == cfg_dir
    other = tmp_path / "elsewhere"
    other.mkdir()
    sched.set_data_dir(str(other))
    assert sched.data_dir() == str(other)
    assert sched.data_dir("t_one") == str(other)
    rep = sched.data_dir_report()
    assert rep["source"] == "plumber" and rep["config_value"] == cfg_dir
    sched.set_data_dir(None)
    assert sched.data_dir() == cfg_dir and sched.data_dir_report()["source"] == "config"


def test_chosen_folder_is_pinned_into_snapshots_and_probes(make_scheduler, tmp_path):
    sched = make_scheduler(backends={"local": real_backend()})
    other = tmp_path / "elsewhere"
    other.mkdir()
    sched.set_data_dir(str(other))
    sched.set_queue(paused=True)
    job = sched.create_job("t_one")
    snap = config_io.read_file(sched.store.get_job(job["id"])["snapshot_path"]).effective("t_one")
    assert config_io.resolved_path(snap, "data_dir", "/x") == str(other)
    ctx = sched.probe_context_for("t_one", "2225", "100", "0")
    assert ctx.data_dir == str(other)


def test_sandbox_ignores_the_chosen_folder(make_scheduler, tmp_path):
    sched = make_scheduler(backends={"fake": FakeBackend(), "local": real_backend()})
    other = tmp_path / "elsewhere"
    other.mkdir()
    sched.set_data_dir(str(other))
    assert sched.data_dir(backend="fake") == sched.sandbox_dir()
    assert sched.data_dir(backend="local") == str(other)


def test_set_data_dir_refuses_missing_folders_and_the_sandbox(make_scheduler, tmp_path):
    sched = make_scheduler(backends={"local": real_backend()})
    with pytest.raises(SchedulerError):
        sched.set_data_dir(str(tmp_path / "nope"))
    with pytest.raises(SchedulerError):
        sched.set_data_dir(sched.sandbox_dir())


def test_candidates_find_sibling_checkouts_and_merge_symlinks(tmp_path):
    root = tmp_path / "vscode"
    a = root / "fluxons" / "fluxpype" / "fluxpype" / "data"
    fill(str(a), mags=3)
    b = root / "fluxons-run" / "fluxpype" / "fluxpype"
    b.mkdir(parents=True)
    os.symlink(a, b / "data")
    empty = root / "fluxons-plumber" / "fluxpype" / "fluxpype" / "data"
    empty.mkdir(parents=True)
    found = datadirs.candidates(str(root / "fluxons-plumber"))
    assert [c["path"] for c in found] == [str(a), str(empty)]
    assert found[0]["has_data"] and found[0]["magnetograms"] == 3 and found[0]["batches"] == 1
    assert found[0]["aliases"] == [str(b / "data")]
    assert not found[1]["has_data"]


def test_api_data_dir_round_trip(make_scheduler, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fluxpype.plumber.server import create_app

    sched = make_scheduler(backends={"local": real_backend()})
    app = create_app(sched, "tok", demo=False, tick_interval=0.1)
    h = {"X-Flux-Token": "tok"}
    other = tmp_path / "elsewhere"
    fill(str(other))
    with TestClient(app, base_url="http://localhost") as c:
        assert c.get("/api/data-dir", headers=h).json()["source"] == "config"
        r = c.put("/api/data-dir", json={"path": str(other)}, headers=h)
        assert r.status_code == 200 and r.json()["current"]["path"] == str(other)
        assert r.json()["current"]["magnetograms"] == 2
        assert c.get("/api/cache", headers=h).json()["data_dir"] == str(other)
        assert c.put("/api/data-dir", json={"path": str(tmp_path / "nope")}, headers=h).status_code == 400
        assert c.put("/api/data-dir", json={"path": None}, headers=h).json()["source"] == "config"
        scopes = {p["name"]: p for p in c.get("/api/cache", headers=h).json()["presets"]}
        assert "t_one" in scopes and scopes["t_one"]["crs"]
