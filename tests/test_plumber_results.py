"""Results of one run: products by stage, made-here vs reused, job settings, job-scoped file access."""

import os
import time

import pytest

from fluxpype.plumber.monitor import ProbeContext
from fluxpype.plumber.results import for_this_nflux, made_here, task_results


def touch(path, mtime=None, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_nflux_filter_only_rejects_other_counts():
    assert for_this_nflux("cr2225_f500_hmi_relaxed.png", "500")
    assert not for_this_nflux("cr2225_f1000_hmi_relaxed.png", "500")
    assert for_this_nflux("cr2225_fluxlight_light_fluxel_301_100_1.0.png", "500")  # no _f<N>_ token: keep
    assert for_this_nflux("distances.png", "500")


def test_made_here():
    assert made_here(100.0, None, None) is None
    assert made_here(50.0, 100.0, 200.0) is False
    assert made_here(150.0, 100.0, 200.0) is True
    assert made_here(98.0, 100.0, 200.0) is True  # clock slack
    assert made_here(1000.0, 100.0, 200.0) is None  # touched long after the run


def test_task_results_groups_products_and_tags_reuse(tmp_path):
    d = str(tmp_path)
    b = os.path.join(d, "batches", "B", "data", "cr2225")
    t0 = time.time()
    old = t0 - 3600
    touch(os.path.join(d, "magnetograms", "CR2225_r4_hmi.fits"), old)
    touch(os.path.join(b, "floc", "floc_cr2225_r4_f500_hmi.dat"), old)
    touch(os.path.join(b, "world", "cr2225_f500_hmi.flux"), t0 + 10)
    touch(os.path.join(b, "world", "cr2225_f500_hmi_relaxed_s840.flux"), t0 + 20)
    touch(os.path.join(b, "world", "cr2225_f1000_hmi_relaxed_s840.flux"), t0 + 20)  # other task
    touch(os.path.join(b, "world", "cr2225_f500_hmi_relaxed.png"), t0 + 30)
    touch(os.path.join(b, "world", "cr2225_f1000_hmi_relaxed.png"), t0 + 30)  # other task
    touch(os.path.join(b, "light", "cr2225_fluxlight_light_x_1.png"), t0 + 40)
    ctx = ProbeContext(data_dir=d, batch="B", cr="2225", nflux="500", adapt="0", inst="hmi")
    r = task_results(ctx, t0, t0 + 60)
    by = {p["id"]: p for p in r["products"]}
    assert [f["made_here"] for f in by["magnetogram"]["files"]] == [False]
    assert [f["name"] for f in by["relaxed_world"]["files"]] == ["cr2225_f500_hmi_relaxed_s840.flux"]
    assert by["relaxed_world"]["files"][0]["made_here"] is True
    assert by["pfss"]["files"] == []  # expected but missing
    plots = [f["name"] for f in r["plots"]]
    assert "cr2225_f500_hmi_relaxed.png" in plots and "cr2225_f1000_hmi_relaxed.png" not in plots
    assert "cr2225_fluxlight_light_x_1.png" not in plots  # listed once, under FLUXlight
    assert r["headline"] == "batches/B/data/cr2225/light/cr2225_fluxlight_light_x_1.png"
    assert all(not f["path"].startswith("/") for p in r["products"] for f in p["files"])


def test_task_results_without_context():
    assert task_results(None, None, None)["products"] == []


# ------------------------------------------------------------------ API

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from fluxpype.plumber.server import create_app  # noqa: E402

H = {"X-Flux-Token": "tok"}


@pytest.fixture
def client(make_scheduler):
    sched = make_scheduler()
    app = create_app(sched, "tok", demo=True, tick_interval=0.1)
    with TestClient(app, base_url="http://localhost") as c:
        c.sched = sched
        yield c


def finished_job(client, overrides=None):
    body = {"profile": "t_one", "backend": "fake", "speed": 300, "overrides": overrides or {}}
    job = client.post("/api/jobs", json=body, headers=H).json()
    end = time.time() + 60
    while time.time() < end:
        j = client.get(f"/api/jobs/{job['id']}", headers=H).json()
        if j["status"] in ("ok", "failed", "partial", "cancelled"):
            return j
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_job_config_returns_the_snapshot_and_marks_overrides(client):
    job = finished_job(client, {"relax_threshold": "1.7"})
    r = client.get(f"/api/jobs/{job['id']}/config", headers=H)
    assert r.status_code == 200, r.text
    cfg = r.json()
    e = {x["key"]: x for x in cfg["entries"]}
    assert cfg["profile"] == "t_one" and cfg["overrides"] == {"relax_threshold": "1.7"}
    assert e["relax_threshold"]["value"] == "1.7" and e["relax_threshold"]["overridden"]
    assert e["relax_threshold"]["differs_from_preset"]
    assert cfg["data_dir"] == client.sched.sandbox_dir()  # the simulated run was pinned to the sandbox
    assert client.get("/api/jobs/nope/config", headers=H).status_code == 404


def test_task_results_and_job_files(client):
    job = finished_job(client)
    tid = job["tasks"][0]["id"]
    r = client.get(f"/api/tasks/{tid}/results", headers=H).json()
    ids = [p["id"] for p in r["products"]]
    assert ids[:2] == ["magnetogram", "footpoints"] and "fluxlight" in ids
    made = [f for p in r["products"] for f in p["files"] if f["made_here"]]
    assert made, "the simulated run wrote files"
    files = client.get(f"/api/jobs/{job['id']}/files", headers=H).json()
    assert files["data_dir"] == r["data_dir"]
    assert all(tid in f["tasks"] for f in files["files"])
    assert files["is_current_data_dir"] is True  # sandboxed backend is active, so the sandbox is current


def test_job_scoped_preview_is_confined_to_the_job_data_dir(client, tmp_path):
    job = finished_job(client)
    tid = job["tasks"][0]["id"]
    r = client.get(f"/api/tasks/{tid}/results", headers=H).json()
    img = next(f for f in r["plots"] + [g for p in r["products"] for g in p["files"]] if f["name"].endswith(".png"))
    base = f"/api/jobs/{job['id']}/data"
    assert client.get(f"{base}/preview", params={"path": img["path"]}, headers=H).json()["view"] == "image"
    raw = client.get(f"{base}/file", params={"path": img["path"]}, headers=H)
    assert raw.status_code == 200 and raw.headers["content-type"] == "image/png"
    (tmp_path / "secret.txt").write_text("no")
    for bad in ("../secret.txt", "/etc/passwd", "a/../../secret.txt"):
        assert client.get(f"{base}/preview", params={"path": bad}, headers=H).status_code == 403
        assert client.get(f"{base}/file", params={"path": bad}, headers=H).status_code == 403
    assert client.get(f"{base}/render", params={"path": img["path"]}, headers=H).status_code == 415
    assert client.get(f"{base}/file", params={"path": img["path"]}).status_code == 401
