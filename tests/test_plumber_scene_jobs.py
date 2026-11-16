"""Scene relaxations as queue jobs (fake backend: a stand-in curve replaces FLUX)."""

import json
import os
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fluxpype.plumber import ask, scene_jobs  # noqa: E402
from fluxpype.plumber.server import create_app  # noqa: E402

TOKEN = "t0k3n-for-tests"
H = {"X-Flux-Token": TOKEN}
ASK = {"X-Flux-Token": TOKEN, "X-Flux-Client": "ask"}


@pytest.fixture
def client(make_scheduler):
    with TestClient(
        create_app(make_scheduler(), TOKEN, demo=True, tick_interval=0.1), base_url="http://localhost"
    ) as c:
        for name, pairing in (("k0", [0, 1, 2, 3]), ("k2", [2, 3, 0, 1])):
            r = c.post(
                "/api/scenes",
                json={"spec": {"name": name, "n_lines": 4, "pairing": pairing, "lid_height": 4}},
                headers=H,
            )
            assert r.status_code == 201, r.text
        yield c


def wait_job(c, jid, secs=60):
    end = time.time() + secs
    while time.time() < end:
        v = c.get(f"/api/jobs/{jid}", headers=H).json()
        if v["status"] in ("ok", "failed", "cancelled", "partial"):
            return v
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def data_dir(c):
    return c.app.state.scheduler.data_dir()


def test_a_lid_sweep_is_one_job_with_a_task_per_lid_and_saves_the_variants(client):
    body = {"mode": "compare", "a": "k0", "b": "k2", "lids": [4, 8, 12.5], "cycles": 45, "steps": 10, "speed": 50}
    r = client.post("/api/scene-jobs", json=body, headers=H)
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["profile"] == "scene-compare" and job["extra"]["kind"] == "scene" and len(job["tasks"]) == 3
    t = job["tasks"]
    assert [x["cr"] for x in t] == [
        "k0 vs k2",
        "k0-lid8 vs k2-lid8",
        "k0-lid12p5 vs k2-lid12p5",
    ]  # lid 4 is the scene's own
    assert {(x["adapt"], x["method"], x["slots"], x["nflux"]) for x in t} == {("scene", "compare", 2, "4")}
    saved = {s["id"]: s["lid_height"] for s in client.get("/api/scenes", headers=H).json()["scenes"]}
    assert saved["k0-lid8"] == 8 and saved["k2-lid12p5"] == 12.5
    assert os.path.exists(os.path.join(data_dir(client), "scenes", "k0-lid8.flux"))
    done = wait_job(client, job["id"])
    assert done["status"] == "ok", [x["status"] for x in done["tasks"]]
    # the job page's own views work on a scene task
    tid = done["tasks"][1]["id"]
    state = client.get(f"/api/tasks/{tid}", headers=H).json()
    assert (
        [s["id"] for s in state["stages"]] == ["relax"]
        and state["stages"][0]["status"] == "ok"
        and state["scene"]["mode"] == "compare"
    )
    series = client.get(f"/api/tasks/{tid}/relax", headers=H).json()
    assert series["n"] == 45 and series["cycles"][-1]["cycle"] == 44
    assert "CYCLE A" in client.get(f"/api/tasks/{tid}/log", headers=H).json()["text"]
    res = client.get(f"/api/tasks/{tid}/scene-result?points=5", headers=H).json()
    assert res["done"] and res["lid"] == 8 and [w["scene"] for w in res["worlds"]] == ["k0-lid8", "k2-lid8"]
    assert len(res["D"]["series"]) <= 5 and res["D"]["final"] == pytest.approx(
        res["worlds"][1]["final_e_simple2"] - res["worlds"][0]["final_e_simple2"]
    )
    assert res["worlds"][0]["energy_settled_at_cycle"] is not None
    table = client.get(f"/api/jobs/{done['id']}/scene-results", headers=H).json()
    assert [r["lid"] for r in table["tasks"]] == [4, 8, 12.5] and all(r["D_final"] is not None for r in table["tasks"])
    assert client.get("/api/tasks/nope/scene-result", headers=H).status_code == 404
    pipeline = client.post("/api/jobs", json={"profile": "t_one", "backend": "fake", "speed": 400}, headers=H).json()
    assert client.get(f"/api/tasks/{pipeline['tasks'][0]['id']}/scene-result", headers=H).status_code == 404
    again = client.post(
        "/api/scene-jobs", json=body, headers=H
    )  # the variants exist now and are reused, not overwritten
    assert again.status_code == 201


def test_a_run_relaxes_one_scene_and_a_bad_request_says_why(client):
    r = client.post(
        "/api/scene-jobs", json={"mode": "run", "scene": "k0", "cycles": 12, "steps": 5, "speed": 50}, headers=H
    )
    job = wait_job(client, r.json()["id"])
    assert job["status"] == "ok" and job["tasks"][0]["slots"] == 1 and job["tasks"][0]["method"] == "run"
    res = client.get(f"/api/tasks/{job['tasks'][0]['id']}/scene-result", headers=H).json()
    assert len(res["worlds"]) == 1 and "D" not in res and res["worlds"][0]["cycles_done"] == 12
    cases = [
        ({"mode": "run", "scene": "nope"}, "unknown scene"),
        ({"mode": "run"}, "scene is required"),
        ({"mode": "run", "scene": "k0", "a": "k2"}, "run takes"),
        ({"mode": "compare", "a": "k0", "b": "k0"}, "two different"),
        ({"mode": "run", "scene": "k0", "cycles": 0}, "cycles"),
        ({"mode": "run", "scene": "k0", "steps": 99999}, "steps"),
        ({"mode": "run", "scene": "k0", "lids": [1.0]}, "lid 1"),  # below max_height x 1.2
        ({"mode": "run", "scene": "k0", "lids": list(range(10, 30))}, "1 to 8 heights"),
        ({"mode": "run", "scene": "k0", "lids": [-3]}, "positive"),
        ({"mode": "run", "scene": "k0", "speed": 5}, "fake backend"),
    ]
    for body, why in cases[:-1]:
        r = client.post("/api/scene-jobs", json=body, headers=H)
        assert r.status_code == 400 and why in r.json()["detail"], (body, r.text)
    assert client.post("/api/scene-jobs", json={"mode": "run", "scene": "k0", "bogus": 1}, headers=H).status_code == 422


def test_an_unfair_pair_is_refused(client):
    client.post("/api/scenes", json={"spec": {"name": "five", "n_lines": 5, "lid_height": 4}}, headers=H)
    r = client.post("/api/scene-jobs", json={"mode": "compare", "a": "k0", "b": "five"}, headers=H)
    assert r.status_code == 400 and "not a fair pair" in r.json()["detail"]


def test_scene_tasks_share_the_cpu_budget_and_run_slots_with_other_tasks(client):
    sched = client.app.state.scheduler
    sched.set_queue(max_parallel=4, cpu_budget=2)
    slow = {"mode": "compare", "a": "k0", "b": "k2", "cycles": 40, "steps": 5, "speed": 0.1}  # about 12 s
    first = client.post("/api/scene-jobs", json=slow, headers=H).json()
    second = client.post("/api/scene-jobs", json=slow, headers=H).json()
    pipeline = client.post("/api/jobs", json={"profile": "t_one", "backend": "fake", "speed": 400}, headers=H).json()
    why = sched.queue_explain()
    assert why[second["tasks"][0]["id"]] == "cpu_budget" and why[pipeline["tasks"][0]["id"]] == "cpu_budget"
    assert (
        sched.queue_state()["slots_used"] == 2 and sched.queue_state()["running"] == 1
    )  # the compare holds both slots
    for j in (first, second, pipeline):
        client.post(f"/api/jobs/{j['id']}/cancel", headers=H)
    end = time.time() + 20
    while time.time() < end and client.get(f"/api/jobs/{first['id']}", headers=H).json()["status"] != "cancelled":
        time.sleep(0.1)
    assert client.get(f"/api/jobs/{first['id']}", headers=H).json()["tasks"][0]["status"] == "cancelled"


def test_a_failed_scene_task_can_be_retried_in_place(client, monkeypatch):
    sched = client.app.state.scheduler
    r = client.post(
        "/api/scene-jobs", json={"mode": "run", "scene": "k0", "cycles": 5, "steps": 5, "speed": 50}, headers=H
    ).json()
    done = wait_job(client, r["id"])
    t = done["tasks"][0]
    sched.store.update_task(
        t["id"], status="failed"
    )  # as if it had failed: the inputs are in the job dir, not the run dir
    sched.store.update_job(r["id"], status="failed")
    assert client.post(f"/api/jobs/{r['id']}/retry", headers=H).status_code == 200
    again = wait_job(client, r["id"])
    assert again["tasks"][0]["status"] == "ok" and again["tasks"][0]["attempt"] == 2


def test_relax_scene_is_a_codesign_proposal_checked_with_the_route_code(client, monkeypatch):
    monkeypatch.setattr(ask, "_question", "q1")
    monkeypatch.setattr(ask, "_mode", "explain")
    body = {
        "tool": "relax_scene",
        "args": {"mode": "compare", "a": "k0", "b": "k2", "lids": [8, 16], "cycles": 10, "steps": 5},
        "question": "q1",
    }
    r = client.post("/api/proposals", json=body, headers=ASK)
    assert r.status_code == 403 and "Co-design" in r.json()["detail"]
    monkeypatch.setattr(ask, "_mode", "codesign")
    bad = client.post("/api/proposals", json={**body, "args": {**body["args"], "b": "nope"}}, headers=ASK)
    assert bad.status_code == 400 and "unknown scene" in bad.json()["detail"]
    r = client.post("/api/proposals", json=body, headers=ASK)
    assert r.status_code == 201, r.text
    p = r.json()
    assert p["args"]["backend"] == "fake" and "2 task(s)" in p["summary"] and "4 lid variant" in p["summary"]
    assert [d["to"] for d in p["diff"]] == ["lid 8", "lid 16"]
    assert not any(
        s["id"].endswith("lid8") for s in client.get("/api/scenes", headers=H).json()["scenes"]
    )  # proposing saved nothing
    jobs_before = len(client.get("/api/jobs", headers=H).json()["jobs"])
    a = client.post(f"/api/proposals/{p['id']}/approve", headers=H)
    assert a.status_code == 200, a.text
    jid = a.json()["result"]["job_id"]
    job = client.get(f"/api/jobs/{jid}", headers=H).json()
    assert len(job["tasks"]) == 2 and job["origin"]["kind"] == "ask" and job["origin"]["proposal"] == p["id"]
    assert len(client.get("/api/jobs", headers=H).json()["jobs"]) == jobs_before + 1
    assert any(s["id"] == "k0-lid16" for s in client.get("/api/scenes", headers=H).json()["scenes"])
    wait_job(client, jid)


def test_plan_names_lid_variants_and_keeps_an_existing_scene_that_differs(client):
    dd = data_dir(client)
    assert scene_jobs.lid_name("arc", 12.5) == "arc-lid12p5" and scene_jobs.lid_name("arc", 3) == "arc-lid3"
    # a scene already named like the variant but with other settings blocks the sweep rather than being overwritten
    client.post("/api/scenes", json={"spec": {"name": "k0-lid9", "n_lines": 7, "lid_height": 9}}, headers=H)
    with pytest.raises(scene_jobs.SceneJobError, match="already exists with different"):
        scene_jobs.plan(dd, {"mode": "run", "scene": "k0", "lids": [9]})
    p = scene_jobs.plan(dd, {"mode": "run", "scene": "k0", "lids": [8, 4]})
    assert [e["lid"] for e in p["entries"]] == [4.0, 8.0] and p["args"]["lids"] == [4.0, 8.0]
    assert not os.path.exists(os.path.join(dd, "scenes", "k0-lid8.flux"))  # planning writes nothing
