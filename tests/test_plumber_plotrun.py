"""Post-hoc plot runs: the registry, input resolution, the runner, and the API around it."""

import fnmatch
import json
import os
import re
import string
import sys
import time

import pytest

from fluxpype.plumber import paths, plotrun
from fluxpype.plumber.monitor import ProbeContext
from fluxpype.plumber.plotcatalog import ALLOWED_PLACEHOLDERS, PLOTS, _p
from fluxpype.plumber.results import task_results

HOC = {
    "fieldmap_topology bmag_scatter bmag_all_profiles wind_profiles_lines wind_profiles_filled light_triplot "
    "fr_expansion bmag_fill world3d world_density world_fluxon_id area_by_method open_area_all_methods"
}
HOC = set(next(iter(HOC)).split())


def touch(path, mtime=None, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def make_ctx(d, **kw):
    base = dict(data_dir=str(d), batch="B", cr="2150", nflux="1000", adapt="0", inst="hmi", reduce="4")
    return ProbeContext(**{**base, **kw})


def entry(i):
    return next(e for e in PLOTS if e["id"] == i)


def test_registry_invariants():
    assert {e["id"] for e in PLOTS if e["post_hoc"]} == HOC
    for e in PLOTS:
        if e["post_hoc"]:
            assert e["command"] and e["post_hoc_reason"] is None and e["command"][0] == "{python}"
            allowed = ALLOWED_PLACEHOLDERS | {f"in_{k}" for k in e["needs"]}
            used = set()
            for tmpl in list(e["command"]) + list(e["needs"].values()):
                used |= {f for _, f, _, _ in string.Formatter().parse(tmpl) if f}
            assert used <= allowed | {"nflux"}, (e["id"], used - allowed)
            for a in e["command"]:
                m = re.match(r"\{(plotting|science)\}/(.+\.py)$", a)
                if m:
                    assert (paths.checkout_root() / "fluxpype" / m.group(1) / m.group(2)).is_file(), a
        else:
            assert e["command"] is None and e["post_hoc_reason"]
        fake = plotrun.fake_name(e["pattern"])
        assert fnmatch.fnmatch(fake, e["pattern"]), (e["id"], fake)
    for e in PLOTS:  # entries of one group are the same command
        if e["group"]:
            assert {tuple(x["command"]) for x in PLOTS if x["group"] == e["group"]} == {tuple(e["command"])}


def test_bmag_scatter_argv(tmp_path):
    ctx = make_ctx(tmp_path)
    f = tmp_path / "batches/B/data/cr2150/wind/cr2150_f1000_radial_bmag.dat"
    touch(str(f))
    e = entry("bmag_scatter")
    found, missing = plotrun.resolve_needs(e, ctx)
    assert missing == [] and found == {"bmag": str(f)}
    argv = plotrun.resolve_command(e, ctx, "/py", found)
    plotting = str(paths.checkout_root() / "fluxpype" / "plotting")
    assert argv == [
        "/py",
        f"{plotting}/plot_bmag.py",
        "--file",
        str(f),
        "--cr",
        "2150",
        "--nwant",
        "1000",
        "--adapt",
        "0",
    ]


def test_world_input_prefers_the_highest_step(tmp_path):
    w = tmp_path / "batches/B/data/cr2150/world"
    for s in (300, 1000, 40):
        touch(str(w / f"cr2150_f1000_hmi_relaxed_s{s}.flux"))
    found, missing = plotrun.resolve_needs(entry("light_triplot"), make_ctx(tmp_path))
    assert missing == [] and found["world"].endswith("_s1000.flux")


def test_missing_inputs_are_listed(tmp_path):
    g = plotrun.generatable(entry("bmag_scatter"), make_ctx(tmp_path))
    assert g["can_generate"] is False
    assert g["missing_inputs"] == ["batches/B/data/cr2150/wind/cr2150_f1000_radial_bmag.dat"]
    assert plotrun.generatable(entry("world_relaxed_narrow"), make_ctx(tmp_path))["can_generate"] is False


def test_outside_inputs_and_bad_values_are_refused(tmp_path):
    d, out = tmp_path / "data", tmp_path / "outside"
    touch(str(out / "x.dat"))
    wind = d / "batches/B/data/cr2150/wind"
    os.makedirs(wind)
    os.symlink(out / "x.dat", wind / "cr2150_f1000_radial_bmag.dat")
    found, missing = plotrun.resolve_needs(entry("bmag_scatter"), make_ctx(d))
    assert found == {} and len(missing) == 1
    with pytest.raises(plotrun.PlotRunError):
        plotrun.context_vars(make_ctx(d, batch="../x"), "py")
    with pytest.raises(plotrun.PlotRunError):
        plotrun.context_vars(make_ctx(d, cr="21;50"), "py")


def test_inputs_written_after_the_task_are_flagged(tmp_path):
    f = tmp_path / "batches/B/data/cr2150/wind/cr2150_f1000_radial_bmag.dat"
    touch(str(f), 1000.0)
    ctx = make_ctx(tmp_path)
    assert plotrun.generatable(entry("bmag_scatter"), ctx, ended=900.0)["inputs_later"] == [
        "batches/B/data/cr2150/wind/cr2150_f1000_radial_bmag.dat"
    ]
    assert plotrun.generatable(entry("bmag_scatter"), ctx, ended=990.0)["inputs_later"] == []


# ---------------------------------------------------------------- the runner

SCRIPT = """
import os, sys, pathlib
mode, target, counter = sys.argv[1:4]
with open(counter, "a") as f:
    f.write("x")
if mode == "write":
    pathlib.Path(target).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(target).write_bytes(b"png")
elif mode == "fail":
    sys.exit(3)
elif mode == "sleep":
    import time; time.sleep(60)
"""


def fake_plot(tmp_path, id, mode, name=None, group=None):
    script = tmp_path / "fake.py"
    script.write_text(SCRIPT)
    name = name or f"{id}.png"
    return _p(
        id,
        "wind_plots",
        id,
        "imgs/t/" + name,
        "test",
        "test",
        post_hoc=True,
        command=["{python}", str(script), mode, "{batch_dir}/imgs/t/" + name, str(tmp_path / f"count_{group or id}")],
        group=group,
        timeout_s=20,
    )


def events(run_dir):
    with open(os.path.join(run_dir, "events.jsonl")) as f:
        return [json.loads(x) for x in f]


def test_run_plots_reports_each_outcome(tmp_path):
    d, run = tmp_path / "data", tmp_path / "run"
    os.makedirs(d), os.makedirs(run)
    plots = [
        fake_plot(tmp_path, "good", "write"),
        fake_plot(tmp_path, "bad", "fail"),
        fake_plot(tmp_path, "quiet", "none"),
    ]
    rc = plotrun.run_plots(str(run), make_ctx(d), ["good", "bad", "quiet"], plots=plots)
    assert rc == 1
    ends = {e["id"]: e for e in events(run) if e["event"] == "plot.end"}
    assert [ends[i]["status"] for i in ("good", "bad", "quiet")] == ["ok", "failed", "failed"]
    assert ends["bad"]["exit_code"] == 3 and "wrote no file" in ends["quiet"]["detail"]
    assert ends["good"]["outputs"] == ["batches/B/imgs/t/good.png"]
    res = json.load(open(run / "result.json"))
    assert res["status"] == "failed" and [p["status"] for p in res["plots"]] == ["ok", "failed", "failed"]
    names = [e["event"] for e in events(run)]
    assert names[0] == "plotrun.start" and names[-1] == "plotrun.end"


def test_a_group_runs_once(tmp_path):
    d, run = tmp_path / "data", tmp_path / "run"
    os.makedirs(d), os.makedirs(run)
    a = fake_plot(tmp_path, "a", "write", "same.png", group="g")
    b = {**fake_plot(tmp_path, "b", "write", "same.png", group="g")}
    assert plotrun.run_plots(str(run), make_ctx(d), ["a", "b"], plots=[a, b]) == 0
    assert (tmp_path / "count_g").read_text() == "x"


def test_existing_target_needs_overwrite_and_is_backed_up(tmp_path):
    d, run, run2 = tmp_path / "data", tmp_path / "run", tmp_path / "run2"
    os.makedirs(run), os.makedirs(run2)
    target = d / "batches/B/imgs/t/good.png"
    touch(str(target), 1000.0, b"old")
    plots = [fake_plot(tmp_path, "good", "write")]
    assert plotrun.run_plots(str(run), make_ctx(d), ["good"], plots=plots) == 1
    end = [e for e in events(run) if e["event"] == "plot.end"][0]
    assert "target exists" in end["detail"] and target.read_bytes() == b"old"
    assert plotrun.run_plots(str(run2), make_ctx(d), ["good"], plots=plots, overwrite=True) == 0
    assert target.read_bytes() == b"png"
    assert (run2 / "replaced/batches/B/imgs/t/good.png").read_bytes() == b"old"


def test_main_cancel_kills_the_running_command(tmp_path):
    import signal
    import subprocess

    d, run = tmp_path / "data", tmp_path / "run"
    os.makedirs(d), os.makedirs(run)
    # main() uses the real registry, so cancel through a tiny driver that uses a fake one
    driver = tmp_path / "drive.py"
    driver.write_text(
        "import sys\nfrom fluxpype.plumber import plotrun\nfrom fluxpype.plumber.monitor import ProbeContext\n"
        "sys.path.insert(0, %r)\nfrom test_plumber_plotrun import fake_plot, make_ctx\nfrom pathlib import Path\n"
        "tp = Path(%r)\nsys.exit(plotrun.run_plots(%r, make_ctx(%r), ['s'], plots=[fake_plot(tp, 's', 'sleep')]))\n"
        % (os.path.dirname(__file__), str(tmp_path), str(run), str(d))
    )
    p = subprocess.Popen([sys.executable, str(driver)], cwd=str(paths.checkout_root()))
    for _ in range(100):
        if (tmp_path / "count_s").exists():
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 143
    assert json.load(open(run / "result.json"))["plots"][0]["status"] == "cancelled"


def test_fake_run_writes_placeholder_images(tmp_path):
    d, run = tmp_path / "data", tmp_path / "run"
    os.makedirs(run)
    ctx = make_ctx(d)
    ids = ["bmag_scatter", "fieldmap_topology"]
    for i in ids:  # inputs
        for rel in entry(i)["needs"].values():
            touch(str(d / plotrun.fake_name(rel.format(cr="2150", nflux="1000", inst="hmi", red="4", batch="B"))))
    assert plotrun.run_plots(str(run), ctx, ids, fake=True) == 0
    assert all(plotrun.targets(entry(i), ctx) for i in ids)


# ---------------------------------------------------------------- results


def test_task_results_tags_posthoc_files(tmp_path):
    t0 = time.time()
    rel = "batches/B/imgs/bmag/cr2150_f1000_radial_bmag.png"
    touch(str(tmp_path / rel), t0 + 5)
    ctx = make_ctx(tmp_path)
    cat = {c["id"]: c for c in task_results(ctx, t0, t0 + 60, posthoc={rel: "p1"})["catalog"]}
    f = cat["bmag_scatter"]["files"][0]
    assert f["origin"] == "posthoc" and f["plot_run"] == "p1"
    assert cat["bmag_scatter"]["can_generate"] is False and cat["bmag_scatter"]["post_hoc"] is True


# ---------------------------------------------------------------- scheduler and API

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


def finished_task(client, profile="t_one", scenario=None):
    body = {"profile": profile, "backend": "fake", "speed": 300, "overrides": {}, "scenario": scenario}
    job = client.post("/api/jobs", json=body, headers=H).json()
    end = time.time() + 60
    while time.time() < end:
        j = client.get(f"/api/jobs/{job['id']}", headers=H).json()
        if j["status"] in ("ok", "failed", "partial", "cancelled"):
            return j, j["tasks"][0]["id"]
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def wait_run(client, tid, rid, final=("ok", "failed", "cancelled", "crashed")):
    end = time.time() + 60
    while time.time() < end:
        v = client.get(f"/api/tasks/{tid}/plots/{rid}", headers=H).json()
        if v["status"] in final:
            return v
        time.sleep(0.1)
    raise AssertionError("plot run did not finish")


def test_plot_run_api_end_to_end(client):
    job, tid = finished_task(client)
    url = f"/api/tasks/{tid}/plots"
    assert client.post(url, json={"plots": ["bmag_scatter"]}).status_code == 401
    assert client.post(url, json={"plots": ["nope"]}, headers=H).status_code == 400
    assert client.post(url, json={"plots": ["world_relaxed_narrow"]}, headers=H).status_code == 400
    assert client.post("/api/tasks/nope/plots", json={"plots": ["bmag_scatter"]}, headers=H).status_code == 404
    r = client.post(url, json={"plots": ["bmag_scatter"], "dry_run": True}, headers=H)
    assert r.status_code == 200
    plan = r.json()["plan"]
    assert plan[0]["id"] == "bmag_scatter" and plan[0]["missing"] == [] and plan[0]["argv"][1].endswith("plot_bmag.py")
    r = client.post(url, json={"plots": ["bmag_scatter", "fieldmap_topology"]}, headers=H)
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    assert client.post(url, json={"plots": "all_possible"}, headers=H).status_code == 409  # one at a time
    view = wait_run(client, tid, rid)
    assert view["status"] == "ok", view
    assert [p["status"] for p in view["plots"]] == ["ok", "ok"]
    res = client.get(f"/api/tasks/{tid}/results", headers=H).json()
    cat = {c["id"]: c for c in res["catalog"]}
    f = cat["bmag_scatter"]["files"][0]
    assert f["origin"] == "posthoc" and f["plot_run"] == rid
    assert cat["bmag_scatter"]["last_run"]["status"] == "ok"
    assert cat["world_relaxed_narrow"]["post_hoc"] is False and cat["world_relaxed_narrow"]["post_hoc_reason"]
    # plot runs never change the task or the job
    assert client.get(f"/api/jobs/{job['id']}", headers=H).json()["status"] == "ok"
    assert client.get(f"/api/tasks/{tid}", headers=H).json()["status"] == "ok"
    assert [x["id"] for x in client.get(url, headers=H).json()["runs"]] == [rid]
    log = client.get(f"{url}/{rid}/log", headers=H)
    assert log.status_code == 200 and "text" in log.json()
    assert client.get(f"{url}/bad/log", headers=H).status_code == 404
    # asking again would replace the file: 409 until overwrite is confirmed
    again = client.post(url, json={"plots": ["bmag_scatter"]}, headers=H)
    assert again.status_code == 409
    r = client.post(url, json={"plots": ["bmag_scatter"], "overwrite": True}, headers=H)
    assert r.status_code == 201
    v = wait_run(client, tid, r.json()["id"])
    assert v["status"] == "ok" and v["plots"][0]["replaced"]
    # everything possible that is still missing, never replacing anything
    r = client.post(url, json={"plots": "all_possible", "stage": "wind_plots"}, headers=H)
    assert r.status_code == 201, r.text
    v = wait_run(client, tid, r.json()["id"])
    assert v["status"] == "ok" and "bmag_scatter" not in [p["id"] for p in v["plots"]]
    assert {"wind_profiles_lines", "wind_profiles_filled"} <= {p["id"] for p in v["plots"]}


def test_a_late_plot_does_not_change_a_finished_tasks_stages(client):
    _, tid = finished_task(client, "t_light", "fail_in_pfss")
    sched = client.sched
    before = sched.task_state(tid)
    ctx = sched.probe_context(sched.store.get_task(tid))
    touch(os.path.join(ctx.data_dir, "batches", ctx.batch, "imgs", "light", f"cr{ctx.cr}_fluxlight_light_x_1.png"))
    sched.relax_series(tid)
    after = sched.task_state(tid)
    assert [(s["id"], s["status"]) for s in after["stages"]] == [(s["id"], s["status"]) for s in before["stages"]]


def test_settings_are_parsed_never_passed_through():
    ok = plotrun.clean_settings(
        "light", {"light_method": "voronoi", "fov": "3", "nxy": "401", "observer_lon": " ", "nz": ""}
    )
    assert ok == {"light_method": '"voronoi"', "fov": "3.0", "nxy": "401"}
    assert plotrun.clean_settings("wind", {"flow_method": "wsa"}) == {"flow_method": "[wsa]"}
    for bad in (
        {"fov": "__import__('os').system('x')"},
        {"fov": "1e9"},
        {"nxy": "3.5"},
        {"nxy": "5000"},
        {"light_method": 'voronoi"\nx = 1'},
        {"unknown_key": "1"},
    ):
        with pytest.raises(plotrun.PlotRunError):
            plotrun.clean_settings("light", bad)
    with pytest.raises(plotrun.PlotRunError):
        plotrun.clean_settings("wind", {"fov": "3"})


def test_light_remake_takes_settings_into_its_own_copy_of_the_config(client):
    job, tid = finished_task(client, "t_light")
    opts = client.get(f"/api/tasks/{tid}/rerun", headers=H).json()
    light = {f["key"]: f for f in opts["light"]["fields"]}
    assert opts["light"]["can"] and light["light_method"]["choices"] and light["fov"]["value"]
    url = f"/api/tasks/{tid}/plots"
    assert client.post(url, json={"plots": ["bmag_scatter"], "settings": {"fov": "3"}}, headers=H).status_code == 400
    assert (
        client.post(url, json={"plots": ["light_triplot"], "settings": {"fov": "evil()"}}, headers=H).status_code == 400
    )
    job_path = client.sched.store.get_job(job["id"])["snapshot_path"]
    before = open(job_path).read()
    body = {"plots": ["light_triplot"], "settings": {"fov": "3", "light_method": "voronoi"}, "overwrite": True}
    dry = client.post(url, json={**body, "dry_run": True}, headers=H).json()["plan"][0]
    assert dry["argv"][-1] == "--force"
    r = client.post(url, json=body, headers=H)
    assert r.status_code == 201, r.text
    view = wait_run(client, tid, r.json()["id"])
    assert view["status"] == "ok" and view["settings"] == {"fov": "3.0", "light_method": "voronoi"}
    snap = open(os.path.join(client.sched.plot_run_dir(tid, view["id"]), "config.snapshot.ini")).read()
    assert re.search(r"^fov\s*=\s*3\.0\s*$", snap, re.M) and re.search(r'^light_method\s*=\s*"voronoi"\s*$', snap, re.M)
    assert open(job_path).read() == before  # the job's own record is untouched


def test_wind_rerun_queues_a_wind_only_job_from_the_jobs_own_config(client):
    job, tid = finished_task(client, "t_light")
    url = f"/api/tasks/{tid}/rerun/wind"
    assert client.post(url, json={}).status_code == 401
    assert client.post(url, json={"settings": {"flow_method": "x;y"}}, headers=H).status_code == 400
    assert client.post("/api/tasks/nope/rerun/wind", json={}, headers=H).status_code == 404
    r = client.post(url, json={"settings": {"flow_method": "wsa"}}, headers=H)
    assert r.status_code == 201, r.text
    new = r.json()
    assert new["id"] != job["id"] and new["extra"]["from_task"] == tid and len(new["tasks"]) == 1
    assert (
        new["tasks"][0]["method"] == "wsa"
        and new["tasks"][0]["cr"] == client.get(f"/api/tasks/{tid}", headers=H).json()["cr"]
    )
    snap = open(client.sched.store.get_job(new["id"])["snapshot_path"]).read()
    assert re.search(r"^wind_only\s*=\s*1\s*$", snap, re.M) and re.search(r"^light\s*=\s*0\s*$", snap, re.M)
    end = time.time() + 60
    while time.time() < end and client.get(f"/api/jobs/{new['id']}", headers=H).json()["status"] not in (
        "ok",
        "failed",
    ):
        time.sleep(0.1)
    done = client.get(f"/api/jobs/{new['id']}", headers=H).json()
    assert done["status"] == "ok", done
    stages = {
        s["id"]: s["status"] for s in client.get(f"/api/tasks/{done['tasks'][0]['id']}", headers=H).json()["stages"]
    }
    assert stages["wind"] != "skipped" and stages["relax"] == "skipped"  # only the end of the pipeline ran
