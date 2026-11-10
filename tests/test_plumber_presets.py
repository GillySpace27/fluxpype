"""Job presets (user presets beside config.ini), the active backend, the sandbox and health checks."""

import os
import time

import pytest

from fluxpype.plumber.backends.fake import FakeBackend
from fluxpype.plumber.backends.local import LocalBackend
from fluxpype.plumber.presets import PresetError, PresetStore


@pytest.fixture
def presets(plumber_home, plumber_config, make_scheduler):
    sched = make_scheduler()
    return sched.presets


# ------------------------------------------------------------------ preset store


def test_create_from_a_builtin_copies_its_own_settings(presets):
    presets.create("my_quick", base="quickstart", values={"fluxon_count": "[200]"})
    user = presets.user_sections()["my_quick"]
    assert user["fluxon_count"] == "[200]" and "rotations" in user
    assert "my_quick" in presets.user_names() and presets.origin("my_quick") == "user"
    assert presets.origin("quickstart") == "builtin"


def test_config_ini_is_never_written(presets, plumber_config):
    before = plumber_config.read_text()
    presets.create("a1", base="quickstart")
    presets.save("a1", {"rotations": "[2150]"})
    presets.rename("a1", "a2")
    presets.delete("a2")
    assert plumber_config.read_text() == before


def test_user_presets_join_the_config_and_inherit_default(presets):
    presets.create("mine", base="quickstart", values={"rotations": "[2160]"})
    from fluxpype.plumber import config_io

    cf = config_io.parse(presets.combined_text())
    eff = cf.effective("mine")
    assert eff["rotations"].value == "[2160]" and "data_dir" in eff  # data_dir comes from DEFAULT


@pytest.mark.parametrize(
    "values, msg",
    [
        ({"data_dir": "/tmp/x"}, "cannot be overridden"),
        ({"rotations": "__import__('os')"}, "not allowed"),
        ({"nonsense_key": "1"}, "not an existing setting"),
    ],
)
def test_values_follow_the_override_rules(presets, values, msg):
    with pytest.raises(PresetError, match=msg):
        presets.create("bad", base="quickstart", values=values)
    assert "bad" not in presets.user_sections()


def test_names_are_checked_and_builtins_are_read_only(presets):
    with pytest.raises(PresetError):
        presets.create("quickstart")
    with pytest.raises(PresetError):
        presets.create("has space")
    with pytest.raises(PresetError, match="read-only"):
        presets.save("quickstart", {"rotations": "[1]"})
    with pytest.raises(PresetError, match="hide"):
        presets.delete("quickstart")


def test_delete_goes_to_trash_and_restores(presets):
    presets.create("tmp1", base="quickstart")
    presets.set_hidden("tmp1", True)
    presets.delete("tmp1")
    assert "tmp1" not in presets.user_sections() and "tmp1" in presets.trashed_sections()
    assert "tmp1" not in presets.hidden()
    assert presets.restore_deleted("tmp1") == "tmp1"
    assert "tmp1" in presets.user_sections() and not presets.trashed_sections()


def test_hide_and_unhide_any_preset(presets):
    presets.set_hidden("quickstart", True)
    assert presets.hidden() == ["quickstart"]
    presets.set_hidden("quickstart", False)
    assert presets.hidden() == []
    with pytest.raises(PresetError):
        presets.set_hidden("nope", True)


def test_export_import_round_trip(presets):
    text = presets.export(["quickstart"])
    assert "[quickstart]" in text and "data_dir" not in text
    added = presets.import_text(text)
    assert added == ["quickstart_2"]
    assert presets.user_sections()["quickstart_2"]["rotations"]
    with pytest.raises(PresetError):
        presets.import_text("[evil]\nrun_script = rm -rf /\n")
    with pytest.raises(PresetError):
        presets.import_text("not an ini file")


def test_jobs_run_user_presets(make_scheduler):
    sched = make_scheduler()
    sched.presets.create("mine", base="t_one")
    job = sched.create_job("mine", backend="fake", extra={"speed": "300"})
    assert job["profile"] == "mine" and len(job["tasks"]) == 1


# ------------------------------------------------------------------ backends and sandbox


def test_active_backend_is_persisted_and_used_for_new_jobs(make_scheduler):
    sched = make_scheduler(backends={"local": LocalBackend(), "fake": FakeBackend()})
    assert sched.active_backend == "local"
    sched.set_active_backend("fake")
    again = make_scheduler(backends={"local": LocalBackend(), "fake": FakeBackend()})
    assert again.active_backend == "fake"
    job = again.create_job("t_one", extra={"speed": "300"})
    assert again.store.get_job(job["id"])["backend"] == "fake"


def test_simulated_runs_write_to_the_sandbox(make_scheduler, plumber_home, tmp_path):
    sched = make_scheduler(backends={"local": LocalBackend(), "fake": FakeBackend()})
    real = sched.data_dir()
    sched.set_active_backend("fake")
    assert sched.data_dir() == str(plumber_home / "sandbox" / "data") != real
    job = sched.create_job("t_one", extra={"speed": "300"})
    snap = open(sched.store.get_job(job["id"])["snapshot_path"]).read()
    assert f"data_dir = {plumber_home / 'sandbox' / 'data'}" in snap
    sched.set_active_backend("local")
    assert sched.data_dir() == real


# ------------------------------------------------------------------ API


@pytest.fixture
def client(make_scheduler):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fluxpype.plumber.server import create_app

    sched = make_scheduler(backends={"local": LocalBackend(), "fake": FakeBackend()})
    app = create_app(sched, "tok", run_loop=False)
    with TestClient(app, base_url="http://localhost", headers={"X-Flux-Token": "tok"}) as c:
        c.sched = sched
        yield c


def test_api_preset_lifecycle(client):
    assert client.post("/api/profiles", json={"name": "mine", "base": "quickstart"}).status_code == 201
    assert client.post("/api/profiles", json={"name": "mine"}).status_code == 400
    assert client.patch("/api/profiles/mine", json={"values": {"rotations": "[2150]"}}).status_code == 200
    detail = client.get("/api/profiles/mine").json()
    assert detail["origin"] == "user" and any(
        e["key"] == "rotations" and e["value"] == "[2150]" for e in detail["entries"]
    )
    assert client.post("/api/profiles/mine/rename", json={"name": "mine2"}).json()["name"] == "mine2"
    assert client.post("/api/profiles/quickstart/hidden", json={"hidden": True}).status_code == 200
    rows = {p["name"]: p for p in client.get("/api/profiles").json()["profiles"]}
    assert rows["quickstart"]["hidden"] and rows["mine2"]["origin"] == "user"
    exp = client.get("/api/profiles-export?names=mine2")
    assert exp.status_code == 200 and "[mine2]" in exp.text and "attachment" in exp.headers["content-disposition"]
    assert client.delete("/api/profiles/mine2").status_code == 200
    assert client.get("/api/profiles").json()["deleted"] == ["mine2"]
    assert client.post("/api/profiles-deleted/mine2/restore").json()["name"] == "mine2"
    assert client.post("/api/profiles-import", json={"text": exp.text}).json()["added"] == ["mine2_2"]
    assert client.delete("/api/profiles/quickstart").status_code == 400


def test_api_preset_rows_carry_cache_counts(client, plumber_config):
    data = client.sched.data_dir()
    os.makedirs(os.path.join(data, "magnetograms"), exist_ok=True)
    rows = {p["name"]: p for p in client.get("/api/profiles").json()["profiles"]}
    c = rows["quickstart"]["cache"]
    assert c["tasks"] >= 1 and set(c["counts"]) == {"magnetogram", "footpoints", "pfss", "world", "relaxed_world"}
    assert rows["Real_Run"]["cache"] is None and rows["Real_Run"]["errors"]


def test_api_backend_switch_and_health(client):
    st = client.get("/api/status").json()
    assert st["active_backend"] == "local" and not st["sandboxed"]
    assert client.put("/api/backend", json={"name": "slurm"}).status_code == 400
    assert client.put("/api/backend", json={"name": "fake"}).json()["active_backend"] == "fake"
    st = client.get("/api/status").json()
    assert st["sandboxed"] and st["data_dir"].endswith(os.path.join("sandbox", "data"))
    h = client.get("/api/health").json()
    names = {b["name"]: b for b in h["backends"]}
    assert names["fake"]["active"] and names["fake"]["status"] == "ok"
    assert names["slurm"]["configured"] is False and names["slurm"]["status"] == "off"
    assert {c["id"] for c in names["local"]["checks"]} >= {"python", "science", "runner", "perl"}
    assert {c["id"] for c in h["system"]} >= {"server", "queue", "config", "data", "mcp"}
    # simulated jobs accept scenarios; the real backend does not
    ok = client.post("/api/jobs", json={"profile": "t_one", "scenario": "ok", "speed": 300})
    assert ok.status_code == 201
    client.put("/api/backend", json={"name": "local"})
    bad = client.post("/api/jobs", json={"profile": "t_one", "scenario": "ok"})
    assert bad.status_code == 400
