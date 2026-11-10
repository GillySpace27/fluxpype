"""SQLite store: persistence, job status derivation, settings."""

import time

import pytest

from fluxpype.studio.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "s.db")
    yield s
    s.close()


def add(store, jid="j1", n=3):
    store.add_job(
        {
            "id": jid,
            "profile": "p",
            "overrides": {"a": "1"},
            "backend": "fake",
            "created": time.time(),
            "snapshot_path": "/x",
            "extra": {"speed": "2"},
        },
        [
            {
                "id": f"{jid}-{i}",
                "idx": i,
                "cr": "1",
                "nflux": "2",
                "adapt": "0",
                "method": "m",
                "run_dir": f"/r/{i}",
                "lock_key": "k1\nk2",
                "slots": 3,
            }
            for i in range(1, n + 1)
        ],
    )


def test_roundtrip(store, tmp_path):
    add(store)
    job = store.get_job("j1")
    assert job["overrides"] == {"a": "1"} and job["extra"] == {"speed": "2"} and job["status"] == "pending"
    tasks = store.tasks_for_job("j1")
    assert [t["idx"] for t in tasks] == [1, 2, 3] and tasks[0]["slots"] == 3 and tasks[0]["attempt"] == 1
    store.close()
    again = Store(tmp_path / "s.db")  # reopened: state survives a restart
    assert again.get_job("j1")["profile"] == "p"
    again.close()
    store.__init__(tmp_path / "s.db")  # leave a live connection for the fixture teardown


def test_status_derivation(store):
    add(store)
    assert store.derive_job_status("j1") == "pending"
    store.update_task("j1-1", status="running")
    assert store.derive_job_status("j1") == "running"
    store.update_task("j1-1", status="ok")
    assert store.derive_job_status("j1") == "running"  # some done, some pending
    for i in (2, 3):
        store.update_task(f"j1-{i}", status="ok")
    assert store.derive_job_status("j1") == "ok"
    store.update_task("j1-3", status="failed")
    assert store.derive_job_status("j1") == "partial"
    for i in (1, 2):
        store.update_task(f"j1-{i}", status="failed")
    assert store.derive_job_status("j1") == "failed"
    for i in (1, 2, 3):
        store.update_task(f"j1-{i}", status="cancelled")
    assert store.derive_job_status("j1") == "cancelled"


def test_settings_and_counts(store):
    assert store.get_setting("x", 5) == 5
    store.set_setting("x", {"a": [1, 2]})
    store.set_setting("x", {"a": [3]})
    assert store.get_setting("x") == {"a": [3]}
    add(store)
    assert store.counts() == {"pending": 3}


def test_delete_cascades(store):
    add(store)
    store.delete_job("j1")
    assert store.tasks_for_job("j1") == [] and store.get_job("j1") is None


def test_failed_insert_rolls_back(store):
    add(store)
    with pytest.raises(Exception):
        add(store)  # duplicate primary key
    assert len(store.tasks_for_job("j1")) == 3
