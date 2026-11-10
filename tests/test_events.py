"""events.py and events.pm: no-op without the env var, valid JSON lines with it,
never raise, and the Perl side agrees with the Python side on the schema."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from fluxpype import events

REPO = Path(__file__).resolve().parent.parent
PERL = shutil.which("perl")
needs_perl = pytest.mark.skipif(PERL is None, reason="no perl on PATH")

REQUIRED = {"v", "t", "type", "pid", "task"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("FLUXPYPE_EVENTS", "FLUXPYPE_TASK_ID"):
        monkeypatch.delenv(k, raising=False)


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_noop_without_env(tmp_path):
    assert events.emit("metric", name="x") is False
    assert not events.enabled()
    assert list(tmp_path.iterdir()) == []


def test_appends_valid_json_lines(tmp_path, monkeypatch):
    path = tmp_path / "e.jsonl"
    monkeypatch.setenv("FLUXPYPE_EVENTS", str(path))
    monkeypatch.setenv("FLUXPYPE_TASK_ID", "abc")
    assert events.emit("stage.start", stage="relax")
    assert events.emit("metric", name="n", value=3.5, unit=None)
    recs = read(path)
    assert [r["type"] for r in recs] == ["stage.start", "metric"]
    for r in recs:
        assert REQUIRED <= set(r)
        assert r["v"] == 1 and r["task"] == "abc" and r["pid"] == os.getpid()
        assert isinstance(r["t"], float)
    assert recs[0]["stage"] == "relax" and recs[1]["unit"] is None


def test_task_is_null_when_unset(tmp_path, monkeypatch):
    path = tmp_path / "e.jsonl"
    monkeypatch.setenv("FLUXPYPE_EVENTS", str(path))
    events.emit("log", level="warn", message="m")
    assert read(path)[0]["task"] is None


def test_never_raises_on_unwritable_path(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXPYPE_EVENTS", str(tmp_path / "no" / "such" / "dir" / "e.jsonl"))
    assert events.emit("metric", name="x") is False
    monkeypatch.setenv("FLUXPYPE_EVENTS", str(tmp_path))  # a directory
    assert events.emit("metric", name="x") is False


def test_unserializable_values_do_not_raise(tmp_path, monkeypatch):
    path = tmp_path / "e.jsonl"
    monkeypatch.setenv("FLUXPYPE_EVENTS", str(path))
    assert events.emit("metric", name="x", value=object())
    assert isinstance(read(path)[0]["value"], str)


@needs_perl
def test_perl_syntax_ok():
    r = subprocess.run([PERL, "-c", str(REPO / "fluxpype" / "events.pm")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def perl_emit(code, env):
    full = dict(os.environ, **env)
    return subprocess.run(
        [PERL, "-I", str(REPO), "-e", "use fluxpype::events; " + code], env=full, capture_output=True, text=True
    )


@needs_perl
def test_perl_matches_python_schema(tmp_path):
    path = tmp_path / "e.jsonl"
    r = perl_emit(
        'emit_event("relax.step", {step=>10, stiffness=>12.5, fmax=>0.25, vertices=>1234, note=>"q\\"uote"});',
        {"FLUXPYPE_EVENTS": str(path), "FLUXPYPE_TASK_ID": "t9"},
    )
    assert r.returncode == 0, r.stderr
    py_path = tmp_path / "py.jsonl"
    os.environ["FLUXPYPE_EVENTS"] = str(py_path)
    try:
        events.emit("relax.step", step=10, stiffness=12.5, fmax=0.25, vertices=1234)
    finally:
        del os.environ["FLUXPYPE_EVENTS"]
    (perl_rec,) = read(path)
    (py_rec,) = read(py_path)
    assert REQUIRED <= set(perl_rec)
    assert set(py_rec) <= set(perl_rec)  # same envelope + same fields
    assert perl_rec["v"] == 1 and perl_rec["task"] == "t9" and isinstance(perl_rec["pid"], int)
    assert isinstance(perl_rec["t"], float)
    assert perl_rec["step"] == 10 and perl_rec["stiffness"] == 12.5 and perl_rec["note"] == 'q"uote'


@needs_perl
def test_perl_noop_without_env_and_never_dies(tmp_path):
    r = perl_emit('emit_event("x"); print "alive";', {"FLUXPYPE_EVENTS": ""})
    assert r.returncode == 0 and r.stdout == "alive"
    r = perl_emit(
        'emit_event("x", {a=>1}); emit_event("y", "notahash"); print "alive";',
        {"FLUXPYPE_EVENTS": str(tmp_path / "missing" / "dir" / "e.jsonl")},
    )
    assert r.returncode == 0 and r.stdout == "alive"


@needs_perl
def test_perl_forked_children_do_not_write(tmp_path):
    path = tmp_path / "e.jsonl"
    r = perl_emit(
        'my $pid = fork(); if ($pid == 0) { emit_event("child"); exit 0 } waitpid($pid, 0); emit_event("parent");',
        {"FLUXPYPE_EVENTS": str(path)},
    )
    assert r.returncode == 0, r.stderr
    assert [x["type"] for x in read(path)] == ["parent"]
