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


STAGES = [
    "magnetogram",
    "footpoints",
    "pfss",
    "world",
    "relax",
    "world_plots",
    "fieldmap",
    "wind",
    "wind_plots",
    "fluxlight",
]


@needs_perl
def test_perl_stage_helpers_time_and_report(tmp_path):
    path = tmp_path / "e.jsonl"
    r = perl_emit(
        'stage_start("a"); stage_detail("a", "reused relaxed world"); stage_end("a"); '
        'stage_skipped("wind_only", "b", "c");',
        {"FLUXPYPE_EVENTS": str(path)},
    )
    assert r.returncode == 0, r.stderr
    recs = read(path)
    assert [(x["type"], x["stage"]) for x in recs] == [
        ("stage.start", "a"),
        ("stage.end", "a"),
        ("stage.end", "b"),
        ("stage.end", "c"),
    ]
    assert recs[1]["status"] == "ok" and recs[1]["detail"] == "reused relaxed world"
    assert isinstance(recs[1]["elapsed_s"], (int, float))
    assert [x["status"] for x in recs[2:]] == ["skipped", "skipped"]


@needs_perl
def test_perl_open_stage_is_closed_as_failed_when_the_process_dies(tmp_path):
    path = tmp_path / "e.jsonl"
    r = perl_emit('stage_start("pfss"); die "boom\\n";', {"FLUXPYPE_EVENTS": str(path)})
    assert r.returncode != 0 and "boom" in r.stderr  # exit status preserved
    last = read(path)[-1]
    assert (last["type"], last["stage"], last["status"]) == ("stage.end", "pfss", "failed")


@needs_perl
def test_perl_blessed_numbers_are_flattened(tmp_path):
    path = tmp_path / "e.jsonl"
    code = (
        'package Num; use overload q("") => sub { ${$_[0]} }, fallback => 1; package main; '
        'my $x = 2.5; emit_event("relax.step", {stiffness => bless(\\$x, "Num")});'
    )
    r = perl_emit(code, {"FLUXPYPE_EVENTS": str(path)})
    assert r.returncode == 0, r.stderr
    assert read(path)[0]["stiffness"] == 2.5


@needs_perl
def test_soft_load_falls_back_to_stubs_when_module_is_missing():
    line = (
        "BEGIN { eval { require fluxpype::events; fluxpype::events->import; 1 } or eval "
        "'sub emit_event {} sub stage_start {} sub stage_end {} sub stage_skipped {} sub stage_detail {} 1'; } "
        'stage_start("a"); stage_end("a"); emit_event("x"); print "alive";'
    )
    r = subprocess.run([PERL, "-I", "/nonexistent", "-e", line], capture_output=True, text=True)
    assert (r.returncode, r.stdout) == (0, "alive"), r.stderr


def test_pipeline_instrumentation_is_balanced():
    """Static check on magnetogram2wind.pdl: every stage id is started once and ended once,
    wind_only skips exactly the first seven, and each edit is guarded by the soft-load line."""
    import re

    text = (REPO / "fluxpype" / "science" / "magnetogram2wind.pdl").read_text()
    starts = re.findall(r'stage_start\("(\w+)"\)', text)
    ends = re.findall(r'stage_end\("(\w+)"\)', text)
    assert starts == [s for s in STAGES]
    assert ends == [s for s in STAGES]
    skipped = re.search(r'stage_skipped\("wind_only", qw\(([^)]*)\)\) if \$wind_only', text).group(1).split()
    assert skipped == STAGES[:7]
    assert "require fluxpype::events" in text


def test_instrumented_modules_use_the_soft_load_line():
    for name in ("relax_pfss_world.pm", "simple_relaxer.pm"):
        text = (REPO / "fluxpype" / "science" / name).read_text()
        assert "require fluxpype::events" in text, name
