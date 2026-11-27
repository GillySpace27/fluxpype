"""A7-05: every entry point resolves the same Perl and Flux.pm for the same config."""

import json
import os
import stat
import sys

import pytest

from fluxpype import flux_runtime as fr

FAKE = """#!/bin/sh
# fake perl: `use Flux` works when FAKE_FLUX_OK=1 or PERL5LIB names a dir with Flux.pm
case "$*" in
  *"use Flux"*)
    if [ "$FAKE_FLUX_OK" = 1 ]; then printf 'ok /default/Flux.pm'; exit 0; fi
    case "$PERL5LIB" in
      *%(lib)s*) printf 'ok %(lib)s/Flux.pm'; exit 0;;
    esac
    exit 2;;
esac
exit 0
"""


@pytest.fixture
def world(tmp_path, monkeypatch):
    lib = tmp_path / "fluxon-perl" / "lib"
    lib.mkdir(parents=True)
    (lib / "Flux.pm").write_text("1;\n")
    perl = tmp_path / "bin" / "perl"
    perl.parent.mkdir()
    perl.write_text(FAKE % {"lib": str(lib)})
    perl.chmod(perl.stat().st_mode | stat.S_IEXEC)
    for k in ("FLUX_PERL", "PERL5LIB", "FAKE_FLUX_OK"):
        monkeypatch.delenv(k, raising=False)
    return {"perl": str(perl), "lib": str(lib), "cfg": {"flux_perl": str(perl), "flux_perl_lib": str(lib)}}


def test_perl_precedence(monkeypatch):
    monkeypatch.delenv("FLUX_PERL", raising=False)
    cfg = {"flux_perl": "~/x/perl"}
    assert fr.resolve_perl({}) == ("perl", "path")
    assert fr.resolve_perl({"flux_perl": ""}) == ("perl", "path")
    assert fr.resolve_perl(cfg) == (os.path.expanduser("~/x/perl"), "config")
    monkeypatch.setenv("FLUX_PERL", "/env/perl")
    assert fr.resolve_perl(cfg) == (os.path.expanduser("~/x/perl"), "config")  # the pipeline never read $FLUX_PERL
    assert fr.resolve_perl(cfg, honor_env=True) == ("/env/perl", "env")
    assert fr.resolve_perl(cfg, "/arg/perl", honor_env=True) == ("/arg/perl", "arg")


def test_pipeline_perl_matches_config_runner():
    from fluxpype.config_runner import _perl

    for cfg in ({}, {"flux_perl": ""}, {"flux_perl": "~/miniconda3a/envs/fluxenv/bin/perl"}):
        assert _perl(cfg) == os.path.expanduser(cfg.get("flux_perl") or "perl")


def test_lib_added_only_when_flux_not_loadable(world, monkeypatch):
    rt = fr.flux_runtime(world["cfg"])
    assert not rt.flux_loadable and rt.perl5lib_added == world["lib"]
    assert rt.env["PERL5LIB"] == world["lib"]
    assert rt.flux_pm == world["lib"] + "/Flux.pm"
    monkeypatch.setenv("FAKE_FLUX_OK", "1")
    rt = fr.flux_runtime(world["cfg"])
    assert rt.flux_loadable and rt.perl5lib_added is None
    assert "PERL5LIB" not in rt.env and rt.flux_pm == "/default/Flux.pm"


def test_existing_perl5lib_is_kept_behind_the_added_dir(world, monkeypatch):
    monkeypatch.setenv("PERL5LIB", "/keep")
    rt = fr.flux_runtime(world["cfg"])
    assert rt.env["PERL5LIB"] == world["lib"] + os.pathsep + "/keep"


@pytest.mark.parametrize("flux_ok", [False, True])
def test_scene_and_pipeline_resolve_identically(world, monkeypatch, tmp_path, flux_ok):
    """The pipeline call, the Plumber scene task call and the scene CLI call agree on Perl, Flux.pm and PERL5LIB."""
    if flux_ok:
        monkeypatch.setenv("FAKE_FLUX_OK", "1")
    cfg = world["cfg"]
    pipeline = fr.flux_runtime(
        cfg, env={**os.environ, "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    )

    from fluxpype.plumber import scene_task

    spec = {
        "run_dir": str(tmp_path / "run"),
        "perl": cfg["flux_perl"],
        "flux_config": {"flux_perl_lib": cfg["flux_perl_lib"]},
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    seen = {}
    monkeypatch.setattr(
        scene_task, "Runner", lambda s, fake, speed: seen.update(spec=s) or (_ for _ in ()).throw(SystemExit)
    )
    before = dict(os.environ)
    try:
        with pytest.raises(SystemExit):
            scene_task.main([str(p)])
        task_pm = seen["spec"]["flux_pm"]
        task_perl = seen["spec"]["perl"]
        task_lib = os.environ.get("PERL5LIB")
    finally:
        os.environ.clear()
        os.environ.update(before)
    cli = fr.flux_runtime(cfg, None, honor_env=True)

    assert task_perl == pipeline.perl == cli.perl == world["perl"]
    assert task_pm == pipeline.flux_pm == cli.flux_pm
    assert task_lib == pipeline.env.get("PERL5LIB") == cli.env.get("PERL5LIB")
    if flux_ok:
        assert task_lib is None
