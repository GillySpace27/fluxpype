"""FLUXPYPE_CONFIG_FILE and FLUXPYPE_LOG_FILE hooks, lazy package import."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def ph(monkeypatch, fake_science):
    monkeypatch.delenv("FLUXPYPE_CONFIG_FILE", raising=False)
    monkeypatch.delenv("FLUXPYPE_CONFIG", raising=False)
    from fluxpype import pipe_helper

    return pipe_helper


def test_import_fluxpype_is_cheap():
    code = (
        "import sys, fluxpype; "
        "bad = [m for m in ('fluxpype.config_runner', 'fluxpype.pipe_helper', 'numpy', 'pandas', 'sunpy') if m in sys.modules]; "  # noqa: E501
        "print(bad)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert r.stdout.strip() == "[]", r.stdout + r.stderr


def test_lazy_attributes_resolve():
    import fluxpype

    assert callable(fluxpype.run)
    assert fluxpype.__all__ == ["config_runner", "pipe_helper", "run"]
    with pytest.raises(AttributeError):
        fluxpype.does_not_exist


def test_env_var_wins_over_argument(ph, tmp_path, monkeypatch):
    snap = tmp_path / "snap.ini"
    snap.write_text("[DEFAULT]\n")
    monkeypatch.setenv("FLUXPYPE_CONFIG_FILE", str(snap))
    assert ph.find_config_file("config.ini") == str(snap)


def test_env_var_missing_file_is_an_error(ph, tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXPYPE_CONFIG_FILE", str(tmp_path / "gone.ini"))
    with pytest.raises(FileNotFoundError, match="FLUXPYPE_CONFIG_FILE"):
        ph.find_config_file("config.ini")


def test_unset_falls_back_to_shipped_file(ph, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    found = Path(ph.find_config_file("config.ini"))
    assert found == REPO / "fluxpype" / "config.ini"


def test_configurations_reads_the_snapshot(ph, tmp_path, monkeypatch):
    snap = tmp_path / "snap.ini"
    snap.write_text(
        "[DEFAULT]\nconfig_name = s\nbase_dir = auto\nrc_path = ~/.zshrc\nfl_mhdlib = ${base_dir}/\n"
        "data_dir = %s\nbatch_name = snapbatch\nrotations = [2300, 2301]\nfluxon_count = [50]\n"
        "adapts = [0]\nflow_method = [parker]\n[s]\n" % (tmp_path / "d")
    )
    monkeypatch.setenv("FLUXPYPE_CONFIG_FILE", str(snap))
    cfg = ph.configurations(debug=False)
    assert cfg["batch_name"] == "snapbatch"
    assert cfg["rotations"] == ["2300", "2301"]
    assert cfg["config_name"] == "s"


def test_log_file_override_is_append_mode(tmp_path, monkeypatch):
    from fluxpype import console

    target = tmp_path / "sub" / "x.log"
    target.parent.mkdir()
    target.write_text("old\n")
    monkeypatch.setenv("FLUXPYPE_LOG_FILE", str(target))
    monkeypatch.setattr(console, "_log_file", None)
    path = console.init_log("whatever")
    console.write_to_log("\x1b[31mnew\x1b[0m\n")
    console.close_log()
    assert path == target
    assert target.read_text() == "old\nnew\n"
