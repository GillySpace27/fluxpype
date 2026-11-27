"""A5-02: config-only callers must not pay for sunpy.net, matplotlib.pyplot and pandas."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HEAVY = ("sunpy.net", "matplotlib.pyplot", "pandas", "drms")


def _loaded_heavy(statement):
    code = f"import sys; {statement}; " f"print(sorted(m for m in {HEAVY!r} if m in sys.modules))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_light_config_module_skips_heavy_imports():
    assert _loaded_heavy("from fluxpype.config import configurations") == "[]"


@pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None for m in ("numpy", "astropy")),
    reason="pipe_helper needs numpy and astropy at import",
)
def test_pipe_helper_import_skips_heavy_imports():
    assert _loaded_heavy("from fluxpype.pipe_helper import configurations") == "[]"


def test_pipe_helper_reexports_config_names(fake_science):
    from fluxpype import config, pipe_helper

    for name in ("configurations", "convert_value", "get_flux_perl_lib", "update_magdir_paths", "find_config_file"):
        assert getattr(pipe_helper, name) is getattr(config, name)
