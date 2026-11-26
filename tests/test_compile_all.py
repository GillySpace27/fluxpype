"""Every module in the package must compile on the running Python.

CI runs 3.10, 3.11 and 3.12, but no test imports fluxpype/science/fluxlight.py, so a 3.12-only f-string there
went unnoticed (audit finding A7-06). Compiling everything catches the next one on every interpreter in the matrix.
"""

import compileall
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "fluxpype"


def test_package_compiles():
    assert compileall.compile_dir(str(PKG), quiet=1, force=True, legacy=False, workers=1, ddir=None)
