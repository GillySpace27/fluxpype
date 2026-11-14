import os
import shutil
import subprocess
import time

import pytest

psutil = pytest.importorskip("psutil")
from fluxpype.plumber.sysmon import external_flux


@pytest.mark.skipif(not shutil.which("perl"), reason="needs perl")
def test_external_flux_lists_perl_flux_runs_outside_plumber_groups():
    p = subprocess.Popen(["perl", "-e", "sleep 30", "world.flux"], start_new_session=True)
    try:
        time.sleep(0.3)
        assert p.pid in [r["pid"] for r in external_flux({})]
        assert p.pid not in [r["pid"] for r in external_flux({os.getpgid(p.pid): "job"})]
    finally:
        p.kill()
