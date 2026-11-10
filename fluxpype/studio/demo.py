"""Demo mode: a throwaway studio home, a sample config and a handful of simulated jobs."""

import os
import tempfile
from pathlib import Path

from . import config_io, paths

DEMO_PROFILES = """

#############################################
## Demo profiles (flux-studio --demo)

[Demo_Sweep]
batch_name = demo_sweep
rotations = [2210, 2218, 2225]
fluxon_count = [300]
adapts = [0]
flow_method = [parker]
concurrency = 8
relax_concurrency = 8

[Demo_Methods]
batch_name = demo_methods
rotations = [2225]
fluxon_count = [500]
adapts = [0, 1]
flow_method = [parker, wsa]
concurrency = 6
relax_concurrency = 6
light = 1

[Demo_WindOnly]
batch_name = demo_sweep
rotations = (2200, 2230, 10)
fluxon_count = [300]
adapts = [0]
flow_method = [parker, ghosts]
wind_only = 1
concurrency = 4
relax_concurrency = 4
"""


def prepare_home() -> Path:
    """Create the temp studio home and return it (also exported as FLUX_STUDIO_HOME)."""
    home = Path(tempfile.mkdtemp(prefix="flux-studio-demo-"))
    os.environ["FLUX_STUDIO_HOME"] = str(home)
    data = home / "data"
    data.mkdir()
    text = paths.shipped_config().read_text(encoding="utf-8")
    text = config_io.set_value(text, "DEFAULT", "data_dir", str(data))
    text = config_io.set_value(text, "DEFAULT", "jsoc_email", "demo@example.com")
    (home / "config.ini").write_text(text.rstrip("\n") + "\n" + DEMO_PROFILES, encoding="utf-8")
    return home


def seed(scheduler) -> None:
    """Queue a mix of finished-fast history and live-speed jobs so every screen has something to show."""
    fast = {"speed": "40"}
    # history (fast): an ok run, a two-task ok run, and a failure
    scheduler.create_job("quickstart", backend="fake", extra={**fast, "scenario": "ok"})
    scheduler.create_job("TestConfig", backend="fake", extra={**fast, "scenario": "ok"})
    scheduler.create_job("wsa", backend="fake", extra={**fast, "scenario": "fail_in_pfss"})
    scheduler.create_job("light", backend="fake", extra={**fast, "scenario": "cached_relax"})
    # live
    scheduler.create_job("Demo_Sweep", backend="fake", extra={"speed": "1", "scenario": "ok"})
    scheduler.create_job("Demo_Methods", backend="fake", extra={"speed": "1", "scenario": "events_dropout"})
    scheduler.create_job("Demo_WindOnly", backend="fake", extra={"speed": "1", "scenario": "ok"})
