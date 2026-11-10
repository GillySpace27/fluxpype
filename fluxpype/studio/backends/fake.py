"""A backend that runs ``fake_flux`` instead of the real pipeline (demo and tests)."""

import sys
from typing import List

from .base import TaskSpec
from .local import REQUIRED_FEATURES, LocalBackend

SCENARIOS = ("ok", "fail_in_pfss", "slow_relax", "no_events", "events_dropout", "cached_relax")


class FakeBackend(LocalBackend):
    name = "fake"
    kind = "fake"
    description = "Simulated runs (no FLUX needed)"

    def probe(self) -> List[str]:
        return list(REQUIRED_FEATURES)

    def command(self, task: TaskSpec) -> List[str]:
        scenario = task.extra.get("scenario", "ok")
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}")
        return [
            sys.executable,
            "-m",
            "fluxpype.studio.fake_flux",
            "--profile",
            task.profile,
            "--cr",
            task.cr,
            "--nflux",
            task.nflux,
            "--adapt",
            task.adapt,
            "--method",
            task.method,
            "--scenario",
            scenario,
            "--speed",
            str(task.extra.get("speed", "1")),
        ]
