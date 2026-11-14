#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""
This is a configuration file for pytest containing customizations and fixtures.

In VSCode, Code Coverage is recorded in config.xml. Delete this file to reset reporting.
"""

from __future__ import annotations

from typing import List

import pytest
from _pytest.nodes import Item

# test_methods.py is an unmodified project-template sample that imports a package
# (python_package) which does not exist in this repository, so it cannot be collected.
collect_ignore = ["test_methods.py"]


def pytest_collection_modifyitems(items: list[Item]):
    for item in items:
        if "spark" in item.nodeid:
            item.add_marker(pytest.mark.spark)
        elif "_int_" in item.nodeid:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def unit_test_mocks(monkeypatch: None):
    """Include Mocks here to execute all commands offline and fast."""
    pass


# ---------------------------------------------------------------------------
# Fluxpype Plumber fixtures


TEST_PROFILES = """

[t_one]
batch_name = tb_one
rotations = [2225]
fluxon_count = [100]
adapts = [0]
flow_method = [parker]
concurrency = 2
relax_concurrency = 2

[t_two]
batch_name = tb_two
rotations = [2225, 2226]
fluxon_count = [100]
adapts = [0]
flow_method = [parker, wsa]
concurrency = 2
relax_concurrency = 2

[t_big]
batch_name = tb_big
rotations = [2301, 2302, 2303]
fluxon_count = [100]
adapts = [0]
flow_method = [parker]
concurrency = 6
relax_concurrency = 6

[t_light]
batch_name = tb_light
rotations = [2310]
fluxon_count = [100]
adapts = [0]
flow_method = [parker]
light = 1
"""


@pytest.fixture
def plumber_home(tmp_path, monkeypatch):
    home = tmp_path / "plumber-home"
    home.mkdir()
    monkeypatch.setenv("FLUX_PLUMBER_HOME", str(home))
    return home


@pytest.fixture
def plumber_config(tmp_path, plumber_home):
    """A config.ini (shipped profiles plus test ones) whose data_dir lives in tmp_path."""
    from fluxpype.plumber import config_io, paths

    text = paths.shipped_config().read_text(encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    text = config_io.set_value(text, "DEFAULT", "data_dir", str(data))
    path = tmp_path / "config.ini"
    path.write_text(text.rstrip("\n") + "\n" + TEST_PROFILES, encoding="utf-8")
    return path


@pytest.fixture
def make_scheduler(plumber_home, plumber_config):
    """Factory for schedulers over one persistent store (call twice to simulate a server restart)."""
    from fluxpype.plumber import paths
    from fluxpype.plumber.backends.fake import FakeBackend
    from fluxpype.plumber.scheduler import Scheduler
    from fluxpype.plumber.store import Store

    made = []

    def make(**kw):
        store = Store(paths.db_path(plumber_home))
        sched = Scheduler(
            store,
            kw.pop("backends", {"fake": FakeBackend()}),
            plumber_config,
            home=plumber_home,
            cpu_budget=kw.pop("cpu_budget", 64),
            **kw,
        )
        made.append(store)
        return sched

    yield make
    for s in made:
        s.close()


def wait_for(sched, cond, timeout=40.0):
    """Tick the scheduler until `cond()` is true (fails the test on timeout)."""
    import time

    end = time.time() + timeout
    while time.time() < end:
        sched.tick()
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("timed out waiting for condition")


@pytest.fixture
def fake_science():
    """Run the test with the science stack faked (see tests/fake_science.py), then undo it."""
    from tests import fake_science as fs

    fs.install()
    yield fs
    fs.uninstall()
