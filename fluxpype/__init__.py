"""FLUXpype: Python wrapper and automation for the FLUX modeling framework.

The heavy science modules are imported lazily, so ``import fluxpype`` stays cheap
(the studio and the event helpers rely on that).
"""

import importlib

__all__ = ["config_runner", "pipe_helper", "run"]


def __getattr__(name):
    if name in ("config_runner", "pipe_helper"):
        return importlib.import_module(f".{name}", __name__)
    if name == "run":
        return importlib.import_module(".config_runner", __name__).run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
