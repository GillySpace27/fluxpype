"""
Stand-ins for the heavy science stack (numpy, pandas, sunpy, ...).

``fluxpype.pipe_helper`` imports all of it at module level, but the pieces the
tests exercise (``configurations``, ``find_config_file``) only need the standard
library. ``install()`` registers a meta-path finder that answers for those
packages with permissive dummy modules, so the real ``pipe_helper`` code can run
in an environment without the science dependencies.
"""

import importlib.abc
import importlib.machinery
import sys
import types
from unittest import mock

FAKE_ROOTS = {
    "numpy",
    "pandas",
    "matplotlib",
    "astropy",
    "sunpy",
    "drms",
    "bs4",
    "lxml",
    "zeep",
    "pfsspy",
    "sunkit_magex",
    "cv2",
    "scipy",
}


class _FakeModule(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = mock.MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FAKE_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _FakeModule(spec.name)

    def exec_module(self, module):
        pass


def install():
    """Idempotently put the fake finder at the end of sys.meta_path."""
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.append(_Finder())


def uninstall():
    """Remove the finder and every fake module (and anything imported on top of them)."""
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _Finder)]
    for name in list(sys.modules):
        if name.split(".")[0] in FAKE_ROOTS or name == "fluxpype.pipe_helper":
            del sys.modules[name]
    import fluxpype

    if hasattr(fluxpype, "pipe_helper"):
        del fluxpype.pipe_helper
