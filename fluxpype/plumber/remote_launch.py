"""Launcher entry for remote backends: run from inside the run folder.

``python -m fluxpype.plumber.remote_launch .`` is ``fluxpype.plumber.launcher``
with one difference: path-valued entries in launch.json's ``env`` that the job
script already exported as absolute paths are not overwritten by the relative
ones, because the runner changes directory to ``cwd`` (the checkout) before it
opens them.
"""

import json
import os
import sys

from . import launcher

ABSOLUTE = ("FLUXPYPE_CONFIG_FILE", "FLUXPYPE_EVENTS", "FLUXPYPE_LOG_FILE")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    run_dir = os.path.abspath(argv[0])
    path = os.path.join(run_dir, "launch.json")
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    env = spec.get("env", {})
    for key in ABSOLUTE:
        if key in env and not os.path.isabs(env[key]):
            env[key] = os.path.join(run_dir, env[key])
    spec["argv"] = [os.path.join(run_dir, a) if a == "scene.json" else a for a in spec["argv"]]
    cwd = spec.get("cwd") or ""
    spec["cwd"] = os.path.expanduser(cwd) if cwd else None
    launcher.write_json_atomic(path, spec)
    return launcher.main([run_dir])


if __name__ == "__main__":
    sys.exit(main())
