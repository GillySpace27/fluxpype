"""
A fake ``perl`` for the runner tests (the real pipeline cannot run here).

Installed first on PATH by the ``fake_perl`` fixture. It answers the runner's
``perl -e "use PDL..."`` / ``"use Flux..."`` probes with ``ok``; given a script
path plus ``CR NFLUX ADAPT METHOD`` it replays a small fixture: log lines, a few
events (written by the real events.pm when a system perl is available), then
exits the way the scenario says.

Scenario selection: ``FAKE_PERL_MODES`` is ``default=ok,2226=fail`` style
(key = CR); modes are ``ok``, ``fail`` (exit 3), ``hang`` (never returns,
leaves a grandchild in the same process group).
"""

import json
import os
import subprocess
import sys
import time


def _mode(cr):
    modes = dict(p.split("=", 1) for p in os.environ.get("FAKE_PERL_MODES", "default=ok").split(",") if "=" in p)
    return modes.get(cr, modes.get("default", "ok"))


def _perl_event(kind, fields):
    perl = "/usr/bin/perl"
    root = os.environ.get("FAKE_PERL_REPO", "")
    if os.path.exists(perl) and root:
        code = "use fluxpype::events; emit_event(shift, {%s});" % ",".join(f"{k}=>{v!r}" for k, v in fields.items())
        subprocess.run([perl, "-I", root, "-e", code, kind], check=False)
    else:
        sys.path.insert(0, root)
        from fluxpype.events import emit

        emit(kind, **fields)


def main(argv):
    if argv and argv[0] == "-e":
        print("ok", end="")
        return 0
    script, cr, nflux, adapt, method = argv[:5]
    log = os.environ.get("FAKE_PERL_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps({"script": script, "args": [cr, nflux, adapt, method]}) + "\n")
    dump = os.environ.get("FAKE_PERL_DUMP")
    if dump:
        keep = {k: v for k, v in os.environ.items() if k.startswith("FLUXPYPE_")}
        with open(dump, "w") as f:
            json.dump(keep, f)

    print(f"(pdl) fake pipeline for CR{cr}", flush=True)
    if os.environ.get("FLUXPYPE_EVENTS"):
        _perl_event("stage.start", {"stage": "'relax'"})
    mode = _mode(cr)
    if mode == "hang":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        pidfile = os.environ.get("FAKE_PERL_PIDFILE")
        if pidfile:
            with open(pidfile, "w") as f:
                json.dump({"perl": os.getpid(), "child": child.pid}, f)
        print("hanging now", flush=True)
        time.sleep(600)
    if mode == "fail":
        print("get_pfss.py failed", flush=True)
        return 3
    print(f"Program Complete for CR{cr} in 1.0 Seconds", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
