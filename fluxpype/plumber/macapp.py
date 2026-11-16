"""Run Plumber like an app on macOS: a launchd agent keeps it running at its fixed address.

`flux-plumb --install` writes ~/Library/LaunchAgents/<LABEL>.plist and loads it. launchd does not read
shell profiles, so the agent records the PATH and PERL5LIB of the shell that ran --install (the one in
which FLUX works). <home>/install.json keeps what was installed, for --status and --uninstall.
The Dock icon is the browser's own "Add to Dock" / "Install app" for the Plumber page: nothing here.
"""

import json
import os
import plistlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LABEL = "com.gillyspace27.fluxpype.plumber"
CAPTURED_ENV = ("PATH", "PERL5LIB", "PDLLIB", "FLUX_PLUMBER_PYTHON", "LANG")


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def install_record_path(home: Path) -> Path:
    return Path(home) / "install.json"


def read_install_record(home: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(install_record_path(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def installed_python(home: Path) -> Optional[str]:
    """The job interpreter recorded by --install, so a hand-started Plumber for the same home matches."""
    rec = read_install_record(home)
    py = rec.get("python") if rec else None
    return py if isinstance(py, str) and os.access(py, os.X_OK) else None


def agent_argv(python: str, home: Path, port: int, job_python: Optional[str], config: Optional[str]) -> List[str]:
    argv = [python, "-m", "fluxpype.plumber.cli", "--headless", "--home", str(home), "--port", str(port)]
    if job_python:
        argv += ["--python", job_python]
    if config:
        argv += ["--config", config]
    return argv


def build_plist(argv: List[str], env: Dict[str, str], workdir: str, home: Path) -> bytes:
    return plistlib.dumps(
        {
            "Label": LABEL,
            "ProgramArguments": argv,
            "EnvironmentVariables": env,
            "WorkingDirectory": workdir,
            "RunAtLoad": True,
            # restart after a crash, not after a clean stop (flux-plumb --stop) or "already running" (exit 3)
            "KeepAlive": {"SuccessfulExit": False},
            "ThrottleInterval": 30,
            "ProcessType": "Interactive",
            "StandardOutPath": str(Path(home) / "launchd.log"),
            "StandardErrorPath": str(Path(home) / "launchd.log"),
        }
    )


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _wait_for_server(home: Path, timeout: float = 20.0) -> Optional[dict]:
    from .cli import read_server_json
    from .launcher import process_alive

    end = time.time() + timeout
    while time.time() < end:
        info = read_server_json(home)
        if info and process_alive(info.get("pid"), info.get("pid_start")):
            return info
        time.sleep(0.3)
    return None


def install(args, home: Path) -> int:
    from . import paths
    from .cli import DEMO_PORT, REAL_PORT, browser_url, lock_is_held, stop_running

    if sys.platform != "darwin":
        print("--install sets up a macOS launchd agent; on Linux use a systemd user service", file=sys.stderr)
        return 2
    if args.demo:
        print("the demo is not installed as a login item; start it by hand with flux-plumb --demo", file=sys.stderr)
        return 2
    port = args.port or REAL_PORT
    if port == DEMO_PORT:
        print(f"port {DEMO_PORT} is the demo's default; pick another", file=sys.stderr)
        return 2
    job_python = args.python or os.environ.get("FLUX_PLUMBER_PYTHON")
    if job_python:
        job_python = str(Path(job_python).expanduser().resolve())
        if not os.access(job_python, os.X_OK):
            print(f"--python {job_python} is not an executable file", file=sys.stderr)
            return 2
    else:
        print(
            "note: no --python given, so FLUX jobs will run on this interpreter "
            f"({sys.executable}); it needs the science stack",
            file=sys.stderr,
        )
    config = str(Path(args.config).expanduser().resolve()) if args.config else None
    env = {k: os.environ[k] for k in CAPTURED_ENV if os.environ.get(k)}
    env["FLUX_PLUMBER_HOME"] = str(home)
    argv = agent_argv(sys.executable, home, port, job_python, config)
    workdir = str(paths.checkout_root())

    # one Plumber per home: hand over from a hand-started one (its detached jobs keep running)
    if lock_is_held(home):
        print("stopping the Plumber that is running now; the agent takes over")
        if stop_running(home) != 0:
            print("could not stop it; stop it yourself and run --install again", file=sys.stderr)
            return 1
    _launchctl("bootout", f"{_domain()}/{LABEL}")  # an older version of the agent, if any
    target = plist_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(build_plist(argv, env, workdir, home))
    r = _launchctl("bootstrap", _domain(), str(target))
    if r.returncode != 0:
        print(f"launchctl bootstrap failed: {(r.stderr or r.stdout).strip()}", file=sys.stderr)
        return 1
    record = {
        "label": LABEL,
        "plist": str(target),
        "argv": argv,
        "python": job_python,
        "port": port,
        "env": env,
        "installed": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    install_record_path(home).write_text(json.dumps(record, indent=2), encoding="utf-8")
    info = _wait_for_server(home)
    if not info:
        print(f"the agent is loaded but Plumber did not come up; see {home}/launchd.log and {home}/server.log")
        return 1
    print(f"Plumber runs at {browser_url(info)} and starts at login.")
    print("Add it to the Dock: open that address, then Safari > File > Add to Dock, or Chrome > Install app.")
    print("Sign this browser in once with: flux-plumb --open")
    return 0


def uninstall(home: Path) -> int:
    if sys.platform != "darwin":
        print("--uninstall removes the macOS launchd agent; nothing to do here", file=sys.stderr)
        return 2
    r = _launchctl("bootout", f"{_domain()}/{LABEL}")
    removed = False
    if plist_path().exists():
        plist_path().unlink()
        removed = True
    try:
        install_record_path(home).unlink()
    except OSError:
        pass
    if r.returncode != 0 and not removed:
        print("no Plumber agent was installed")
        return 1
    print("removed the Plumber agent; Plumber has stopped (running jobs keep running)")
    return 0


def status_lines(home: Path) -> List[str]:
    from .cli import browser_url, read_server_json
    from .launcher import process_alive

    out = []
    info = read_server_json(home)
    if info and process_alive(info.get("pid"), info.get("pid_start")):
        kind = "demo" if info.get("demo") else "real"
        out.append(f"running: {browser_url(info)} ({kind}, pid {info.get('pid')}, since {info.get('started')})")
    else:
        out.append(f"not running for {home}")
    rec = read_install_record(home)
    if sys.platform == "darwin":
        loaded = _launchctl("print", f"{_domain()}/{LABEL}").returncode == 0
        out.append(f"launchd agent: {'loaded' if loaded else 'not loaded'} ({plist_path()})")
    if rec:
        out.append(f"installed {rec.get('installed')}: jobs run {rec.get('python') or 'on the Plumber interpreter'}")
        missing = [p for p in [rec.get("argv", [None])[0], rec.get("python")] if p and not os.access(p, os.X_OK)]
        for p in missing:
            out.append(f"WARNING: {p} no longer exists; rerun flux-plumb --install")
        path_dirs = (rec.get("env") or {}).get("PATH", "").split(os.pathsep)
        perl = next((str(Path(d) / "perl") for d in path_dirs if d and (Path(d) / "perl").is_file()), None)
        out.append(f"perl for jobs: {perl or 'not found on the recorded PATH'}")
    else:
        out.append("not installed as a login item (flux-plumb --install)")
    return out


def print_status(home: Path) -> int:
    for line in status_lines(home):
        print(line)
    return 0
