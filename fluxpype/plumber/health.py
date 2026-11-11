"""
Health of Plumber's parts, for the status indicators in the UI.

Every check returns ``{"id", "label", "status", "detail"}`` with status one of ``ok``, ``warn``,
``fail``, ``off`` (not set up) or ``unknown``. Backend checks run the job interpreter and Perl the
way a job would (same environment), so a green light means a job would get past imports. Results
are cached for ``TTL`` seconds because some checks start interpreters.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from . import config_io

TTL = 120.0
STARTED = time.time()
PLACEHOLDER_BACKENDS = [
    ("ssh", "SSH host", "Run on a workstation you can reach over SSH"),
    ("slurm", "Slurm cluster", "Submit tasks as batch jobs on an HPC queue"),
]


def check(id_: str, label: str, status: str, detail: str = "") -> Dict[str, str]:
    return {"id": id_, "label": label, "status": status, "detail": detail}


def _run(argv: List[str], timeout: float = 60.0, cwd: str = None, env: dict = None):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", f"{argv[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout:.0f} s"
    except OSError as e:
        return 126, "", str(e)


def _last_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1][:300] if lines else ""


def local_backend_checks(backend, sched) -> List[Dict[str, str]]:
    from . import paths
    from .backends.local import REQUIRED_FEATURES

    py = backend.python
    env = {**os.environ, **backend.environment_probe()} if hasattr(backend, "environment_probe") else dict(os.environ)
    env["PYTHONPATH"] = str(paths.checkout_root()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    out = []
    if not (os.path.isfile(py) and os.access(py, os.X_OK)):
        return [check("python", "Job Python", "fail", f"{py} is not an executable file")]
    code, so, se = _run([py, "-c", "import sys; print(sys.version.split()[0])"], 20, env=env)
    out.append(
        check("python", "Job Python", "ok" if code == 0 else "fail", f"{py} ({so})" if code == 0 else _last_line(se))
    )
    code, so, se = _run(
        [py, "-c", "import numpy, sunpy, pfsspy; print('numpy', numpy.__version__, 'sunpy', sunpy.__version__)"],
        90,
        cwd=str(paths.checkout_root()),
        env=env,
    )
    out.append(
        check(
            "science",
            "Science stack",
            "ok" if code == 0 else "fail",
            so if code == 0 else (_last_line(se) or "import failed") + f" (in {py})",
        )
    )
    features = backend.probe()
    missing = [f for f in REQUIRED_FEATURES if f not in features]
    out.append(
        check(
            "runner",
            "fluxpype runner",
            "ok" if not missing else "fail",
            "supports events, task selection and timeouts" if not missing else "too old; missing " + ", ".join(missing),
        )
    )
    perl = shutil.which("perl", path=env.get("PATH"))
    if not perl:
        out.append(check("perl", "Perl with PDL", "fail", "no perl on Plumber's PATH"))
    else:
        code, so, se = _run([perl, "-MPDL", "-e", "print $PDL::VERSION"], 30, env=env)
        out.append(
            check(
                "perl",
                "Perl with PDL",
                "ok" if code == 0 else "fail",
                f"PDL {so} ({perl})" if code == 0 else f"{perl} cannot load PDL: {(se.splitlines() or [''])[0][:200]}",
            )
        )
    return out


def backend_view(name: str, backend, sched, deep: bool) -> Dict[str, Any]:
    caps = backend.capabilities()
    kind = getattr(backend, "kind", name)
    if kind == "fake":
        checks = [check("sim", "Simulator", "ok", "built in; runs write to a sandbox, never your data folder")]
        label = "Simulated"
    else:
        checks = local_backend_checks(backend, sched) if deep else []
        label = "This computer" if kind == "local" else name
    worst = _worst([c["status"] for c in checks]) if checks else "unknown"
    return {
        "name": name,
        "kind": kind,
        "label": label,
        "description": caps.description,
        "slots": caps.slots,
        "configured": True,
        "active": name == sched.active_backend,
        "sandboxed": sched.is_sandboxed(name),
        "data_dir": sched.data_dir(backend=name),
        "status": worst,
        "checks": checks,
    }


def _worst(statuses: List[str]) -> str:
    for s in ("fail", "warn", "unknown", "ok"):
        if s in statuses:
            return s
    return "ok"


def system_checks(sched, version: str) -> List[Dict[str, str]]:
    out = [
        check(
            "server", "Plumber server", "ok", f"version {version}, pid {os.getpid()}, up {_dur(time.time() - STARTED)}"
        )
    ]
    q = sched.queue_state()
    out.append(
        check(
            "queue",
            "Queue",
            "warn" if q["paused"] else "ok",
            ("paused: nothing new starts. " if q["paused"] else "")
            + f"{q['running']} running, {q['pending']} queued, up to {q['max_parallel']} at once",
        )
    )
    try:
        cf = sched.parsed()
        broken = [n for n in cf.profiles if config_io.expand_tasks(cf.effective(n)).errors]
        out.append(
            check(
                "config",
                "Config and presets",
                "ok",  # presets that cannot run are flagged on their own tiles, not as a system fault
                f"{len(cf.profiles)} presets from {sched.config_path}"
                + (f"; {len(sched.presets.user_names())} yours" if sched.presets.user_names() else "")
                + (f"; {len(broken)} cannot run as written" if broken else ""),
            )
        )
    except Exception as e:  # noqa: BLE001
        out.append(check("config", "Config and presets", "fail", f"cannot read {sched.config_path}: {e}"))
    data = sched.data_dir()
    elsewhere = ""
    if not sched.is_sandboxed():
        from . import datadirs

        here = datadirs.describe(data)
        if not here["has_data"]:
            try:
                other = next(
                    (c for c in sched.data_dir_report()["candidates"] if c["has_data"] and c["real"] != here["real"]),
                    None,
                )
            except Exception:  # noqa: BLE001
                other = None
            if other:
                elsewhere = (
                    f"; no runs here, but {other['path']} holds {other['magnetograms']} magnetograms and "
                    f"{other['batches']} batches: choose it under Settings, Data folder"
                )
    if not os.path.isdir(data):
        out.append(
            check(
                "data",
                "Data folder",
                "warn",
                f"{data} does not exist yet" + (elsewhere or "; the first run creates it"),
            )
        )
    else:
        free = shutil.disk_usage(data).free
        writable = os.access(data, os.W_OK)
        status = "fail" if not writable else "warn" if free < 5e9 or elsewhere else "ok"
        sand = " (sandbox for simulated runs)" if sched.is_sandboxed() else ""
        out.append(
            check(
                "data",
                "Data folder",
                status,
                f"{data}{sand}; {free / 1e9:.0f} GB free" + ("" if writable else "; not writable") + elsewhere,
            )
        )
    return out


def _dur(s: float) -> str:
    s = int(s)
    return (
        f"{s // 86400}d {s % 86400 // 3600}h"
        if s >= 86400
        else f"{s // 3600}h {s % 3600 // 60}m" if s >= 3600 else f"{s // 60}m"
    )


def extras(home) -> List[Dict[str, str]]:
    import importlib.util

    out = []
    mcp = importlib.util.find_spec("mcp") is not None and importlib.util.find_spec("httpx") is not None
    out.append(
        check(
            "mcp",
            "AI agent link (MCP)",
            "ok" if mcp else "off",
            "MCP SDK installed" if mcp else 'not installed: pip install -e ".[mcp]"',
        )
    )
    if sys.platform == "darwin":
        from . import macapp

        rec = macapp.read_install_record(home)
        out.append(
            check(
                "login",
                "Start at login",
                "ok" if rec else "off",
                "launchd agent installed" if rec else "not set up: flux-plumb --install",
            )
        )
    return out


class Health:
    def __init__(self, sched, version: str):
        self.sched = sched
        self.version = version
        self._deep: Dict[str, Any] = {}
        self._at = 0.0
        self.lock = threading.Lock()

    def invalidate(self) -> None:
        with self.lock:
            self._at = 0.0

    def _deep_backends(self) -> Dict[str, List[Dict[str, str]]]:
        jobs = {n: b for n, b in self.sched.backends.items() if getattr(b, "kind", "") != "fake"}
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
            futs = {n: ex.submit(backend_view, n, b, self.sched, True) for n, b in jobs.items()}
            return {n: f.result() for n, f in futs.items()}

    def report(self, refresh: bool = False) -> Dict[str, Any]:
        with self.lock:
            if refresh or time.time() - self._at > TTL:
                self._deep = self._deep_backends()
                self._at = time.time()
            deep, at = dict(self._deep), self._at
        backends = []
        for n, b in self.sched.backends.items():
            view = deep.get(n) or backend_view(n, b, self.sched, False)
            view = {**view, "active": n == self.sched.active_backend, "data_dir": self.sched.data_dir(backend=n)}
            backends.append(view)
        for name, label, desc in PLACEHOLDER_BACKENDS:
            backends.append(
                {
                    "name": name,
                    "kind": name,
                    "label": label,
                    "description": desc,
                    "slots": 0,
                    "configured": False,
                    "active": False,
                    "sandboxed": False,
                    "data_dir": None,
                    "status": "off",
                    "checks": [
                        check("setup", "Connection", "off", "not set up yet (coming with the cluster backends)")
                    ],
                }
            )
        system = system_checks(self.sched, self.version) + extras(self.sched.home)
        return {
            "active_backend": self.sched.active_backend,
            "backends": backends,
            "system": system,
            "checked_at": at,
            "overall": _worst(
                [c["status"] for c in system if c["status"] != "off"]
                + [next(b["status"] for b in backends if b["active"])]
            ),
        }
