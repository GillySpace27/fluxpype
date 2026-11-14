"""External runs: work that happens outside Plumber's own queue but should still be visible in it.

A script, a terminal session or another Claude writes one small JSON file per run into
`<plumber home>/external_runs/<id>.json`. Plumber and the Ask agent (Plum) only read these files; nothing here
starts, stops or manages a process. Fields (all optional except `id`):

    id        short slug, letters digits - _ . only
    title     one line
    status    running | ok | failed | cancelled | note
    host      where it runs
    cwd       working directory
    pid       process id, for a liveness check on the same machine
    started, updated   ISO timestamps (updated is set on every write)
    log       path to a log file; the last lines are served on request
    report    path to a result or report file
    summary   a sentence or two of what it found so far
    progress  list of {"label": str, "value": number or str}
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
STATUSES = ("running", "ok", "failed", "cancelled", "note")
KEYS = ("title", "status", "host", "cwd", "pid", "started", "log", "report", "summary", "progress")
MAX_RUNS = 500


def runs_dir(home) -> Path:
    return Path(home) / "external_runs"


def valid_id(run_id: str) -> bool:
    return bool(ID_RE.match(run_id or ""))


def _clean(body: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in KEYS:
        v = body.get(k)
        if v is None:
            continue
        if k == "status" and v not in STATUSES:
            raise ValueError("status must be one of " + ", ".join(STATUSES))
        if k == "pid":
            v = int(v)
        elif k == "progress":
            if not isinstance(v, list) or len(v) > 50:
                raise ValueError("progress must be a list of at most 50 rows")
            v = [{"label": str(r.get("label", ""))[:80], "value": r.get("value")} for r in v if isinstance(r, dict)]
        else:
            v = str(v)[:2000]
        out[k] = v
    return out


def write(home, run_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create or update a run (fields not in `body` keep their value) and return it."""
    if not valid_id(run_id):
        raise ValueError("id must be 1 to 64 characters: letters, digits, dot, dash, underscore")
    d = runs_dir(home)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_id}.json"
    cur = _read(path) or {"id": run_id, "started": _now()}
    cur.update(_clean(body))
    cur["id"] = run_id
    cur["updated"] = _now()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=1))
    os.replace(tmp, path)
    return view(cur)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and valid_id(str(data.get("id", ""))) else None


def _alive(pid: Any) -> Optional[bool]:
    """True/False for a pid on this machine, None when unknown."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def view(run: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(run)
    alive = _alive(run.get("pid"))
    if alive is not None:
        out["pid_alive"] = alive
    return out


def get(home, run_id: str) -> Optional[Dict[str, Any]]:
    if not valid_id(run_id):
        return None
    run = _read(runs_dir(home) / f"{run_id}.json")
    return view(run) if run else None


def list_runs(home, limit: int = 100) -> List[Dict[str, Any]]:
    d = runs_dir(home)
    if not d.is_dir():
        return []
    rows = []
    for p in d.glob("*.json"):
        run = _read(p)
        if run:
            rows.append(view(run))
    rows.sort(key=lambda r: str(r.get("updated", "")), reverse=True)
    return rows[: min(max(limit, 1), MAX_RUNS)]


def delete(home, run_id: str) -> bool:
    if not valid_id(run_id):
        return False
    try:
        (runs_dir(home) / f"{run_id}.json").unlink()
    except OSError:
        return False
    return True


def log_tail(run: Dict[str, Any], lines: int = 40) -> Optional[str]:
    """Last lines of the run's log file, or None if it has none or cannot be read."""
    path = run.get("log")
    if not path:
        return None
    try:
        with open(os.path.expanduser(path), "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    return "\n".join(data.splitlines()[-max(1, min(lines, 400)) :])
