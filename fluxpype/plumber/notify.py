"""Plumber's event log producer and the one-line push format (stdlib only).

The ``Notifier`` runs as a step of the server loop whether or not a browser is connected. ``observe()`` turns store
transitions (each row's ``status`` against its ``notified`` column) into outcome events; ``telemetry()`` adds stall,
resume and degraded events for running tasks. Events go into the store's ``events`` table (``INSERT OR IGNORE`` on a
unique key, so one outcome per attempt is a database constraint) and out through ``publish``.

``format_line`` is the push contract shared by ``watch`` and the MCP ``get_notifications`` tool: ``key=value`` pairs,
ASCII, one line, built only from validated enums, ids and integers. No log text, paths or detail ever reach it.
"""

import json
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional

from .monitor import STAGE_IDS

FINISHED = ("ok", "failed", "timeout", "cancelled", "crashed")
JOB_END = ("ok", "partial", "failed", "cancelled")
JOB_STATUSES = ("pending", "running") + JOB_END
SEVS = ("info", "success", "warn", "error")

# kind -> severity
KINDS = {
    "job_queued": "info",
    "task_started": "info",
    "task_ok": "success",
    "task_failed": "error",
    "task_timeout": "error",
    "task_crashed": "error",
    "task_cancelled": "info",
    "job_ok": "success",
    "job_partial": "error",
    "job_failed": "error",
    "job_cancelled": "info",
    "task_stalled": "warn",
    "task_resumed": "info",
    "task_degraded": "warn",
    "queue_paused": "info",
    "queue_resumed": "info",
    "backend_down": "error",
    "backend_up": "info",
    "plumber_started": "info",
}
# kinds the web inbox lists (web/src/lib/live.tsx keeps the same set)
INBOX = {
    "task_failed",
    "task_timeout",
    "task_crashed",
    "job_ok",
    "job_partial",
    "job_failed",
    "task_stalled",
    "backend_down",
}
TASK_OUTCOME = {
    "ok": "task_ok",
    "failed": "task_failed",
    "timeout": "task_timeout",
    "crashed": "task_crashed",
    "cancelled": "task_cancelled",
}
JOB_OUTCOME = {"ok": "job_ok", "partial": "job_partial", "failed": "job_failed", "cancelled": "job_cancelled"}
# what `watch` prints by default
WATCH_KINDS = set(TASK_OUTCOME.values()) | set(JOB_OUTCOME.values())
WATCH_KINDS |= {"task_stalled", "task_resumed", "backend_down", "backend_up", "plumber_started"}

ID_RE = re.compile(r"^[a-z0-9]{6,24}$")  # origin session, tab and call ids
JOB_RE = re.compile(r"^j[0-9a-f]{8}$")
TASK_RE = re.compile(r"^j[0-9a-f]{8}-\d{2,4}$")
BACKEND_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
VERSION_RE = re.compile(r"^[0-9A-Za-z.+_-]{1,40}$")
TELEMETRY_GAP_S = 30.0  # a longer gap between telemetry passes means the machine slept or the loop starved


def _int(lo: int, hi: int) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


# every `f` field a line may carry, with its check; anything else is dropped and flagged `bad=1`
FIELDS = {
    "attempt": _int(1, 9999),
    "exit": _int(-1024, 1024),
    "stage": lambda v: v in STAGE_IDS,
    "reason": lambda v: v in ("exit", "submit", "stage"),
    "launcher_lost": _int(0, 1),
    "tasks": _int(0, 1_000_000),
    "backend": lambda v: isinstance(v, str) and bool(BACKEND_RE.match(v)),
    "ok": _int(0, 1_000_000),
    "failed": _int(0, 1_000_000),
    "cancelled": _int(0, 1_000_000),
    "timeout": _int(0, 1_000_000),
    "crashed": _int(0, 1_000_000),
    "running": _int(0, 1_000_000),
    "pending": _int(0, 1_000_000),
    "quiet_s": _int(0, 100_000_000),
    "limit_s": _int(0, 100_000_000),
    "version": lambda v: isinstance(v, str) and bool(VERSION_RE.match(v)),
    "reattached": _int(0, 1_000_000),
}
FIELD_ORDER = list(FIELDS)


def clean_id(value: Any) -> Optional[str]:
    """`value` if it is a valid origin id, else None (dropped silently)."""
    return value if isinstance(value, str) and ID_RE.match(value) else None


def _origin(job: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    o = (job or {}).get("origin") or {}
    out = {"kind": o["kind"]} if o.get("kind") in ("web", "mcp", "ask") else {}
    out.update({k: o[k] for k in ("session", "tab") if clean_id(o.get(k))})
    return out


def _ev(kind: str, key: str, job_id=None, task_id=None, origin=None, **f) -> Dict[str, Any]:
    return {
        "kind": kind,
        "sev": KINDS[kind],
        "key": key,
        "job_id": job_id,
        "task_id": task_id,
        "origin": origin or {},
        "f": {k: v for k, v in f.items() if v is not None},
    }


def submit_failed(row: Dict[str, Any]) -> bool:
    """The backend refused to start the task (scheduler._launch): failed with no process and no exit code."""
    return row.get("status") == "failed" and row.get("exit_code") is None and not row.get("pid")


def task_transition(notified: Optional[str], row: Dict[str, Any], job: Optional[Dict[str, Any]] = None) -> List[dict]:
    """Events for a task whose store status moved from `notified` to `row["status"]`.

    `row` is a tasks row, optionally with `stage` (first failed stage) and `launcher_lost` added by the caller."""
    status = row.get("status")
    if status == notified:
        return []
    tid, jid = row["id"], row["job_id"]
    a = int(row.get("attempt") or 1)
    origin = _origin(job)
    out = []
    # a task that ran and finished between two passes still gets its start (the key makes this idempotent)
    caught_up = status in FINISHED and notified in (None, "pending") and row.get("started") and not submit_failed(row)
    if status == "running" or caught_up:
        out.append(_ev("task_started", f"task_started|{tid}|{a}", jid, tid, origin, attempt=a))
    if status in FINISHED:
        code = row.get("exit_code")
        f: Dict[str, Any] = {"attempt": a}
        if status in ("ok", "failed", "timeout") and isinstance(code, int):
            f["exit"] = code
        if status == "failed":
            f["reason"] = "submit" if submit_failed(row) else ("stage" if code == 0 else "exit")
            if row.get("stage") in STAGE_IDS:
                f["stage"] = row["stage"]
        if row.get("launcher_lost"):
            f["launcher_lost"] = 1
        out.append(_ev(TASK_OUTCOME[status], f"task_end|{tid}|{a}", jid, tid, origin, **f))
    return out


def job_counts(tasks: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    tasks = list(tasks)
    return {
        s: sum(1 for t in tasks if t.get("status") == s) for s in ("ok", "failed", "cancelled", "timeout", "crashed")
    }


def job_transition(notified: Optional[str], job: Dict[str, Any], tasks: List[Dict[str, Any]]) -> List[dict]:
    """Events for a job whose store status moved from `notified` (None: never seen) to `job["status"]`."""
    status = job.get("status")
    if status == notified:
        return []
    jid, origin = job["id"], _origin(job)
    out = []
    if notified is None:
        out.append(
            _ev("job_queued", f"job_queued|{jid}", jid, None, origin, tasks=len(tasks), backend=job.get("backend"))
        )
    if status in JOB_END:
        r = max([int(t.get("attempt") or 1) for t in tasks] or [1])
        out.append(_ev(JOB_OUTCOME[status], f"job_end|{jid}|{r}", jid, None, origin, **job_counts(tasks)))
    return out


# ---------------------------------------------------------------------------------------------- the push format


def _utc(t: Any) -> Optional[str]:
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t < 1e11:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def format_line(ev: Dict[str, Any]) -> str:
    """One ``flux seq=.. kind=..`` line. Every value is checked here; a failing field is left out and `bad=1` added."""
    parts, bad = ["flux"], False

    def put(name, value, ok):
        nonlocal bad
        if ok:
            parts.append(f"{name}={value}")
        else:
            bad = True

    seq = ev.get("seq")
    put("seq", seq, _int(0, 10**15)(seq))
    t = _utc(ev.get("t"))
    put("t", t, t is not None)
    kind, sev = ev.get("kind"), ev.get("sev")
    put("kind", kind, kind in KINDS)
    put("sev", sev, sev in SEVS)
    job, task = ev.get("job_id"), ev.get("task_id")
    job_ok = isinstance(job, str) and bool(JOB_RE.match(job))
    task_ok = isinstance(task, str) and bool(TASK_RE.match(task))
    if job is not None:
        put("job", job, job_ok)
    if task is not None:
        put("task", task, task_ok)
    f = ev.get("f") if isinstance(ev.get("f"), dict) else {}
    if ev.get("f") is not None and not isinstance(ev.get("f"), dict):
        bad = True
    for k in FIELD_ORDER:
        if k in f:
            put(k, f[k], FIELDS[k](f[k]))
    if any(k not in FIELDS for k in f):
        bad = True
    if kind in KINDS and KINDS[kind] == "error" and kind.startswith("task_") and task_ok:
        parts.append(f"next=get_task:{task}")
    elif kind in JOB_OUTCOME.values() and job_ok:
        parts.append(f"next=get_job:{job}")
    if bad:
        parts.append("bad=1")
    return " ".join(parts)


def format_state(view: Dict[str, Any], seq: Any) -> str:
    """``flux state job=.. status=..`` for one job view (`GET /api/jobs/{id}`), with the event cursor `seq`."""
    parts, bad = ["flux", "state"], False
    job, status = view.get("id"), view.get("status")
    if isinstance(job, str) and JOB_RE.match(job):
        parts.append(f"job={job}")
    else:
        bad = True
    if status in JOB_STATUSES:
        parts.append(f"status={status}")
    else:
        bad = True
    tasks = view.get("tasks") if isinstance(view.get("tasks"), list) else []
    c = view.get("counts") if isinstance(view.get("counts"), dict) else {}
    ints = {
        "tasks": len(tasks),
        "ok": c.get("ok", 0),
        "failed": sum(c.get(k, 0) for k in ("failed", "timeout", "crashed") if isinstance(c.get(k, 0), int)),
        "cancelled": c.get("cancelled", 0),
        "running": c.get("running", 0),
        "pending": c.get("pending", 0),
        "seq": seq,
    }
    for k, v in ints.items():
        if _int(0, 10**15)(v):
            parts.append(f"{k}={v}")
        else:
            bad = True
    if bad:
        parts.append("bad=1")
    return " ".join(parts)


def format_done(code: int, ok: int, other: int, seq: Any, job: Optional[str] = None) -> str:
    """The last line `watch` prints: its exit code and how the watched jobs ended."""
    parts = ["flux", "done", f"exit={int(code)}", f"ok={int(ok)}", f"other={int(other)}"]
    if _int(0, 10**15)(seq):
        parts.append(f"seq={seq}")
    if isinstance(job, str) and JOB_RE.match(job):
        parts.append(f"next=get_job:{job}")
    return " ".join(parts)


# ---------------------------------------------------------------------------------------------- run.end


def last_run_end(run_dir: str, limit: int = 65536) -> Optional[Dict[str, Any]]:
    """The last ``run.end`` record in the task's events.jsonl (its last 64 KB), or None."""
    try:
        with open(os.path.join(run_dir, "events.jsonl"), "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(f.tell() - limit, 0))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("type") == "run.end":
            return rec
    return None


# ---------------------------------------------------------------------------------------------- the producer


class Notifier:
    """Writes events into the store from transitions and telemetry; `publish(event)` sends each new one live."""

    def __init__(self, sched, publish: Callable[[Dict[str, Any]], None] = None, version: str = "dev", stall_after=None):
        self.sched, self.store = sched, sched.store
        self.publish = publish or (lambda ev: None)
        self.version = version
        self.stall_after = stall_after
        self._lock = threading.Lock()
        self._paused: Optional[bool] = None
        self._down: set = set()
        self._degraded: Dict[str, bool] = {}
        self._last_tel: Optional[float] = None
        self._woke_at = 0.0
        self._trimmed = 0.0

    def emit(self, ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.store.add_event(ev)
        if row is not None:
            try:
                self.publish(row)
            except Exception:  # noqa: BLE001 - a dead socket must not stop the log
                pass
        return row

    def log_id(self) -> str:
        lid = self.store.get_setting("events_log_id")
        if not lid:
            lid = uuid.uuid4().hex[:12]
            self.store.set_setting("events_log_id", lid)
        return lid

    def start(self) -> None:
        """Once per server start: log id, retention, and a `plumber_started` event."""
        self.log_id()
        self.store.trim_events()
        self._trimmed = time.time()
        reattached = len(self.store.tasks_with_status("running"))
        v = self.version if VERSION_RE.match(str(self.version)) else None
        self.emit(_ev("plumber_started", f"plumber_started|{int(time.time())}", version=v, reattached=reattached))

    # -- outcomes from store transitions
    def observe(self) -> None:
        with self._lock:
            for j in self.store._all("SELECT id FROM jobs WHERE notified IS NULL ORDER BY created"):
                job = self.store.get_job(j["id"])  # announce a new job before its tasks (the key dedupes later)
                if job is not None:
                    for ev in job_transition(None, {**job, "status": "pending"}, self.store.tasks_for_job(j["id"])):
                        self.emit(ev)
            for t in self.store._all("SELECT * FROM tasks WHERE notified IS NOT status ORDER BY created, idx"):
                job = self.store.get_job(t["job_id"])
                row = dict(t)
                try:
                    summary = json.loads(t["summary_json"]) if t["summary_json"] else {}
                except ValueError:
                    summary = {}
                row["stage"] = next(
                    (s.get("id") for s in summary.get("stages", []) if s.get("status") == "failed"),
                    summary.get("current_stage"),
                )
                row["launcher_lost"] = bool(summary.get("launcher_lost"))
                for ev in task_transition(t["notified"], row, job):
                    self.emit(ev)
                self.store._exec(
                    "UPDATE tasks SET notified=? WHERE id=? AND status=?", (t["status"], t["id"], t["status"])
                )
            for j in self.store._all(
                "SELECT id, status, notified FROM jobs WHERE notified IS NOT status ORDER BY created"
            ):
                job = self.store.get_job(j["id"])
                if job is None:
                    continue
                for ev in job_transition(j["notified"], job, self.store.tasks_for_job(j["id"])):
                    self.emit(ev)
                self.store._exec(
                    "UPDATE jobs SET notified=? WHERE id=? AND status=?", (j["status"], j["id"], j["status"])
                )
            paused = bool(self.store.get_setting("paused"))
            if self._paused is not None and paused != self._paused:
                n = self.store.count_events("queue|") + 1
                self.emit(_ev("queue_paused" if paused else "queue_resumed", f"queue|{n}"))
            self._paused = paused
            down = set(getattr(self.sched, "backend_unknown", {}) or {})
            for b in sorted(down - self._down):
                n = self.store.count_events(f"backend_down|{b}|") + 1
                self.emit(_ev("backend_down", f"backend_down|{b}|{n}", backend=b if BACKEND_RE.match(b) else None))
            for b in sorted(self._down - down):
                n = self.store.count_events(f"backend_up|{b}|") + 1
                self.emit(_ev("backend_up", f"backend_up|{b}|{n}", backend=b if BACKEND_RE.match(b) else None))
            self._down = down

    # -- stall, resume and degraded for running tasks
    def stall_limit(self, job: Dict[str, Any]) -> float:
        setting = self.store.get_setting("stall_after_s")
        if setting:
            return float(setting)
        if self.stall_after:
            return float(self.stall_after)
        return min(max(0.4 * self.sched._timeout(job), 120.0), 1800.0)

    @staticmethod
    def last_output(t: Dict[str, Any]) -> float:
        """Newest mtime of the files a running task writes (survives a Plumber restart); at least its start."""
        newest = t.get("started") or 0.0
        for name in ("events.jsonl", "run.log", "stdout.log"):
            try:
                newest = max(newest, os.path.getmtime(os.path.join(t["run_dir"], name)))
            except OSError:
                pass
        return newest

    def telemetry(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        with self._lock:
            if self._last_tel is not None and now - self._last_tel > TELEMETRY_GAP_S:
                self._woke_at = now  # slept: give output a full stall window before judging it quiet
            self._last_tel = now
            if now - self._trimmed > 3600:
                self.store.trim_events(now)
                self._trimmed = now
            running = self.store.tasks_with_status("running")
            for tid in [k for k in self._degraded if k not in {t["id"] for t in running}]:
                self._degraded.pop(tid, None)
            poll_state = getattr(self.sched, "poll_state", {}) or {}
            for t in running:
                if poll_state.get(t["id"]) == "queued":
                    continue
                job = self.store.get_job(t["job_id"])
                if job is None:
                    continue
                try:
                    state = self.sched.task_state(t["id"])
                except Exception:  # noqa: BLE001 - a task that vanished is the scheduler's business
                    continue
                stage = state.get("current_stage") if state.get("current_stage") in STAGE_IDS else None
                tid, a, origin = t["id"], int(t["attempt"] or 1), _origin(job)
                limit = self.stall_limit(job)
                quiet = now - self.last_output(t)
                stalls = self.store.count_events(f"task_stalled|{tid}|{a}|")
                resumes = self.store.count_events(f"task_resumed|{tid}|{a}|")
                if stalls > resumes:
                    if quiet <= limit:
                        key = f"task_resumed|{tid}|{a}|{resumes + 1}"
                        self.emit(_ev("task_resumed", key, t["job_id"], tid, origin, stage=stage))
                elif quiet > limit and now - self._woke_at >= limit:
                    key = f"task_stalled|{tid}|{a}|{stalls + 1}"
                    f = {"stage": stage, "quiet_s": int(quiet), "limit_s": int(limit)}
                    self.emit(_ev("task_stalled", key, t["job_id"], tid, origin, **f))
                degraded = bool(state.get("degraded"))
                prev = self._degraded.get(tid)
                n = self.store.count_events(f"task_degraded|{tid}|{a}|")
                if prev is None:
                    prev = n > 0
                if degraded and not prev:
                    self.emit(
                        _ev("task_degraded", f"task_degraded|{tid}|{a}|{n + 1}", t["job_id"], tid, origin, stage=stage)
                    )
                self._degraded[tid] = degraded
