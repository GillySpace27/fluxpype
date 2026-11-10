"""
The job queue: turns jobs into tasks, starts them through a backend while respecting
a CPU budget and output locks, and reconciles their state.

``tick()`` is synchronous and idempotent; the server calls it from an asyncio loop, and
the tests call it directly. All state lives in the SQLite store and in each task's run
directory, so a restarted server simply keeps ticking: every task marked ``running`` is
polled on the next tick and reattached or finalized (the launcher keeps jobs alive while
the server is down).
"""

import json
import os
import shutil
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from . import config_io, paths
from .backends.base import Backend, Handle, TaskSpec
from .monitor import ProbeContext, TaskMonitor
from .store import FINISHED, Store

CANCEL_GRACE_S = 10.0


class SchedulerError(ValueError):
    pass


def _truthy(value: str) -> bool:
    return str(value).strip().lower() not in ("", "0", "false", "none", "no")


class Scheduler:
    def __init__(
        self,
        store: Store,
        backends: Dict[str, Backend],
        config_path,
        home=None,
        default_backend: str = None,
        cpu_budget: int = None,
        stale_after: float = 120.0,
        on_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.store = store
        self.backends = backends
        self.config_path = str(config_path)
        self.home = home or paths.studio_home()
        self.default_backend = default_backend or next(iter(backends))
        self.stale_after = stale_after
        self.on_change = on_change or (lambda kind, payload: None)
        self.lock = threading.RLock()
        self._monitors: Dict[str, TaskMonitor] = {}
        self._cancel_at: Dict[str, float] = {}
        self._features: Dict[str, List[str]] = {}
        if self.store.get_setting("cpu_budget") is None:
            self.store.set_setting("cpu_budget", cpu_budget or os.cpu_count() or 1)
        elif cpu_budget:
            self.store.set_setting("cpu_budget", cpu_budget)
        if self.store.get_setting("max_parallel") is None:
            self.store.set_setting("max_parallel", 1)
        if self.store.get_setting("paused") is None:
            self.store.set_setting("paused", False)

    # ------------------------------------------------------------------ config
    def config_text(self) -> str:
        with open(os.path.expanduser(self.config_path), "r", encoding="utf-8") as f:
            return f.read()

    def parsed(self) -> config_io.ConfigFile:
        return config_io.parse(self.config_text())

    def data_dir(self, profile: str = None) -> str:
        cf = self.parsed()
        eff = cf.effective(profile) if profile and profile in cf.sections else dict(cf.defaults)
        return config_io.resolved_path(eff, "data_dir", str(paths.checkout_root()))

    # ------------------------------------------------------------------ settings
    def queue_state(self) -> Dict[str, Any]:
        counts = self.store.counts()
        return {
            "paused": bool(self.store.get_setting("paused")),
            "max_parallel": int(self.store.get_setting("max_parallel")),
            "cpu_budget": int(self.store.get_setting("cpu_budget")),
            "running": counts.get("running", 0),
            "pending": counts.get("pending", 0),
            "slots_used": sum(t["slots"] for t in self.store.tasks_with_status("running")),
            "backends": {n: b.capabilities().__dict__ for n, b in self.backends.items()},
            "default_backend": self.default_backend,
        }

    def set_queue(self, paused=None, max_parallel=None, cpu_budget=None) -> Dict[str, Any]:
        with self.lock:
            if paused is not None:
                self.store.set_setting("paused", bool(paused))
            if max_parallel is not None:
                if not 1 <= int(max_parallel) <= 64:
                    raise SchedulerError("max_parallel must be between 1 and 64")
                self.store.set_setting("max_parallel", int(max_parallel))
            if cpu_budget is not None:
                if not 1 <= int(cpu_budget) <= 4096:
                    raise SchedulerError("cpu_budget must be between 1 and 4096")
                self.store.set_setting("cpu_budget", int(cpu_budget))
        state = self.queue_state()
        self.on_change("queue_state", state)
        self.tick()
        return state

    # ------------------------------------------------------------------ jobs
    def _features_for(self, backend: str) -> List[str]:
        if backend not in self._features:
            self._features[backend] = self.backends[backend].probe()
        return self._features[backend]

    def create_job(
        self,
        profile: str,
        overrides: Optional[Dict[str, str]] = None,
        backend: str = None,
        tasks: Optional[List[List[str]]] = None,
        extra: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Snapshot the config, expand the profile into tasks and queue them."""
        backend = backend or self.default_backend
        if backend not in self.backends:
            raise SchedulerError(f"unknown backend {backend!r}")
        missing = [f for f in ("only", "events", "config_file", "log_file", "task_timeout") if f not in self._features_for(backend)]
        if missing:
            raise SchedulerError(
                "the fluxpype install behind this backend is too old (missing: " + ", ".join(missing) + "); update fluxpype"
            )
        text = self.config_text()
        cf = config_io.parse(text)
        if profile not in cf.sections:
            raise SchedulerError(f"unknown profile {profile!r}")
        try:
            snapshot = config_io.make_snapshot(text, profile, overrides or {}, str(paths.checkout_root()))
        except config_io.ConfigError as e:
            raise SchedulerError(str(e)) from e
        eff = config_io.parse(snapshot).effective(profile)
        plan = config_io.expand_tasks(eff)
        if plan.errors:
            raise SchedulerError("; ".join(plan.errors))
        chosen = plan.tasks
        if tasks is not None:
            wanted = {tuple(str(x) for x in t) for t in tasks}
            chosen = [t for t in plan.tasks if (t[1], t[2], t[0], t[3]) in wanted]  # wanted: (cr, nflux, adapt, method)
            if not chosen:
                raise SchedulerError("none of the requested tasks are in this profile")
        job_id = "j" + uuid.uuid4().hex[:8]
        job_dir = os.path.join(paths.runs_dir(self.home), job_id)
        os.makedirs(job_dir, exist_ok=True)
        snap_path = os.path.join(job_dir, "config.snapshot.ini")
        with open(snap_path, "w", encoding="utf-8") as f:
            f.write(snapshot)
        data_dir = config_io.resolved_path(eff, "data_dir", str(paths.checkout_root()))
        batch = config_io.resolve(eff, "batch_name")
        slots = self._slots(eff)
        rows = []
        for i, (adapt, cr, nflux, method) in enumerate(chosen, 1):
            tid = f"{job_id}-{i:02d}"
            inst = config_io.task_instrument(adapt)
            keys = [f"out:{data_dir}|{batch}|{cr}", f"cache:{data_dir}|{cr}|{inst}"]
            rows.append(
                {
                    "id": tid,
                    "idx": i,
                    "cr": cr,
                    "nflux": nflux,
                    "adapt": adapt,
                    "method": method,
                    "run_dir": os.path.join(job_dir, tid),
                    "lock_key": "\n".join(keys),
                    "slots": slots,
                }
            )
        job = {
            "id": job_id,
            "profile": profile,
            "overrides": overrides or {},
            "backend": backend,
            "created": time.time(),
            "snapshot_path": snap_path,
            "extra": extra or {},
        }
        self.store.add_job(job, rows)
        self.on_change("job_state", self.job_view(job_id))
        self.tick()
        return self.job_view(job_id)

    @staticmethod
    def _slots(eff) -> int:
        def num(key):
            try:
                return int(float(config_io.resolve(eff, key) or 0))
            except ValueError:
                return 0

        return max(num("relax_concurrency"), num("concurrency"), 1)

    def job_view(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.store.get_job(job_id)
        if not job:
            return None
        tasks = self.store.tasks_for_job(job_id)
        job["status"] = self.store.derive_job_status(job_id)
        job["tasks"] = [self.task_row_view(t) for t in tasks]
        job["counts"] = {s: sum(1 for t in tasks if t["status"] == s) for s in ("pending", "running", *FINISHED)}
        return job

    @staticmethod
    def task_row_view(t: Dict[str, Any]) -> Dict[str, Any]:
        keep = ("id", "job_id", "idx", "cr", "nflux", "adapt", "method", "status", "exit_code", "created", "started", "ended", "slots", "attempt")
        return {k: t[k] for k in keep}

    def cancel_job(self, job_id: str) -> None:
        with self.lock:
            for t in self.store.tasks_for_job(job_id):
                self._cancel_task(t)
        self._publish_job(job_id)
        self.tick()

    def cancel_task(self, task_id: str) -> None:
        t = self.store.get_task(task_id)
        if not t:
            raise SchedulerError("no such task")
        with self.lock:
            self._cancel_task(t)
        self._publish_job(t["job_id"])
        self.tick()

    def _cancel_task(self, t: Dict[str, Any]) -> None:
        if t["status"] == "pending":
            self.store.update_task(t["id"], status="cancelled", ended=time.time())
        elif t["status"] == "running":
            job = self.store.get_job(t["job_id"])
            backend = self.backends[job["backend"]]
            self._cancel_at.setdefault(t["id"], time.time())
            backend.cancel(Handle(t["id"], t["run_dir"]))

    def delete_job(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if not job:
            raise SchedulerError("no such job")
        if any(t["status"] in ("pending", "running") for t in self.store.tasks_for_job(job_id)):
            raise SchedulerError("cancel the job before deleting it")
        self.store.delete_job(job_id)
        shutil.rmtree(os.path.dirname(job["snapshot_path"]), ignore_errors=True)
        self.on_change("job_removed", {"id": job_id})

    def retry_job(self, job_id: str) -> Dict[str, Any]:
        """Re-queue the job's failed, timed-out, cancelled and crashed tasks (old run kept aside)."""
        job = self.store.get_job(job_id)
        if not job:
            raise SchedulerError("no such job")
        with self.lock:
            n = 0
            for t in self.store.tasks_for_job(job_id):
                if t["status"] in ("failed", "timeout", "cancelled", "crashed"):
                    old = f"{t['run_dir']}.attempt{t['attempt']}"
                    if os.path.isdir(t["run_dir"]):
                        shutil.move(t["run_dir"], old)
                    self._monitors.pop(t["id"], None)
                    self._cancel_at.pop(t["id"], None)
                    self.store.update_task(
                        t["id"],
                        status="pending",
                        pid=None,
                        pid_start=None,
                        exit_code=None,
                        started=None,
                        ended=None,
                        summary_json=None,
                        attempt=t["attempt"] + 1,
                    )
                    n += 1
            if n == 0:
                raise SchedulerError("nothing to retry")
            self.store.update_job(job_id, status="pending", ended=None)
        self._publish_job(job_id)
        self.tick()
        return self.job_view(job_id)

    # ------------------------------------------------------------------ the loop
    def _handle(self, t: Dict[str, Any]) -> Handle:
        return Handle(t["id"], t["run_dir"])

    def _backend_for(self, t: Dict[str, Any]) -> Backend:
        return self.backends[self.store.get_job(t["job_id"])["backend"]]

    def tick(self) -> None:
        with self.lock:
            changed_jobs = set()
            changed_jobs |= self._reconcile_running()
            changed_jobs |= self._start_pending()
            for jid in changed_jobs:
                self._publish_job(jid)
            if changed_jobs:
                self.on_change("queue_state", self.queue_state())

    def _reconcile_running(self) -> set:
        changed = set()
        for t in self.store.tasks_with_status("running"):
            backend = self._backend_for(t)
            st = backend.poll(self._handle(t))
            if st.state in ("running", "starting"):
                if st.pid and not t["pid"]:
                    self.store.update_task(t["id"], pid=st.pid, pid_start=st.pid_start)
                if st.cancelled:
                    first = self._cancel_at.setdefault(t["id"], time.time())
                    if time.time() - first > CANCEL_GRACE_S:
                        backend.cancel(self._handle(t), force=True)
                continue
            now = time.time()
            if st.state == "exited":
                code = st.exit_code
                if st.cancelled:
                    status = "cancelled"
                elif code == 0:
                    status = "ok"
                elif code == 124:
                    status = "timeout"
                else:
                    status = "failed"
            else:  # lost
                status, code = ("cancelled" if st.cancelled else "crashed"), st.exit_code
            ended = st.ended or now
            self.store.update_task(t["id"], status=status, exit_code=code, ended=ended, pid=st.pid, pid_start=st.pid_start, started=t["started"] or st.started)
            self._cancel_at.pop(t["id"], None)
            final = self.task_state(t["id"], force_status=status)
            self.store.update_task(t["id"], summary_json=json.dumps(final))
            self.on_change("task_state", final)
            changed.add(t["job_id"])
        return changed

    def _running_keys(self) -> set:
        keys = set()
        for t in self.store.tasks_with_status("running"):
            keys.update(k for k in t["lock_key"].split("\n") if k)
        return keys

    def _start_pending(self) -> set:
        changed = set()
        if self.store.get_setting("paused"):
            return changed
        budget = int(self.store.get_setting("cpu_budget"))
        max_parallel = int(self.store.get_setting("max_parallel"))
        running = self.store.tasks_with_status("running")
        used = sum(min(t["slots"], budget) for t in running)
        held = self._running_keys()
        count = len(running)
        for t in self.store.tasks_with_status("pending"):
            if count >= max_parallel:
                break
            slots = min(t["slots"], budget)
            keys = {k for k in t["lock_key"].split("\n") if k}
            if keys & held:
                continue
            if count > 0 and used + slots > budget:
                continue
            self._launch(t)
            used += slots
            held |= keys
            count += 1
            changed.add(t["job_id"])
        return changed

    def _launch(self, t: Dict[str, Any]) -> None:
        job = self.store.get_job(t["job_id"])
        backend = self.backends[job["backend"]]
        timeout = self._timeout(job)
        spec = TaskSpec(
            task_id=t["id"],
            job_id=t["job_id"],
            run_dir=t["run_dir"],
            snapshot_path=job["snapshot_path"],
            profile=job["profile"],
            cr=t["cr"],
            nflux=t["nflux"],
            adapt=t["adapt"],
            method=t["method"],
            timeout_s=timeout,
            extra={k: str(v) for k, v in job["extra"].items()},
        )
        now = time.time()
        try:
            backend.submit(spec)
        except Exception as e:  # noqa: BLE001 - a launch failure must fail the task, not the loop
            os.makedirs(t["run_dir"], exist_ok=True)
            with open(os.path.join(t["run_dir"], "launcher.log"), "a") as f:
                f.write(f"submit failed: {e}\n")
            self.store.update_task(t["id"], status="failed", started=now, ended=now, exit_code=None)
            return
        self.store.update_task(t["id"], status="running", started=now)
        if job["status"] in ("pending",):
            self.store.update_job(job["id"], status="running", started=job["started"] or now)
        self.on_change("task_state", self.task_state(t["id"]))

    def _timeout(self, job) -> int:
        try:
            eff = config_io.read_file(job["snapshot_path"]).effective(job["profile"])
            return int(float(config_io.resolve(eff, "task_timeout") or 2000))
        except Exception:
            return 2000

    def _publish_job(self, job_id: str) -> None:
        view = self.job_view(job_id)
        if not view:
            return
        status = view["status"]
        fields = {"status": status}
        if status in ("ok", "failed", "cancelled", "partial") and not view["ended"]:
            fields["ended"] = time.time()
        if status == "running" and not view["started"]:
            fields["started"] = time.time()
        self.store.update_job(job_id, **fields)
        view = self.job_view(job_id)
        self.on_change("job_state", view)

    # ------------------------------------------------------------------ state for the UI
    def probe_context(self, t: Dict[str, Any]) -> Optional[ProbeContext]:
        job = self.store.get_job(t["job_id"])
        try:
            eff = config_io.read_file(job["snapshot_path"]).effective(job["profile"])
        except Exception:
            return None
        return ProbeContext(
            data_dir=config_io.resolved_path(eff, "data_dir", str(paths.checkout_root())),
            batch=config_io.resolve(eff, "batch_name"),
            cr=t["cr"],
            nflux=t["nflux"],
            adapt=t["adapt"],
            inst=config_io.task_instrument(t["adapt"]),
            reduce=config_io.resolve(eff, "mag_reduce") or "4",
            adapt_select=config_io.resolve(eff, "adapt_select") or "2",
        )

    def probe_context_for(self, profile: str, cr: str, nflux: str, adapt: str) -> Optional[ProbeContext]:
        """Probe context for a task of `profile` as the shipped config defines it (profile preview)."""
        try:
            eff = self.parsed().effective(profile)
        except Exception:
            return None
        return ProbeContext(
            data_dir=config_io.resolved_path(eff, "data_dir", str(paths.checkout_root())),
            batch=config_io.resolve(eff, "batch_name"),
            cr=cr,
            nflux=nflux,
            adapt=adapt,
            inst=config_io.task_instrument(adapt),
            reduce=config_io.resolve(eff, "mag_reduce") or "4",
            adapt_select=config_io.resolve(eff, "adapt_select") or "2",
        )

    def monitor_for(self, t: Dict[str, Any]) -> TaskMonitor:
        mon = self._monitors.get(t["id"])
        if mon is None:
            if len(self._monitors) > 40:
                for tid in [k for k, v in self._monitors.items() if self.store.get_task(k) and self.store.get_task(k)["status"] in FINISHED][:20]:
                    self._monitors.pop(tid, None)
            mon = TaskMonitor(t["run_dir"], self.probe_context(t), stale_after=self.stale_after)
            self._monitors[t["id"]] = mon
        return mon

    def task_state(self, task_id: str, force_status: str = None) -> Dict[str, Any]:
        t = self.store.get_task(task_id)
        if not t:
            raise SchedulerError("no such task")
        status = force_status or t["status"]
        finished = status in FINISHED
        if finished and t["summary_json"] and not force_status and task_id not in self._monitors:
            cached = json.loads(t["summary_json"])
            cached["status"] = status
            return cached
        mon = self.monitor_for(t)
        alive = None
        if status == "running":
            st = self._backend_for(t).poll(self._handle(t))
            alive = st.state in ("running", "starting")
        state = mon.refresh(
            alive=alive,
            outcome=status if finished else None,
            started=t["started"],
            ended=t["ended"],
        )
        job = self.store.get_job(t["job_id"])
        state.update(
            {
                "id": t["id"],
                "job_id": t["job_id"],
                "profile": job["profile"],
                "idx": t["idx"],
                "cr": t["cr"],
                "nflux": t["nflux"],
                "adapt": t["adapt"],
                "method": t["method"],
                "status": status,
                "started": t["started"],
                "ended": t["ended"],
                "exit_code": t["exit_code"],
                "slots": t["slots"],
                "attempt": t["attempt"],
                "elapsed_s": ((t["ended"] or time.time()) - t["started"]) if t["started"] else None,
            }
        )
        return state

    def relax_series(self, task_id: str, after: int = 0) -> Dict[str, Any]:
        t = self.store.get_task(task_id)
        if not t:
            raise SchedulerError("no such task")
        mon = self.monitor_for(t)
        mon.refresh(probe=False, alive=None, outcome=t["status"] if t["status"] in FINISHED else None, started=t["started"])
        return mon.relax_series(after)
