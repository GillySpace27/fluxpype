"""
The job queue: turns jobs into tasks, starts them through a backend while respecting
a CPU budget and output locks, and reconciles their state.

``tick()`` is synchronous and idempotent; the server calls it from an asyncio loop, and
the tests call it directly. All state lives in the SQLite store and in each task's run
directory, so a restarted server simply keeps ticking: every task marked ``running`` is
polled on the next tick and reattached or finalized (the launcher keeps jobs alive while
the server is down).
"""

import dataclasses
import glob
import json
import os
import re
import shutil
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from . import config_io, notify, paths
from .backends.base import Backend, Handle, TaskSpec
from .monitor import ProbeContext, TaskMonitor
from .store import FINISHED, Store

CANCEL_GRACE_S = 10.0


class SchedulerError(ValueError):
    pass


class SchedulerConflict(SchedulerError):
    """The request is fine but not now (the API answers 409)."""


RUN_ID_RE = re.compile(r"^p\d{14}[0-9a-f]{4}$")


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
        presets=None,
    ):
        self.store = store
        self.backends = backends
        self.config_path = str(config_path)
        self.home = home or paths.plumber_home()
        self.default_backend = default_backend or next(iter(backends))
        from .presets import PresetStore

        self.presets = presets or PresetStore(self.home, self.config_path, store)
        self.stale_after = stale_after
        self.on_change = on_change or (lambda kind, payload: None)
        self.lock = threading.RLock()
        self._monitors: Dict[str, TaskMonitor] = {}
        self._cancel_at: Dict[str, float] = {}
        self._features: Dict[str, List[str]] = {}
        self._plotruns: Dict[str, Dict[str, Any]] = {}  # active post-hoc plot runs by id
        self.backend_unknown: Dict[str, float] = {}  # backend -> when it first answered `unknown` (notify.py)
        self.poll_state: Dict[str, str] = {}  # running task -> the state its last poll returned
        self._find_plot_runs()
        if self.store.get_setting("cpu_budget") is None:
            self.store.set_setting("cpu_budget", cpu_budget or os.cpu_count() or 1)
        elif cpu_budget:
            self.store.set_setting("cpu_budget", cpu_budget)
        if self.store.get_setting("max_parallel") is None:
            self.store.set_setting("max_parallel", 1)
        if self.store.get_setting("paused") is None:
            self.store.set_setting("paused", False)
        if default_backend or self.store.get_setting("active_backend") not in self.backends:
            self.store.set_setting("active_backend", self.default_backend)

    # ------------------------------------------------------------------ config
    def config_text(self) -> str:
        """config.ini plus the user's presets: what previews read and what job snapshots are made from."""
        return self.presets.combined_text()

    def parsed(self) -> config_io.ConfigFile:
        return config_io.parse(self.config_text())

    # ------------------------------------------------------------------ backends
    @property
    def active_backend(self) -> str:
        name = self.store.get_setting("active_backend")
        return name if name in self.backends else self.default_backend

    def set_active_backend(self, name: str) -> Dict[str, Any]:
        if name not in self.backends:
            raise SchedulerError(f"unknown backend {name!r}")
        self.store.set_setting("active_backend", name)
        state = self.queue_state()
        self.on_change("queue_state", state)
        return state

    def is_sandboxed(self, backend: str = None) -> bool:
        """Simulated backends never write to the real data directory."""
        b = self.backends.get(backend or self.active_backend)
        return bool(b is not None and getattr(b, "kind", "") == "fake")

    def sandbox_dir(self) -> str:
        path = os.path.join(str(self.home), "sandbox", "data")
        os.makedirs(path, exist_ok=True)
        return path

    def data_dir_setting(self) -> Optional[str]:
        """The data folder chosen in Plumber for real runs, or None to use config.ini's data_dir."""
        val = self.store.get_setting("data_dir")
        return str(val) if val else None

    def set_data_dir(self, path: Optional[str]) -> Optional[str]:
        """Pin real runs to `path` (an existing folder), or None to go back to config.ini's data_dir."""
        if path is None or not str(path).strip():
            self.store.set_setting("data_dir", None)
            return None
        full = os.path.abspath(os.path.expanduser(str(path).strip()))
        if not os.path.isdir(full):
            raise SchedulerError(f"{full} is not a folder")
        if os.path.realpath(full) == os.path.realpath(os.path.join(str(self.home), "sandbox", "data")):
            raise SchedulerError("that is the sandbox for simulated runs; choose your real data folder")
        self.store.set_setting("data_dir", full)
        return full

    def pinned_data_dir(self, backend: str = None) -> Optional[str]:
        """The folder a job on `backend` must write to regardless of config.ini, if any."""
        if self.is_sandboxed(backend):
            return self.sandbox_dir()
        return self.data_dir_setting()

    def config_data_dir(self) -> str:
        """data_dir as config.ini's [DEFAULT] gives it."""
        return config_io.resolved_path(dict(self.parsed().defaults), "data_dir", str(paths.checkout_root()))

    def data_dir_report(self) -> Dict[str, Any]:
        """Where real runs read and write, why, and other folders on this machine that hold data."""
        from . import datadirs

        chosen = self.data_dir_setting()
        from_config = self.config_data_dir()
        current = chosen or from_config
        return {
            "current": datadirs.describe(current),
            "source": "plumber" if chosen else "config",
            "config_value": from_config,
            "candidates": datadirs.candidates(
                config_io.auto_base_dir(str(paths.checkout_root())), [current, from_config]
            ),
        }

    def data_dir(self, profile: str = None, backend: str = None) -> str:
        pinned = self.pinned_data_dir(backend)
        if pinned:
            return pinned
        cf = self.parsed()
        eff = cf.effective(profile) if profile and profile in cf.sections else dict(cf.defaults)
        return config_io.resolved_path(eff, "data_dir", str(paths.checkout_root()))

    def all_data_dirs(self) -> List[str]:
        dirs = {self.sandbox_dir()}
        if self.data_dir_setting():
            dirs.add(self.data_dir_setting())
        try:
            cf = self.parsed()
            dirs.add(config_io.resolved_path(dict(cf.defaults), "data_dir", str(paths.checkout_root())))
            for prof in cf.profiles:
                dirs.add(config_io.resolved_path(cf.effective(prof), "data_dir", str(paths.checkout_root())))
        except Exception:  # noqa: BLE001 - a broken config must not hide the sandbox
            pass
        return sorted(dirs)

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
            "active_backend": self.active_backend,
        }

    @staticmethod
    def check_queue(max_parallel=None, cpu_budget=None) -> None:
        if max_parallel is not None and not 1 <= int(max_parallel) <= 64:
            raise SchedulerError("max_parallel must be between 1 and 64")
        if cpu_budget is not None and not 1 <= int(cpu_budget) <= 4096:
            raise SchedulerError("cpu_budget must be between 1 and 4096")

    def set_queue(self, paused=None, max_parallel=None, cpu_budget=None) -> Dict[str, Any]:
        self.check_queue(max_parallel, cpu_budget)
        with self.lock:
            if paused is not None:
                self.store.set_setting("paused", bool(paused))
            if max_parallel is not None:
                self.store.set_setting("max_parallel", int(max_parallel))
            if cpu_budget is not None:
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
        base_text: Optional[str] = None,
        dry_run: bool = False,
        origin: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Snapshot the config (`base_text`, else the preset file as it is now), expand the profile into tasks and queue them.

        `dry_run` runs every check and returns {"task_count": n} without writing or queueing anything."""
        backend = backend or self.active_backend
        if backend not in self.backends:
            raise SchedulerError(f"unknown backend {backend!r}")
        missing = [
            f
            for f in ("only", "events", "config_file", "log_file", "task_timeout")
            if f not in self._features_for(backend)
        ]
        if missing:
            raise SchedulerError(
                "the fluxpype install behind this backend is too old (missing: "
                + ", ".join(missing)
                + "); update fluxpype"
            )
        text = base_text or self.config_text()
        cf = config_io.parse(text)
        if profile not in cf.sections:
            raise SchedulerError(f"unknown profile {profile!r}")
        try:
            snapshot = config_io.make_snapshot(text, profile, overrides or {}, str(paths.checkout_root()))
        except config_io.ConfigError as e:
            raise SchedulerError(str(e)) from e
        # simulated runs write into the sandbox, never the real data_dir; real runs go to the folder chosen in Plumber
        pinned = self.pinned_data_dir(backend)
        if pinned:
            snapshot = config_io.set_value(snapshot, "DEFAULT", "data_dir", pinned)
            if "data_dir" in config_io.parse(snapshot).sections.get(profile, {}):
                snapshot = config_io.set_value(snapshot, profile, "data_dir", pinned)
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
        if dry_run:
            return {"task_count": len(chosen)}
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
            "origin": origin or {},  # who started it: a routing label built by the server, never an authority
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
        keep = (
            "id",
            "job_id",
            "idx",
            "cr",
            "nflux",
            "adapt",
            "method",
            "status",
            "exit_code",
            "created",
            "started",
            "ended",
            "slots",
            "attempt",
        )
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
        self._no_plot_run_for(job_id)
        self.store.delete_job(job_id)
        self.store.ack_events(job=job_id)  # its inbox rows would point nowhere
        shutil.rmtree(os.path.dirname(job["snapshot_path"]), ignore_errors=True)
        self.on_change("job_removed", {"id": job_id})

    def retry_job(self, job_id: str, by: Optional[str] = None) -> Dict[str, Any]:
        """Re-queue the job's failed, timed-out, cancelled and crashed tasks (old run kept aside).

        `by` (web, mcp, ask) is recorded in the job's origin as `retried_by`."""
        job = self.store.get_job(job_id)
        if not job:
            raise SchedulerError("no such job")
        self._no_plot_run_for(job_id)
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
                        notified=None,  # the next attempt's events are announced even if it ends before a pass
                    )
                    n += 1
            if n == 0:
                raise SchedulerError("nothing to retry")
            fields: Dict[str, Any] = {"status": "pending", "ended": None, "notified": None}
            if by:
                fields["origin_json"] = json.dumps({**job["origin"], "retried_by": by})
            self.store.update_job(job_id, **fields)
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
            self._reconcile_plot_runs()
            changed_jobs |= self._start_pending()
            for jid in changed_jobs:
                self._publish_job(jid)
            if changed_jobs:
                self.on_change("queue_state", self.queue_state())

    def _reconcile_running(self) -> set:
        changed = set()
        known = set()
        running = self.store.tasks_with_status("running")
        for tid in [k for k in self.poll_state if k not in {t["id"] for t in running}]:
            self.poll_state.pop(tid, None)
        for t in running:
            backend = self._backend_for(t)
            st = backend.poll(self._handle(t))
            self.poll_state[t["id"]] = st.state
            bname = self.store.get_job(t["job_id"])["backend"]
            if st.state == "unknown":  # the backend cannot tell now: never finalize, flag the backend
                self.backend_unknown.setdefault(bname, time.time())
                continue
            known.add(bname)
            if st.state in ("running", "starting", "queued"):
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
                end = None if st.cancelled else notify.last_run_end(t["run_dir"])
                if end and end.get("status") in ("ok", "failed", "timeout"):
                    # only the launcher died; the runner said how it ended
                    status, code = end["status"], end.get("exit_code")
            ended = st.ended or now
            self.store.update_task(
                t["id"],
                status=status,
                exit_code=code,
                ended=ended,
                pid=st.pid,
                pid_start=st.pid_start,
                started=t["started"] or st.started,
            )
            self._cancel_at.pop(t["id"], None)
            final = self.task_state(t["id"], force_status=status)
            if status == "ok" and any(s["status"] == "failed" for s in final["stages"]):
                # the process exited 0 but a stage failed (the runner can swallow the error): never call that success
                status = "failed"
                self.store.update_task(t["id"], status=status)
                final = self.task_state(t["id"], force_status=status)
            if st.state == "lost" and end:
                final["launcher_lost"] = True
            self.store.update_task(t["id"], summary_json=json.dumps(final))
            self.on_change("task_state", final)
            changed.add(t["job_id"])
        for name in [b for b in self.backend_unknown if b in known]:
            self.backend_unknown.pop(name, None)
        return changed

    def _running_keys(self) -> set:
        keys = set()
        for t in self.store.tasks_with_status("running"):
            keys.update(k for k in t["lock_key"].split("\n") if k)
        self._reconcile_plot_runs()
        keys.update(r["key"] for r in self._plotruns.values())
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

    def probe_context_for(
        self, profile: str, cr: str, nflux: str, adapt: str, backend: str = None, parsed=None
    ) -> Optional[ProbeContext]:
        """Probe context for a task of `profile` as the config defines it (preset preview)."""
        try:
            eff = (parsed or self.parsed()).effective(profile)
        except Exception:
            return None
        return ProbeContext(
            data_dir=(
                self.pinned_data_dir(backend) or config_io.resolved_path(eff, "data_dir", str(paths.checkout_root()))
            ),
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
                for tid in [
                    k
                    for k, v in self._monitors.items()
                    if self.store.get_task(k) and self.store.get_task(k)["status"] in FINISHED
                ][:20]:
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
        if finished and t["summary_json"] and not force_status:  # a finished task's stages never change afterwards
            cached = json.loads(t["summary_json"])
            cached["status"] = status
            return cached
        mon = self.monitor_for(t)
        alive = None
        if status == "running":
            st = self._backend_for(t).poll(self._handle(t))
            alive = None if st.state in ("queued", "unknown") else st.state in ("running", "starting")
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
        mon.refresh(
            probe=False, alive=None, outcome=t["status"] if t["status"] in FINISHED else None, started=t["started"]
        )
        return mon.relax_series(after)

    # ------------------------------------------------------------------ post-hoc plot runs
    def _find_plot_runs(self) -> None:
        """After a restart: plot runs whose process has not written an exit code are still active."""
        for launch in glob.glob(os.path.join(str(paths.runs_dir(self.home)), "*", "*", "plots", "p*", "launch.json")):
            d = os.path.dirname(launch)
            rid = os.path.basename(d)
            if os.path.exists(os.path.join(d, "exit_code")) or not RUN_ID_RE.match(rid):
                continue
            try:
                req = json.load(open(os.path.join(d, "request.json"), encoding="utf-8"))
                c = req["ctx"]
                key = f"out:{c['data_dir']}|{c['batch']}|{c['cr']}"
                self._plotruns[rid] = {"task_id": req["task_id"], "job_id": req["job_id"], "dir": d, "key": key}
            except (OSError, ValueError, KeyError):
                continue

    def _no_plot_run_for(self, job_id: str) -> None:
        with self.lock:
            self._reconcile_plot_runs()  # "active" is decided now, not at the last tick
            busy = any(r["job_id"] == job_id for r in self._plotruns.values())
        if busy:
            raise SchedulerConflict("a plot run for this job is still going; cancel it first")

    def _plot_dir(self, task_id: str, rid: str) -> str:
        t = self.store.get_task(task_id)
        if not t:
            raise SchedulerError("no such task")
        d = os.path.join(t["run_dir"], "plots", rid)
        if not RUN_ID_RE.match(rid) or not os.path.isfile(os.path.join(d, "request.json")):
            raise SchedulerError("no such plot run")
        return d

    def list_plot_runs(self, task_id: str) -> List[Dict[str, Any]]:
        from . import plotrun

        t = self.store.get_task(task_id)
        if not t:
            raise SchedulerError("no such task")
        dirs = glob.glob(os.path.join(t["run_dir"], "plots", "p*", "request.json"))
        return [plotrun.read_view(os.path.dirname(p)) for p in sorted(dirs, reverse=True)]

    def plot_run(self, task_id: str, rid: str) -> Dict[str, Any]:
        from . import plotrun

        return plotrun.read_view(self._plot_dir(task_id, rid))

    def plot_run_dir(self, task_id: str, rid: str) -> str:
        return self._plot_dir(task_id, rid)

    def rerun_options(self, task_id: str) -> Dict[str, Any]:
        """For the two regenerate panels: each editable setting with the value this job ran with, and whether it can run now."""
        from . import plotrun
        from .plotcatalog import PLOTS

        with self.lock:
            t = self.store.get_task(task_id)
            if not t:
                raise SchedulerError("no such task")
            job = self.store.get_job(t["job_id"])
            ctx = self.probe_context(t)
            if ctx is None:
                raise SchedulerError("the job's configuration snapshot is unreadable")
            eff = config_io.read_file(job["snapshot_path"]).effective(job["profile"])
            _, missing = plotrun.resolve_needs(next(e for e in PLOTS if e["id"] == "light_triplot"), ctx)
        out: Dict[str, Any] = {}
        for group, fields in plotrun.SETTINGS.items():
            rows = []
            for f in fields:
                cur = t["method"] if f["key"] == "flow_method" else config_io.resolve(eff, f["key"])
                rows.append({**f, "value": (cur or "").strip().strip('"[]') or f["default"]})
            out[group] = {"fields": rows, "can": not missing, "missing": missing}
        return out

    def rerun_wind(self, task_id: str, settings: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Queue a new job that redoes only the wind and wind plots of a finished task, in the same batch.

        It runs the job's own frozen config with wind_only=1 (the pipeline's way of skipping everything before wind) and
        light=0; the earlier stages' files are read, not rebuilt. The old wind files of the same method are replaced."""
        from . import plotrun
        from .plotcatalog import PLOTS

        clean = plotrun.clean_settings("wind", settings)
        with self.lock:
            t = self.store.get_task(task_id)
            if not t:
                raise SchedulerError("no such task")
            if t["status"] not in FINISHED:
                raise SchedulerConflict("the task has not finished")
            job = self.store.get_job(t["job_id"])
            ctx = self.probe_context(t)
            if ctx is None:
                raise SchedulerError("the job's configuration snapshot is unreadable")
            _, missing = plotrun.resolve_needs(next(e for e in PLOTS if e["id"] == "light_triplot"), ctx)
            if missing:
                raise SchedulerError("the relaxed world this wind is made from is missing: " + ", ".join(missing))
            with open(job["snapshot_path"], encoding="utf-8") as f:
                text = f.read()
        method = clean.get("flow_method", "").strip("[]") or t["method"]
        extra = {k: v for k, v in job["extra"].items() if k == "speed"}
        extra["from_task"] = task_id
        # set_value adds a key the profile lacks; the overrides argument only accepts keys it already has
        for k, v in {"wind_only": "1", "light": "0", **clean}.items():
            text = config_io.set_value(text, job["profile"], k, v)
        return self.create_job(
            job["profile"], {}, job["backend"], [[t["cr"], t["nflux"], t["adapt"], method]], extra, base_text=text
        )

    def cancel_plot_run(self, task_id: str, rid: str) -> Dict[str, Any]:
        d = self._plot_dir(task_id, rid)
        job = self.store.get_job(self.store.get_task(task_id)["job_id"])
        self.backends[job["backend"]].cancel(Handle(rid, d))
        return self.plot_run(task_id, rid)

    def posthoc_outputs(self, t: Dict[str, Any]) -> Dict[str, str]:
        """{data-dir-relative path: plot run id} for plots made afterwards, the newest run winning."""
        out: Dict[str, str] = {}
        for view in reversed(self.list_plot_runs(t["id"])):
            for p in view["plots"]:
                if p["status"] == "ok":
                    out.update({rel: view["id"] for rel in p["outputs"]})
        return out

    def _reconcile_plot_runs(self) -> None:
        from . import plotrun

        for rid, r in list(self._plotruns.items()):
            view = plotrun.read_view(r["dir"])
            if view != r.get("view"):
                r["view"] = view
                self.on_change("plot_run", view)
            if view["status"] not in ("starting", "running"):
                self._plotruns.pop(rid, None)

    def start_plot_run(
        self,
        task_id: str,
        plots,
        stage: Optional[str] = None,
        overwrite: bool = False,
        dry_run: bool = False,
        include_legacy: bool = False,
        settings: Optional[Dict[str, str]] = None,
    ):
        """Remake plots of a finished task from the files on disk. Never changes the task's or job's status.

        `settings` (plotrun.SETTINGS) are written into this run's own copy of the job's config, so the job's snapshot stays as it ran.
        """
        from . import plotrun
        from .plotcatalog import PLOTS

        with self.lock:
            t = self.store.get_task(task_id)
            if not t:
                raise SchedulerError("no such task")
            job = self.store.get_job(t["job_id"])
            backend = self.backends[job["backend"]]
            ctx = self.probe_context(t)
            if ctx is None:
                raise SchedulerError("the job's configuration snapshot is unreadable")
            if not hasattr(backend, "python"):
                raise SchedulerError("plot runs need a backend that runs on this machine")
            by_id = {e["id"]: e for e in PLOTS}
            try:
                if plots == "all_possible":
                    ids = []
                    for e in PLOTS:
                        mates = [m for m in PLOTS if m["group"] == e["group"]] if e["group"] else [e]
                        if (
                            e["post_hoc"]
                            and (include_legacy or not e["legacy"])
                            and (stage is None or e["stage"] == stage)
                            and plotrun.generatable(e, ctx)["can_generate"]
                            and not any(plotrun.targets(m, ctx) for m in mates)
                        ):
                            ids.append(e["id"])
                else:
                    if not isinstance(plots, list) or not plots:
                        raise SchedulerError("nothing to generate")
                    ids = [i for i in plots if isinstance(i, str)]
                    for i in ids:  # a group is one command, so it makes (and would replace) all its members
                        g = by_id.get(i, {}).get("group")
                        ids += [m["id"] for m in PLOTS if g and m["group"] == g and m["id"] not in ids]
                ids = list(dict.fromkeys(ids))
                if not ids:
                    raise SchedulerError("nothing to generate")
                items = plotrun.plan(ctx, ids, backend.python)
                groups = {by_id[i].get("settings") for i in ids}
                if settings and (len(groups) != 1 or None in groups):
                    raise plotrun.PlotRunError("these plots take no settings")
                clean = plotrun.clean_settings(groups.pop(), settings) if settings else {}
            except plotrun.PlotRunError as e:
                raise SchedulerError(str(e)) from e
            later = {i: plotrun.generatable(by_id[i], ctx, t["ended"])["inputs_later"] for i in ids}
            if dry_run:
                return [
                    {
                        "id": it["id"],
                        "title": by_id[it["id"]]["title"],
                        "argv": it["argv"],
                        "missing": it["missing"],
                        "conflicts": it["conflicts"],
                        "inputs_later": later[it["id"]],
                        "overwrite_risk": by_id[it["id"]]["overwrite_risk"],
                    }
                    for it in items
                ]
            if t["status"] not in FINISHED:
                raise SchedulerConflict("the task has not finished")
            self._reconcile_plot_runs()
            if any(r["task_id"] == task_id for r in self._plotruns.values()):
                raise SchedulerConflict("a plot run for this task is already going")
            key = f"out:{ctx.data_dir}|{ctx.batch}|{ctx.cr}"
            if any(key in x["lock_key"].split("\n") for x in self.store.tasks_with_status("running")):
                raise SchedulerConflict("a running task is writing to this output; wait for it to finish")
            missing = [f"{it['id']}: {', '.join(it['missing'])}" for it in items if it["missing"]]
            if missing:
                raise SchedulerError("missing inputs: " + "; ".join(missing))
            clash = [c for it in items for c in it["conflicts"]] + [f for v in later.values() for f in v]
            if clash and not overwrite:
                raise SchedulerConflict("would replace or use changed files: " + ", ".join(dict.fromkeys(clash)))
            rid = "p" + time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]
            d = os.path.join(t["run_dir"], "plots", rid)
            os.makedirs(d, exist_ok=True)
            snap_text = open(job["snapshot_path"], encoding="utf-8").read()
            for k, v in (
                ("rotations", t["cr"]),
                ("fluxon_count", t["nflux"]),
                ("adapts", t["adapt"]),
                ("flow_method", t["method"]),
            ):
                snap_text = config_io.set_value(snap_text, job["profile"], k, f"[{v}]")
            for k, v in clean.items():
                snap_text = config_io.set_value(snap_text, job["profile"], k, v)
            snap = os.path.join(d, "config.snapshot.ini")
            with open(snap, "w", encoding="utf-8") as f:
                f.write(snap_text)
            fake = backend.kind == "fake"
            req = {
                "task_id": task_id,
                "job_id": t["job_id"],
                "plots": ids,
                "overwrite": bool(overwrite),
                "settings": {k: v.strip('"[]') for k, v in clean.items()},  # as shown to the user
                "fake": fake,
                "python": backend.python,
                "created": time.time(),
                "ctx": dataclasses.asdict(ctx),
            }
            with open(os.path.join(d, "request.json"), "w", encoding="utf-8") as f:
                json.dump(req, f)
            root = str(paths.checkout_root())
            env = {
                "FLUXPYPE_CONFIG": job["profile"],
                "FLUXPYPE_CONFIG_FILE": snap,
                "FLUXPYPE_EVENTS": os.path.join(d, "events.jsonl"),
                "FLUXPYPE_TASK_ID": task_id,
                "PYTHONPATH": root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
                "PYTHONUNBUFFERED": "1",
                "MPLBACKEND": "Agg",
                "PATH": os.path.dirname(backend.python) + os.pathsep + os.environ.get("PATH", ""),
            }
            from .backends.local import spawn

            spawn(d, [backend.python, "-m", "fluxpype.plumber.plotrun", d], env, root)
            self._plotruns[rid] = {"task_id": task_id, "job_id": t["job_id"], "dir": d, "key": key}
        view = plotrun.read_view(d)
        self._plotruns[rid]["view"] = view
        self.on_change("plot_run", view)
        return view
