"""
Per-task state, built from three independent sources.

1. ``events.jsonl`` (primary): schema v1 lines, tailed by byte offset, complete lines only.
2. The log (``run.log``, ANSI stripped): regexes over the banners and progress lines the
   pipeline already prints. The patterns are written from the print statements in the
   sources, not from a captured real log (see the test docstrings).
3. File probes: the artifacts each stage should leave under ``data_dir``.

Every stage and the relaxation numbers carry a ``source`` of ``event``, ``log``,
``file`` or ``none``. Events win while they flow. When they go quiet while the log
keeps growing the task is ``degraded`` and log-derived fields take over. When an event
says a stage finished ok, its probe must agree or the stage gets ``verified: false``.
Stdlib only.
"""

import glob
import json
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .. import names
from .. import stages as stage_manifest

# the stage list, labels and log markers come from the manifest (A2-09)
STAGES = [(st.id, st.label) for st in stage_manifest.STAGES]
STAGE_IDS = stage_manifest.STAGE_IDS
LABELS = stage_manifest.LABELS

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"

# stage start markers in the log, in pipeline order (a later marker also ends earlier stages)
LOG_STARTS = stage_manifest.log_markers("log_start")
LOG_REUSED = stage_manifest.log_markers("log_reused")
LOG_FAILURES = stage_manifest.log_markers("log_failure") + [
    (None, re.compile(r"Error executing command.*")),
    (None, re.compile(r"PDL script failed with exit code \d+")),
]
RE_STEP = re.compile(r"^step (\d+): time is now")
RE_STEP_STATS = re.compile(
    r"\((\d+) vertices; stiffness=\s*"
    + _NUM
    + r"%, fmax=\s*"
    + _NUM
    + r", f_av=\s*"
    + _NUM
    + r", fs_av=\s*"
    + _NUM
    + r", n_av=\s*"
    + _NUM
)
RE_CYCLE = re.compile(r"Cycle (\d+), Cumulative relaxation time: " + _NUM + r" seconds, (\d+) steps")
RE_FIXES = re.compile(r"Cycle (\d+) :: Curvature changed (\d+) vertices, proximity changed (\d+) vertices")
RE_TARGET = re.compile(r"Relaxing PFSS model for (\d+) steps to " + _NUM + r" stiffness")
RE_DONE = re.compile(r"Program Complete for CR(\d+) in ([\d.]+) Seconds")


@dataclass
class ProbeContext:
    """Where this task's artifacts should appear (all values already resolved)."""

    data_dir: str
    batch: str
    cr: str
    nflux: str
    adapt: str
    inst: str
    reduce: str = "4"
    adapt_select: str = "2"

    def fields(self) -> Dict[str, str]:
        return stage_manifest.fields_for(
            self.data_dir, self.batch, self.cr, self.nflux, self.inst, self.reduce, self.adapt_select
        )

    def patterns(self) -> Dict[str, List[str]]:
        """Glob patterns per stage id that has probes (from the stage manifest)."""
        f = self.fields()
        return {s.id: stage_manifest.probes(s.id, f) for s in stage_manifest.STAGES if s.probes}

    def image_globs(self) -> List[str]:
        return [os.path.join(self.data_dir, "batches", self.batch, "data", f"cr{self.cr}", "**", "*.png")]


def _first_match(patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[-1]
    return None


def first_match_for_context(ctx: "ProbeContext") -> Dict[str, bool]:
    """Which cacheable intermediates already exist for a task (used by the profile preview)."""
    pats = ctx.patterns()
    return {
        "magnetogram": _first_match(pats["magnetogram"]) is not None,
        "footpoints": _first_match(pats["footpoints"]) is not None,
        "pfss": _first_match(pats["pfss"]) is not None,
        "world": _first_match(pats["world"]) is not None,
        "relaxed_world": _first_match(pats["relax"]) is not None,
    }


class _Tail:
    """Incremental line reader: remembers the offset, yields complete lines only."""

    def __init__(self, path: str):
        self.path = path
        self.offset = 0
        self.carry = b""
        self.size = 0

    def read(self) -> List[str]:
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read(8 * 1024 * 1024)
        except OSError:
            return []
        if not chunk:
            return []
        data = self.carry + chunk
        cut = max(data.rfind(b"\n"), data.rfind(b"\r"))
        if cut < 0:
            self.carry = data
            self.offset += len(chunk)
            return []
        self.carry = data[cut + 1 :]
        self.offset += len(chunk)
        self.size = self.offset
        text = data[: cut + 1].decode("utf-8", "replace")
        return [ln for ln in re.split(r"[\r\n]", text) if ln != ""]


class TaskMonitor:
    def __init__(
        self, run_dir: str, probe: Optional[ProbeContext] = None, stale_after: float = 120.0, min_stale: float = 10.0
    ):
        self.run_dir = run_dir
        # refresh() advances the tail offsets; two threads in it at once duplicate or drop lines (A3-01)
        # _lock guards the ingested lists (also taken by series readers); _refresh_lock serializes whole refreshes,
        # so a slow probe or artifact scan never holds up a chart read
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self.probe = probe
        self.stale_after = stale_after
        self.min_stale = min_stale
        self.events_path = os.path.join(run_dir, "events.jsonl")
        self.log_path = os.path.join(run_dir, "run.log")
        self._ev = _Tail(self.events_path)
        self._lg = _Tail(self.log_path)
        # events state
        self.ev_lines = 0
        self.ev_bad = 0
        self.ev_stage: Dict[str, Dict[str, Any]] = {}
        self.ev_run_start: Optional[Dict[str, Any]] = None
        self.ev_run_end: Optional[Dict[str, Any]] = None
        self.ev_steps: List[Dict[str, Any]] = []
        self.ev_cycles: List[Dict[str, Any]] = []
        self.ev_metrics: Dict[str, Dict[str, Any]] = {}
        self.ev_artifacts: List[Dict[str, Any]] = []
        self.ev_logs: List[Dict[str, Any]] = []
        self.ev_last_change: Optional[float] = None
        self.ev_step_times: List[float] = []
        # log state
        self.lg_lines = 0
        self.lg_last_change: Optional[float] = None
        self.lg_marks: Dict[str, float] = {}
        self.lg_order: List[str] = []
        self.lg_reused: Dict[str, str] = {}
        self.lg_failures: List[Dict[str, Any]] = []
        self.lg_steps: List[Dict[str, Any]] = []
        self.lg_cycles: List[Dict[str, Any]] = []
        self.lg_done: Optional[float] = None
        self.lg_target: Optional[float] = None
        self._pending_step: Optional[int] = None
        self.lg_relax_done = False
        # probes
        self._probe_cache: Dict[str, Any] = {}
        self._probe_at = 0.0
        self._artifact_cache: List[Dict[str, Any]] = []
        self._art_at = 0.0

    # ------------------------------------------------------------------ ingest
    def _ingest_events(self, now: float):
        new = self._ev.read()
        if not new:
            return
        self.ev_last_change = now
        for line in new:
            try:
                e = json.loads(line)
                kind = e["type"]
            except (ValueError, KeyError, TypeError):
                self.ev_bad += 1
                continue
            self.ev_lines += 1
            if kind == "run.start":
                self.ev_run_start = e
            elif kind == "run.end":
                self.ev_run_end = e
            elif kind == "stage.start":
                self.ev_stage.setdefault(e.get("stage"), {}).update(start=e.get("t"), status="running", end=None)
            elif kind == "stage.end":
                st = self.ev_stage.setdefault(e.get("stage"), {})
                st.update(
                    end=e.get("t"), status=e.get("status", "ok"), elapsed=e.get("elapsed_s"), detail=e.get("detail")
                )
                if e.get("status") == "skipped" and st.get("start") is None:
                    st["start"] = None
            elif kind == "relax.step":
                self.ev_steps.append(e)
                self.ev_step_times.append(float(e.get("t", now)))
            elif kind == "relax.cycle":
                self.ev_cycles.append(e)
            elif kind == "metric":
                self.ev_metrics[str(e.get("name"))] = e
            elif kind == "artifact":
                self.ev_artifacts.append(e)
            elif kind == "log":
                self.ev_logs.append(e)

    def _ingest_log(self, now: float):
        new = self._lg.read()
        if not new:
            return
        self.lg_last_change = now
        for raw in new:
            line = _ANSI.sub("", raw).rstrip()
            if not line.strip():
                continue
            self.lg_lines += 1
            self._parse_log_line(line, now)

    def _parse_log_line(self, line: str, now: float):
        if self._pending_step is not None:
            m = RE_STEP_STATS.search(line)
            if m:
                v, s, fm, fa, fsa, na = m.groups()
                self.lg_steps.append(
                    {
                        "step": self._pending_step,
                        "t": now,
                        "vertices": int(v),
                        "stiffness": float(s),
                        "fmax": float(fm),
                        "f_av": float(fa),
                        "fs_av": float(fsa),
                        "n_av": float(na),
                    }
                )
            self._pending_step = None
        m = RE_STEP.match(line)
        if m:
            self._pending_step = int(m.group(1))
            return
        for stage, rx in LOG_STARTS:
            if rx.search(line):
                if stage not in self.lg_marks:
                    self.lg_marks[stage] = now
                    self.lg_order.append(stage)
                break
        for stage, rx in LOG_REUSED:
            if rx.search(line):
                self.lg_reused[stage] = rx.pattern.replace("\\", "")[:60]
        for stage, rx in LOG_FAILURES:
            m = rx.search(line)
            if m:
                self.lg_failures.append({"stage": stage, "message": m.group(0).strip(), "t": now})
        m = RE_CYCLE.search(line)
        if m:
            self.lg_cycles.append(
                {"cycle": int(m.group(1)), "cum_time_s": float(m.group(2)), "steps": int(m.group(3)), "t": now}
            )
        m = RE_FIXES.search(line)
        if m:
            c = int(m.group(1))
            for cyc in self.lg_cycles:
                if cyc["cycle"] == c:
                    cyc.update(nc_curve=int(m.group(2)), nc_proxi=int(m.group(3)))
        m = RE_TARGET.search(line)
        if m:
            self.lg_target = float(m.group(2))
        if RE_DONE.search(line):
            self.lg_done = now
        if re.search(r"Relaxation Complete in \d+ Steps", line):
            self.lg_relax_done = True

    # ------------------------------------------------------------------ probes
    def _probe_stages(self, now: float, force: bool = False) -> Dict[str, Optional[Dict[str, Any]]]:
        if not self.probe:
            return {}
        if not force and now - self._probe_at < 2.0 and self._probe_cache:
            return self._probe_cache
        out: Dict[str, Optional[Dict[str, Any]]] = {}
        for stage, pats in self.probe.patterns().items():
            hit = _first_match(pats)
            if hit is None:
                out[stage] = {"ok": False}
            else:
                try:
                    out[stage] = {"ok": True, "path": hit, "mtime": os.path.getmtime(hit)}
                except OSError:
                    out[stage] = {"ok": False}
        self._probe_cache, self._probe_at = out, now
        return out

    def _images(self, now: float, started: Optional[float]) -> List[Dict[str, Any]]:
        if not self.probe:
            return []
        if now - self._art_at < 3.0 and self._artifact_cache:
            return self._artifact_cache
        found = []
        for pat in self.probe.image_globs():
            for path in glob.glob(pat, recursive=True)[:200]:
                try:
                    mt = os.path.getmtime(path)
                except OSError:
                    continue
                if started is not None and mt < started - 5:
                    continue  # left over from an earlier run
                if not names.is_this_task(os.path.basename(path), self.probe.nflux, self.probe.inst):
                    continue  # another task's plot (same rotation, different fluxon count or instrument)
                found.append({"path": path, "kind": "image", "t": mt, "source": "file"})
        self._artifact_cache, self._art_at = found, now
        return found

    # ------------------------------------------------------------------ relax
    def relax_points(self) -> List[Dict[str, Any]]:
        pts = [
            {
                **{k: e.get(k) for k in ("step", "stiffness", "fmax", "f_av", "fs_av", "n_av", "vertices", "t")},
                "s": "event",
            }
            for e in self.ev_steps
        ]
        last = max((p["step"] for p in pts if p["step"] is not None), default=-1)
        pts += [{**p, "s": "log"} for p in self.lg_steps if p["step"] > last]
        return pts

    def relax_cycles(self) -> List[Dict[str, Any]]:
        cyc = [
            {
                **{
                    k: e.get(k)
                    for k in (
                        "cycle",
                        "threshold",
                        "stiffness",
                        "cum_time_s",
                        "steps",
                        "nc_curve",
                        "nc_proxi",
                        "broken",
                        "t",
                    )
                },
                "s": "event",
            }
            for e in self.ev_cycles
        ]
        have = {c["cycle"] for c in cyc}
        cyc += [{**c, "s": "log"} for c in self.lg_cycles if c["cycle"] not in have]
        return cyc

    def relax_series(self, after: int = 0) -> Dict[str, Any]:
        """Chart data: points after index `after`, all cycles, and the total point count."""
        with self._lock:
            pts = self.relax_points()
            return {"n": len(pts), "points": pts[after:], "cycles": self.relax_cycles()}

    # ------------------------------------------------------------------ state
    def _stale_threshold(self) -> float:
        times = self.ev_step_times[-20:]
        if len(times) >= 4:
            gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
            if gaps:
                return min(self.stale_after, max(3 * statistics.median(gaps), self.min_stale))
        return self.stale_after

    def refresh(self, *args, **kwargs) -> Dict[str, Any]:
        """Read whatever is new and return the task's state (see `_refresh`); one thread at a time."""
        with self._refresh_lock:
            return self._refresh(*args, **kwargs)

    def _refresh(
        self,
        now: Optional[float] = None,
        alive: Optional[bool] = None,
        outcome: Optional[str] = None,
        started: Optional[float] = None,
        ended: Optional[float] = None,
        probe: bool = True,
    ) -> Dict[str, Any]:
        """Read whatever is new and return the task's state as a plain dict.

        `alive` is the backend's view of the process (None when unknown); `outcome` is set
        once the task is finished: ok | failed | timeout | cancelled | crashed.
        """
        now = time.time() if now is None else now
        with self._lock:
            self._ingest_events(now)
            self._ingest_log(now)
        finished = outcome is not None
        if finished:  # the final state must not reuse a probe taken while the task was still running
            self._probe_at = self._art_at = 0.0
        has_events = self.ev_lines > 0
        # staleness: events went quiet while the log kept moving
        degraded = False
        reason = None
        if has_events and not finished and self.ev_run_end is None:
            thr = self._stale_threshold()
            quiet = now - (self.ev_last_change or now)
            log_moved = self.lg_last_change is not None and (
                self.ev_last_change is None or self.lg_last_change > self.ev_last_change
            )
            if quiet > thr and log_moved and alive is not False:
                degraded = True
                reason = f"no events for {int(quiet)} s while the log is still advancing"
        mode = (
            "events" if has_events and not degraded else "degraded" if degraded else "log" if self.lg_lines else "none"
        )

        probes = self._probe_stages(now, force=finished) if probe else {}
        stages = self._resolve_stages(mode, finished, outcome, probes, started)
        run_end = self.ev_run_end
        current = next((s["id"] for s in stages if s["status"] == "running"), None)

        pts = self.relax_points()
        latest = pts[-1] if pts else None
        steps_per_s = None
        if len(pts) >= 2 and pts[-1].get("t") and pts[-2].get("t") and pts[-1]["t"] > pts[-2]["t"]:
            window = pts[-min(len(pts), 6) :]
            dt = window[-1]["t"] - window[0]["t"]
            if dt > 0 and window[-1]["step"] is not None and window[0]["step"] is not None:
                steps_per_s = (window[-1]["step"] - window[0]["step"]) / dt
        cycles = self.relax_cycles()
        relax_src = "none" if latest is None else latest["s"]
        metrics = [
            {"name": m.get("name"), "value": m.get("value"), "unit": m.get("unit"), "source": "event"}
            for m in self.ev_metrics.values()
        ]
        warnings = [f"{e.get('level', 'warn')}: {e.get('message')}" for e in self.ev_logs][-10:]
        warnings += [f.get("message") for f in self.lg_failures][-5:]
        artifacts = [
            {
                "path": a.get("path"),
                "kind": a.get("kind", "image"),
                "t": a.get("t"),
                "source": "event",
                **({"pair": a["pair"]} if a.get("pair") else {}),  # the AIA file that goes with a model image (A4-03)
            }
            for a in self.ev_artifacts
        ]
        seen = {a["path"] for a in artifacts}
        artifacts += [a for a in self._images(now, started) if a["path"] not in seen]
        artifacts.sort(key=lambda a: a.get("t") or 0)

        return {
            "mode": mode,
            "degraded": degraded,
            "degraded_reason": reason,
            "current_stage": current,
            "stages": stages,
            "relax": {
                "n_points": len(pts),
                "latest": latest,
                "cycle": cycles[-1] if cycles else None,
                "cycles": len(cycles),
                "threshold": (
                    cycles[-1].get("threshold") if cycles and cycles[-1].get("threshold") else self.lg_target
                ),
                "steps_per_s": steps_per_s,
                "source": relax_src,
            },
            "metrics": metrics,
            "warnings": warnings,
            "artifacts": artifacts[-60:],
            "run": {
                "start": self.ev_run_start,
                "end": run_end,
                "log_done": self.lg_done is not None,
            },
            "sources": {
                "events": {
                    "present": has_events,
                    "lines": self.ev_lines,
                    "bad_lines": self.ev_bad,
                    "last_change": self.ev_last_change,
                    "age_s": None if self.ev_last_change is None else round(now - self.ev_last_change, 1),
                },
                "log": {
                    "present": self.lg_lines > 0,
                    "lines": self.lg_lines,
                    "bytes": self._lg.offset,
                    "last_change": self.lg_last_change,
                    "age_s": None if self.lg_last_change is None else round(now - self.lg_last_change, 1),
                },
                "files": {
                    "checked": bool(probes),
                    "ok": sorted(k for k, v in probes.items() if v and v.get("ok")),
                    "missing": sorted(k for k, v in probes.items() if v and not v.get("ok")),
                },
                "process": {"alive": alive},
            },
            "outcome": outcome,
            "observed_at": now,
        }

    def _resolve_stages(self, mode, finished, outcome, probes, started) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        log_order = [s for s in STAGE_IDS if s in self.lg_marks]
        last_log = log_order[-1] if log_order else None
        for stage in STAGE_IDS:
            ev = self.ev_stage.get(stage)
            row = {
                "id": stage,
                "label": LABELS[stage],
                "status": "pending",
                "source": "none",
                "started": None,
                "ended": None,
                "elapsed_s": None,
                "detail": None,
                "verified": None,
            }
            if ev and ev.get("status"):
                row.update(
                    status=ev["status"],
                    source="event",
                    started=ev.get("start"),
                    ended=ev.get("end"),
                    elapsed_s=ev.get("elapsed"),
                    detail=ev.get("detail"),
                )
            else:
                self._fill_from_log(row, stage, last_log, log_order)
            out.append(row)
        # an event-known stage that is still "running" while the log shows progress beyond it
        if mode == "degraded":
            for i, row in enumerate(out):
                if row["source"] == "event" and row["status"] == "running":
                    later = [s for s in log_order if STAGE_IDS.index(s) > i]
                    if later:
                        row.update(status="ok", source="log", ended=self.lg_marks[later[0]])
        # file evidence for stages nothing else spoke for
        furthest = max((i for i, r in enumerate(out) if r["status"] in ("running", "ok", "failed")), default=-1)
        for i, row in enumerate(out):
            pr = probes.get(row["id"])
            if row["status"] == "pending" and pr and pr.get("ok") and (finished or i < furthest):
                fresh = started is None or pr["mtime"] >= started - 2
                row.update(
                    status="ok" if fresh else "skipped",
                    source="file",
                    ended=pr["mtime"],
                    detail=None if fresh else "output already on disk",
                )
            if row["source"] == "event" and row["status"] == "ok" and pr is not None:
                row["verified"] = bool(pr.get("ok"))
        # close out running stages and unexplained gaps once the task is over
        if finished:
            for row in out:
                if row["status"] == "running":
                    row["status"] = "ok" if outcome == "ok" else "failed"
                    if outcome != "ok":
                        named = next((f["message"] for f in self.lg_failures if f["stage"] == row["id"]), None)
                        row["detail"] = (
                            row["detail"]
                            or named
                            or {
                                "timeout": "task timed out",
                                "cancelled": "cancelled",
                                "crashed": "process vanished",
                            }.get(outcome, "task failed")
                        )
                elif row["status"] == "pending" and outcome == "ok":
                    row.update(status="skipped", detail="no evidence it ran")
        # failure text from the log goes on the stage it names
        for f in self.lg_failures:
            row = next((r for r in out if r["id"] == f["stage"]), None)
            if row and row["status"] in ("running", "failed", "pending") and (finished or row["status"] != "pending"):
                row["status"] = "failed"
                row["detail"] = row["detail"] or f["message"]
                if row["source"] == "none":
                    row["source"] = "log"
        # euv is optional: a task whose run never reported it does not get a permanently grey stage
        return [r for r in out if r["id"] != "euv" or "euv" in self.ev_stage]

    def _fill_from_log(self, row, stage, last_log, log_order):
        if stage in self.lg_marks:
            idx = STAGE_IDS.index(stage)
            row["started"] = self.lg_marks[stage]
            later = [s for s in log_order if STAGE_IDS.index(s) > idx]
            row["source"] = "log"
            if later or self.lg_done is not None:
                row["status"] = "ok"
                row["ended"] = self.lg_marks[later[0]] if later else self.lg_done
                row["elapsed_s"] = round(row["ended"] - row["started"], 2)
            else:
                row["status"] = "running"
            if stage in self.lg_reused:
                row["detail"] = "reused existing output"
            if stage == "relax" and stage in self.lg_reused:
                row["detail"] = "reused relaxed world"
        elif log_order:
            idx = STAGE_IDS.index(stage)
            first = STAGE_IDS.index(log_order[0])
            if idx < first and stage != "fluxlight":
                row.update(status="skipped", source="log", detail="no output before " + log_order[0])
