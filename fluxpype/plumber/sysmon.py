"""Machine load for the sidebar monitor: CPU, memory, network, disk I/O, GPU, storage and the busiest processes.

Uses psutil when it is installed. Without it CPU falls back to the load average, storage still works, and the rest is
reported as missing (``psutil: false``). Samples are taken on request (at most one a second) and the last
``HISTORY`` are kept, so a freshly opened page can draw its charts at once.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:  # optional; the monitor degrades
    psutil = None

HISTORY = 120
_UTIL = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_MODEL = re.compile(r'"model"\s*=\s*"([^"]+)"')


def _run(argv: List[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def read_gpu() -> Optional[Dict[str, Any]]:
    """GPU utilisation: Apple's IOAccelerator on macOS, nvidia-smi elsewhere; None when neither answers."""
    if sys.platform == "darwin":
        out = _run(["ioreg", "-r", "-d", "1", "-w0", "-c", "IOAccelerator"])
        u = _UTIL.search(out)
        if u:
            m = _MODEL.search(out)
            return {"name": m.group(1) if m else "GPU", "pct": float(u.group(1)), "mem_used": None, "mem_total": None}
    elif shutil.which("nvidia-smi"):
        out = _run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
        )
        try:
            name, pct, used, total = [x.strip() for x in out.splitlines()[0].split(",")]
            return {"name": name, "pct": float(pct), "mem_used": float(used) * 2**20, "mem_total": float(total) * 2**20}
        except (IndexError, ValueError):
            pass
    return None


class SysMon:
    """`volumes` returns [(label, path)] to report storage for; `jobs` returns {process-group id: task label}."""

    def __init__(
        self, volumes: Callable[[], List[Tuple[str, str]]], jobs: Callable[[], Dict[int, str]] = lambda: {}
    ):
        self.volumes, self.jobs = volumes, jobs
        self.hist: Deque[Dict[str, Any]] = deque(maxlen=HISTORY)
        self._t = 0.0
        self._cores: List[float] = []
        self._io: Optional[Tuple[float, float, float, float, float]] = None  # t, net rx, net tx, disk read, disk write
        self._gpu: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
        if psutil:
            psutil.cpu_percent(None, percpu=True)  # the first call has no interval to measure over

    # --------------------------------------------------------------- one sample
    def sample(self) -> Dict[str, Any]:
        now = time.time()
        if self.hist and now - self._t < 1.0:
            return self.hist[-1]
        self._t = now
        s: Dict[str, Any] = {"t": now, "cpu": None, "mem": None, "net_rx": None, "net_tx": None, "disk_r": None, "disk_w": None, "gpu": None}
        if psutil:
            cores = self._cores = psutil.cpu_percent(None, percpu=True)
            s["cpu"] = sum(cores) / len(cores) if cores else None
            s["mem"] = psutil.virtual_memory().percent
            n, d = psutil.net_io_counters(), psutil.disk_io_counters()
            cur = (now, n.bytes_recv, n.bytes_sent, d.read_bytes if d else 0, d.write_bytes if d else 0)
            if self._io and cur[0] > self._io[0]:
                dt = cur[0] - self._io[0]
                s["net_rx"], s["net_tx"] = (cur[1] - self._io[1]) / dt, (cur[2] - self._io[2]) / dt
                s["disk_r"], s["disk_w"] = (cur[3] - self._io[3]) / dt, (cur[4] - self._io[4]) / dt
            self._io = cur
        else:
            s["cpu"] = min(100.0, os.getloadavg()[0] / (os.cpu_count() or 1) * 100)  # a load, not a utilisation
        if now - self._gpu[0] > 2.0:
            self._gpu = (now, read_gpu())
        s["gpu"] = self._gpu[1]["pct"] if self._gpu[1] else None
        self.hist.append(s)
        return s

    # --------------------------------------------------------------- the full picture
    def snapshot(self, procs: bool = False) -> Dict[str, Any]:
        cur = self.sample()
        out: Dict[str, Any] = {
            "psutil": psutil is not None,
            "now": cur,
            "history": list(self.hist),
            "cpu_count": os.cpu_count(),
            "load": list(os.getloadavg()),
            "gpu": self._gpu[1],
            "storage": self.storage(),
        }
        if psutil:
            out["cores"] = self._cores
            vm, sw = psutil.virtual_memory(), psutil.swap_memory()
            out["memory"] = {"total": vm.total, "used": vm.total - vm.available, "swap_used": sw.used, "swap_total": sw.total}
            if procs:
                out["processes"] = self.processes()
        return out

    def storage(self) -> List[Dict[str, Any]]:
        seen, rows = set(), []
        for label, path in self.volumes():
            try:
                dev = os.stat(path).st_dev
                if dev in seen:
                    continue
                u = shutil.disk_usage(path)
            except OSError:
                continue
            seen.add(dev)
            rows.append({"label": label, "path": path, "total": u.total, "used": u.used, "free": u.free, "pct": u.used / u.total * 100 if u.total else 0})
        return rows

    def processes(self, top: int = 8) -> Dict[str, Any]:
        """The busiest processes, plus Plumber's running tasks summed over their process groups."""
        groups = self.jobs()
        rows, tasks = [], {}
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
            i = p.info
            if i["memory_info"] is None:
                continue
            row = {"pid": i["pid"], "name": i["name"] or "?", "cpu": i["cpu_percent"] or 0.0, "rss": i["memory_info"].rss}
            try:
                label = groups.get(os.getpgid(i["pid"]))
            except OSError:
                label = None
            if label:
                t = tasks.setdefault(label, {"label": label, "cpu": 0.0, "rss": 0, "procs": 0})
                t["cpu"] += row["cpu"]
                t["rss"] += row["rss"]
                t["procs"] += 1
            rows.append({**row, "task": label})
        rows.sort(key=lambda r: r["cpu"], reverse=True)
        return {"top": rows[:top], "tasks": sorted(tasks.values(), key=lambda t: -t["cpu"])}
