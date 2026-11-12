"""What an AI agent can do through Plumber's MCP server, and a record of what it has done.

``TOOLS`` mirrors the tools ``mcp_server.build_server`` registers (a test keeps the two in step). ``AgentLog`` keeps
one entry per tool call (the MCP server tags every HTTP request with a call id), backed by a JSON-lines file with one
line per request so the record survives a restart. Stdlib only.
"""

import json
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List

# name, kind (read: looks only; write: queues or changes something; stop: cancels running work), plain-language summary
TOOLS = [
    ("status", "read", "See the queue, backends, running tasks and recent results"),
    ("list_profiles", "read", "List the saved profiles"),
    ("get_profile", "read", "Read one profile and preview the tasks it expands to"),
    ("submit_job", "write", "Queue a new job from a profile"),
    ("list_jobs", "read", "List recent jobs"),
    ("get_job", "read", "Read one job with a per-task stage summary"),
    ("get_task", "read", "Read one task: stages, relaxation, warnings, images"),
    ("get_relax_series", "read", "Read a task's relaxation history"),
    ("read_task_log", "read", "Read or search a task's log"),
    ("wait_for_task", "read", "Wait for a task to finish or a stage to end"),
    ("cancel", "stop", "Cancel a job or a task"),
    ("retry_failed", "write", "Re-queue the failed tasks of a job"),
    ("set_queue", "write", "Pause or resume the queue, change how many tasks run at once"),
    ("cache_summary", "read", "See what is in the data cache"),
    ("get_task_image", "read", "Look at one of a task's images"),
]

MAX_CALLS = 1000  # calls kept in memory (and the file is trimmed to about twice the request lines this implies)
FILE_LINES = 4000
LEGACY_GAP_S = 10  # requests of one tool this close together, with no call id, are one call
_REAL_ID = re.compile(r"^[0-9a-f]{12}$")


class AgentLog:
    def __init__(self, path: str):
        self.path = path
        self.calls: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._legacy: Dict[str, str] = {}  # tool -> the untagged call its latest request joined
        lines: List[str] = []
        try:
            with open(path, "r", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            pass
        if len(lines) > FILE_LINES:  # keep the newest half
            lines = lines[-FILE_LINES // 2 :]
            try:
                with open(path, "w") as f:
                    f.write("\n".join(lines) + "\n")
            except OSError:
                pass
        for ln in lines:
            try:
                self._add(json.loads(ln))
            except (ValueError, TypeError, KeyError):
                continue

    def _add(self, r: Dict[str, Any]) -> None:
        cid = r["call"]
        if not _REAL_ID.match(cid):  # an MCP server from before calls were tagged sends one id per request
            prev = self._legacy.get(r["tool"])
            if prev in self.calls and r["ts"] - self.calls[prev]["last"] <= LEGACY_GAP_S:
                cid = prev  # a wait_for_task that polls every 1.5 s is one call, not one a poll
            self._legacy[r["tool"]] = cid
        c = self.calls.get(cid)
        if c is None:
            c = self.calls[cid] = {
                "call": cid,
                "ts": r["ts"],
                "last": r["ts"],
                "tool": r["tool"],
                "args": r.get("args", ""),
                "requests": 0,
                "ms": 0,
                "status": 0,
                "paths": [],
                "via": r.get("via", "mcp"),
            }
            while len(self.calls) > MAX_CALLS:
                self.calls.popitem(last=False)
        c["requests"] += 1
        c["last"] = r["ts"]
        c["ms"] += int(r.get("ms", 0))
        c["status"] = max(c["status"], int(r.get("status", 0)))
        if len(c["paths"]) < 4:
            c["paths"].append(f"{r.get('method', '')} {r.get('path', '')}".strip())

    def record(self, call: str, tool: str, args: str, method: str, path: str, status: int, ms: float, via: str = "mcp") -> None:
        r = {
            "ts": time.time(),
            "call": call,
            "tool": tool,
            "args": args,
            "method": method,
            "path": path,
            "status": status,
            "ms": round(ms),
            "via": via,
        }
        self._add(r)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(r) + "\n")
        except OSError:
            pass  # the in-memory record still works

    def summary(self, limit: int = 150) -> Dict[str, Any]:
        calls = list(self.calls.values())
        per: Dict[str, Dict[str, Any]] = {}
        for c in calls:
            if c["via"] == "ask":  # the Ask bar's own reads are not the agent's doing
                continue
            p = per.setdefault(c["tool"], {"calls": 0, "errors": 0, "last": None, "last_status": None})
            p["calls"] += 1
            p["errors"] += c["status"] >= 400
            p["last"], p["last_status"] = c["ts"], c["status"]
        tools = [
            {"name": n, "kind": k, "what": w, **per.pop(n, {"calls": 0, "errors": 0, "last": None, "last_status": None})}
            for n, k, w in TOOLS
        ]
        # tools this table does not know (a newer MCP server): still shown, so the record is complete
        tools += [{"name": n, "kind": "read", "what": "", **p} for n, p in per.items() if n]
        return {
            "tools": tools,
            "calls": [dict(c) for c in reversed(calls[-limit:])],
            "total": len(calls),
            "since": calls[0]["ts"] if calls else None,
        }
