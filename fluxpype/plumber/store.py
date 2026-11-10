"""SQLite job store (stdlib only, WAL)."""

import json
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    overrides_json TEXT NOT NULL DEFAULT '{}',
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    created REAL NOT NULL,
    started REAL,
    ended REAL,
    snapshot_path TEXT NOT NULL,
    extra_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    cr TEXT NOT NULL,
    nflux TEXT NOT NULL,
    adapt TEXT NOT NULL,
    method TEXT NOT NULL,
    status TEXT NOT NULL,
    pid INTEGER,
    pid_start TEXT,
    exit_code INTEGER,
    created REAL NOT NULL,
    started REAL,
    ended REAL,
    run_dir TEXT NOT NULL,
    lock_key TEXT NOT NULL DEFAULT '',
    slots INTEGER NOT NULL DEFAULT 1,
    attempt INTEGER NOT NULL DEFAULT 1,
    summary_json TEXT
);
CREATE INDEX IF NOT EXISTS tasks_job ON tasks(job_id);
CREATE INDEX IF NOT EXISTS tasks_status ON tasks(status);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

ACTIVE = ("pending", "running")
FINISHED = ("ok", "failed", "timeout", "cancelled", "crashed")


class Store:
    def __init__(self, path):
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)

    def close(self):
        with self._lock:
            self._db.close()

    # -- generic helpers
    def _all(self, sql: str, args: Iterable = ()) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, tuple(args)).fetchall()]

    def _one(self, sql: str, args: Iterable = ()) -> Optional[Dict[str, Any]]:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    def _exec(self, sql: str, args: Iterable = ()):
        with self._lock:
            self._db.execute(sql, tuple(args))

    # -- settings
    def get_setting(self, key: str, default=None):
        row = self._one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value) -> None:
        self._exec(
            "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    # -- jobs
    def add_job(self, job: Dict[str, Any], tasks: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._db.execute("BEGIN")
            try:
                self._db.execute(
                    "INSERT INTO jobs(id, profile, overrides_json, backend, status, created, snapshot_path, extra_json) "  # noqa: E501
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        job["id"],
                        job["profile"],
                        json.dumps(job.get("overrides", {})),
                        job["backend"],
                        "pending",
                        job["created"],
                        job["snapshot_path"],
                        json.dumps(job.get("extra", {})),
                    ),
                )
                for t in tasks:
                    self._db.execute(
                        "INSERT INTO tasks(id, job_id, idx, cr, nflux, adapt, method, status, created, run_dir, lock_key, slots) "  # noqa: E501
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            t["id"],
                            job["id"],
                            t["idx"],
                            t["cr"],
                            t["nflux"],
                            t["adapt"],
                            t["method"],
                            "pending",
                            job["created"],
                            t["run_dir"],
                            t.get("lock_key", ""),
                            t.get("slots", 1),
                        ),
                    )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    @staticmethod
    def _job_row(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        row["overrides"] = json.loads(row.pop("overrides_json"))
        row["extra"] = json.loads(row.pop("extra_json"))
        return row

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._job_row(self._one("SELECT * FROM jobs WHERE id=?", (job_id,)))

    def list_jobs(self, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._all("SELECT * FROM jobs ORDER BY created DESC, rowid DESC LIMIT ?", (limit,))
        return [self._job_row(r) for r in rows]

    def update_job(self, job_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def delete_job(self, job_id: str) -> None:
        self._exec("DELETE FROM jobs WHERE id=?", (job_id,))

    # -- tasks
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._one("SELECT * FROM tasks WHERE id=?", (task_id,))

    def tasks_for_job(self, job_id: str) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM tasks WHERE job_id=? ORDER BY idx", (job_id,))

    def tasks_with_status(self, *statuses: str) -> List[Dict[str, Any]]:
        marks = ",".join("?" for _ in statuses)
        return self._all(f"SELECT * FROM tasks WHERE status IN ({marks}) ORDER BY created, job_id, idx", statuses)

    def recent_finished(self, limit: int = 20) -> List[Dict[str, Any]]:
        marks = ",".join("?" for _ in FINISHED)
        return self._all(
            f"SELECT * FROM tasks WHERE status IN ({marks}) ORDER BY ended DESC LIMIT ?", (*FINISHED, limit)
        )

    def update_task(self, task_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE tasks SET {cols} WHERE id=?", (*fields.values(), task_id))

    def counts(self) -> Dict[str, int]:
        rows = self._all("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    def derive_job_status(self, job_id: str) -> str:
        """pending | running | ok | failed | cancelled | partial, from the job's tasks."""
        statuses = [t["status"] for t in self.tasks_for_job(job_id)]
        if not statuses:
            return "pending"
        if "running" in statuses:
            return "running"
        if "pending" in statuses:
            return "running" if any(s in FINISHED for s in statuses) else "pending"
        if all(s == "ok" for s in statuses):
            return "ok"
        if all(s == "cancelled" for s in statuses):
            return "cancelled"
        if "ok" in statuses:
            return "partial"
        return "failed"

    @staticmethod
    def now() -> float:
        return time.time()
