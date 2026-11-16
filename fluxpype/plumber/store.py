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
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    t REAL NOT NULL,
    kind TEXT NOT NULL,
    sev TEXT NOT NULL,
    key TEXT NOT NULL UNIQUE,
    job_id TEXT,
    task_id TEXT,
    origin_kind TEXT,
    session TEXT,
    tab TEXT,
    f_json TEXT NOT NULL DEFAULT '{}',
    acked INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id);
CREATE INDEX IF NOT EXISTS events_session ON events(session, seq);
"""
# columns added after the first release: CREATE TABLE IF NOT EXISTS never alters an existing table
NEW_COLUMNS = (
    ("jobs", "origin_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("jobs", "notified", "TEXT"),
    ("tasks", "notified", "TEXT"),
)
SEV_RANK = {"info": 0, "success": 1, "warn": 2, "error": 3}

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
        self._migrate()

    def _migrate(self):
        """Add missing columns. A new `notified` starts equal to `status`, so an upgrade announces nothing old."""
        with self._lock:
            for table, col, decl in NEW_COLUMNS:
                have = {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}
                if col not in have:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
                    if col == "notified":
                        self._db.execute(f"UPDATE {table} SET notified = status")

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
                    "INSERT INTO jobs(id, profile, overrides_json, backend, status, created, snapshot_path, extra_json, origin_json) "  # noqa: E501
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        job["id"],
                        job["profile"],
                        json.dumps(job.get("overrides", {})),
                        job["backend"],
                        "pending",
                        job["created"],
                        job["snapshot_path"],
                        json.dumps(job.get("extra", {})),
                        json.dumps(job.get("origin") or {}),
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
        row["origin"] = json.loads(row.pop("origin_json", None) or "{}")
        row.pop("notified", None)
        return row

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._job_row(self._one("SELECT * FROM jobs WHERE id=?", (job_id,)))

    def list_jobs(self, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._all("SELECT * FROM jobs ORDER BY created DESC, rowid DESC LIMIT ?", (limit,))
        return [self._job_row(r) for r in rows]

    def run_stats(self) -> Dict[str, Dict[str, Any]]:
        """Per preset: how many jobs, how many finished ok, and the latest job's status and time."""
        rows = self._all(
            "SELECT profile, COUNT(*) AS jobs, SUM(status = 'ok') AS ok, MAX(created) AS last_at FROM jobs GROUP BY profile"
        )
        out = {}
        for r in rows:
            last = self._one(
                "SELECT status, COALESCE(ended, created) AS at FROM jobs WHERE profile = ? ORDER BY created DESC LIMIT 1",
                (r["profile"],),
            )
            out[r["profile"]] = {
                "jobs": r["jobs"],
                "ok": r["ok"] or 0,
                "last_status": last["status"] if last else None,
                "last_at": last["at"] if last else None,
            }
        return out

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

    # -- events (see notify.py)
    @staticmethod
    def _event_row(r: Dict[str, Any]) -> Dict[str, Any]:
        origin = {"kind": r["origin_kind"]} if r["origin_kind"] else {}
        origin.update({k: r[k] for k in ("session", "tab") if r[k]})
        return {
            "seq": r["seq"],
            "t": r["t"],
            "kind": r["kind"],
            "sev": r["sev"],
            "key": r["key"],
            "job_id": r["job_id"],
            "task_id": r["task_id"],
            "origin": origin,
            "f": json.loads(r["f_json"]),
            "acked": bool(r["acked"]),
        }

    def add_event(self, ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert one event; None when its key already exists (INSERT OR IGNORE).

        The NOT EXISTS guard keeps a known duplicate from using up a seq (an ignored AUTOINCREMENT insert does), so
        seqs stay contiguous and a client can treat a gap as missed events."""
        origin = ev.get("origin") or {}
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO events(t, kind, sev, key, job_id, task_id, origin_kind, session, tab, f_json) "
                "SELECT ?,?,?,?,?,?,?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM events WHERE key = ?)",
                (
                    ev.get("t") or time.time(),
                    ev["kind"],
                    ev["sev"],
                    ev["key"],
                    ev.get("job_id"),
                    ev.get("task_id"),
                    origin.get("kind"),
                    origin.get("session"),
                    origin.get("tab"),
                    json.dumps(ev.get("f") or {}, sort_keys=True),
                    ev["key"],
                ),
            )
            if not cur.rowcount:
                return None
            row = self._db.execute("SELECT * FROM events WHERE seq=?", (cur.lastrowid,)).fetchone()
        return self._event_row(dict(row))

    def events(
        self,
        since: Optional[int] = None,
        before: Optional[int] = None,
        limit: int = 100,
        min_sev: Optional[str] = None,
        job: Optional[str] = None,
        session: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """`since`: ascending after it; `before`: descending below it; neither: the newest, descending."""
        where, args = [], []
        if since is not None:
            where.append("seq > ?")
            args.append(since)
        elif before is not None:
            where.append("seq < ?")
            args.append(before)
        if min_sev in SEV_RANK:
            sevs = [s for s, r in SEV_RANK.items() if r >= SEV_RANK[min_sev]]
            where.append(f"sev IN ({','.join('?' for _ in sevs)})")
            args += sevs
        if job:
            where.append("job_id = ?")
            args.append(job)
        if session:
            where.append("session = ?")
            args.append(session)
        sql = "SELECT * FROM events" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY seq " + ("ASC" if since is not None else "DESC") + " LIMIT ?"
        return [self._event_row(r) for r in self._all(sql, (*args, limit))]

    def last_seq(self) -> int:
        return (self._one("SELECT MAX(seq) AS m FROM events") or {}).get("m") or 0

    def count_events(self, key_prefix: str) -> int:
        """Events whose key starts with `key_prefix` (episode numbers for stalls, backends and the queue)."""
        like = key_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        return self._one("SELECT COUNT(*) AS n FROM events WHERE key LIKE ? ESCAPE '\\'", (like,))["n"]

    def event_counts(self) -> Dict[str, int]:
        upto = int(self.get_setting("events_read_upto", 0) or 0)
        unread = self._one("SELECT COUNT(*) AS n FROM events WHERE sev IN ('warn','error') AND seq > ?", (upto,))["n"]
        attention = self._one("SELECT COUNT(*) AS n FROM events WHERE sev = 'error' AND acked = 0")["n"]
        return {"unread": unread, "attention": attention, "read_upto": upto}

    def mark_read(self, upto: int) -> None:
        if int(upto) > int(self.get_setting("events_read_upto", 0) or 0):
            self.set_setting("events_read_upto", int(upto))

    def ack_events(self, seqs: Iterable[int] = (), all_: bool = False, job: Optional[str] = None) -> int:
        with self._lock:
            if all_ or job:
                sql, args = "UPDATE events SET acked=1 WHERE acked=0", []
                if job:
                    sql, args = sql + " AND job_id=?", [job]
                return self._db.execute(sql, args).rowcount
            seqs = [int(s) for s in seqs][:1000]
            if not seqs:
                return 0
            marks = ",".join("?" for _ in seqs)
            return self._db.execute(f"UPDATE events SET acked=1 WHERE seq IN ({marks})", seqs).rowcount

    def trim_events(self, now: Optional[float] = None, keep: int = 50000) -> int:
        """Drop info/success older than 7 days and anything older than 30 days, then keep the newest `keep`."""
        now = now or time.time()
        with self._lock:
            n = self._db.execute(
                "DELETE FROM events WHERE (sev IN ('info','success') AND t < ?) OR t < ?",
                (now - 7 * 86400, now - 30 * 86400),
            ).rowcount
            n += self._db.execute(
                "DELETE FROM events WHERE seq <= (SELECT seq FROM events ORDER BY seq DESC LIMIT 1 OFFSET ?)", (keep,)
            ).rowcount
        return n

    @staticmethod
    def now() -> float:
        return time.time()
