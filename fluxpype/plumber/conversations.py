"""Ask conversations, kept under the Plumber home so a Co-design session survives a page reload and a restart.

One file per conversation (``ask_conversations/<id>.json``, 0600): the mode and model it was started with, the CLI
session id it maps to, the turns (question, answer, cost, proposals) and the running cost. The browser makes the id;
the CLI session id is Plumber's own (a uuid), so nothing the person types ever reaches a flag. The file holds what
the person and Plum said, never tool results (untrusted, and bulky). Stdlib only.
"""

import json
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

DIR = "ask_conversations"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,40}$")
KEEP_FILES = 100  # oldest conversations dropped past this
KEEP_TURNS = 200
ANSWER_CAP = 20000  # characters of one stored answer


class ConversationError(ValueError):
    pass


class Conversations:
    def __init__(self, home: str):
        self.dir = os.path.join(str(home), DIR)
        self.lock = threading.RLock()

    def _path(self, cid: str) -> str:
        if not ID_RE.match(cid or ""):
            raise ConversationError("a conversation id is 8 to 40 letters, digits, dash or underscore")
        return os.path.join(self.dir, cid + ".json")

    def load(self, cid: str) -> Optional[Dict[str, Any]]:
        try:
            with open(self._path(cid), encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            return None
        return rec if isinstance(rec, dict) and rec.get("id") == cid else None

    def _save(self, rec: Dict[str, Any]) -> None:
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        path = self._path(rec["id"])
        tmp = path + ".tmp"
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(rec, f)
            os.replace(tmp, path)
        except OSError:
            pass  # the panel still has the turns until it is reloaded

    def get_or_create(self, cid: str, mode: str, model: str) -> Dict[str, Any]:
        with self.lock:
            rec = self.load(cid)
            if rec:
                return rec
            now = time.time()
            rec = {
                "id": cid,
                "mode": mode,
                "model": model,
                "session": str(uuid.uuid4()),  # the CLI session this conversation resumes
                "started": False,  # True once a turn has reached the CLI with this session id
                "created": now,
                "updated": now,
                "cost": 0.0,
                "turns": [],
                "woken_jobs": [],
            }
            self._save(rec)
            self._prune()
            return rec

    def _prune(self) -> None:
        try:
            files = sorted(
                (os.path.join(self.dir, n) for n in os.listdir(self.dir) if n.endswith(".json")), key=os.path.getmtime
            )
        except OSError:
            return
        for p in files[: max(0, len(files) - KEEP_FILES)]:
            try:
                os.remove(p)
            except OSError:
                pass

    def update(self, cid: str, **fields: Any) -> Optional[Dict[str, Any]]:
        with self.lock:
            rec = self.load(cid)
            if not rec:
                return None
            rec.update(fields, updated=time.time())
            self._save(rec)
            return rec

    def add_turn(self, cid: str, turn: Dict[str, Any], cost: float = 0.0) -> Optional[Dict[str, Any]]:
        with self.lock:
            rec = self.load(cid)
            if not rec:
                return None
            turn = {**turn, "answer": str(turn.get("answer", ""))[:ANSWER_CAP]}
            rec["turns"] = (rec["turns"] + [turn])[-KEEP_TURNS:]
            rec["cost"] = round(float(rec.get("cost", 0)) + float(cost or 0), 6)
            rec["updated"] = time.time()
            self._save(rec)
            return rec

    def list(self) -> List[Dict[str, Any]]:
        out = []
        try:
            names = [n for n in os.listdir(self.dir) if n.endswith(".json")]
        except OSError:
            return out
        for n in names:
            rec = self.load(n[:-5])
            if rec:
                out.append(
                    {k: rec.get(k) for k in ("id", "mode", "model", "created", "updated", "cost")}
                    | {"turns": len(rec["turns"])}
                )
        return sorted(out, key=lambda r: -(r["updated"] or 0))

    def find_by_turn(self, question_id: str) -> Optional[str]:
        """The conversation whose turn carries this question id (what a job started from its proposal records)."""
        if not question_id:
            return None
        for row in self.list():
            rec = self.load(row["id"])
            if rec and any(t.get("id") == question_id for t in rec["turns"]):
                return rec["id"]
        return None

    def mark_woken(self, cid: str, job_id: str) -> bool:
        """Remember that a job has woken this conversation; False when it already did (a job wakes Plum once)."""
        with self.lock:
            rec = self.load(cid)
            if not rec or job_id in rec.get("woken_jobs", []):
                return False
            rec["woken_jobs"] = (rec.get("woken_jobs", []) + [job_id])[-200:]
            self._save(rec)
            return True
