"""Changes the Ask bar's model proposed, held until a person runs or rejects them.

The Ask CLI's MCP server may not change Plumber: server.py refuses every write it sends except ``POST /api/proposals``.
Its write tools post there instead. Plumber checks the arguments with the code the real route uses, stores the
normalized arguments here and returns an id. A click on the card (``POST /api/proposals/{id}/approve``, refused to the
Ask client) runs exactly the stored arguments, once. The Ask client may also withdraw its own pending proposal
(``withdraw``, or ``supersedes`` on a new one); a withdrawn proposal never runs. Stdlib only.
"""

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

TOOLS = ("submit_job", "cancel", "retry_failed", "set_queue", "create_preset", "create_scene")
MAX_PENDING = 20
PER_QUESTION = 6
WITHDRAW_PER_QUESTION = 6
EXPIRY_S = 24 * 3600
ARGS_CAP = 4000  # characters of JSON
KEEP = 200  # decided proposals kept, newest, so the next question can read what became of them


class ProposalError(ValueError):
    def __init__(self, message: str, code: int = 400):
        super().__init__(message)
        self.code = code


def load(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
    except (OSError, ValueError):
        return []
    return [p for p in items if isinstance(p, dict) and "id" in p] if isinstance(items, list) else []


def for_question(path: str, question: str) -> List[str]:
    """Ids of the proposals one Ask question made, oldest first."""
    return [p["id"] for p in load(path) if question and p.get("question") == question]


class ProposalStore:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.items = load(path)  # oldest first

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.items, f)
            os.replace(tmp, self.path)
        except OSError:
            pass  # the in-memory record still works until a restart

    def _expire(self) -> None:
        now = time.time()
        for p in self.items:
            if p["status"] == "pending" and now - p["created"] > EXPIRY_S:
                p["status"], p["decided"] = "expired", p["created"] + EXPIRY_S

    def _find(self, pid: str) -> Dict[str, Any]:
        p = next((p for p in self.items if p["id"] == pid), None)
        if p is None:
            raise ProposalError("unknown proposal", 404)
        return p

    def list(self) -> List[Dict[str, Any]]:
        with self.lock:
            self._expire()
            return [dict(p) for p in reversed(self.items)]

    def pending(self) -> List[Dict[str, Any]]:
        return [p for p in self.list() if p["status"] == "pending"]

    def get(self, pid: str) -> Dict[str, Any]:
        with self.lock:
            return dict(self._find(pid))

    def _withdrawable(self, pid: str, question: str) -> Dict[str, Any]:
        """The Ask bar's own pending proposal, from this question or an earlier one; call under the lock."""
        p = self._find(pid)
        if p.get("origin") != "ask":
            raise ProposalError("only a proposal the Ask bar made can be withdrawn", 403)
        if p["status"] != "pending":
            raise ProposalError(f"this proposal is {p['status']}, not pending", 409)
        if sum(q.get("withdrawn_in") == question for q in self.items) >= WITHDRAW_PER_QUESTION:
            raise ProposalError(f"at most {WITHDRAW_PER_QUESTION} withdrawals per question", 429)
        return p

    @staticmethod
    def _close(p: Dict[str, Any], question: str, reason: str, by: Optional[str] = None) -> None:
        p.update(status="withdrawn", decided=time.time(), reason=reason[:300], superseded_by=by, withdrawn_in=question)

    def withdraw(self, pid: str, question: str, reason: str = "") -> Dict[str, Any]:
        """Close a pending Ask proposal without running it. Same lock as claim, so a running one cannot be withdrawn."""
        with self.lock:
            self._expire()
            p = self._withdrawable(pid, question)
            self._close(p, question, reason or "withdrawn by the Ask bar")
            self._save()
            return dict(p)

    def add(
        self,
        tool: str,
        args: Dict[str, Any],
        summary: str,
        question: str,
        diff: Optional[list] = None,
        supersedes: Optional[str] = None,
    ) -> Dict[str, Any]:
        if tool not in TOOLS:
            raise ProposalError(f"unknown tool {tool!r}")
        if len(json.dumps(args)) > ARGS_CAP:
            raise ProposalError(f"the arguments are over {ARGS_CAP} characters")
        with self.lock:
            self._expire()
            pending = [p for p in self.items if p["status"] == "pending"]
            if len(pending) >= MAX_PENDING:
                raise ProposalError(
                    f"{MAX_PENDING} proposals are already waiting; the person must run or reject some first", 429
                )
            if sum(p.get("question") == question for p in self.items) >= PER_QUESTION:
                raise ProposalError(f"at most {PER_QUESTION} proposals per question", 429)
            old = self._withdrawable(supersedes, question) if supersedes else None
            p = {
                "id": uuid.uuid4().hex[:12],
                "tool": tool,
                "args": args,
                "summary": summary,
                "diff": diff or [],
                "origin": "ask",
                "question": question,
                "created": time.time(),
                "status": "pending",
                "decided": None,
                "result": None,
                "error": None,
                "supersedes": supersedes,
            }
            self.items.append(p)
            if old:
                self._close(old, question, "replaced by a revised proposal", p["id"])
            decided = [q for q in self.items if q["status"] != "pending"]
            if len(decided) > KEEP:
                drop = {q["id"] for q in decided[: len(decided) - KEEP]}
                self.items = [q for q in self.items if q["id"] not in drop]
            self._save()
            return dict(p)

    def claim(self, pid: str) -> Dict[str, Any]:
        """Mark a pending proposal as running and return it; a second claim fails, so a proposal runs at most once."""
        with self.lock:
            self._expire()
            p = self._find(pid)
            if p["status"] != "pending":
                raise ProposalError(f"this proposal is {p['status']}, not pending", 409)
            p["status"] = "running"
            self._save()
            return dict(p)

    def finish(self, pid: str, status: str, result: Any = None, error: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            p = self._find(pid)
            p.update(status=status, decided=time.time(), result=result, error=error)
            self._save()
            return dict(p)

    def reject(self, pid: str) -> Dict[str, Any]:
        with self.lock:
            self._expire()
            p = self._find(pid)
            if p["status"] != "pending":
                raise ProposalError(f"this proposal is {p['status']}, not pending", 409)
            p.update(status="rejected", decided=time.time())
            self._save()
            return dict(p)
