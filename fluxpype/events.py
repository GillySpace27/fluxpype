"""
Structured progress events for FLUXpype.

``emit(type, **fields)`` appends one JSON line to the file named by the
``FLUXPYPE_EVENTS`` environment variable. When the variable is unset it does
nothing, so terminal runs behave exactly as before. Every failure is swallowed:
telemetry must never break a pipeline run.

The schema is documented in ``docs/gui/EVENTS.md``. This module uses only the
standard library.
"""

import json
import os
import time

SCHEMA_VERSION = 1


def enabled():
    """True when events are being collected for this process."""
    return bool(os.environ.get("FLUXPYPE_EVENTS"))


def emit(type, **fields):  # noqa: A002 - "type" is the schema's field name
    """Append one event line. Returns True if a line was written."""
    path = os.environ.get("FLUXPYPE_EVENTS")
    if not path:
        return False
    try:
        record = {
            "v": SCHEMA_VERSION,
            "t": time.time(),
            "type": type,
            "pid": os.getpid(),
            "task": os.environ.get("FLUXPYPE_TASK_ID") or None,
        }
        record.update(fields)
        line = (json.dumps(record, separators=(",", ":"), default=str) + "\n").encode("utf-8")
        # one O_APPEND write per line: atomic for lines below PIPE_BUF on local filesystems
        fd = os.open(os.path.expanduser(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
        return True
    except Exception:
        return False
