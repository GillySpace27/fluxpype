"""
Unified console output and log file handling for FLUXpype.

Provides Rich-colored output to the terminal and a plain-text log file
for each run. All output (Python and subprocess) is teed to both.
"""

import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from rich.console import Console

# ANSI escape sequence pattern for stripping colors in log file
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Module-level state
_log_file = None
_console = Console()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes for plain-text log output."""
    return _ANSI_RE.sub("", text)


def init_log(batch_name: str = "run") -> Path:
    """
    Initialize log file for this run.

    Creates fluxpype/logs/ if needed and opens a timestamped log file.
    Format: fluxpype_YYYY-MM-DD_HH-MM-SS_{batch_name}.log

    Returns
    -------
    Path
        Path to the log file.
    """
    global _log_file
    if _log_file is not None:
        return Path(_log_file.name)

    # logs dir: fluxpype/logs (project root)
    package_dir = Path(__file__).parent
    project_root = package_dir.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_batch = re.sub(r"[^\w\-.]", "_", str(batch_name))
    log_path = logs_dir / f"fluxpype_{timestamp}_{safe_batch}.log"
    _log_file = open(log_path, "w", encoding="utf-8")
    return log_path


def get_log_file():
    """Return the current log file handle, or None if not initialized."""
    return _log_file


def write_to_log(text: str) -> None:
    """Write plain text to the log file (strips ANSI)."""
    if _log_file is not None:
        _log_file.write(strip_ansi(text))
        _log_file.flush()


def emit(msg: str = "", style=None, end: str = "\n") -> None:
    """
    Print to console (with Rich) and tee to log file (plain text).

    Parameters
    ----------
    msg : str
        Message to display.
    style : str, optional
        Rich style string (e.g. 'cyan', 'bold green').
    end : str
        Line ending (default newline).
    """
    if style:
        _console.print(msg, style=style, end=end)
    else:
        _console.print(msg, end=end)
    write_to_log(msg + end)


def emit_raw(text: str) -> None:
    """
    Emit raw text (e.g. from subprocess) to both console and log.
    Console gets it as-is; log gets ANSI stripped.
    """
    _console.print(text, end="")
    write_to_log(strip_ansi(text))


def close_log() -> None:
    """Close the log file."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


@contextmanager
def log_session(batch_name: str = "run"):
    """
    Context manager to initialize and close a log session.

    Usage
    -----
    with log_session(batch_name):
        # ... run pipeline ...
    """
    path = init_log(batch_name)
    try:
        yield path
    finally:
        close_log()
