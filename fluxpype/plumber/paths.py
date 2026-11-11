"""Where the plumber keeps its state."""

import os
from pathlib import Path


def plumber_home() -> Path:
    """Plumber state directory: $FLUX_PLUMBER_HOME or ~/.fluxpype/plumber (created on demand)."""
    home = Path(os.environ.get("FLUX_PLUMBER_HOME") or Path.home() / ".fluxpype" / "plumber")
    home.mkdir(parents=True, exist_ok=True)
    return home


def db_path(home: Path = None) -> Path:
    return (home or plumber_home()) / "plumber.db"


def runs_dir(home: Path = None) -> Path:
    path = (home or plumber_home()) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def token_path(home: Path = None) -> Path:
    return (home or plumber_home()) / "token"


def checkout_root() -> Path:
    """The repository checkout that holds the fluxpype package (the runner's cwd)."""
    return Path(__file__).resolve().parent.parent.parent


def shipped_config() -> Path:
    return Path(__file__).resolve().parent.parent / "config.ini"
