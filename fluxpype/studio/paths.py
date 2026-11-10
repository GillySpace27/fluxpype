"""Where the studio keeps its state."""

import os
from pathlib import Path


def studio_home() -> Path:
    """Studio state directory: $FLUX_STUDIO_HOME or ~/.fluxpype/studio (created on demand)."""
    home = Path(os.environ.get("FLUX_STUDIO_HOME") or Path.home() / ".fluxpype" / "studio")
    home.mkdir(parents=True, exist_ok=True)
    return home


def db_path(home: Path = None) -> Path:
    return (home or studio_home()) / "studio.db"


def runs_dir(home: Path = None) -> Path:
    path = (home or studio_home()) / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def token_path(home: Path = None) -> Path:
    return (home or studio_home()) / "token"


def checkout_root() -> Path:
    """The repository checkout that holds the fluxpype package (the runner's cwd)."""
    return Path(__file__).resolve().parent.parent.parent


def shipped_config() -> Path:
    return Path(__file__).resolve().parent.parent / "config.ini"
