"""
Find FLUX data folders on this machine, so Plumber can offer the one that holds your runs.

A data folder is what config.ini's ``data_dir`` points at: ``magnetograms/``, ``pfss/`` and
``batches/<batch>/data/cr<N>/``. Discovery looks at ``<base>/fluxpype/fluxpype/data`` for every
sibling of the checkout's base folder (for example every ``~/vscode/*``), plus the folders Plumber
already knows. It only lists directories; it never opens files.
"""

import os
from typing import Dict, Iterable, List


def _count(path: str, suffixes) -> int:
    try:
        return sum(1 for e in os.scandir(path) if e.name.endswith(suffixes))
    except OSError:
        return 0


def _dirs(path: str) -> int:
    try:
        return sum(1 for e in os.scandir(path) if e.is_dir() and not e.name.startswith("."))
    except OSError:
        return 0


def describe(path: str) -> Dict:
    """What a folder holds, by the pipeline's layout. Cheap: three directory listings."""
    exists = os.path.isdir(path)
    out = {
        "path": path,
        "real": os.path.realpath(path) if exists else path,
        "exists": exists,
        "magnetograms": _count(os.path.join(path, "magnetograms"), (".fits", ".fts", ".fits.gz", ".fts.gz")),
        "pfss": _count(os.path.join(path, "pfss"), ".pkl"),
        "batches": _dirs(os.path.join(path, "batches")),
    }
    out["has_data"] = bool(out["magnetograms"] or out["pfss"] or out["batches"])
    return out


def candidates(base_dir: str, known: Iterable[str] = ()) -> List[Dict]:
    """Data folders worth offering, one per real location, those with data first."""
    paths = [p for p in known if p]
    parent = os.path.dirname(os.path.abspath(base_dir))
    try:
        for e in sorted(os.scandir(parent), key=lambda e: e.name):
            if e.is_dir() and not e.name.startswith("."):
                paths.append(os.path.join(e.path, "fluxpype", "fluxpype", "data"))
    except OSError:
        pass
    seen: Dict[str, Dict] = {}
    for p in paths:
        d = describe(p)
        if not d["exists"]:
            continue
        if d["real"] in seen:
            known_as = seen[d["real"]]
            if p != known_as["path"] and p not in known_as.get("aliases", []):
                known_as.setdefault("aliases", []).append(p)
            continue
        seen[d["real"]] = d
    return sorted(seen.values(), key=lambda d: (not d["has_data"], -(d["magnetograms"] + d["pfss"]), d["path"]))
