"""
Index of ``data_dir`` and a safe trash.

Files are classified by their names (the same conventions the pipeline uses) into
typed records. Nothing here ever deletes: "remove" moves a file into
``data_dir/.flux_trash/<timestamp>/`` (confined to ``data_dir`` by realpath), and only
"empty trash" removes data, and only inside that trash folder.
"""

import json
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

TRASH = ".flux_trash"

KINDS = [
    "magnetogram",
    "pfss_pickle",
    "footpoints",
    "world",
    "relaxed_world",
    "wind",
    "plots",
    "fluxlight",
    "other",
]

_RULES = [
    ("magnetogram", re.compile(r"^CR(?P<cr>\d+)_(?:r(?P<reduce>\d+)_hmi|rf(?P<sel>\d+)_adapt)\.fits$")),
    ("pfss_pickle", re.compile(r"^pfss_cr(?P<cr>\d+)_r(?P<reduce>\d+)_(?P<inst>\w+)\.pkl$")),
    (
        "footpoints",
        re.compile(r"^floc_cr(?P<cr>\d+)_(?:r(?P<reduce>\d+)|rf(?P<sel>\d+))_f(?P<n>\d+)_(?P<inst>\w+)\.dat$"),
    ),
    ("relaxed_world", re.compile(r"^cr(?P<cr>\d+)_f(?P<n>\d+)_(?P<inst>\w+?)_relaxed_s(?P<steps>\d+)\.flux$")),
    ("world", re.compile(r"^cr(?P<cr>\d+)_f(?P<n>\d+)_(?P<inst>\w+)\.flux$")),
    ("fluxlight", re.compile(r"^cr(?P<cr>\d+)_fluxlight_.*\.png$")),
    ("plots", re.compile(r"^.*\.(?:png|pdf|jpg|jpeg|svg)$")),
]
_WIND_DIR = re.compile(r"(?:^|/)batches/(?P<batch>[^/]+)/data/cr(?P<cr>\d+)/wind/")
_BATCH = re.compile(r"(?:^|/)batches/(?P<batch>[^/]+)/")
_CR_IN_PATH = re.compile(r"(?:^|/)cr(?P<cr>\d+)(?:/|$)")


class CacheError(ValueError):
    pass


def classify(rel: str) -> Dict[str, Any]:
    """Kind and parameters for a path relative to data_dir."""
    name = os.path.basename(rel)
    info: Dict[str, Any] = {"kind": "other"}
    if rel.startswith(TRASH):
        return {"kind": "trash"}
    for kind, rx in _RULES:
        m = rx.match(name)
        if m:
            info = {"kind": kind, **{k: v for k, v in m.groupdict().items() if v is not None}}
            break
    else:
        if _WIND_DIR.search(rel):
            info = {"kind": "wind"}
    if _WIND_DIR.search(rel) and info["kind"] in ("other", "plots"):
        info["kind"] = "wind" if info["kind"] == "other" else "plots"
    if "inst" not in info:
        if name.endswith("_hmi.fits"):
            info["inst"] = "hmi"
        elif name.endswith("_adapt.fits"):
            info["inst"] = "adapt"
    m = _BATCH.search(rel)
    if m:
        info["batch"] = m.group("batch")
    if "cr" not in info:
        m = _CR_IN_PATH.search(rel)
        if m:
            info["cr"] = m.group("cr")
    return info


def _confine(data_dir: str, rel: str) -> str:
    """Absolute path of `rel` inside data_dir, or CacheError if it escapes (.., symlinks, absolute)."""
    root = os.path.realpath(data_dir)
    if not rel or os.path.isabs(rel):
        raise CacheError(f"not a path inside the data directory: {rel!r}")
    parts = rel.replace("\\", "/").split("/")
    if ".." in parts:
        raise CacheError(f"not a path inside the data directory: {rel!r}")
    candidate = os.path.join(root, *parts)
    # the parent chain must resolve inside root (a symlinked directory must not lead out)
    parent = os.path.realpath(os.path.dirname(candidate))
    if parent != root and not parent.startswith(root + os.sep):
        raise CacheError(f"path escapes the data directory: {rel!r}")
    if os.path.islink(candidate):
        target = os.path.realpath(candidate)
        if target != root and not target.startswith(root + os.sep):
            raise CacheError(f"symlink leads outside the data directory: {rel!r}")
    first = parts[0]
    if first == TRASH:
        raise CacheError("that path is already in the trash")
    if os.path.realpath(candidate) == root:
        raise CacheError("refusing to touch the data directory itself")
    return candidate


class Cache:
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self._records: Optional[List[Dict[str, Any]]] = None
        self._scanned_at = 0.0

    # ------------------------------------------------------------------ scanning
    def scan(self, max_files: int = 200000) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        root = self.data_dir
        if os.path.isdir(root):
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                if os.path.relpath(dirpath, root) == ".":
                    dirnames[:] = [d for d in dirnames if d != TRASH]
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    try:
                        st = os.lstat(full)
                    except OSError:
                        continue
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    rec = {
                        "id": rel,
                        "path": rel,
                        "name": fn,
                        "size": st.st_size,
                        "mtime": st.st_mtime,
                        **classify(rel),
                    }
                    records.append(rec)
                    if len(records) >= max_files:
                        break
        self._records = sorted(records, key=lambda r: (-r["mtime"], r["path"]))
        self._scanned_at = time.time()
        return self._records

    def records(self, rescan: bool = False) -> List[Dict[str, Any]]:
        if rescan or self._records is None:
            self.scan()
        return self._records or []

    def summary(self, rescan: bool = False) -> Dict[str, Any]:
        recs = self.records(rescan)
        tiles = {k: {"kind": k, "count": 0, "size": 0} for k in KINDS}
        for r in recs:
            t = tiles.setdefault(r["kind"], {"kind": r["kind"], "count": 0, "size": 0})
            t["count"] += 1
            t["size"] += r["size"]
        return {
            "data_dir": self.data_dir,
            "exists": os.path.isdir(self.data_dir),
            "total_count": len(recs),
            "total_size": sum(r["size"] for r in recs),
            "tiles": [tiles[k] for k in KINDS],
            "scanned_at": self._scanned_at,
            "trash": self.trash_summary(),
        }

    # ------------------------------------------------------------------ trash
    def _trash_root(self) -> str:
        return os.path.join(os.path.realpath(self.data_dir), TRASH)

    def move_to_trash(self, rels: List[str]) -> Dict[str, Any]:
        """Move files into a fresh timestamped trash folder. All-or-nothing validation first."""
        targets = [(rel, _confine(self.data_dir, rel)) for rel in rels]
        for rel, full in targets:
            if not os.path.lexists(full):
                raise CacheError(f"no such file: {rel}")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = os.path.join(self._trash_root(), stamp)
        n = 1
        while os.path.exists(folder):
            n += 1
            folder = os.path.join(self._trash_root(), f"{stamp}-{n}")
        moved, total = [], 0
        for rel, full in targets:
            dest = os.path.join(folder, *rel.replace("\\", "/").split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                total += os.lstat(full).st_size
            except OSError:
                pass
            shutil.move(full, dest)
            moved.append(rel)
        manifest = os.path.join(folder, "manifest.json")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({"trashed": time.time(), "files": moved}, f)
        self._records = None
        return {"folder": os.path.basename(folder), "moved": moved, "size": total}

    def list_trash(self) -> List[Dict[str, Any]]:
        out = []
        root = self._trash_root()
        if not os.path.isdir(root):
            return out
        for stamp in sorted(os.listdir(root), reverse=True):
            folder = os.path.join(root, stamp)
            if not os.path.isdir(folder):
                continue
            for dirpath, _, filenames in os.walk(folder):
                for fn in filenames:
                    if fn == "manifest.json" and dirpath == folder:
                        continue
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, folder).replace(os.sep, "/")
                    st = os.lstat(full)
                    out.append(
                        {
                            "id": f"{stamp}/{rel}",
                            "folder": stamp,
                            "path": rel,
                            "size": st.st_size,
                            "mtime": st.st_mtime,
                            **classify(rel),
                        }
                    )
        return out

    def trash_summary(self) -> Dict[str, Any]:
        items = self.list_trash()
        return {"count": len(items), "size": sum(i["size"] for i in items)}

    def restore(self, ids: List[str]) -> List[str]:
        root = self._trash_root()
        restored = []
        for item in ids:
            parts = item.replace("\\", "/").split("/", 1)
            if len(parts) != 2 or ".." in item.split("/"):
                raise CacheError(f"bad trash id: {item!r}")
            stamp, rel = parts
            src = os.path.join(root, stamp, *rel.split("/"))
            parent = os.path.realpath(os.path.dirname(src))
            if parent != os.path.realpath(root) and not parent.startswith(os.path.realpath(root) + os.sep):
                raise CacheError(f"bad trash id: {item!r}")
            if not os.path.lexists(src):
                raise CacheError(f"not in the trash: {item}")
            dest = _confine(self.data_dir, rel)
            if os.path.lexists(dest):
                raise CacheError(f"{rel} already exists; not overwriting")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(src, dest)
            restored.append(rel)
        self._prune_empty_trash_dirs()
        self._records = None
        return restored

    def _prune_empty_trash_dirs(self) -> None:
        root = self._trash_root()
        if not os.path.isdir(root):
            return
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            if dirpath == root:
                continue
            left = [f for f in filenames if not (f == "manifest.json" and os.path.dirname(dirpath) == root)]
            if not left and not [d for d in dirnames if os.path.isdir(os.path.join(dirpath, d))]:
                shutil.rmtree(dirpath, ignore_errors=True)

    def empty_trash(self) -> Dict[str, Any]:
        root = self._trash_root()
        removed, size = 0, 0
        if (
            os.path.isdir(root)
            and not os.path.islink(root)
            and os.path.realpath(root).startswith(os.path.realpath(self.data_dir) + os.sep)
        ):
            for entry in os.listdir(root):
                full = os.path.join(root, entry)
                for dirpath, _, filenames in os.walk(full):
                    for fn in filenames:
                        if fn == "manifest.json" and dirpath == full:
                            continue
                        try:
                            size += os.lstat(os.path.join(dirpath, fn)).st_size
                        except OSError:
                            pass
                        removed += 1
                shutil.rmtree(full, ignore_errors=True)
        self._records = None
        return {"removed": removed, "size": size}
