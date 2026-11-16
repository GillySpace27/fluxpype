"""Where scenes live: `<data dir>/scenes/<id>.scene.json` (the spec) beside `<id>.flux` (its compiled world).

The folder sits in the data directory so Plumber's cache browser and 3-D viewer can already open the world. A scene
is never overwritten here: a new version needs a new id, so a run that points at a scene keeps meaning one thing.

An edit (`edit`) saves a new scene under a new id and writes `<id>.ops.json` beside it: the parent scene, the operations
that made it, who made them and when. Scenes therefore form a tree of immutable versions. Undo is "open the parent";
two edits of the same parent are two variants; nothing is ever deleted or rewritten.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from .checks import comparable, summarize
from .ops import OpError, apply_ops
from .planar import compile_flux
from .spec import PlanarArcadeSpec

ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
FOLDER = "scenes"


class SceneError(ValueError):
    pass


def folder(data_dir: str) -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(data_dir)), FOLDER)


def check_id(scene_id: str) -> str:
    if not ID_RE.match(scene_id or ""):
        raise SceneError("a scene name starts with a letter and uses letters, digits, dash and underscore (48 at most)")
    return scene_id


def prepare(spec_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a scene and compile it, without writing: the spec, the .flux text and its construction checks.

    Raises SceneError for a bad spec or a world that fails its own construction checks.
    """
    try:
        spec = PlanarArcadeSpec.from_dict(spec_dict)
    except (ValueError, TypeError) as e:
        raise SceneError(str(e))
    check_id(spec.name)
    text = compile_flux(spec)
    report = summarize(text)
    if report["problems"]:
        raise SceneError("the compiled world fails its construction checks: " + "; ".join(report["problems"]))
    return {"spec": spec, "text": text, "report": report}


def save(data_dir: str, spec_dict: Dict[str, Any]) -> Dict[str, Any]:
    made = prepare(spec_dict)
    spec: PlanarArcadeSpec = made["spec"]
    d = folder(data_dir)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, spec.name)
    if os.path.exists(base + ".scene.json"):
        raise SceneError(f"scene {spec.name!r} already exists; save a new version under a new name")
    for path, body in ((base + ".flux", made["text"]), (base + ".scene.json", json.dumps(spec.to_dict(), indent=1))):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(body)
        os.replace(tmp, path)
    return {
        "id": spec.name,
        "hash": spec.content_hash(),
        "world": f"{FOLDER}/{spec.name}.flux",
        "report": made["report"],
    }


def get(data_dir: str, scene_id: str) -> Optional[Dict[str, Any]]:
    try:
        check_id(scene_id)
        with open(os.path.join(folder(data_dir), scene_id + ".scene.json")) as f:
            spec = PlanarArcadeSpec.from_dict(json.load(f))
    except (OSError, ValueError, SceneError):
        return None
    return {"id": scene_id, "hash": spec.content_hash(), "world": f"{FOLDER}/{scene_id}.flux", "spec": spec.to_dict()}


def list_scenes(data_dir: str) -> List[Dict[str, Any]]:
    d = folder(data_dir)
    if not os.path.isdir(d):
        return []
    rows = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".scene.json"):
            row = get(data_dir, fn[: -len(".scene.json")])
            if row:
                s = row["spec"]
                rows.append(
                    {
                        **{k: row[k] for k in ("id", "hash", "world")},
                        "lines": s["n_lines"],
                        "lid_height": s["lid_height"],
                        "notes": s["notes"],
                        "modified": os.path.getmtime(os.path.join(d, fn)),
                    }
                )
    return sorted(rows, key=lambda r: -r["modified"])


def _meta_path(data_dir: str, scene_id: str) -> str:
    return os.path.join(folder(data_dir), scene_id + ".ops.json")


def _meta(data_dir: str, scene_id: str) -> Optional[Dict[str, Any]]:
    try:
        with open(_meta_path(data_dir, scene_id)) as f:
            m = json.load(f)
        return m if isinstance(m, dict) else None
    except (OSError, ValueError):
        return None


def _next_name(data_dir: str, scene_id: str) -> str:
    """`arc` -> `arc-v2`, `arc-v2` -> `arc-v3` (the first free number for this scene's root name)."""
    base = re.sub(r"-v\d+$", "", scene_id)
    n = 2
    while os.path.exists(os.path.join(folder(data_dir), f"{base}-v{n}.scene.json")) and n < 10000:
        n += 1
    return check_id(f"{base}-v{n}")[:48]


def plan_edit(data_dir: str, scene_id: str, ops: List[Dict[str, Any]], name: Optional[str] = None) -> Dict[str, Any]:
    """What an edit would do, without writing: the new spec, its world and checks, the fields changed and whether the
    new world is a fair pair with its parent for an energy comparison (only connectivity may differ).
    """
    parent = get(data_dir, scene_id)
    if not parent:
        raise SceneError(f"unknown scene {scene_id!r}")
    try:
        new_spec, changes = apply_ops(parent["spec"], ops)
    except OpError as e:
        raise SceneError(str(e))
    if not changes:
        raise SceneError("these operations change nothing")
    new_name = check_id(name) if name else _next_name(data_dir, scene_id)
    if os.path.exists(os.path.join(folder(data_dir), new_name + ".scene.json")):
        raise SceneError(f"scene {new_name!r} already exists; choose another name")
    new_spec["name"] = new_name
    made = prepare(new_spec)
    why = comparable(compile_flux(PlanarArcadeSpec.from_dict(parent["spec"])), made["text"])
    return {
        **made,
        "id": new_name,
        "parent": scene_id,
        "changes": changes,
        "fair_pair_with_parent": {"fair": not why, "why": why},
    }


def edit(
    data_dir: str,
    scene_id: str,
    ops: List[Dict[str, Any]],
    label: Optional[str] = None,
    name: Optional[str] = None,
    author: str = "api",
) -> Dict[str, Any]:
    """Apply operations to a saved scene and save the result as a new version (the parent is left as it is)."""
    plan = plan_edit(data_dir, scene_id, ops, name)
    saved = save(data_dir, plan["spec"].to_dict())
    meta = {
        "parent": scene_id,
        "ops": ops,
        "changes": plan["changes"],
        "label": (label or "")[:120],
        "author": author if author in ("web", "mcp", "ask", "api") else "api",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = _meta_path(data_dir, saved["id"])
    with open(path + ".tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(path + ".tmp", path)
    lid = plan["spec"].lid_height
    return {
        **saved,
        "parent": scene_id,
        "changes": plan["changes"],
        "fair_pair_with_parent": plan["fair_pair_with_parent"],
        "lid_height": lid,
    }


def _node(data_dir: str, scene_id: str) -> Optional[Dict[str, Any]]:
    row = get(data_dir, scene_id)
    if not row:
        return None
    m = _meta(data_dir, scene_id) or {}
    return {
        "id": scene_id,
        "hash": row["hash"],
        "parent": m.get("parent"),
        "label": m.get("label", ""),
        "author": m.get("author"),
        "created": m.get("created"),
        "ops": m.get("ops", []),
        "changes": m.get("changes", []),
    }


def history(data_dir: str, scene_id: str) -> Optional[Dict[str, Any]]:
    """The versions leading to a scene (oldest first, ending with it) and the versions made from it."""
    node = _node(data_dir, scene_id)
    if not node:
        return None
    chain = [node]
    while chain[0]["parent"] and len(chain) < 200:
        up = _node(data_dir, chain[0]["parent"])
        if not up or any(up["id"] == c["id"] for c in chain):
            break
        chain.insert(0, up)
    d = folder(data_dir)
    kids = []
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".ops.json"):
                m = _meta(data_dir, fn[: -len(".ops.json")])
                if m and m.get("parent") == scene_id:
                    kid = _node(data_dir, fn[: -len(".ops.json")])
                    if kid:
                        kids.append(kid)
    return {"id": scene_id, "versions": chain, "children": kids}
