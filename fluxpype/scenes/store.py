"""Where scenes live: `<data dir>/scenes/<id>.scene.json` (the spec) beside `<id>.flux` (its compiled world).

The folder sits in the data directory so Plumber's cache browser and 3-D viewer can already open the world. A scene
is never overwritten here: a new version needs a new id, so a run that points at a scene keeps meaning one thing.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from .checks import summarize
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
                        "wall_radius": s.get("wall_radius"),
                        "notes": s["notes"],
                        "modified": os.path.getmtime(os.path.join(d, fn)),
                    }
                )
    return sorted(rows, key=lambda r: -r["modified"])
