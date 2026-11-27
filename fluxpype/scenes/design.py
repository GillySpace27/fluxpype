"""Validate and preview a scene spec without saving or running it.

One implementation behind both the MCP tools (`validate_scene`, `preview_scene`) and the read-only HTTP routes
(`POST /api/scenes/validate`, `POST /api/scenes/preview`), so a person at the web form and an agent see the same JSON.
"""

from typing import Any, Dict, Optional

from . import checks, store
from .store import SceneError

NOTE = "Connectivity is hand-set and these are construction checks only; nothing here says how the world relaxes."
REPORT_KEYS = ("lines", "vertices", "net_flux", "max_height", "min_line_separation", "photosphere", "lid")


def validate(
    spec: Dict[str, Any], compare_spec: Optional[Dict[str, Any]] = None, compare_id: str = ""
) -> Dict[str, Any]:
    """Compile `spec` and run its construction checks. An invalid spec gives {"ok": False, "problem": ...} (not an
    exception). `compare_spec` (a saved scene's spec, named `compare_id`) adds the fair-pair verdict; a bad compare
    spec raises SceneError."""
    try:
        made = store.prepare(spec)
    except SceneError as e:
        return {"ok": False, "problem": str(e)}
    rep = made["report"]
    out: Dict[str, Any] = {
        "ok": True,
        "name": made["spec"].name,
        "content_hash": made["spec"].content_hash(),
        "pairing": made["spec"].pairs(),
        "report": {k: rep[k] for k in REPORT_KEYS},
        "note": NOTE,
    }
    if compare_spec is not None:
        try:
            text_b = store.prepare(compare_spec)["text"]
        except SceneError as e:
            raise SceneError(f"{compare_id}: {e}")
        why = checks.comparable(made["text"], text_b)
        out["fair_pair_with"] = {"scene": compare_id, "fair": not why, "why_not": why}
    return out


def preview(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The compiled world in brief (header lines, bounding box, first lines). Raises SceneError for an invalid spec."""
    made = store.prepare(spec)
    return {"name": made["spec"].name, "pairing": made["spec"].pairs(), **checks.preview(made["text"])}
