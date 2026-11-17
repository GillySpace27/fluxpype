"""Typed edit operations on a planar scene: the one way the panel, Plum and scripts change a scene.

An operation is a small JSON object, so the same edit can come from a form field, from a Plum proposal or from a script:

    {"op": "set", "field": "lid_height", "value": 8}      change one spec field
    {"op": "shear", "k": 2}                                pair each positive footpoint with the one k steps along
                                                           (k = 0 puts the arcs straight across again)
    {"op": "shear_footpoints", "offset": 0.5}              slide the negative row along the PIL by `offset` and keep
                                                           each line on the footpoints it has now (the stressed state)
    {"op": "potential"}                                    swap the lines for the traced potential field of the
                                                           footpoints as they are (the reference state; see potential_trace.py)

A stressed scene and the potential version of it have the same footpoints, photosphere and lid, so they are a fair
pair for `relax compare`: the difference is energy above the computed potential seed's relaxed state, at that lid.

`apply_ops` is a pure function on a spec dict. It never writes anything; `store.edit` saves the result as a new
version. Every edit comes back with the list of fields it changed, so a reader can see exactly what an edit did.
"""

from typing import Any, Dict, List, Tuple

from .spec import PlanarArcadeSpec, shift_pairing

MAX_OPS = 20

# every spec field except the schema version and the name (a version's name is its id, chosen by the store)
EDITABLE = (
    "n_lines",
    "separation",
    "spacing",
    "shear_offset",
    "pairing",
    "flux_per_line",
    "height_ratio",
    "max_height",
    "vertices_per_line",
    "photosphere_z",
    "lid_height",
    "seed",
    "notes",
)


class OpError(ValueError):
    pass


def _one(spec: Dict[str, Any], op: Any) -> None:
    if not isinstance(op, dict) or "op" not in op:
        raise OpError('each operation is an object with an "op" key')
    kind = op["op"]
    if kind == "set":
        field = op.get("field")
        if field not in EDITABLE:
            raise OpError(f"cannot set {field!r}; the editable fields are: {', '.join(EDITABLE)}")
        if "value" not in op:
            raise OpError('"set" needs a "value" (use null to clear lid_height or pairing)')
        spec[field] = op["value"]
        # a pairing is a permutation of the lines, so it cannot survive a change in their number
        if field == "n_lines" and spec.get("pairing") is not None and len(spec["pairing"]) != op["value"]:
            spec["pairing"] = None
    elif kind == "shear":
        k = op.get("k")
        if not isinstance(k, int) or isinstance(k, bool):
            raise OpError('"shear" needs an integer "k"')
        n = spec.get("n_lines")
        if not isinstance(n, int) or not 0 <= k < n:
            raise OpError(f"shear k must be from 0 to n_lines - 1 ({n} lines here)")
        spec["pairing"] = None if k == 0 else shift_pairing(n, k)
        spec["seed"] = "arc"
    elif kind == "shear_footpoints":
        # the physical shear: slide the negative row along the PIL and keep every line on its own footpoints
        off = op.get("offset")
        if isinstance(off, bool) or not isinstance(off, (int, float)):
            raise OpError('"shear_footpoints" needs a number "offset" (distance along the PIL, model units)')
        try:
            before = PlanarArcadeSpec.from_dict(dict(spec))
        except (ValueError, TypeError) as e:
            raise OpError(str(e))
        spec["pairing"] = before.pairs()  # freeze the connectivity it has now, traced or hand-set
        spec["seed"] = "arc"
        spec["shear_offset"] = float(off)
    elif kind == "potential":
        # the reference state for the footpoints as they are: traced potential lines, derived pairing
        spec["pairing"] = None
        spec["seed"] = "potential"
    else:
        raise OpError(f"unknown operation {kind!r}; use set, shear, shear_footpoints or potential")


def apply_ops(spec: Dict[str, Any], ops: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The edited spec (validated) and the fields that changed, as [{"field", "from", "to"}]."""
    if not isinstance(ops, list) or not ops:
        raise OpError("give a non-empty list of operations")
    if len(ops) > MAX_OPS:
        raise OpError(f"at most {MAX_OPS} operations in one edit")
    new = {k: (list(v) if isinstance(v, list) else v) for k, v in spec.items()}
    for op in ops:
        _one(new, op)
    try:
        PlanarArcadeSpec.from_dict(new)
    except (ValueError, TypeError) as e:
        raise OpError(str(e))
    changes = [{"field": k, "from": spec.get(k), "to": new.get(k)} for k in EDITABLE if spec.get(k) != new.get(k)]
    return new, changes
