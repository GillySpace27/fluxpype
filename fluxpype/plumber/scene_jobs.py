"""Scene relaxations as queue jobs: planning, saving the lid variants, and reading results back.

A saved scene (``scenes/<id>.scene.json`` with its ``.flux``) can be relaxed with FLUX headless, one world (``run``) or
a fair pair compared at once (``compare``), as ``python -m fluxpype.scenes.relax`` does. Here that runs as an ordinary
Plumber job so it shows on the job page, takes run slots and the CPU budget like a pipeline task, and can be proposed
by Plum. How it sits in the queue (scheduler.create_scene_job):

* a job with the reserved profile ``scene-relax`` or ``scene-compare``; its ``extra`` carries ``kind: scene``;
* one task per scene (``run``) or per pair (``compare``), and per lid height when a sweep is asked for. The task's
  ``cr`` holds the scene label, ``adapt`` is ``scene``, ``method`` is the mode: the store's columns are text, so no
  schema change is needed;
* ``run`` takes one slot, ``compare`` two (both worlds relax at once);
* the task's inputs live in ``<job dir>/scene-<task id>.json`` (the job dir survives a retry, the run dir does not);
  its outputs (events, log, ``series.jsonl``, ``result.json``, the relaxed worlds) are in the task's run dir.

A lid sweep needs one world per lid, so each lid height makes a derived scene ``<scene>-lid<height>`` (the saved spec
with ``lid_height`` changed, never overwritten) saved when the job is created. Stdlib plus fluxpype.scenes.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ..scenes import checks, relax as scene_relax, store as scene_store
from ..scenes.spec import PlanarArcadeSpec

PROFILES = {"run": "scene-relax", "compare": "scene-compare"}
MAX_CYCLES, MAX_STEPS, MAX_LIDS, MAX_TASKS = 1000, 2000, 8, 16
DEFAULT_TIMEOUT_S = 6 * 3600


class SceneJobError(ValueError):
    pass


def is_scene_job(job: Optional[Dict[str, Any]]) -> bool:
    return bool(job) and (job.get("extra") or {}).get("kind") == "scene"


def lid_name(scene: str, lid: float) -> str:
    """The id of the variant of `scene` with its B2 lid at `lid`: 12 -> scene-lid12, 12.5 -> scene-lid12p5."""
    text = ("%g" % lid).replace(".", "p").replace("-", "m")
    return f"{scene}-lid{text}"


def _spec(data_dir: str, scene_id: str) -> Dict[str, Any]:
    try:
        row = scene_store.get(data_dir, scene_id)
    except scene_store.SceneError as e:
        raise SceneJobError(str(e))
    if not row:
        raise SceneJobError(f"unknown scene {scene_id!r}")
    return row["spec"]


def _variant(data_dir: str, scene_id: str, lid: float) -> Tuple[str, Dict[str, Any], bool]:
    """`(id, spec dict, exists)` of `scene_id` with its lid at `lid`. The scene itself when it already has that lid."""
    spec = _spec(data_dir, scene_id)
    if spec["lid_height"] is not None and abs(spec["lid_height"] - lid) < 1e-9:
        return scene_id, spec, True
    name = lid_name(scene_id, lid)
    notes = (spec.get("notes") or "") + f" [lid variant of {scene_id}, lid at {lid:g}]"
    try:
        made = PlanarArcadeSpec.from_dict({**spec, "name": name, "lid_height": lid, "notes": notes.strip()})
        scene_store.check_id(name)
    except (ValueError, scene_store.SceneError) as e:
        raise SceneJobError(f"lid {lid:g} for scene {scene_id}: {e}")
    have = scene_store.get(data_dir, name)
    if have and have["hash"] != made.content_hash():
        raise SceneJobError(
            f"a scene named {name} already exists with different settings; rename it or choose other lids"
        )
    return name, made.to_dict(), bool(have)


def plan(data_dir: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Check a relax request and expand it; writes nothing. Returns the normalized arguments, one `entry` per task
    (`label`, the scenes with their spec and whether each exists yet, the lid) and a plain summary."""
    allowed = {"mode", "scene", "a", "b", "lids", "cycles", "steps"}
    if set(args) - allowed:
        raise SceneJobError("unknown argument(s): " + ", ".join(sorted(set(args) - allowed)))
    mode = args.get("mode")
    if mode not in PROFILES:
        raise SceneJobError("mode is run (one scene) or compare (a fair pair)")
    names = ["scene"] if mode == "run" else ["a", "b"]
    stray = {"scene"} if mode == "compare" else {"a", "b"}
    if any(args.get(k) for k in stray):
        raise SceneJobError("run takes `scene`; compare takes `a` and `b`")
    ids = []
    for k in names:
        if not isinstance(args.get(k), str) or not args[k]:
            raise SceneJobError(f"{k} is required for {mode}")
        ids.append(args[k])
    if mode == "compare" and ids[0] == ids[1]:
        raise SceneJobError("compare needs two different scenes")
    cycles, steps = args.get("cycles", 150), args.get("steps", 100)
    for key, val, hi in (("cycles", cycles, MAX_CYCLES), ("steps", steps, MAX_STEPS)):
        if isinstance(val, bool) or not isinstance(val, int) or not 1 <= val <= hi:
            raise SceneJobError(f"{key} is a whole number from 1 to {hi}")
    lids = args.get("lids")
    if lids is not None:
        if not isinstance(lids, list) or not 1 <= len(lids) <= MAX_LIDS:
            raise SceneJobError(f"lids is a list of 1 to {MAX_LIDS} heights")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 < x < 1e6 for x in lids):
            raise SceneJobError("each lid height is a positive number")
        lids = sorted({float(x) for x in lids})
    entries: List[Dict[str, Any]] = []
    for lid in lids if lids is not None else [None]:
        scenes = []
        for sid in ids:
            if lid is None:
                scenes.append({"id": sid, "spec": _spec(data_dir, sid), "exists": True})
            else:
                vid, vspec, exists = _variant(data_dir, sid, lid)
                scenes.append({"id": vid, "spec": vspec, "exists": exists})
        for sc in scenes:  # every world must compile and pass its own construction checks
            try:
                sc["text"] = scene_store.prepare(sc["spec"])["text"]
            except scene_store.SceneError as e:
                raise SceneJobError(f"scene {sc['id']}: {e}")
        if mode == "compare":
            why = checks.comparable(scenes[0]["text"], scenes[1]["text"])
            if why:
                raise SceneJobError(f"{scenes[0]['id']} and {scenes[1]['id']} are not a fair pair: " + "; ".join(why))
        entry_lid = lid if lid is not None else scenes[0]["spec"]["lid_height"]
        entries.append(
            {
                "label": " vs ".join(s["id"] for s in scenes),
                "scenes": scenes,
                "lid": entry_lid,
                "lines": scenes[0]["spec"]["n_lines"],
            }
        )
    if len(entries) > MAX_TASKS:
        raise SceneJobError(f"at most {MAX_TASKS} tasks per job")
    norm: Dict[str, Any] = {"mode": mode, "cycles": cycles, "steps": steps}
    norm.update({k: v for k, v in zip(names, ids)})
    if lids is not None:
        norm["lids"] = lids
    new = [s["id"] for e in entries for s in e["scenes"] if not s["exists"]]
    what = ids[0] if mode == "run" else f"{ids[0]} and {ids[1]}"
    summary = (
        f"Relax {what} ({mode}) for {cycles} cycles of {steps} steps"
        + (f" at {len(lids)} lid height(s): {', '.join('%g' % x for x in lids)}" if lids else "")
        + f" as {len(entries)} task(s) in the queue"
        + (f"; first saves {len(new)} lid variant scene(s)" if new else "")
    )
    diff = [
        {"key": e["label"], "from": None, "to": "lid " + ("none" if e["lid"] is None else "%g" % e["lid"])}
        for e in entries
    ]
    return {"args": norm, "entries": entries, "summary": summary, "diff": diff}


def save_variants(data_dir: str, entries: List[Dict[str, Any]]) -> None:
    """Write the lid variant scenes the plan needs and that do not exist yet (never over an existing one)."""
    for e in entries:
        for sc in e["scenes"]:
            if not sc["exists"] and not scene_store.get(data_dir, sc["id"]):
                scene_store.save(data_dir, sc["spec"])
                sc["exists"] = True


def world_path(data_dir: str, scene_id: str) -> str:
    return os.path.join(scene_store.folder(data_dir), scene_id + ".flux")


# ------------------------------------------------------------------ results


def read_series(run_dir: str) -> Dict[str, Any]:
    """What a scene task has produced: the finished `result.json`, or the cycles so far from `series.jsonl`."""
    try:
        with open(os.path.join(run_dir, "result.json"), encoding="utf-8") as f:
            return {**json.load(f), "done": True}
    except (OSError, ValueError):
        pass
    worlds: Dict[int, List[Dict[str, Any]]] = {}
    try:
        with open(os.path.join(run_dir, "series.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                worlds.setdefault(int(rec.pop("w", 0)), []).append(rec)
    except OSError:
        return {}
    return {"done": False, "worlds": [{"series": worlds[w]} for w in sorted(worlds)]}


def thin(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """At most `n` rows, evenly spaced, always keeping the first and the last."""
    if len(rows) <= n:
        return rows
    idx = sorted({round(i * (len(rows) - 1) / (n - 1)) for i in range(n)})
    return [rows[i] for i in idx]


def brief(res: Dict[str, Any], points: int = 30) -> Dict[str, Any]:
    """A result shaped for reading: the settled cycles, the final numbers, a thinned energy series and, for a compare,
    the energy difference B - A (called D) with the cycle it settled at."""
    worlds = res.get("worlds") or []
    out: Dict[str, Any] = {k: res.get(k) for k in ("mode", "lid", "cycles", "steps", "ok", "done") if k in res}
    out["worlds"] = []
    for i, w in enumerate(worlds):
        ser = [r for r in w.get("series", []) if r.get("cycle", -1) >= 0]
        last = ser[-1] if ser else {}
        out["worlds"].append(
            {
                "scene": w.get("id"),
                "cycles_done": len(ser),
                "final_e_simple2": last.get("e_simple2"),
                "final_top_z": last.get("top_z"),
                "final_stiffness": last.get("stiffness"),
                "energy_settled_at_cycle": w.get("settled_at"),
                "series": [
                    {k: r.get(k) for k in ("cycle", "e_simple2", "top_z", "stiffness")} for r in thin(ser, points)
                ],
            }
        )
    d = res.get("difference")
    if d is not None:
        out["D"] = {
            "definition": "e_simple2 of B minus A, per cycle (real fluxons only)",
            "final": d[-1]["d_e_simple2"] if d else None,
            "settled_at_cycle": res.get("d_settled_at"),
            "value_at_settle": res.get("d_at_settle"),
            "series": thin(d, points),
        }
    elif len(worlds) == 2 and all(w.get("series") for w in worlds):  # a compare that is still running
        ea = {r["cycle"]: r["e_simple2"] for r in worlds[0]["series"]}
        d = [
            {"cycle": r["cycle"], "d_e_simple2": r["e_simple2"] - ea[r["cycle"]]}
            for r in worlds[1]["series"]
            if r["cycle"] in ea and r["cycle"] >= 0
        ]
        out["D"] = {
            "definition": "B minus A so far",
            "final": d[-1]["d_e_simple2"] if d else None,
            "series": thin(d, points),
        }
    return out


def settle(series: List[float]) -> Optional[int]:
    return scene_relax.settled_at(series)
