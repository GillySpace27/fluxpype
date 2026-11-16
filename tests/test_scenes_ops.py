"""Scene edit operations and the version tree in the scene store."""

import json
import os

import pytest

from fluxpype.scenes import store
from fluxpype.scenes.ops import OpError, apply_ops
from fluxpype.scenes.spec import PlanarArcadeSpec, shift_pairing


def _spec(**kw):
    return PlanarArcadeSpec(name="arc", n_lines=5, **kw).to_dict()


def test_set_and_shear_report_exactly_what_changed():
    base = _spec()
    new, changes = apply_ops(base, [{"op": "set", "field": "lid_height", "value": 8}, {"op": "shear", "k": 2}])
    assert new["lid_height"] == 8 and new["pairing"] == shift_pairing(5, 2)
    assert {c["field"] for c in changes} == {"lid_height", "pairing"}
    assert base["lid_height"] is None and base["pairing"] is None  # the input is left as it was


def test_shear_zero_straightens_the_arcade_again():
    sheared, _ = apply_ops(_spec(), [{"op": "shear", "k": 3}])
    straight, changes = apply_ops(sheared, [{"op": "shear", "k": 0}])
    assert straight["pairing"] is None and changes == [{"field": "pairing", "from": shift_pairing(5, 3), "to": None}]


def test_changing_the_line_count_drops_a_pairing_that_no_longer_fits():
    sheared, _ = apply_ops(_spec(), [{"op": "shear", "k": 1}])
    new, _ = apply_ops(sheared, [{"op": "set", "field": "n_lines", "value": 7}])
    assert new["n_lines"] == 7 and new["pairing"] is None


@pytest.mark.parametrize(
    "ops",
    [
        [],
        "set",
        [{"field": "n_lines", "value": 3}],
        [{"op": "set", "field": "name", "value": "x"}],
        [{"op": "set", "field": "lid_height"}],
        [{"op": "shear", "k": 5}],
        [{"op": "shear", "k": True}],
        [{"op": "twist"}],
        [{"op": "set", "field": "n_lines", "value": -1}],
        [{"op": "shear", "k": 0}] * 21,
    ],
)
def test_bad_operations_are_refused(ops):
    with pytest.raises(OpError):
        apply_ops(_spec(), ops)


def test_an_edit_saves_a_new_version_and_leaves_the_parent_alone(tmp_path):
    d = str(tmp_path)
    store.save(d, _spec())
    before = open(os.path.join(store.folder(d), "arc.scene.json")).read()
    out = store.edit(d, "arc", [{"op": "shear", "k": 2}], label="shear by two", author="web")
    assert out["id"] == "arc-v2" and out["parent"] == "arc"
    assert out["fair_pair_with_parent"] == {"fair": True, "why": []}
    assert open(os.path.join(store.folder(d), "arc.scene.json")).read() == before
    meta = json.load(open(os.path.join(store.folder(d), "arc-v2.ops.json")))
    assert meta["parent"] == "arc" and meta["label"] == "shear by two" and meta["author"] == "web"
    assert store.get(d, "arc-v2")["spec"]["pairing"] == shift_pairing(5, 2)
    # editing a version numbers the next one from the same root
    assert store.edit(d, "arc-v2", [{"op": "set", "field": "lid_height", "value": 6}])["id"] == "arc-v3"
    assert [r["id"] for r in store.list_scenes(d)].count("arc-v3") == 1


def test_a_change_to_the_box_is_flagged_as_not_a_fair_pair(tmp_path):
    d = str(tmp_path)
    store.save(d, _spec())
    out = store.edit(d, "arc", [{"op": "set", "field": "lid_height", "value": 4}])
    assert out["fair_pair_with_parent"]["fair"] is False and out["fair_pair_with_parent"]["why"]
    assert out["lid_height"] == 4


def test_plan_edit_writes_nothing_and_refuses_no_op_edits_and_taken_names(tmp_path):
    d = str(tmp_path)
    store.save(d, _spec())
    plan = store.plan_edit(d, "arc", [{"op": "shear", "k": 1}])
    assert plan["id"] == "arc-v2" and not os.path.exists(os.path.join(store.folder(d), "arc-v2.scene.json"))
    with pytest.raises(store.SceneError, match="change nothing"):
        store.plan_edit(d, "arc", [{"op": "shear", "k": 0}])
    with pytest.raises(store.SceneError, match="already exists"):
        store.plan_edit(d, "arc", [{"op": "shear", "k": 1}], name="arc")
    with pytest.raises(store.SceneError, match="unknown scene"):
        store.plan_edit(d, "nope", [{"op": "shear", "k": 1}])
    with pytest.raises(store.SceneError):
        store.plan_edit(d, "arc", [{"op": "shear", "k": 9}])


def test_history_walks_back_to_the_root_and_lists_children(tmp_path):
    d = str(tmp_path)
    store.save(d, _spec())
    store.edit(d, "arc", [{"op": "shear", "k": 1}])
    store.edit(d, "arc-v2", [{"op": "shear", "k": 2}], name="arc-k2")
    store.edit(d, "arc", [{"op": "set", "field": "lid_height", "value": 8}])
    h = store.history(d, "arc-k2")
    assert [v["id"] for v in h["versions"]] == ["arc", "arc-v2", "arc-k2"] and h["children"] == []
    assert h["versions"][0]["parent"] is None and h["versions"][-1]["changes"][0]["field"] == "pairing"
    assert sorted(c["id"] for c in store.history(d, "arc")["children"]) == ["arc-v2", "arc-v3"]
    assert store.history(d, "nope") is None
