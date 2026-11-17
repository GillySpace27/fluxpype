"""The potential-field seed for planar scenes, and the two edit operations built on it."""

import itertools
import math

import pytest

from fluxpype.scenes import PlanarArcadeSpec, compile_flux, summarize
from fluxpype.scenes import potential
from fluxpype.scenes.checks import comparable
from fluxpype.scenes.ops import OpError, apply_ops


def test_a_single_bipole_line_is_symmetric_and_lands():
    spec = PlanarArcadeSpec(n_lines=1, seed="potential").validate()
    seed = spec.potential()
    assert seed["pairing"] == [0] and seed["moved"] == 0 and seed["not_landed"] == 0
    line = seed["lines"][0]
    mid = line[len(line) // 2]
    assert abs(mid[0]) < 0.02 and abs(mid[1]) < 1e-9  # the top is over the PIL
    assert mid[2] == pytest.approx(seed["top_z"], rel=1e-3)
    for a, b in zip(line, reversed(line)):  # mirror image about x = 0
        assert a[0] == pytest.approx(-b[0], abs=0.02) and a[2] == pytest.approx(b[2], abs=0.02)


def test_an_unsheared_row_traces_to_the_identity_pairing():
    spec = PlanarArcadeSpec(n_lines=9, seed="potential").validate()
    seed = spec.potential()
    assert seed["pairing"] == list(range(9)) and seed["moved"] == 0 and seed["not_landed"] == 0
    assert summarize(compile_flux(spec))["problems"] == []


def test_the_lid_images_leave_almost_no_field_through_the_lid():
    sources = [(-1.0, 0.0, 1.0), (1.0, 0.0, -1.0)]
    with_lid = potential.field((0.3, 0.2, 4.0), sources, lid=4.0)
    without = potential.field((0.3, 0.2, 4.0), sources)
    assert abs(with_lid[2]) < 0.05 * abs(without[2])


def test_assign_matches_brute_force():
    cost = [[4.0, 1.0, 3.0, 9.0], [2.0, 0.5, 5.0, 1.0], [3.0, 2.0, 2.0, 7.0], [6.0, 1.5, 0.2, 2.5]]
    best = min(itertools.permutations(range(4)), key=lambda p: sum(cost[i][p[i]] for i in range(4)))
    got = potential._assign(cost)
    assert sum(cost[i][got[i]] for i in range(4)) == pytest.approx(sum(cost[i][best[i]] for i in range(4)))
    assert sorted(got) == [0, 1, 2, 3]


def test_a_potential_seed_refuses_a_hand_set_pairing_and_an_unknown_seed():
    with pytest.raises(ValueError, match="derives its own pairing"):
        PlanarArcadeSpec(n_lines=3, seed="potential", pairing=[1, 0, 2]).validate()
    with pytest.raises(ValueError, match="seed must be"):
        PlanarArcadeSpec(seed="dipole").validate()


def test_arc_scenes_keep_their_hash_and_text():
    assert PlanarArcadeSpec().content_hash() == "7543ea935e08"  # the hash before seeds existed
    assert "hand-set" in compile_flux(PlanarArcadeSpec())
    assert PlanarArcadeSpec(seed="potential").content_hash() != PlanarArcadeSpec().content_hash()


def test_a_sheared_potential_seed_is_always_a_valid_world_and_says_what_it_moved():
    spec = PlanarArcadeSpec(n_lines=9, seed="potential", shear_offset=1.5, lid_height=8).validate()
    seed = spec.potential()
    assert sorted(seed["pairing"]) == list(range(9))
    text = compile_flux(spec)
    assert summarize(text)["problems"] == []
    assert f"{seed['moved']} moved" in text and "connectivity is traced" in text
    assert all(z > 0 for ln in seed["lines"] for (_x, _y, z) in ln) and seed["top_z"] < 8


def test_shear_footpoints_then_potential_is_a_fair_pair():
    base = PlanarArcadeSpec(name="pil", n_lines=7, lid_height=8, seed="potential").to_dict()
    stressed, changes = apply_ops(base, [{"op": "shear_footpoints", "offset": 0.75}])
    assert stressed["shear_offset"] == 0.75 and stressed["seed"] == "arc"
    assert stressed["pairing"] == list(range(7))  # each line stays on the footpoints it had
    assert {c["field"] for c in changes} == {"shear_offset", "pairing", "seed"}
    reference, _ = apply_ops(stressed, [{"op": "potential"}])
    assert reference["seed"] == "potential" and reference["pairing"] is None
    a = compile_flux(PlanarArcadeSpec.from_dict(stressed))
    b = compile_flux(PlanarArcadeSpec.from_dict(reference))
    assert comparable(a, b) == []


def test_shear_footpoints_needs_a_number():
    with pytest.raises(OpError, match="offset"):
        apply_ops(PlanarArcadeSpec().to_dict(), [{"op": "shear_footpoints", "offset": "far"}])


def test_the_trace_is_keyed_on_the_lid():
    low = PlanarArcadeSpec(n_lines=3, seed="potential", lid_height=2.5).validate().potential()
    high = PlanarArcadeSpec(n_lines=3, seed="potential", lid_height=20).validate().potential()
    assert low["top_z"] < high["top_z"] and not math.isclose(low["top_z"], high["top_z"], rel_tol=1e-4)


def test_a_lid_variant_of_a_potential_scene_is_retraced_under_the_new_lid(tmp_path):
    from fluxpype.plumber import scene_jobs
    from fluxpype.scenes import store

    dd = str(tmp_path)
    store.save(dd, PlanarArcadeSpec(name="ref", n_lines=3, seed="potential", lid_height=20).to_dict())
    name, spec, exists = scene_jobs._variant(dd, "ref", 2.5)
    assert name == "ref-lid2p5" and not exists and spec["seed"] == "potential"
    low = PlanarArcadeSpec.from_dict(spec).potential()["top_z"]
    assert low < PlanarArcadeSpec(n_lines=3, seed="potential", lid_height=20).potential()["top_z"]
    # the lid is a mirror wall for the traced field too, so a low lid flattens the seed under it
    store.save(dd, PlanarArcadeSpec(name="tall", n_lines=9, seed="potential", shear_offset=1.5, lid_height=8).to_dict())
    _name, squashed, _ = scene_jobs._variant(dd, "tall", 1.9)
    assert PlanarArcadeSpec.from_dict(squashed).potential()["top_z"] < 1.9
