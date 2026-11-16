"""Planar arcade scenes: the spec, the compiler and the construction checks."""

import json

import pytest

from fluxpype.scenes import PlanarArcadeSpec, compile_flux, shift_pairing, summarize
from fluxpype.scenes import checks
from fluxpype.scenes.__main__ import main


def test_shift_pairing_is_a_permutation_and_k_zero_is_the_identity():
    assert shift_pairing(5, 0) == [0, 1, 2, 3, 4]
    assert shift_pairing(5, 2) == [2, 3, 4, 1, 0]
    for n in (3, 9, 20):
        for k in range(n):
            assert sorted(shift_pairing(n, k)) == list(range(n))
    with pytest.raises(ValueError):
        shift_pairing(4, 4)


def test_compiled_world_passes_the_construction_checks():
    text = compile_flux(PlanarArcadeSpec(n_lines=5))
    s = summarize(text)
    assert s["problems"] == [] and s["lines"] == 5 and s["vertices"] == 5 * 19
    assert s["net_flux"] == 0 and 0 < s["max_height"] <= 1.5
    assert s["photosphere"] == "GLOBAL PHOTOSPHERE 0 0 -0.001 0 0 1" and s["lid"] is None
    assert s["min_line_separation"] > 0.05


def test_the_format_matches_the_menagerie_layout():
    lines = compile_flux(PlanarArcadeSpec(n_lines=2, vertices_per_line=3)).splitlines()
    body = [l for l in lines if l and not l.startswith("#")]
    assert body[:3] == [
        "FRAME 0",
        "GLOBAL FORCES f_pressure_equi f_curvature f_vertex",
        "GLOBAL PHOTOSPHERE 0 0 -0.001 0 0 1",
    ]
    assert body[3].startswith("NEW 2 -1 -0.125 0 1") and body[4].startswith("NEW 3 1 -0.125 0 -1")
    assert body[5] == "LINE 2 2 3 1" and body[6].startswith("VERTEX 2 3 1 ") and body[8].startswith("VERTEX 2 5 3 ")
    assert body[9].startswith("NEW 4 ") and body[11] == "LINE 6 4 5 1"  # next fluxon label follows the vertices


def test_lid_is_a_b2_plane_pointing_down():
    s = summarize(compile_flux(PlanarArcadeSpec(lid_height=3.0)))
    assert s["lid"] == "GLOBAL B2 PLANE 0 0 3 0 0 -1"


def test_sheared_and_perpendicular_worlds_are_a_fair_pair():
    n = 9
    a = compile_flux(PlanarArcadeSpec(n_lines=n, pairing=shift_pairing(n, 0), shear_offset=0.1))
    b = compile_flux(PlanarArcadeSpec(n_lines=n, pairing=shift_pairing(n, 2), shear_offset=0.1))
    assert checks.comparable(a, b) == []
    assert a != b and summarize(b)["problems"] == []


def test_a_different_photosphere_or_vertex_count_is_not_a_fair_pair():
    base = compile_flux(PlanarArcadeSpec())
    why = checks.comparable(base, compile_flux(PlanarArcadeSpec(photosphere_z=-0.01, vertices_per_line=31)))
    assert "global photosphere differs" in why and "vertices per line differ" in why
    assert "footpoint positions or fluxes differ" in checks.comparable(
        base, compile_flux(PlanarArcadeSpec(shear_offset=0.3))
    )


def test_validation_rejects_bad_scenes():
    for bad in (
        dict(n_lines=0),
        dict(spacing=-1),
        dict(pairing=[0, 0, 1, 2, 3]),
        dict(photosphere_z=0.5),
        dict(lid_height=1.6),  # below 1.2 x max_height
        dict(vertices_per_line=2),
    ):
        with pytest.raises(ValueError):
            PlanarArcadeSpec(**bad).validate()


def test_json_round_trip_and_hash_ignores_name_and_notes():
    spec = PlanarArcadeSpec(n_lines=4, pairing=shift_pairing(4, 1), lid_height=3)
    again = PlanarArcadeSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again == spec
    renamed = PlanarArcadeSpec.from_dict({**spec.to_dict(), "name": "other", "notes": "x"})
    assert renamed.content_hash() == spec.content_hash()
    assert PlanarArcadeSpec.from_dict({**spec.to_dict(), "flux_per_line": 2}).content_hash() != spec.content_hash()
    with pytest.raises(ValueError):
        PlanarArcadeSpec.from_dict({**spec.to_dict(), "colour": "red"})


def test_cli_compiles_a_scene_file(tmp_path, capsys):
    assert main(["example", "--lines", "4", "--shear", "1", "--lid", "3"]) == 0
    spec_path = tmp_path / "s.json"
    spec_path.write_text(capsys.readouterr().out)
    out = tmp_path / "s.flux"
    assert main(["compile", str(spec_path), str(out)]) == 0
    assert out.read_text().startswith("# scene planar-arcade") and json.loads(capsys.readouterr().out)["lines"] == 4


def test_store_saves_lists_and_never_overwrites(tmp_path):
    from fluxpype.scenes import store

    saved = store.save(str(tmp_path), {"name": "a1", "n_lines": 3})
    assert saved["world"] == "scenes/a1.flux" and (tmp_path / "scenes" / "a1.flux").exists()
    assert [r["id"] for r in store.list_scenes(str(tmp_path))] == ["a1"]
    assert store.get(str(tmp_path), "a1")["spec"]["n_lines"] == 3 and store.get(str(tmp_path), "../x") is None
    with pytest.raises(store.SceneError, match="already exists"):
        store.save(str(tmp_path), {"name": "a1"})
    with pytest.raises(store.SceneError):
        store.save(str(tmp_path), {"name": "a2", "colour": "red"})
