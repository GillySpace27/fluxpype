"""FLUX world-file parser: a tiny fixture, then the real sample files against the counts the Flux library reports."""

import os

import pytest

from fluxpype.plumber import fluxfile

WORLD = "/Users/gilly/vscode/fluxons/fluxpype/fluxpype/data/batches/fluxlight/data/cr2150/world"

FIXTURE = """# comment
FRAME 0
GLOBAL BOUNDARY  SPHERE 0 0 0 1 0 0
GLOBAL SCALING D 2
NEW\t100\t0.6\t0.0\t0.8\t-1\t(null)\t0
NEW\t200\t0.0\t0.6\t0.8\t1\t(null)\t0
LINE\t-10\t-3\t-4\t-10\t-11\t0            0 0 0   0 0 0
LINE\t101\t-302\t-301\t200\t100\t1            0.0 0.6 0.8   0.6 0.0 0.8
\tVERTEX\t101\t102\t2\t0.4\t0.4\t1.5
\tVERTEX\t101\t103\t1\t0.0\t0.7\t1.2

VNEIGHBOR 102 103
WHATEVER 1 2 3
VERTEX 101 104 oops
"""


def test_fixture():
    w = fluxfile.parse_text(FIXTURE)
    assert w["frame"] == 0 and w["boundary"] == {"type": "SPHERE", "args": [0, 0, 0, 1, 0, 0]}
    c = w["counts"]
    assert (c["vertices"], c["fluxons"], c["concentrations"]) == (4 + 2, 2, 2 + 8)
    assert c["vneighbor_rows"] == 1 and c["unknown_rows"] == {"WHATEVER": 1} and c["bad_rows"] == 1
    fl = w["fluxons"]  # the pseudo-fluxon is counted but not drawn; interior vertices are ordered by pos
    assert fl["id"] == [101] and fl["offsets"] == [0, 4]
    assert fl["xyz"] == [0, 0.6, 0.8, 0, 0.7, 1.2, 0.4, 0.4, 1.5, 0.6, 0, 0.8]
    assert w["concentrations"]["flux"] == [-1, 1]


def test_payload_decimates_and_says_so(tmp_path):
    p = tmp_path / "w.flux"
    p.write_text(FIXTURE.replace("VERTEX 101 104 oops", ""))
    assert fluxfile.payload(str(p))["decimated"] is None
    d = fluxfile.payload(str(p), max_vertices=2)
    assert d["decimated"]["step"] == 2 and d["decimated"]["fluxons_total"] == 1 and len(d["fluxons"]["id"]) == 1


@pytest.mark.skipif(not os.path.isdir(WORLD), reason="sample world files not on this machine")
@pytest.mark.parametrize(
    "name,vertices", [("cr2150_f1000_hmi.flux", 5497), ("cr2150_f1000_hmi_relaxed_s300.flux", 9554)]
)
def test_real_files_match_flux_library_counts(name, vertices):
    c = fluxfile.parse_file(os.path.join(WORLD, name))["counts"]
    assert (c["vertices"], c["fluxons"], c["concentrations"]) == (vertices, 545, 1042)
    assert c["bad_rows"] == 0


def test_planar_scene_world_parses_with_its_photosphere_and_lid():
    from fluxpype.scenes import PlanarArcadeSpec, compile_flux

    w = fluxfile.parse_text(compile_flux(PlanarArcadeSpec(n_lines=4, vertices_per_line=5, lid_height=3.0)))
    assert w["boundary"] is None and w["counts"]["bad_rows"] == 0
    assert w["planes"] == [
        {"role": "photosphere", "origin": [0, 0, -0.001], "normal": [0, 0, 1]},
        {"role": "b2", "origin": [0, 0, 3], "normal": [0, 0, -1]},
    ]
    fl = w["fluxons"]
    assert len(fl["id"]) == 4 and fl["offsets"][-1] == 4 * (5 + 2)  # footpoints at both ends, 5 interior vertices
    assert w["concentrations"]["flux"].count(1) == 4 and w["concentrations"]["flux"].count(-1) == 4


def test_a_global_row_that_is_not_a_plane_adds_no_plane():
    assert (
        fluxfile.parse_text("GLOBAL FORCES f_curvature\nGLOBAL B2 NONE\nGLOBAL PHOTOSPHERE 0 0 0 0 0 0\n")["planes"]
        == []
    )
