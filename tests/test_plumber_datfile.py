import os

import pytest

from fluxpype.plumber import datfile

SAMPLE = "/Users/gilly/vscode/fluxons/fluxpype/fluxpype/data/batches/fluxlight/data/cr2150"


def test_headerless_whitespace_table():
    p = datfile.parse_text("1 2.5 -1\n3 4.5 1\n\n# note\n5 6\n")
    assert p["rows"] == 2 and p["bad_rows"] == 1 and not p["header"]
    assert [c["name"] for c in p["columns"]] == ["col1", "col2", "col3"]
    assert p["columns"][2]["integer"] and p["columns"][0]["sorted"] and p["columns"][1]["max"] == 4.5


def test_header_comma_and_nan(tmp_path):
    f = tmp_path / "results_x_full_velocity.dat"
    f.write_text("fluxon_position,radius,velocity\n0,1.0,0\n0,2.0,nan\n1,1.5,3\n")
    d = datfile.payload(str(f))
    assert d["header"] and [c["name"] for c in d["columns"]] == ["fluxon_position", "radius", "velocity"]
    assert d["data"][2] == [0, None, 3] and d["columns"][2]["nonfinite"] == 1
    assert d["kind"]["id"] == "velocity"
    assert datfile.payload(str(f), max_rows=2)["decimated"] == {"step": 2, "rows_kept": 2, "rows_total": 3}


def test_unknown_name_has_no_kind(tmp_path):
    f = tmp_path / "mystery.dat"
    f.write_text("1 2\n3 4\n")
    assert datfile.payload(str(f))["kind"] is None


# name, rows, columns, kind id (real files written by the pipeline; counts agree with wc -l)
REAL = [
    ("floc/floc_cr2150_r4_f1000_hmi.dat", 1022, 3, "footpoints"),
    ("floc/floc_open_cr2150_r4_f1000_hmi.dat", 2171, 5, "open_lines"),
    ("floc/floc_closed_cr2150_r4_f1000_hmi.dat", 3410, 5, "closed_lines"),
    ("wind/cr2150_f1000_radial_bmag.dat", 52, 9, "bmag"),
    ("wind/cr2150_f1000_radial_bmag_all.dat", 1243, 13, "bmag_all"),
    ("wind/cr2150_f1000_radial_fr.dat", 1243, 9, "fr"),
    ("wind/cr2150_f1000_radial_wind_parker.dat", 42, 9, "wind"),
    ("wind/full_velocity_profiles/results_parker_full_velocity.dat", 21042, 3, "velocity"),
]


@pytest.mark.skipif(not os.path.isdir(SAMPLE), reason="real sample files not on this machine")
@pytest.mark.parametrize("name,rows,cols,kind", REAL)
def test_real_files(name, rows, cols, kind):
    d = datfile.payload(os.path.join(SAMPLE, name))
    assert (d["rows"], len(d["columns"]), d["bad_rows"], d["kind"]["id"]) == (rows, cols, 0, kind)
    assert all(c["nonfinite"] == 0 for c in d["columns"])
