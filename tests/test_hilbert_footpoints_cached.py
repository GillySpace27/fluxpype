"""A5-07: a cached footpoint file must report its row count, not a third of its first column."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCIENCE = REPO / "fluxpype" / "science"


def _have_pdl():
    perl = shutil.which("perl")
    if not perl:
        return False
    return subprocess.run([perl, "-MPDL", "-MPDL::IO::Misc", "-e", "1"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _have_pdl(), reason="needs perl with PDL")


def test_cached_footpoints_count_rows(tmp_path):
    # Stubs for FLUX-side modules that the cached branch never calls.
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "pipe_helper.pm").write_text(
        "package pipe_helper; use Exporter 'import'; our @EXPORT_OK=('shorten_path');"
        " sub shorten_path { return $_[0] } 1;\n"
    )
    (stubs / "fluxon_placement_hilbert.pm").write_text(
        "package fluxon_placement_hilbert; use Exporter 'import'; our @EXPORT_OK=('fluxon_placement_hilbert');"
        " sub fluxon_placement_hilbert { die 'fresh branch must not run' } 1;\n"
    )
    floc = tmp_path / "floc.dat"
    floc.write_text("".join(f"{i}.5 {i}.25 {i}.75\n" for i in range(5)))
    code = (
        "use get_hilbert_footpoints;"
        "my ($f, $n) = get_hilbert_footpoints::get_hilbert_footpoints(flocdir=>'%s', flocfile=>'floc.dat',"
        " n_fluxons_wanted=>5, force_process_magnetogram=>0);"
        'print "N=$n DIMS=", join(",", $f->dims), "\\n";'
    ) % tmp_path
    out = subprocess.run(
        ["perl", "-I", str(stubs), "-I", str(SCIENCE), "-e", code], capture_output=True, text=True, cwd=tmp_path
    )
    assert out.returncode == 0, out.stderr
    assert "N=5 DIMS=3,5" in out.stdout
