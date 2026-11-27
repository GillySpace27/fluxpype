"""Offline checks for the realflux harness (no FLUX needed): golden data integrity, build keys, provenance fallback."""

from __future__ import annotations

import stat

from fluxpype.scenes import relax as R
from tests.realflux import support


def test_vendored_goldens_match_their_recorded_sha():
    g = support.load_golden()
    for name, spec in g["fixtures"].items():
        old = [b for b in spec["builds"] if b["key"]["pdl"] == "2.106"]
        assert old and support.sha256(support.DATA / "relaxed" / f"{name}.relaxed.flux") == old[0]["sha256"], name
        assert (support.DATA / "fixtures" / f"{name}.flux").is_file()


def test_topology_counts_fluxons():
    counts = {
        n: support.topology(support.DATA / "relaxed" / f"{n}.relaxed.flux")[0]
        for n in ("simple-4", "planar_shear2", "twisted")
    }
    assert counts == {"simple-4": 6, "planar_shear2": 11, "twisted": 27}  # grep of LINE records, audit A6-03


def test_build_key_matching_is_strict_about_unknowns():
    key = {"pdl": "2.085", "perl": "5.38.2", "flux_pm_sha256": "ab", "fluxon_mhd_sha": None}
    assert support.key_matches({"pdl": "2.085", "perl": "5.38.2"}, key)
    assert not support.key_matches({"pdl": "2.106", "perl": "5.38.2"}, key)
    assert not support.key_matches({"pdl": "2.085", "fluxon_mhd_sha": "3f65ebd"}, key)  # build commit unknown
    assert support.build_key({"fluxon_mhd_sha": "3f65ebd0123"})["fluxon_mhd_sha"] == "3f65ebd"


def test_shipped_builds_are_only_the_audited_one():
    """A key is recorded by `record_build` after repeated identical runs, never by a test run (A6-03)."""
    for spec in support.load_golden()["fixtures"].values():
        for b in spec["builds"]:
            assert b.get("runs"), "every recorded build says how many identical runs justified it"


def test_provenance_reads_the_probe_and_adds_the_commit(tmp_path):
    fake = tmp_path / "perl"
    fake.write_text(
        '#!/bin/sh\ncat >/dev/null\necho \'PROVENANCE {"pdl":"2.085","perl":"5.38.2","flux_pm":"/x/Flux.pm","flux_pm_sha256":"ab"}\'\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    got = R.provenance(str(fake), env={"FLUXON_MHD_SHA": "3f65ebd"})
    assert got == {
        "pdl": "2.085",
        "perl": "5.38.2",
        "flux_pm": "/x/Flux.pm",
        "flux_pm_sha256": "ab",
        "fluxon_mhd_sha": "3f65ebd",
    }
    assert R.provenance(str(fake), env={})["fluxon_mhd_sha"] is None


def test_provenance_is_empty_when_perl_cannot_load_flux(tmp_path):
    assert R.provenance(str(tmp_path / "no-such-perl")) == {}
    bad = tmp_path / "perl"
    bad.write_text("#!/bin/sh\necho nothing useful\n")
    bad.chmod(bad.stat().st_mode | stat.S_IXUSR)
    assert R.provenance(str(bad)) == {}
