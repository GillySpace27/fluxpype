"""Golden relax runs against the real FLUX (audit A6-02).

Tier S: when the loaded build matches a recorded build key, the relaxed world must be byte-identical (SHA256) to the
recorded one. A failure there is a physics change, or heap noise (A1-05); look at it.
Tier L: on any build, the topology must match the golden world and the final energy and stiffness must stay inside
the per-fixture band. Relaxed worlds depend on the PDL version (A6-03), so vertex counts and positions are not
compared across builds.
"""

from __future__ import annotations

import pytest

from . import support

FIXTURES = ["simple-4", "planar_shear2", "twisted"]


def _rel(a, b):
    return abs(a - b) / abs(b)


@pytest.mark.parametrize("name", FIXTURES)
def test_tier_s_exact_sha_on_a_recorded_build(name, key, golden, relaxed):
    spec = golden["fixtures"][name]
    build = support.find_build(spec, key)
    if build is None:
        pytest.skip(
            f"no recorded build key for {name} matches {key}; Tier L still applies. After repeated identical runs, "
            "record it with: python -m tests.realflux.record_build"
        )
    res = relaxed(name)
    assert res["ok"], res
    assert (
        res["sha256"] == build["sha256"]
    ), f"{name}: relaxed world differs from the golden on a recorded build {build['key']}"


@pytest.mark.parametrize("name", FIXTURES)
def test_tier_l_topology_and_band(name, golden, relaxed, record_property):
    spec = golden["fixtures"][name]
    res = relaxed(name)
    assert res["ok"], res
    want_n, want_lines = support.topology(support.DATA / "relaxed" / f"{name}.relaxed.flux")
    got_n, got_lines = support.topology(res["relaxed"])
    assert got_n == want_n, f"{name}: {got_n} fluxons, golden has {want_n}"
    assert got_lines == want_lines, f"{name}: fluxon labels, end concentrations or flux per fluxon changed"

    ref, final = spec["reference"], res["final"]
    for field in ("e_simple2", "stiffness"):
        change = _rel(final[field], ref[field])
        record_property(f"{field}_change", f"{change:+.4f}")
        band = spec["bands"][field]
        if band is not None:
            assert change <= band, f"{name} {field}: {final[field]} vs golden {ref[field]} ({change:.1%} > {band:.0%})"
    record_property("vertices", f"{final['n']} (golden {ref['n']})")


def test_round_trip_is_stable_after_one_pass(tmp_path, prov):
    """A1-07 G7: write(read(x)) is a fixed point, so comparisons may normalise through one core write first."""
    for name in FIXTURES:
        src = support.DATA / "relaxed" / f"{name}.relaxed.flux"
        one, two = tmp_path / f"{name}.1.flux", tmp_path / f"{name}.2.flux"
        support.roundtrip(src, one)
        support.roundtrip(one, two)
        assert support.sha256(one) == support.sha256(two), name
