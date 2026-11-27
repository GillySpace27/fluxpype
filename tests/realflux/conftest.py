"""realflux: golden tests that run FLUX itself.

Every test under tests/realflux carries the `realflux` marker and skips (never fails) when the Perl that `use Flux`
(`$FLUX_PERL`, else `perl`) cannot load PDL and Flux, so the ordinary CI matrix stays green. Run them with
`python -m pytest -m realflux tests/realflux`; add `--realflux-slow` for the 50 s twisted fixture. Set
`FLUXON_MHD_SHA` to the fluxon-mhd commit the build came from, or no build key can match (Tier S then skips).

Not covered yet (follow-ups, see the audit): A5-08, a real ADAPT CR2225 run (about 4 min, needs data downloads, so a
nightly job and no network test here); A1-07 G4 parallel relax and G5 pipeline relaxer (need a harness that calls
simple_relaxer without PFSS), G6 field snapshot, and G8 teardown cases (the A1-03 and A1-02 recipes).
"""

from __future__ import annotations

import pytest

from fluxpype.scenes import relax as R

from . import support


def pytest_addoption(parser):
    parser.addoption("--realflux-slow", action="store_true", help="also run the slow realflux fixtures (twisted)")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/realflux/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.realflux)


@pytest.fixture(scope="session")
def prov():
    p = R.provenance()
    if not p:
        pytest.skip("FLUX is not available: `perl -MPDL -MFlux` fails (set FLUX_PERL to the Perl that has it)")
    return p


@pytest.fixture(scope="session")
def key(prov):
    return support.build_key(prov)


@pytest.fixture(scope="session")
def golden():
    return support.load_golden()


@pytest.fixture(scope="session")
def relaxed(prov, golden, tmp_path_factory, request):
    """Relax each fixture once per session with the documented golden command; shared by Tier S and Tier L."""
    cache = {}

    def get(name):
        spec = golden["fixtures"][name]
        if spec.get("slow") and not request.config.getoption("--realflux-slow"):
            pytest.skip(f"{name} takes about a minute; pass --realflux-slow")
        if name not in cache:
            out = tmp_path_factory.mktemp(name) / f"{name}.relaxed.flux"
            world = support.DATA / "fixtures" / f"{name}.flux"
            cache[name] = support.run_relax(world, spec["cycles"], spec["steps"], out)
        return cache[name]

    return get
