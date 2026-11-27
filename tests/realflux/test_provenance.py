"""A7-07: the relax driver's result JSON says which FLUX build produced it."""

from __future__ import annotations

import json

from fluxpype.scenes import relax as R

from . import support


def test_result_json_provenance_matches_the_loaded_build(tmp_path, prov, capsys):
    world = support.DATA / "fixtures" / "simple-4.flux"
    rc = R.main(
        ["run", str(world), "--cycles", "1", "--steps", "10", "--out", str(tmp_path / "o.flux"), "--no-register"]
    )
    assert rc == 0
    got = json.loads(capsys.readouterr().out)["provenance"]
    assert got == prov
    assert got["pdl"] and got["perl"] and got["flux_pm"] and len(got["flux_pm_sha256"]) == 64
