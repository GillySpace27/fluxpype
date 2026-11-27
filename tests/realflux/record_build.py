"""Record the loaded FLUX build as strict (Tier S) in tests/data/realflux/golden.json.

    FLUXON_MHD_SHA=3f65ebd python -m tests.realflux.record_build [--runs 3] [--fixtures simple-4 planar_shear2]

Explicit on purpose (audit A6-03): the fixture is relaxed `--runs` times and a key is written only when every run
gives the same SHA256. A fixture that shows two SHAs (planar_shear2 on PDL 2.085 alternates between two) is reported
and left to Tier L. The fluxon-mhd commit must be known (`$FLUXON_MHD_SHA` or `--fluxon-sha`) because a key without
it would match any source revision.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from fluxpype.scenes import relax as R

from . import support


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tests.realflux.record_build", description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=3, help="identical runs required before a key is written (default 3)")
    ap.add_argument("--fixtures", nargs="*", help="default: every fixture except the slow ones")
    ap.add_argument("--fluxon-sha", help="fluxon-mhd commit this build was made from (default $FLUXON_MHD_SHA)")
    ap.add_argument("--slow", action="store_true", help="include the slow fixtures")
    a = ap.parse_args(argv)
    if a.runs < 2:
        ap.error("--runs must be at least 2")

    env = dict(os.environ)
    if a.fluxon_sha:
        env["FLUXON_MHD_SHA"] = a.fluxon_sha
    prov = R.provenance(env=env)
    if not prov:
        print("FLUX is not available (perl -MPDL -MFlux fails); nothing recorded", file=sys.stderr)
        return 2
    if not prov.get("fluxon_mhd_sha"):
        print("fluxon-mhd commit unknown: set FLUXON_MHD_SHA or pass --fluxon-sha; nothing recorded", file=sys.stderr)
        return 2
    key = support.build_key(prov)
    g = support.load_golden()
    names = a.fixtures or [n for n, s in g["fixtures"].items() if a.slow or not s.get("slow")]
    status = 0
    for name in names:
        spec = g["fixtures"][name]
        world = support.DATA / "fixtures" / f"{name}.flux"
        shas = []
        with tempfile.TemporaryDirectory() as td:
            for i in range(a.runs):
                res = support.run_relax(world, spec["cycles"], spec["steps"], Path(td) / f"{i}.flux")
                if not res["ok"]:
                    print(f"{name}: run {i + 1} failed; nothing recorded", file=sys.stderr)
                    status = 1
                    break
                shas.append(res["sha256"])
        if len(shas) != a.runs:
            continue
        if len(set(shas)) != 1:
            print(f"{name}: {len(set(shas))} different SHA256 in {a.runs} runs; not recorded (Tier L only)")
            status = 1
            continue
        existing = support.find_build(spec, key)
        if existing is not None and existing["key"].get("flux_pm_sha256") == key["flux_pm_sha256"]:
            print(f"{name}: this build is already recorded; not changed")
            continue
        spec.setdefault("builds", []).append(
            {"key": key, "sha256": shas[0], "runs": f"{a.runs} of {a.runs} identical", "flux_pm": prov["flux_pm"]}
        )
        print(f"{name}: recorded {shas[0][:12]} for {key}")
    support.save_golden(g)
    return status


if __name__ == "__main__":
    sys.exit(main())
