"""Helpers for the realflux golden suite: build key, topology reader, relax runner, golden.json access.

Everything here is plain Python; only `run_relax` and `roundtrip` start the FLUX Perl.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fluxpype.scenes import relax as R

DATA = Path(__file__).resolve().parent.parent / "data" / "realflux"
GOLDEN_JSON = DATA / "golden.json"

# The fields that identify a build. An entry in golden.json matches when every field it names equals the loaded
# build's value (the 2.106 entry predates Flux.pm hashing and names three).
KEY_FIELDS = ("pdl", "perl", "flux_pm_sha256", "fluxon_mhd_sha")


def load_golden() -> Dict:
    return json.loads(GOLDEN_JSON.read_text())


def save_golden(g: Dict) -> None:
    GOLDEN_JSON.write_text(json.dumps(g, indent=1) + "\n")


def build_key(prov: Dict) -> Dict:
    """The comparable part of `relax.provenance()`."""
    key = {k: prov.get(k) for k in KEY_FIELDS}
    if key["fluxon_mhd_sha"]:
        key["fluxon_mhd_sha"] = key["fluxon_mhd_sha"][:7]  # compare short shas, however the CI spelled it
    return key


def key_matches(entry_key: Dict, key: Dict) -> bool:
    """True when every field the entry names equals the build's; an unknown (null) build value never matches."""
    return all(key.get(k) is not None and key.get(k) == v for k, v in entry_key.items())


def find_build(fixture: Dict, key: Dict) -> Optional[Dict]:
    for b in fixture.get("builds", []):
        if key_matches(b["key"], key):
            return b
    return None


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def topology(path) -> Tuple[int, List[Tuple]]:
    """(fluxon count, sorted connectivity) from the LINE records of a .flux file.

    A LINE record is `LINE label start_vertex end_vertex start_conc end_conc flux x0 y0 z0 x1 y1 z1`. Vertex ids and
    positions depend on how many vertices fix_curvature / fix_proximity left, so they are not part of the topology;
    the label, both end concentrations and the flux per fluxon are.
    """
    lines = []
    for ln in Path(path).read_text().splitlines():
        t = ln.split()
        if t and t[0] == "LINE":
            lines.append((int(t[1]), int(t[4]), int(t[5]), float(t[6])))
    return len(lines), sorted(lines)


def run_relax(world, cycles: int, steps: int, out, perl: Optional[str] = None) -> Dict:
    """The documented golden command, in process: serial, fluxpype force list, no Plumber registration."""
    res = R.relax(str(world), out=str(out), cycles=cycles, steps=steps, perl=perl)
    final = res["series"][-1] if res["series"] else {}
    return {"ok": res["ok"], "relaxed": res["relaxed"], "final": final, "sha256": sha256(out) if res["ok"] else None}


ROUNDTRIP = r"""
use strict; use warnings;
use PDL; use Flux; use Flux::World qw(read_world);
our @KEEP;   # Flux objects crash in DESTROY; keep every one alive until exit
my ($in, $out) = @ARGV;
my $w = read_world($in); push @KEEP, $w;
$w->write_world($out);
print "DONE\n";
"""


def roundtrip(src, dst, perl: Optional[str] = None) -> None:
    """read_world then write_world through FLUX's own Perl (exit-time crashes after DONE are ignored)."""
    r = subprocess.run(
        [R.find_perl(perl), "-", str(src), str(dst)], input=ROUNDTRIP, capture_output=True, text=True, timeout=120
    )
    if "DONE" not in r.stdout:
        raise RuntimeError(f"round trip failed: {r.stderr[-500:]}")
