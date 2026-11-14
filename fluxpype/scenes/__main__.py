"""python -m fluxpype.scenes compile spec.json out.flux   |   example [--shear K] [--lid Z] > spec.json"""

import argparse
import json
import sys

from . import PlanarArcadeSpec, compile_flux, shift_pairing, summarize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fluxpype.scenes", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="write the .flux world for a scene file and print its construction checks")
    c.add_argument("spec")
    c.add_argument("out")
    e = sub.add_parser("example", help="print an example scene file")
    e.add_argument("--lines", type=int, default=9)
    e.add_argument("--shear", type=int, default=0, help="pairing shift in lines (0 = perpendicular arcs)")
    e.add_argument("--lid", type=float, default=None, help="B2 lid height")
    e.add_argument("--wall", type=float, default=None, help="B2 cylinder wall radius (open top)")
    a = ap.parse_args(argv)
    if a.cmd == "example":
        spec = PlanarArcadeSpec(
            n_lines=a.lines, pairing=shift_pairing(a.lines, a.shear), lid_height=a.lid, wall_radius=a.wall
        )
        print(json.dumps(spec.validate().to_dict(), indent=1))
        return 0
    with open(a.spec) as f:
        spec = PlanarArcadeSpec.from_dict(json.load(f))
    text = compile_flux(spec)
    with open(a.out, "w") as f:
        f.write(text)
    report = summarize(text)
    print(json.dumps(report, indent=1))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
