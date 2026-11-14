"""Relax a `.flux` world headless with FLUX, the way fluxpype does, and say when it has stopped changing.

    python -m fluxpype.scenes.relax run WORLD.flux [--cycles 150] [--steps 100] [--out RELAXED.flux]
    python -m fluxpype.scenes.relax compare A.flux B.flux [same options]

`run` relaxes one world: fluxpype's force list (`f_pressure_equi2b f_curvature f_vertex4 b_eqa`), `do_steps`
steps per cycle, `fix_curvature(0.3, 0.1)` and `fix_proximity(2.5)` after cycles 0 to 2 (the `config.ini`
defaults), serial. After each cycle it records the stiffness (f_av / fs_av * 100, as fluxpype computes it), the
magnetic energy from `e_simple2` and the top of the field (highest vertex z). The energy and the top z count real
fluxons only (label > 0); FLUX's image fluxons (labels below 0) are left out.

`compare` refuses two worlds that are not a fair pair (`checks.comparable`: same footpoints, photosphere, lid
and vertex counts) and otherwise relaxes both at once and reports the energy difference B - A per cycle.

Convergence is two-sided: a series counts as settled at cycle c when every value from c - window to c + window is
within `tol` (relative) of the value at c. A one-sided "flat over the last 10 cycles" rule fires at turning points;
this one cannot, but it needs `window` cycles after the answer, so the last `window` cycles can never be the
answer. Settled means the numbers stopped moving, not that they are physically right.

FLUX needs its own Perl (the one that can `use Flux`): `--perl` or `$FLUX_PERL`, else `perl` on PATH. FLUX's
Perl objects often crash when the process exits (SIGSEGV after all output is written); the driver prints `DONE`
first, and a crash after `DONE` is not treated as a failure.

Each run registers itself in Plumber's external-run registry (a JSON file under the Plumber home, the same
thing `PUT /api/external-runs/<id>` writes) so it shows up on the Runs page and to Plum. `--no-register` skips it.
"""

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .checks import comparable

FORCES = ("f_pressure_equi2b", "f_curvature", "f_vertex4", "b_eqa")

# Read by `perl -` from stdin. Prints one "CYCLE {json}" line per cycle and "DONE" before writing nothing else.
DRIVER = r"""
use strict; use warnings;
use PDL; use Flux; use Flux::World qw(read_world);
use Time::HiRes qw(clock_gettime);
$| = 1;
my ($in, $out, $ncyc, $do_steps, $stop_fix, $fc_hi, $fc_lo, $fp) = @ARGV;
my @FORCES = qw(%FORCES%);
our @KEEP;   # Flux objects crash in DESTROY; keep every one alive until exit
sub real { grep { $_->{line}{label} > 0 } @_ }
sub e2 {
    my $w = shift; my $tmp = "$out.etmp.flux"; $w->write_world($tmp);
    my $c = read_world($tmp); push @KEEP, $c; unlink $tmp;
    $c->forces(@FORCES, 'e_simple2'); $c->update_neighbors(-1); $c->update_force(0);
    my @v = real($c->vertices); push @KEEP, \@v;
    my $E = 0; $E += $_->{energy} for @v; return $E;
}
sub topz { my @v = real($_[0]->vertices); push @KEEP, \@v; my $m = -1e300; for (@v) { my $z = $_->{x}->at(2); $m = $z if $z > $m } $m }
my $w = read_world($in); push @KEEP, $w;
$w->forces(@FORCES); $w->{concurrency} = 0;
my $dt = $w->{dtau} || 0.1;
printf "CYCLE {\"cycle\":-1,\"steps\":0,\"stiffness\":null,\"e_simple2\":%.10g,\"top_z\":%.10g,\"wall_s\":0}\n", e2($w), topz($w);
my $wall = 0;
for my $cycle (0 .. $ncyc - 1) {
    my $t0 = clock_gettime();
    for (1 .. $do_steps) {
        $w->update_neighbors(-1) if ($w->{rel_step} || 0) == 0;
        $w->update_force(0); $w->relax_step($dt); $w->{rel_step}++;
    }
    my $h = $w->fw_stats;
    if ($cycle <= $stop_fix) { $w->fix_curvature($fc_hi, $fc_lo); $w->fix_proximity($fp); }
    $wall += clock_gettime() - $t0;
    printf "CYCLE {\"cycle\":%d,\"steps\":%d,\"stiffness\":%.10g,\"e_simple2\":%.10g,\"top_z\":%.10g,\"wall_s\":%.3f,\"n\":%d}\n",
        $cycle, $w->{rel_step}, $h->{f_av} / $h->{fs_av} * 100, e2($w), topz($w), $wall, $h->{n};
}
$w->write_world($out);
print "DONE\n";
""".replace("%FORCES%", " ".join(FORCES))


def settled_at(values: Sequence[Optional[float]], window: int = 10, tol: float = 0.01) -> Optional[int]:
    """First index c with every value in [c - window, c + window] within `tol` (relative) of values[c]; else None."""
    n = len(values)
    for c in range(window, n - window):
        ref = values[c]
        if ref is None or not math.isfinite(ref):
            continue
        scale = abs(ref) or 1.0
        if all(v is not None and abs(v - ref) <= tol * scale for v in values[c - window : c + window + 1]):
            return c
    return None


def parse_line(line: str) -> Optional[Dict]:
    """The record on a driver "CYCLE {...}" line, else None."""
    if not line.startswith("CYCLE "):
        return None
    try:
        return json.loads(line[6:])
    except ValueError:
        return None


def find_perl(perl: Optional[str] = None) -> str:
    return perl or os.environ.get("FLUX_PERL") or "perl"


def _register(run_id: str, body: Dict) -> None:
    """Best effort: a registry write must never stop a relaxation."""
    try:
        from ..plumber import external_runs
        from ..plumber.paths import plumber_home

        external_runs.write(plumber_home(), run_id, body)
    except Exception as e:  # noqa: BLE001  ponytail: any failure here only costs visibility
        print(f"(could not register in Plumber: {e})", file=sys.stderr)


def relax(
    world: str,
    out: Optional[str] = None,
    cycles: int = 150,
    steps: int = 100,
    perl: Optional[str] = None,
    log: Optional[str] = None,
    run_id: Optional[str] = None,
    on_cycle: Optional[Callable[[Dict], None]] = None,
    stop_fixing_after: int = 2,
    fix_curvature=(0.3, 0.1),
    fix_proximity: float = 2.5,
    popen=None,
) -> Dict:
    """Relax `world`; returns {"series": [...], "relaxed": path, "ok": bool, "returncode": int, "done": bool}.

    `series[0]` is the starting world (cycle -1, no stiffness). `popen` is replaceable for tests.
    """
    world = str(Path(world).resolve())
    out = (
        str(Path(out).resolve())
        if out
        else world[:-5] + ".relaxed.flux" if world.endswith(".flux") else world + ".relaxed"
    )
    log = str(Path(log).resolve()) if log else out + ".log"
    args = [
        find_perl(perl),
        "-",
        world,
        out,
        str(cycles),
        str(steps),
        str(stop_fixing_after),
        str(fix_curvature[0]),
        str(fix_curvature[1]),
        str(fix_proximity),
    ]
    series: List[Dict] = []
    done = False
    t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(log, "w") as logf:
        proc = (popen or subprocess.Popen)(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path(out).parent),
        )
        if run_id:
            _register(
                run_id,
                {
                    "title": f"relax {Path(world).name}",
                    "status": "running",
                    "host": socket.gethostname(),
                    "cwd": str(Path(out).parent),
                    "pid": proc.pid,
                    "started": t0,
                    "log": log,
                },
            )
        proc.stdin.write(DRIVER)
        proc.stdin.close()
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            rec = parse_line(line)
            if rec is not None:
                series.append(rec)
                if on_cycle:
                    on_cycle(rec)
                if run_id and rec["cycle"] >= 0:
                    _register(
                        run_id,
                        {
                            "progress": [
                                {"label": "cycle", "value": f"{rec['cycle'] + 1}/{cycles}"},
                                {"label": "stiffness %", "value": round(rec["stiffness"], 5)},
                                {"label": "e_simple2", "value": round(rec["e_simple2"], 6)},
                                {"label": "top z", "value": round(rec["top_z"], 4)},
                            ]
                        },
                    )
            elif line.strip() == "DONE":
                done = True
        rc = proc.wait()
    ok = done and len(series) == cycles + 1  # a crash after DONE is FLUX's exit-time segfault, not a failure
    e = [r["e_simple2"] for r in series[1:]]
    s = settled_at(e)
    if run_id:
        _register(
            run_id,
            {
                "status": "ok" if ok else "failed",
                "report": out,
                "summary": (f"{len(series) - 1} cycles; e_simple2 {e[-1]:.6g}; " if e else "no cycles; ")
                + (f"settled from cycle {s}" if s is not None else "energy not settled")
                + ("" if ok else f"; exit {rc}, done={done}"),
            },
        )
    return {"series": series, "relaxed": out, "log": log, "ok": ok, "returncode": rc, "done": done, "settled_at": s}


def difference(a: List[Dict], b: List[Dict]) -> List[Dict]:
    """Per cycle: B - A of e_simple2, for cycles present in both series."""
    ea = {r["cycle"]: r["e_simple2"] for r in a}
    return [{"cycle": r["cycle"], "d_e_simple2": r["e_simple2"] - ea[r["cycle"]]} for r in b if r["cycle"] in ea]


def _run_id(world: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in Path(world).stem)[:40]
    return f"scene-relax-{stem}-{int(time.time())}"


def _summary(res: Dict) -> Dict:
    last = res["series"][-1] if res["series"] else {}
    return {
        **{k: res[k] for k in ("relaxed", "log", "ok", "returncode", "settled_at")},
        "cycles": len(res["series"]) - 1,
        "final": last,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fluxpype.scenes.relax", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, nargs in (("run", 1), ("compare", 2)):
        p = sub.add_parser(name)
        p.add_argument("worlds", nargs=nargs)
        p.add_argument("--cycles", type=int, default=150)
        p.add_argument("--steps", type=int, default=100, help="relaxation steps per cycle (fluxpype do_steps)")
        p.add_argument("--perl", help="the Perl that can `use Flux` (default $FLUX_PERL, else perl)")
        p.add_argument("--series", help="write the per-cycle series as JSON here")
        p.add_argument("--no-register", action="store_true", help="do not list the run in Plumber")
        if name == "run":
            p.add_argument("--out", help="relaxed world (default WORLD.relaxed.flux)")
    a = ap.parse_args(argv)
    common = dict(cycles=a.cycles, steps=a.steps, perl=a.perl)

    if a.cmd == "run":
        w = a.worlds[0]
        res = relax(w, out=a.out, run_id=None if a.no_register else _run_id(w), **common)
        if a.series:
            Path(a.series).write_text(json.dumps(res["series"], indent=1))
        print(json.dumps(_summary(res), indent=1))
        return 0 if res["ok"] else 1

    wa, wb = a.worlds
    why = comparable(Path(wa).read_text(), Path(wb).read_text())
    if why:
        print("not a fair pair: " + "; ".join(why), file=sys.stderr)
        return 2
    results: Dict[str, Dict] = {}
    threads = [
        threading.Thread(
            target=lambda w=w: results.__setitem__(w, relax(w, run_id=None if a.no_register else _run_id(w), **common))
        )
        for w in (wa, wb)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ra, rb = results[wa], results[wb]
    d = difference(ra["series"][1:], rb["series"][1:])
    s = settled_at([r["d_e_simple2"] for r in d])
    if a.series:
        Path(a.series).write_text(json.dumps({"a": ra["series"], "b": rb["series"], "difference": d}, indent=1))
    print(
        json.dumps(
            {
                "a": _summary(ra),
                "b": _summary(rb),
                "difference_final": d[-1]["d_e_simple2"] if d else None,
                "difference_settled_at": s,
                "difference_at_settle": d[s]["d_e_simple2"] if s is not None else None,
            },
            indent=1,
        )
    )
    return 0 if ra["ok"] and rb["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
