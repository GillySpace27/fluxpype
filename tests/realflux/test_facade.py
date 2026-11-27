"""The flux_api facade (audit A1-01, A1-09): the same relax as the scenes driver, made only of facade calls.

`FACADE_DRIVER` repeats scenes/relax.py's DRIVER (same core calls in the same order, same per-cycle measurements, so
the same heap history) with every core call routed through `flux_api`. Two things are proved:
 - the relaxed world passes the same topology and band checks as the golden (Tier L); its SHA256 is recorded next
   to the scenes driver's (the two drivers agree when the run is the common outcome; A1-05 makes some runs differ);
 - the process exits 0 after DONE, where a driver that lets Perl tear the worlds down exits 139 (A1-03).
The fork-child rule needs no FLUX and runs on every CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fluxpype.scenes import relax as R

from . import support

SCIENCE = Path(__file__).resolve().parents[2] / "fluxpype" / "science"

FACADE_DRIVER = r"""
use strict; use warnings;
use PDL; use Flux; use Flux::World qw(read_world);
use Time::HiRes qw(clock_gettime);
use flux_api qw(load keep relax_cycle fix stats save done finish);
$| = 1;
my ($in, $out, $ncyc, $do_steps, $stop_fix, $fc_hi, $fc_lo, $fp) = @ARGV;
my @FORCES = qw(%FORCES%);
sub real { grep { $_->{line}{label} > 0 } @_ }
sub e2 {
    my $w = shift; my $tmp = "$out.etmp.flux"; $w->write_world($tmp);
    my $c = load($tmp); unlink $tmp;
    $c->forces(@FORCES, 'e_simple2'); $c->update_neighbors(-1); $c->update_force(0);
    my @v = real($c->vertices); keep(\@v);
    my $E = 0; $E += $_->{energy} for @v; return $E;
}
sub topz { my @v = real($_[0]->vertices); keep(\@v); my $m = -1e300; for (@v) { my $z = $_->{x}->at(2); $m = $z if $z > $m } $m }
my $w = load($in);
$w->forces(@FORCES); $w->{concurrency} = 0;
my $dt = $w->{dtau} || 0.1;   # read once: the core changes dtau while it relaxes
printf "CYCLE {\"cycle\":-1,\"steps\":0,\"stiffness\":null,\"e_simple2\":%.10g,\"top_z\":%.10g,\"wall_s\":0}\n", e2($w), topz($w);
for my $cycle (0 .. $ncyc - 1) {
    relax_cycle($w, $do_steps, $dt);
    my $h = stats($w);
    fix($w, $fc_hi, $fc_lo, $fp) if $cycle <= $stop_fix;
    printf "CYCLE {\"cycle\":%d,\"steps\":%d,\"stiffness\":%.10g,\"e_simple2\":%.10g,\"top_z\":%.10g,\"wall_s\":0,\"n\":%d}\n",
        $cycle, $w->{rel_step}, $h->{f_av} / $h->{fs_av} * 100, e2($w), topz($w), $h->{n};
}
save($w, $out);
done();
finish();
""".replace("%FORCES%", " ".join(R.FORCES))


def run_facade(world, cycles, steps, out, perl=None):
    args = [
        R.find_perl(perl),
        f"-I{SCIENCE}",
        "-",
        str(world),
        str(out),
        str(cycles),
        str(steps),
        "2",
        "0.3",
        "0.1",
        "2.5",
    ]
    return subprocess.run(
        args, input=FACADE_DRIVER, capture_output=True, text=True, timeout=300, cwd=str(Path(out).parent)
    )


@pytest.mark.parametrize("name", ["simple-4", "planar_shear2", "twisted"])
def test_facade_run_matches_the_golden_and_exits_zero(name, prov, golden, relaxed, tmp_path, record_property):
    spec = golden["fixtures"][name]
    base = relaxed(name)  # the scenes driver (skips twisted without --realflux-slow)
    out = tmp_path / f"{name}.facade.flux"
    r = run_facade(support.DATA / "fixtures" / f"{name}.flux", spec["cycles"], spec["steps"], out)
    assert r.stdout.rstrip().endswith("DONE"), r.stderr[-500:]
    assert r.returncode == 0, f"facade run exited {r.returncode}: {r.stderr[-500:]}"
    # Same checks as Tier L. An exact SHA match with the scenes driver is recorded but not asserted: planar_shear2
    # and simple-4 each end in one of two worlds from run to run with the scenes driver itself (A1-05).
    record_property("sha_equals_scenes_driver", str(support.sha256(out) == base["sha256"]))
    want_n, want_lines = support.topology(support.DATA / "relaxed" / f"{name}.relaxed.flux")
    got_n, got_lines = support.topology(out)
    assert (got_n, got_lines) == (want_n, want_lines), f"{name}: fluxon labels, end concentrations or flux changed"
    final = [R.parse_line(ln) for ln in r.stdout.splitlines() if R.parse_line(ln)][-1]
    for field in ("e_simple2", "stiffness"):
        band = spec["bands"][field]
        if band is not None:
            change = abs(final[field] - spec["reference"][field]) / abs(spec["reference"][field])
            assert change <= band, f"{name} {field}: {final[field]} ({change:.1%} > {band:.0%})"


def test_fork_child_leaves_without_destructors_and_parent_is_unaffected(tmp_path):
    """Rule 2 (A1-09 #2): a marked child exits through POSIX::_exit, so its END-time and global-destruction code never runs."""
    prog = r"""
use strict; use warnings; use flux_api qw(fork_child);
package Noisy { sub new { bless {}, shift } sub DESTROY { print "DESTROY ", ($$ == $::PARENT ? "parent" : "child"), "\n" } }
$| = 1; our $PARENT = $$; our $obj = Noisy->new;
my $pid = fork; die unless defined $pid;
if ($pid == 0) { fork_child(); exit 3 }
waitpid($pid, 0); printf "child status %d\n", $? >> 8;
"""
    r = subprocess.run(["perl", f"-I{SCIENCE}", "-e", prog], capture_output=True, text=True, timeout=30)
    assert "DESTROY child" not in r.stdout, r.stdout
    assert "child status 3" in r.stdout and "DESTROY parent" in r.stdout, r.stdout + r.stderr


def test_finish_exits_zero_after_flush():
    r = subprocess.run(
        ["perl", f"-I{SCIENCE}", "-e", 'use flux_api qw(done finish); print "a"; done(); finish(); print "never"'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0 and r.stdout.endswith("DONE\n") and "never" not in r.stdout, r
