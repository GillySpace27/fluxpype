"""Headless scene relaxation: convergence rule, driver output parsing and the CLI, with the Perl side mocked.

Nothing here needs PDL or FLUX; a fake process replays the lines the Perl driver prints.
"""

import io
import json

import pytest

from fluxpype.plumber import external_runs
from fluxpype.scenes import PlanarArcadeSpec, compile_flux, shift_pairing
from fluxpype.scenes import relax as R


def test_settled_at_is_two_sided_and_ignores_a_turning_point():
    # falls to a minimum at index 20, rises again, flat from 40; the first index flat 10 either side is 50
    v = [10 - 0.4 * i for i in range(21)] + [2 + 0.2 * i for i in range(1, 20)] + [6.0] * 30
    assert R.settled_at(v, window=10, tol=0.01) == 50
    assert R.settled_at(v[:45], window=10, tol=0.01) is None  # nothing at the turning point
    assert R.settled_at([1.0] * 15, window=10) is None  # needs window values on both sides
    assert R.settled_at([5.0] * 21, window=10) == 10
    assert R.settled_at([1.0, None] + [1.0] * 30, window=10) == 12


def test_parse_line_reads_cycle_records_only():
    assert R.parse_line('CYCLE {"cycle": 3, "e_simple2": 1.5}') == {"cycle": 3, "e_simple2": 1.5}
    assert R.parse_line("Read 'w.flux': 193 vertices") is None
    assert R.parse_line("CYCLE {broken") is None


def _cycles(n, e0=2.0, rate=0.0):
    lines = ['CYCLE {"cycle":-1,"steps":0,"stiffness":null,"e_simple2":9,"top_z":1,"wall_s":0}\n']
    for c in range(n):
        lines.append(
            json.dumps(
                {
                    "cycle": c,
                    "steps": 100 * (c + 1),
                    "stiffness": 1.0,
                    "e_simple2": e0 + rate * c,
                    "top_z": 1.5,
                    "wall_s": c,
                    "n": 100,
                }
            ).join(["CYCLE ", "\n"])
        )
    return lines


class FakeProc:
    def __init__(self, lines, rc):
        self.stdin, self.stdout, self.pid, self.rc = io.StringIO(), iter(lines), 4242, rc

    def wait(self):
        return self.rc


def fake_popen(lines, rc=0, seen=None):
    def popen(args, **kw):
        if seen is not None:
            seen.append(args)
        return FakeProc(lines, rc)

    return popen


def test_relax_tolerates_the_exit_time_segfault_after_done(tmp_path):
    w = tmp_path / "w.flux"
    w.write_text("x")
    seen = []
    res = R.relax(str(w), cycles=25, popen=fake_popen(_cycles(25) + ["DONE\n"], rc=-11, seen=seen))
    assert res["ok"] and res["returncode"] == -11 and len(res["series"]) == 26
    assert res["settled_at"] == 10 and res["relaxed"].endswith("w.relaxed.flux")
    assert seen[0][1:4] == ["-", str(w.resolve()), res["relaxed"]] and seen[0][4:6] == ["25", "100"]
    assert "CYCLE" in open(res["log"]).read()


def test_relax_is_a_failure_without_done_or_with_missing_cycles(tmp_path):
    w = tmp_path / "w.flux"
    w.write_text("x")
    assert not R.relax(str(w), cycles=5, popen=fake_popen(_cycles(5), rc=-11))["ok"]
    assert not R.relax(str(w), cycles=5, popen=fake_popen(_cycles(3) + ["DONE\n"]))["ok"]


def test_relax_registers_progress_and_result_in_plumber(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUX_PLUMBER_HOME", str(tmp_path / "home"))
    w = tmp_path / "w.flux"
    w.write_text("x")
    R.relax(str(w), cycles=25, run_id="scene-relax-test", popen=fake_popen(_cycles(25) + ["DONE\n"]))
    run = external_runs.get(tmp_path / "home", "scene-relax-test")
    assert run["status"] == "ok" and run["pid"] == 4242
    assert {"label": "cycle", "value": "25/25"} in run["progress"]
    assert "settled from cycle 10" in run["summary"]


def test_difference_pairs_cycles():
    a = [{"cycle": 0, "e_simple2": 1.0}, {"cycle": 1, "e_simple2": 0.9}]
    b = [{"cycle": 0, "e_simple2": 1.5}, {"cycle": 1, "e_simple2": 1.2}, {"cycle": 2, "e_simple2": 1.1}]
    assert [round(r["d_e_simple2"], 9) for r in R.difference(a, b)] == [0.5, 0.3]


def _world(tmp_path, name, **kw):
    p = tmp_path / name
    p.write_text(compile_flux(PlanarArcadeSpec(n_lines=5, **kw)))
    return str(p)


def test_compare_refuses_an_unfair_pair(tmp_path, capsys):
    a = _world(tmp_path, "a.flux", lid_height=3.0)
    b = _world(tmp_path, "b.flux", lid_height=4.0)
    assert R.main(["compare", a, b, "--no-register"]) == 2
    assert "global b2 differs" in capsys.readouterr().err


def test_compare_reports_the_difference_series(tmp_path, monkeypatch, capsys):
    a = _world(tmp_path, "a.flux", lid_height=3.0)
    b = _world(tmp_path, "b.flux", lid_height=3.0, pairing=shift_pairing(5, 1))

    def popen(args, **kw):
        lines = _cycles(30, e0=2.0) if args[2].endswith("a.flux") else _cycles(30, e0=2.5)
        return FakeProc(lines + ["DONE\n"], 0)

    monkeypatch.setattr(R.subprocess, "Popen", popen)
    out = tmp_path / "series.json"
    assert R.main(["compare", a, b, "--cycles", "30", "--no-register", "--series", str(out)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["difference_final"] == pytest.approx(0.5) and rep["difference_settled_at"] == 10
    assert len(json.loads(out.read_text())["difference"]) == 30
