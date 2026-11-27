"""stage_timing reads per-stage wall time from a recorded events.jsonl (A5-01).

The fixture is synthetic: its stage times copy the A5-01 audit table (quoted, run2,
100 fluxons, relax_concurrency 4) so the expected values are known, not measured here."""

import json

from fluxpype import stage_timing

QUOTED = {"magnetogram": 3.10, "pfss": 4.59, "relax": 83.77, "wind": 108.94, "fluxlight": 19.73}


def fixture():
    t, recs = 1000.0, [{"type": "run.start", "t": 1000.0, "task": "a"}]
    for name, wall in QUOTED.items():
        recs.append({"type": "stage.start", "t": t, "stage": name, "task": "a"})
        t += wall
        recs.append({"type": "stage.end", "t": t, "stage": name, "status": "ok", "elapsed_s": wall, "task": "a"})
    recs.append({"type": "stage.end", "t": t, "stage": "wind_plots", "status": "skipped", "elapsed_s": 0, "task": "a"})
    recs.append({"type": "run.end", "t": t, "status": "ok", "elapsed_s": 236.3, "task": "a"})
    return [json.dumps(r) for r in recs]


def test_table_matches_quoted_values():
    s = stage_timing.summarize(fixture())
    by = {r["stage"]: r for r in s["stages"]}
    assert {k: by[k]["wall_s"] for k in QUOTED} == QUOTED
    assert by["wind_plots"]["status"] == "skipped"
    assert s["total_s"] == 236.3 and s["total_source"] == "run.end elapsed_s"
    assert by["relax"]["share"] == round(83.77 / 236.3, 3)


def test_falls_back_to_timestamps_and_tolerates_junk():
    lines = ["not json", json.dumps({"type": "stage.start", "t": 10.0, "stage": "x"})]
    lines.append(json.dumps({"type": "stage.end", "t": 12.5, "stage": "x", "status": "ok"}))
    s = stage_timing.summarize(lines)
    assert s["bad_lines"] == 1 and s["stages"][0]["wall_s"] == 2.5
    assert s["total_source"].startswith("sum of stages")


def test_task_filter_and_cli(tmp_path, capsys):
    p = tmp_path / "e.jsonl"
    p.write_text("\n".join(fixture()) + "\n")
    assert stage_timing.summarize(fixture(), task="zzz")["stages"] == []
    assert stage_timing.main([str(p)]) == 0
    assert "relax" in capsys.readouterr().out
    assert stage_timing.main([str(tmp_path / "missing")]) == 2
