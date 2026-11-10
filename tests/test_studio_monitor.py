"""
Monitor: events-only, log-only, mixed, stale events (degradation), crash and
verification. The log fixtures are SYNTHESIZED FROM SOURCE (the print statements in
the .pdl/.pm/.py files), not captured from a real run; replace them with a real
`quickstart` log when one is available.
"""

import json
import os

import pytest

from fluxpype.studio.monitor import STAGE_IDS, ProbeContext, TaskMonitor

BAR = "~" * 63


def ev(etype, t, /, **fields):
    return json.dumps({"v": 1, "t": t, "type": etype, "pid": 1, "task": "x", **fields}) + "\n"


class Run:
    def __init__(self, tmp_path):
        self.dir = tmp_path / "run"
        self.dir.mkdir()
        self.data = tmp_path / "data"
        self.data.mkdir()
        self.ctx = ProbeContext(str(self.data), "b", "2225", "100", "0", "hmi")

    def events(self, *lines, mode="a"):
        with open(self.dir / "events.jsonl", mode) as f:
            f.write("".join(lines))

    def log(self, text, mode="a"):
        with open(self.dir / "run.log", mode) as f:
            f.write(text)

    def mon(self, **kw):
        return TaskMonitor(str(self.dir), self.ctx, **kw)

    def touch(self, rel):
        p = self.data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


def stage(state, sid):
    return next(s for s in state["stages"] if s["id"] == sid)


def banner(title):
    return f"\n{BAR}\n{title}\n{BAR}\n"


STEP = "\nstep {n}: time is now {t:10.4g} \n({v} vertices; stiffness={s:7.3g}%, fmax={fm:7.3g}, f_av={fa:7.3g}, fs_av={fs:7.3g}, n_av={na:7.3g} max_angle:  80.00, mean_angle:  31.00)\n"  # noqa: E501


def full_log():
    return (
        "LIBS ARE LOADED\n\n\nTHE CR IS 2225\n\n"
        + banner("(pdl) Tracing Magnetogram to get 100 Footpoints")
        + "\n\tFound a 97 footpoint file on Disk:\n"
        + banner("(py) Running PFSS Code to Trace Footpoints into the Corona")
        + banner("(pdl) Converting PFSS Fieldlines into FLUX Fluxons")
        + banner("(pdl) Relax the Initial World State from PFSS")
        + "\n\tRelaxing PFSS model for 100 steps to 1.5 stiffness...\n"
        + STEP.format(n=0, t=0.1, v=4100, s=100, fm=0.2, fa=0.02, fs=0.02, na=7.1)
        + STEP.format(n=20, t=2.1, v=4100, s=62.5, fm=0.12, fa=0.012, fs=0.02, na=7.0)
        + "\tCycle 0, Cumulative relaxation time: 12.50 seconds, 0 steps\n"
        + "\nCycle 0 :: Curvature changed 12 vertices, proximity changed 3 vertices\n"
        + "\tRelaxation Complete in 100 Steps \n"
        + banner("(py) Plotting Magnetic Field Maps")
        + banner("(pdl) Calculating Coronal Plasma Parameters for CR2225")
        + banner("(py) Plotting Solar Wind Maps for CR2225")
        + "\tProgram Complete for CR2225 in 61.5 Seconds, press RETURN to exit...\n"
    )


def test_events_only(run):
    run.events(
        ev("run.start", 100, cr="2225", profile="p"),
        ev("stage.start", 101, stage="magnetogram"),
        ev("stage.end", 103, stage="magnetogram", status="ok", elapsed_s=2, detail=None),
        ev("stage.start", 103, stage="footpoints"),
        ev("metric", 104, name="n_actual", value=97, unit="fluxons"),
        ev("stage.end", 105, stage="footpoints", status="ok", elapsed_s=2, detail=None),
        ev("stage.start", 105, stage="pfss"),
    )
    s = run.mon().refresh(now=200, alive=True)
    assert s["mode"] == "events" and not s["degraded"]
    assert stage(s, "magnetogram")["status"] == "ok" and stage(s, "magnetogram")["source"] == "event"
    assert stage(s, "pfss")["status"] == "running" and s["current_stage"] == "pfss"
    assert stage(s, "wind")["status"] == "pending" and stage(s, "wind")["source"] == "none"
    assert s["metrics"][0]["name"] == "n_actual" and s["metrics"][0]["value"] == 97


def test_relax_events_feed_the_series(run):
    lines = [ev("stage.start", 10, stage="relax")]
    for i in range(5):
        lines.append(
            ev(
                "relax.step",
                11 + i,
                step=i * 20,
                stiffness=100 / (i + 1),
                fmax=0.1,
                f_av=0.01,
                fs_av=0.02,
                n_av=7,
                vertices=4000,
            )
        )
    lines.append(
        ev(
            "relax.cycle",
            17,
            cycle=0,
            threshold=1.65,
            stiffness=20.0,
            cum_time_s=7,
            steps=0,
            nc_curve=3,
            nc_proxi=1,
            broken=0,
        )
    )
    run.events(*lines)
    mon = run.mon()
    s = mon.refresh(now=20, alive=True)
    assert s["relax"]["n_points"] == 5 and s["relax"]["source"] == "event"
    assert s["relax"]["latest"]["step"] == 80 and s["relax"]["cycle"]["threshold"] == 1.65
    series = mon.relax_series(after=3)
    assert series["n"] == 5 and [p["step"] for p in series["points"]] == [60, 80]
    assert series["cycles"][0]["nc_curve"] == 3


def test_partial_lines_wait_for_their_newline(run):
    run.events(ev("stage.start", 1, stage="pfss")[:-12])
    mon = run.mon()
    assert mon.refresh(now=2)["sources"]["events"]["lines"] == 0
    run.events(ev("stage.start", 1, stage="pfss")[-12:])
    # the completed line is now a whole record (the first fragment was never consumed on its own)
    s = mon.refresh(now=3)
    assert s["sources"]["events"]["lines"] == 1 and stage(s, "pfss")["status"] == "running"


def test_garbage_lines_are_counted_not_fatal(run):
    run.events("not json\n", "{}\n", ev("stage.start", 1, stage="pfss"))
    s = run.mon().refresh(now=2)
    assert s["sources"]["events"]["bad_lines"] == 2 and stage(s, "pfss")["status"] == "running"


def test_log_only_finished_task(run):
    run.log(full_log())
    s = run.mon().refresh(now=50, outcome="ok", started=0)
    assert s["mode"] == "log" and s["sources"]["events"]["present"] is False
    for sid in STAGE_IDS[:9]:
        assert stage(s, sid)["status"] == "ok" and stage(s, sid)["source"] == "log", sid
    assert stage(s, "fluxlight")["status"] == "skipped"
    assert s["relax"]["n_points"] == 2 and s["relax"]["source"] == "log"
    assert s["relax"]["latest"]["stiffness"] == 62.5 and s["relax"]["latest"]["vertices"] == 4100
    assert s["relax"]["cycle"]["nc_curve"] == 12
    assert s["run"]["log_done"] is True


def test_log_only_running_task(run):
    run.log(
        "THE CR IS 2225\n"
        + banner("(pdl) Tracing Magnetogram to get 100 Footpoints")
        + banner("(py) Running PFSS Code to Trace Footpoints into the Corona")
    )
    s = run.mon().refresh(now=5, alive=True, started=0)
    assert stage(s, "magnetogram")["status"] == "ok" and stage(s, "footpoints")["status"] == "ok"
    assert stage(s, "pfss")["status"] == "running" and s["current_stage"] == "pfss"
    assert stage(s, "world")["status"] == "pending"


def test_log_with_wind_only_skips_the_early_stages(run):
    run.log(
        "THE CR IS 2225\n"
        + banner("(pdl) Calculating Coronal Plasma Parameters for CR2225")
        + banner("(py) Plotting Solar Wind Maps for CR2225")
    )
    s = run.mon().refresh(now=5, outcome="ok", started=0)
    # the magnetogram marker prints before the wind banner, but the stages between never ran
    assert stage(s, "footpoints")["status"] != "running"
    assert stage(s, "wind")["status"] == "ok"


def test_ansi_and_carriage_returns_are_handled(run):
    run.log("\x1b[96m" + banner("(pdl) Relax the Initial World State from PFSS") + "\x1b[0m")
    run.log(
        "\rProcessing:  50%|#####     |\r"
        + STEP.format(n=40, t=4.1, v=10, s=5, fm=1, fa=1, fs=1, na=1).replace("\n", "\r\n")
    )
    s = run.mon().refresh(now=5, alive=True)
    assert stage(s, "relax")["status"] == "running" and s["relax"]["latest"]["step"] == 40


def test_nan_stiffness_in_the_log_does_not_crash(run):
    run.log(STEP.format(n=0, t=0, v=1, s=float("nan"), fm=1, fa=1, fs=1, na=1))
    s = run.mon().refresh(now=1, alive=True)
    assert s["relax"]["n_points"] == 1


def test_events_win_over_the_log_and_cached_hints_are_kept(run):
    run.events(
        ev("stage.start", 1, stage="relax"),
        ev("stage.end", 5, stage="relax", status="ok", elapsed_s=4, detail="reused relaxed world"),
    )
    run.log(
        banner("(pdl) Relax the Initial World State from PFSS") + "\n\tSkipped relaxation! Already have relaxed file.\n"
    )
    s = run.mon().refresh(now=10, alive=True)
    st = stage(s, "relax")
    assert st["source"] == "event" and st["status"] == "ok" and st["detail"] == "reused relaxed world"


def test_event_ok_stage_is_verified_against_files(run):
    run.events(
        ev("stage.start", 1, stage="pfss"), ev("stage.end", 2, stage="pfss", status="ok", elapsed_s=1, detail=None)
    )
    s = run.mon().refresh(now=3, alive=True, started=0)
    assert stage(s, "pfss")["verified"] is False  # events say ok but there is no pickle on disk
    run.touch("pfss/pfss_cr2225_r4_hmi.pkl")
    s = run.mon().refresh(now=3, alive=True, started=0)
    assert stage(s, "pfss")["verified"] is True
    assert stage(s, "world_plots")["verified"] is None  # no probe for this stage


def test_files_fill_in_for_a_stage_nobody_reported(run):
    run.events(
        ev("stage.start", 1, stage="magnetogram"),
        ev("stage.end", 2, stage="magnetogram", status="ok", elapsed_s=1, detail=None),
        ev("stage.start", 3, stage="world"),
    )
    run.touch("batches/b/data/cr2225/floc/floc_cr2225_r4_f100_hmi.dat")
    s = run.mon().refresh(now=5, alive=True, started=0)
    fp = stage(s, "footpoints")
    assert fp["status"] == "ok" and fp["source"] == "file"


def test_old_output_on_disk_is_reported_as_already_there(run):
    run.events(ev("stage.start", 1, stage="world"))
    p = run.data / "batches/b/data/cr2225/floc/floc_cr2225_r4_f100_hmi.dat"
    p.parent.mkdir(parents=True)
    p.write_text("x")
    os.utime(p, (50, 50))
    s = run.mon().refresh(now=5, alive=True, started=1000)
    fp = stage(s, "footpoints")
    assert fp["status"] == "skipped" and fp["detail"] == "output already on disk"


def test_stale_events_degrade_and_recover(run):
    run.events(
        ev("stage.start", 1, stage="relax"),
        ev("relax.step", 2, step=0, stiffness=90.0, fmax=1, f_av=1, fs_av=1, n_av=1, vertices=5),
    )
    run.log(
        banner("(pdl) Relax the Initial World State from PFSS")
        + STEP.format(n=0, t=0, v=5, s=90, fm=1, fa=1, fs=1, na=1)
    )
    mon = run.mon(stale_after=30, min_stale=5)
    s = mon.refresh(now=100, alive=True)
    assert not s["degraded"]
    # events go quiet, the log keeps moving
    run.log(STEP.format(n=20, t=2, v=5, s=70, fm=1, fa=1, fs=1, na=1))
    s = mon.refresh(now=110, alive=True)
    assert not s["degraded"]  # quiet for 10 s only (< 30)
    run.log(STEP.format(n=40, t=4, v=5, s=50, fm=1, fa=1, fs=1, na=1))
    s = mon.refresh(now=160, alive=True)
    assert s["degraded"] and s["mode"] == "degraded" and "no events" in s["degraded_reason"]
    # log-derived relaxation numbers take over for steps the events never reported
    assert s["relax"]["latest"]["step"] == 40 and s["relax"]["source"] == "log"
    assert [p["s"] for p in mon.relax_points()] == ["event", "log", "log"]
    # events resume: back to normal
    run.events(ev("relax.step", 170, step=60, stiffness=30.0, fmax=1, f_av=1, fs_av=1, n_av=1, vertices=5))
    s = mon.refresh(now=171, alive=True)
    assert not s["degraded"] and s["relax"]["latest"]["step"] == 60


def test_quiet_events_with_a_quiet_log_are_not_degraded(run):
    run.events(ev("stage.start", 1, stage="wind"))
    run.log(banner("(pdl) Calculating Coronal Plasma Parameters for CR2225"))
    mon = run.mon(stale_after=30)
    mon.refresh(now=0, alive=True)
    assert not mon.refresh(now=500, alive=True)["degraded"]  # nothing is advancing: that is a stall, not degradation


def test_stale_threshold_follows_the_observed_print_interval(run):
    lines = [
        ev("relax.step", 10 + 2 * i, step=i, stiffness=50, fmax=1, f_av=1, fs_av=1, n_av=1, vertices=1)
        for i in range(8)
    ]
    run.events(*lines)
    mon = run.mon(stale_after=120, min_stale=5)
    mon.refresh(now=100, alive=True)
    assert mon._stale_threshold() == pytest.approx(6.0)  # 3 x the 2 s interval


def test_crash_closes_the_running_stage(run):
    run.events(ev("stage.start", 1, stage="relax"))
    s = run.mon().refresh(now=5, alive=False, outcome="crashed", started=0)
    st = stage(s, "relax")
    assert st["status"] == "failed" and "vanished" in st["detail"] and s["outcome"] == "crashed"


def test_failure_banner_in_the_log_marks_the_stage(run):
    run.log(
        "THE CR IS 2225\n"
        + banner("(py) Running PFSS Code to Trace Footpoints into the Corona")
        + "get_pfss.py failed (status 256)\n"
    )
    s = run.mon().refresh(now=5, outcome="failed", started=0)
    st = stage(s, "pfss")
    assert st["status"] == "failed" and "get_pfss.py failed" in st["detail"]
    assert any("get_pfss.py failed" in w for w in s["warnings"])


def test_successful_task_leaves_no_pending_chips(run):
    run.events(
        ev("stage.start", 1, stage="wind"), ev("stage.end", 2, stage="wind", status="ok", elapsed_s=1, detail=None)
    )
    s = run.mon().refresh(now=5, outcome="ok", started=0)
    assert all(st["status"] != "pending" for st in s["stages"])


def test_artifacts_from_events_and_images_on_disk(run):
    run.events(ev("artifact", 5, path="/x/cr2225_fluxlight_a.png", kind="image"))
    run.touch("batches/b/data/cr2225/world/w.png")
    s = run.mon().refresh(now=10, alive=True, started=0)
    srcs = {a["path"].rsplit("/", 1)[-1]: a["source"] for a in s["artifacts"]}
    assert srcs == {"cr2225_fluxlight_a.png": "event", "w.png": "file"}


def test_probe_context_patterns_cover_adapt_names():
    ctx = ProbeContext("/d", "b", "2225", "100", "1", "adapt", adapt_select="3")
    pats = ctx.patterns()
    assert pats["magnetogram"] == ["/d/magnetograms/CR2225_rf3_adapt.fits"]
    assert pats["relax"][0].endswith("cr2225_f100_adapt_relaxed_s*.flux")
