"""
A stand-in for ``flux-config-run --only ...`` that replays a realistic run.

It prints the same banners as the real pipeline (taken from the print
statements in the .pdl/.pm/.py sources), writes schema v1 events to
``$FLUXPYPE_EVENTS`` and leaves placeholder artifacts under ``data_dir`` where
the real stages would. Time is compressed with ``--speed`` (1 means a task takes
about a minute). Scenarios exercise the plumber's failure and fallback paths:

* ``ok``              everything succeeds
* ``fail_in_pfss``    the PFSS stage fails and the task exits 1
* ``failure_swallowed`` the PFSS stage fails but the process still exits 0 (as the real runner can)
* ``slow_relax``      many more relaxation cycles
* ``no_events``       no events at all: only the log and files tell the story
* ``events_dropout``  events stop partway through relaxation while the log goes on
* ``cached_relax``    the relaxed world already exists, so relaxation is skipped

Used by ``flux-plumb --demo`` and by the tests; stdlib only.
"""

import argparse
import math
import os
import random
import re
import signal
import socket
import sys
import time

from . import config_io, paths
from ..events import emit

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
BAR = "~" * 63


class Fake:
    def __init__(self, args):
        self.a = args
        self.speed = max(float(args.speed), 0.001)
        self.rng = random.Random(f"{args.cr}-{args.nflux}-{args.adapt}-{args.method}")
        self.scenario = args.scenario
        self.events_on = self.scenario != "no_events" and bool(os.environ.get("FLUXPYPE_EVENTS"))
        self.dropout = False
        self.t0 = time.time()
        self.open_stage = None
        self.log_path = os.environ.get("FLUXPYPE_LOG_FILE")
        self.log = open(self.log_path, "a", encoding="utf-8") if self.log_path else None
        profile = args.profile or os.environ.get("FLUXPYPE_CONFIG") or "DEFAULT"
        snap = os.environ.get("FLUXPYPE_CONFIG_FILE") or str(paths.shipped_config())
        cf = config_io.read_file(snap)
        eff = cf.effective(profile) if profile in cf.sections else dict(cf.defaults)
        checkout = str(paths.checkout_root())
        self.eff = eff
        self.data_dir = config_io.resolved_path(eff, "data_dir", checkout)
        self.batch = config_io.resolve(eff, "batch_name") or "batch"
        self.reduce = config_io.resolve(eff, "mag_reduce") or "4"
        self.adapt_sel = config_io.resolve(eff, "adapt_select") or "2"
        self.inst = config_io.task_instrument(args.adapt)
        self.do_steps = int(float(config_io.resolve(eff, "do_steps") or 100))
        self.threshold = float(config_io.resolve(eff, "relax_threshold") or 1.5)
        self.max_cycles = int(float(config_io.resolve(eff, "max_cycles") or 20))
        self.stop_fix = int(float(config_io.resolve(eff, "stop_fixing_after") or 2))
        self.light = config_io.resolve(eff, "light") in ("1", "true")
        self.wind_only = config_io.resolve(eff, "wind_only") in ("1", "true")
        self.images = self.speed < 50

    # -- output helpers
    def say(self, text="", end="\n"):
        sys.stdout.write(text + end)
        sys.stdout.flush()
        if self.log:
            self.log.write(_ANSI.sub("", text + end))
            self.log.flush()

    def event(self, etype, /, **fields):
        if self.events_on and not self.dropout:
            emit(etype, **fields)

    def sleep(self, seconds):
        time.sleep(seconds / self.speed)

    def banner(self, title):
        self.say(f"\n{BAR}\n{title}\n{BAR}\n")

    # -- artifacts
    def touch(self, rel, size=4096, content=None):
        path = os.path.join(self.data_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            if content is not None:
                f.write(content)
            else:
                f.truncate(size)  # sparse placeholder with a realistic apparent size
        return path

    def image(self, rel, kind):
        try:
            from . import fake_images

            data = fake_images.render(kind, seed=f"{self.a.cr}-{self.a.nflux}-{kind}", small=not self.images)
        except Exception:
            data = None
        return self.touch(rel, content=data) if data else self.touch(rel, size=2048)

    # -- stage plumbing
    def stage(self, name):
        self.open_stage = name
        self.event("stage.start", stage=name)

    def end(self, name, status="ok", detail=None, started=None):
        self.event(
            "stage.end",
            stage=name,
            status=status,
            elapsed_s=round(time.time() - (started or self.t0), 2),
            detail=detail,
        )
        self.open_stage = None

    # -- the run
    def run(self):
        a = self.a
        cr, n, inst = a.cr, a.nflux, self.inst
        world_dir = f"batches/{self.batch}/data/cr{cr}/world"
        self.say(f"Log file: {self.log_path}\n")
        self.say(f"Executing: perl fluxpype/science/magnetogram2wind.pdl {cr} {n} {a.adapt} {a.method}\n")
        self.event(
            "run.start",
            cr=cr,
            nflux=n,
            adapt=a.adapt,
            method=a.method,
            profile=a.profile,
            host=socket.gethostname(),
            fluxpype_sha=None,
        )
        self.say("\nLIBS ARE LOADED\nFLUX is located at: /opt/flux/lib/Flux.pm\nReading configurations...Success!\n")
        self.say(f"\n\n\nTHE CR IS {cr}\n\n\n")

        if self.wind_only:
            for s in ("magnetogram", "footpoints", "pfss", "world", "relax", "world_plots", "fieldmap"):
                self.event("stage.end", stage=s, status="skipped", elapsed_s=0, detail="wind_only")
        else:
            # 1 magnetogram
            t = time.time()
            self.stage("magnetogram")
            self.sleep(2.2)
            mag = f"CR{cr}_r{self.reduce}_hmi.fits" if inst == "hmi" else f"CR{cr}_rf{self.adapt_sel}_adapt.fits"
            self.touch(f"magnetograms/{mag}", size=38_000_000)
            self.end("magnetogram", started=t)

            # 2 footpoints
            t = time.time()
            self.stage("footpoints")
            self.banner(f"(pdl) Tracing Magnetogram to get {n} Footpoints")
            self.say("\n\tSearching for file...Not found.\n\n\tRunning Hilbert Tracer...")
            self.sleep(1.6)
            n_actual = int(int(n) * self.rng.uniform(0.96, 1.0))
            self.say(f"\n\t\tPlacing {n} footpoints using Hilbert Curves...")
            for i in range(1, 4):
                self.say(f"\t\t    Placing footpoints:: iter: {i}/10. {n_actual + 40 - i * 14} footpoints")
                self.sleep(0.5)
            self.say("Success!\n\n\t\tWriting Result...Success! File saved to:")
            red = self.reduce
            floc = (
                f"floc_cr{cr}_r{red}_f{n}_hmi.dat"
                if inst == "hmi"
                else f"floc_cr{cr}_rf{self.adapt_sel}_f{n}_adapt.dat"
            )
            self.touch(f"batches/{self.batch}/data/cr{cr}/floc/{floc}", size=n_actual * 96)
            self.say(f"\n\t\t\t {floc}\n\n\t\t\t```````````````````````````````\n")
            self.end("footpoints", started=t)
            self.event("metric", name="n_actual", value=n_actual, unit="fluxons")

            # 3 pfss
            t = time.time()
            self.stage("pfss")
            self.banner("(py) Running PFSS Code to Trace Footpoints into the Corona")
            self.say("\n\tTracing Open and Closed Fluxons...", end="")
            self.sleep(2.0)
            if self.scenario in ("fail_in_pfss", "failure_swallowed"):
                self.say(
                    '\nTraceback (most recent call last):\n  File "get_pfss.py", line 262, in <module>\nValueError: not enough footpoints in open flux'  # noqa: E501
                )
                self.say("get_pfss.py failed (status 256)")
                self.end("pfss", status="failed", detail="get_pfss.py failed (status 256)", started=t)
                return self.finish(
                    0 if self.scenario == "failure_swallowed" else 1, "PDL script failed with exit code 255"
                )
            self.say("\n\t\tNeed to trace? True: Done!")
            self.sleep(1.5)
            self.touch(f"pfss/pfss_cr{cr}_r{red}_{inst}.pkl", size=118_000_000)
            self.say(f"\t\tFootpoints:\t Open: {int(n_actual * 0.62)}, Closed: {int(n_actual * 0.38)}")
            self.say("\n\t\t\t```````````````````````````````\n")
            self.end("pfss", started=t)

            # 4 world
            t = time.time()
            self.stage("world")
            self.banner("(pdl) Converting PFSS Fieldlines into FLUX Fluxons")
            self.say("\n \n\tGenerating open fluxons from \n\t\tpfss_open.dat\n\t ")
            self.sleep(1.8)
            self.touch(f"{world_dir}/cr{cr}_f{n}_{inst}.flux", size=n_actual * 3200)
            self.say("\n\tSaving the Initial World...\n\t    Saved to world.flux\n")
            self.say("\n\t\t\t```````````````````````````````\n \n\n\n")
            self.end("world", started=t)

            # 5 relax
            t = time.time()
            self.stage("relax")
            self.banner("(pdl) Relax the Initial World State from PFSS")
            reused = self.scenario == "cached_relax"
            if reused:
                self.say(f"\tFound a relaxed file: cr{cr}_f{n}_{inst}_relaxed_s840.flux\n")
                self.say("\n\tSkipped relaxation! Already have relaxed file.\n")
                self.sleep(1.5)
                self.say("\n\tWorlds Successfully loaded!\n")
                self.end("relax", detail="reused relaxed world", started=t)
            else:
                self.say("\tNo relaxed file found, so we will relax the world (fairly slow).\n")
                steps = self.relax(n_actual)
                self.touch(f"{world_dir}/cr{cr}_f{n}_{inst}_relaxed_s{steps}.flux", size=n_actual * 3400)
                self.say(f"\n\n\tSaving the World to cr{cr}_f{n}_{inst}_relaxed_s{steps}.flux")
                if self.dropout:
                    self.dropout = False
                self.end("relax", started=t)

            # 6 world plots
            t = time.time()
            self.stage("world_plots")
            self.sleep(2.4)
            self.image(f"batches/{self.batch}/data/cr{cr}/world/cr{cr}_f{n}_{inst}_relaxed.png", "world")
            self.end("world_plots", started=t)

            # 7 fieldmap
            t = time.time()
            self.stage("fieldmap")
            self.banner("(py) Plotting Magnetic Field Maps")
            self.say("\tMaking Magnetogram with Footpoints...\n\tPlotting...", end="")
            self.sleep(1.8)
            self.image(f"batches/{self.batch}/data/cr{cr}/fieldmap/cr{cr}_f{n}_{inst}_fieldmap.png", "fieldmap")
            self.say("\n\t\t    Success!\n")
            self.end("fieldmap", started=t)

        # 8 wind
        t = time.time()
        self.stage("wind")
        self.banner(f"(pdl) Calculating Coronal Plasma Parameters for CR{cr}")
        self.say("\n\tUpdating neighbors...", end="")
        self.sleep(1.2)
        self.say("Done!\n\t\tNumber of fluxons: " + n)
        self.say("\n\tRadial Magnetic Field (B) Calculation...Done!")
        self.sleep(1.5)
        self.say("\n\tRadial Expansion Factor (Fr) Calculation...Done! Saved to \n\tfr.dat")
        self.say("\n\n\tRadial Wind Speed Calculation...")
        self.sleep(3.0)
        self.touch(f"batches/{self.batch}/data/cr{cr}/wind/cr{cr}_f{n}_{a.method}_wind.dat", size=int(n) * 2400)
        self.say("\t\t\t```````````````````````````````\n\n\n\n")
        self.end("wind", started=t)

        # 9 wind plots
        t = time.time()
        self.stage("wind_plots")
        self.banner(f"(py) Plotting Solar Wind Maps for CR{cr}")
        self.sleep(3.2)
        self.image(f"batches/{self.batch}/data/cr{cr}/wind/cr{cr}_f{n}_{a.method}_wind.png", "wind")
        self.say("\nDone with wind plotting!\n```````````````````````````````\n")
        self.end("wind_plots", started=t)

        # 10 fluxlight
        if self.light:
            t = time.time()
            self.stage("fluxlight")
            self.sleep(3.5)
            path = self.image(
                f"batches/{self.batch}/data/cr{cr}/light/cr{cr}_fluxlight_light_{a.method}_1.png", "light"
            )
            self.event("artifact", path=path, kind="image")
            self.end("fluxlight", started=t)
        else:
            self.event("stage.end", stage="fluxlight", status="skipped", elapsed_s=0, detail="light=0")

        elapsed = time.time() - self.t0
        self.say("-" * 98)
        self.say(f"\tProgram Complete for CR{cr} in {elapsed * self.speed:.1f} Seconds, press RETURN to exit...")
        self.say("-" * 98)
        return self.finish(0)

    def finish(self, code, message=None):
        if message:
            self.say(message)
        status = "ok" if code == 0 else "failed"
        self.event("run.end", status=status, exit_code=code, elapsed_s=round(time.time() - self.t0, 2))
        if code == 0:
            self.say("All 1 jobs succeeded.")
        else:
            self.say(
                f"Error: Rotation {self.a.cr}, Fluxon {self.a.nflux}, Adapt {self.a.adapt}, Method {self.a.method}: {message}"  # noqa: E501
            )
            self.say("FAILED 1 of 1 jobs:")
        return code

    # relaxation, modeled on simple_relaxer and relax_pfss_world
    def relax(self, n_actual):
        slow = self.scenario == "slow_relax"
        tau = 420.0 if slow else 175.0
        print_n = max(self.do_steps // 5, 1)
        threshold = self.threshold
        verts0 = max(n_actual * 41, 100)
        step, cycle, stiff, cum = 0, 0, 100.0, 0.0
        bump = 0.0
        do_steps = self.do_steps
        started = time.time()
        interval = 0.55 if not slow else 0.5
        while stiff > threshold and cycle < self.max_cycles:
            self.say(f"\n\tRelaxing PFSS model for {do_steps} steps to {threshold:.3g} stiffness...\n")
            if self.scenario == "events_dropout" and cycle == 2:
                self.dropout = True
            end_step = step + do_steps
            if step == 0:
                self.say("\tComputing initial fast neighbor search...")
            while step < end_step:
                if step % print_n == 0:
                    stiff = (
                        100.0 * math.exp(-step / tau) * (1 + bump)
                        + 1.1
                        + self.rng.uniform(-0.04, 0.04) * (1 + stiff / 40)
                    )
                    stiff = max(stiff, 0.2)
                    verts = int(verts0 * (1 + 0.045 * cycle))
                    fs_av = 2.4e-2 * (1 + 0.1 * math.sin(step / 70))
                    f_av = fs_av * stiff / 100
                    fmax = f_av * (7.5 + self.rng.uniform(-1.2, 1.2))
                    n_av = 7.1 + 0.4 * math.exp(-step / 300)
                    self.say(
                        f"\nstep {step}: time is now {step * 0.1:10.4g} \n({verts} vertices; stiffness={stiff:7.3g}%, "
                        f"fmax={fmax:7.3g}, f_av={f_av:7.3g}, fs_av={fs_av:7.3g}, n_av={n_av:7.3g} "
                        f"max_angle: {80 + self.rng.uniform(-6, 6):6.2f}, mean_angle: {31 + self.rng.uniform(-2, 2):6.2f})"  # noqa: E501
                    )
                    self.event(
                        "relax.step",
                        step=step,
                        stiffness=stiff,
                        fmax=fmax,
                        f_av=f_av,
                        fs_av=fs_av,
                        n_av=n_av,
                        vertices=verts,
                    )
                    self.sleep(interval)
                step += print_n if print_n > 1 else 1
            threshold *= 1.1
            cum = time.time() - started
            self.say(
                f"\tCycle {cycle}, Cumulative relaxation time: {cum * self.speed:.2f} seconds, {cycle * do_steps} steps"
            )
            nc_curve = nc_proxi = 0
            if cycle <= self.stop_fix:
                nc_curve = int(verts0 * self.rng.uniform(0.04, 0.09))
                nc_proxi = int(verts0 * self.rng.uniform(0.0, 0.02))
                self.say("\nFixing vertex separation....")
                self.say(
                    f"\nCycle {cycle} :: Curvature changed {nc_curve} vertices, proximity changed {nc_proxi} vertices"
                )
                bump = 0.35
            else:
                self.say(f"\nCycle {cycle} :: Skipping fix of vertex separation.")
                do_steps = self.do_steps * 2
                bump = 0.0
            self.event(
                "relax.cycle",
                cycle=cycle,
                threshold=threshold,
                stiffness=stiff,
                cum_time_s=round(cum * self.speed, 2),
                steps=cycle * do_steps,
                nc_curve=nc_curve,
                nc_proxi=nc_proxi,
                broken=0,
            )
            cycle += 1
        self.say(f"\tRelaxation Complete in {step} Steps ")
        return step


def main(argv=None):
    p = argparse.ArgumentParser(prog="fake_flux")
    p.add_argument("--profile", default="")
    p.add_argument("--cr", required=True)
    p.add_argument("--nflux", required=True)
    p.add_argument("--adapt", default="0")
    p.add_argument("--method", default="parker")
    p.add_argument("--scenario", default="ok")
    p.add_argument("--speed", default="1")
    args = p.parse_args(argv)
    fake = Fake(args)

    def on_term(signum, frame):
        # like the real pipeline: an open stage is closed as failed, no run.end (the runner is gone)
        if fake.open_stage:
            fake.event("stage.end", stage=fake.open_stage, status="failed", detail="terminated")
        fake.say("terminated")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    code = fake.run()
    if fake.log:
        fake.log.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
