"""The process behind a scene relaxation task: ``python -m fluxpype.plumber.scene_task SPEC.json [--fake]``.

The scheduler starts it like any task (backends/local.py). It reads the task's spec (scene_jobs.py), relaxes one world
or a fair pair with ``fluxpype.scenes.relax`` and reports the way a pipeline run does: Plumber's events
(``$FLUXPYPE_EVENTS``: a relaxation stage, a stiffness point per cycle of the first world, metrics at the end), a
log (``$FLUXPYPE_LOG_FILE`` and stdout), ``series.jsonl`` while it runs and ``result.json`` when it ends, all in the
task's run directory. Exit 0 when every world finished, 1 when one did not, 124 on the task timeout.

``--fake`` replaces FLUX with a deterministic stand-in curve (the demo and the tests; the energies mean nothing).
"""

import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..events import emit
from ..scenes import relax as scene_relax

BYTES = "utf-8"


class Fake:
    """Stands in for ``subprocess.Popen`` running the Perl driver: prints the same CYCLE lines, then DONE.

    The curve is an exponential approach to a limit that depends on the world's text, so two connectivities differ
    and a higher lid lowers the limit (a stand-in for the lid dependence, not a measurement)."""

    def __init__(self, world_text: str, cycles: int, delay: float):
        h = int(hashlib.sha256(world_text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        lid = next((l for l in world_text.splitlines() if l.startswith("GLOBAL B2")), "").split()
        try:
            z = float(lid[4])  # GLOBAL B2 PLANE px py pz nx ny nz
        except (IndexError, ValueError):
            z = 0.0
        self.e0, self.einf = 12.0 + h, 8.0 + 0.5 * h + (3.0 / (1.0 + z) if z else 1.0)
        self.cycles, self.delay = cycles, delay
        self.pid = os.getpid()
        self.stdin = type("In", (), {"write": lambda s, x: None, "close": lambda s: None})()
        self.stdout = self._lines()

    def _lines(self):
        for c in range(-1, self.cycles):
            time.sleep(self.delay)
            e = self.einf + (self.e0 - self.einf) * math.exp(-max(c, 0) / 4.0) if c >= 0 else self.e0
            stiff = "null" if c < 0 else "%.6g" % (100 * math.exp(-c / 4.0) + 0.5)
            yield (
                f'CYCLE {{"cycle":{c},"steps":{(c + 1) * 100},"stiffness":{stiff},"e_simple2":{e:.8g},'
                f'"top_z":{1.5 + 0.01 * c:.6g},"wall_s":{0.1 * (c + 1):.3f}}}\n'
            )
        yield "DONE\n"

    def wait(self) -> int:
        return 0


class Runner:
    def __init__(self, spec: Dict[str, Any], fake: bool, speed: float):
        self.spec, self.fake, self.speed = spec, fake, max(speed, 0.001)
        self.run_dir = Path(spec["run_dir"])
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.log_path = os.environ.get("FLUXPYPE_LOG_FILE")
        self.log = open(self.log_path, "a", encoding=BYTES) if self.log_path else None
        self.series = open(self.run_dir / "series.jsonl", "a", encoding=BYTES)

    def say(self, text: str) -> None:
        with self.lock:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
            if self.log:
                self.log.write(text + "\n")
                self.log.flush()

    def on_cycle(self, w: int, rec: Dict[str, Any]) -> None:
        tag = "AB"[w] if len(self.spec["worlds"]) == 2 else ""
        self.say(f"CYCLE{' ' + tag if tag else ''} {json.dumps(rec)}")
        with self.lock:
            self.series.write(json.dumps({"w": w, **rec}) + "\n")
            self.series.flush()
        if w == 0 and rec["cycle"] >= 0:  # the job page's relaxation chart follows the first world
            emit("relax.step", step=rec["steps"], stiffness=rec["stiffness"], vertices=None)
            emit(
                "relax.cycle",
                cycle=rec["cycle"],
                threshold=None,
                stiffness=rec["stiffness"],
                cum_time_s=rec.get("wall_s"),
                steps=rec["steps"],
                nc_curve=0,
                nc_proxi=0,
                broken=0,
            )

    def relax_one(self, w: int, out: Dict[int, Dict[str, Any]]) -> None:
        world = self.spec["worlds"][w]
        kw: Dict[str, Any] = {}
        if self.fake:
            kw["popen"] = lambda *a, **k: Fake(Path(world).read_text(), self.spec["cycles"], 0.03 / self.speed)
        try:
            out[w] = scene_relax.relax(
                world,
                out=str(self.run_dir / f"relaxed-{'AB'[w]}.flux"),
                cycles=self.spec["cycles"],
                steps=self.spec["steps"],
                perl=self.spec.get("perl"),
                log=str(self.run_dir / f"relax-{'AB'[w]}.log"),
                on_cycle=lambda rec, w=w: self.on_cycle(w, rec),
                **kw,
            )
        except Exception:  # noqa: BLE001 - say why in the log; the task then fails because this world has no result
            self.say(traceback.format_exc())

    def run(self) -> int:
        s = self.spec
        emit("run.start", mode=s["mode"], scenes=s["ids"], lid=s.get("lid"))
        emit("stage.start", stage="relax")
        self.say(f"Relaxing {' and '.join(s['ids'])} ({s['mode']}): {s['cycles']} cycles of {s['steps']} steps")
        results: Dict[int, Dict[str, Any]] = {}
        threads = [threading.Thread(target=self.relax_one, args=(w, results)) for w in range(len(s["worlds"]))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ok = len(results) == len(threads) and all(r["ok"] for r in results.values())
        res: Dict[str, Any] = {
            "mode": s["mode"],
            "lid": s.get("lid"),
            "cycles": s["cycles"],
            "steps": s["steps"],
            "ok": ok,
            "fake": self.fake,
            "worlds": [
                {
                    "id": s["ids"][w],
                    "ok": r["ok"],
                    "series": r["series"],
                    "settled_at": r["settled_at"],
                    "relaxed": r["relaxed"],
                }
                for w, r in sorted(results.items())
            ],
        }
        if len(results) == 2:
            d = scene_relax.difference(results[0]["series"][1:], results[1]["series"][1:])
            sa = scene_relax.settled_at([r["d_e_simple2"] for r in d])
            res.update(difference=d, d_settled_at=sa, d_at_settle=d[sa]["d_e_simple2"] if sa is not None else None)
        tmp = self.run_dir / "result.json.tmp"
        tmp.write_text(json.dumps(res), encoding=BYTES)
        os.replace(tmp, self.run_dir / "result.json")
        for w, r in sorted(results.items()):
            last = r["series"][-1] if r["series"] else {}
            emit("metric", name=f"e_simple2 {s['ids'][w]}", value=last.get("e_simple2"), unit="")
            emit("metric", name=f"energy settled at cycle {s['ids'][w]}", value=r["settled_at"], unit="cycle")
        if "difference" in res:
            emit(
                "metric",
                name="D = E(B) - E(A)",
                value=res["difference"][-1]["d_e_simple2"] if res["difference"] else None,
                unit="",
            )
            emit("metric", name="D settled at cycle", value=res["d_settled_at"], unit="cycle")
        emit("stage.end", stage="relax", status="ok" if ok else "failed", elapsed_s=round(time.time() - self.t0, 2))
        self.say("All worlds relaxed." if ok else "FAILED: a world did not finish relaxing.")
        emit(
            "run.end",
            status="ok" if ok else "failed",
            exit_code=0 if ok else 1,
            elapsed_s=round(time.time() - self.t0, 2),
        )
        return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    spec = json.loads(Path(argv[0]).read_text(encoding=BYTES))
    fake = "--fake" in argv
    speed = float(argv[argv.index("--speed") + 1]) if "--speed" in argv else 1.0
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if not fake:  # FLUX's Perl: the pipeline runner puts Flux.pm on PERL5LIB the same way (config_runner.py)
        try:
            from ..pipe_helper import get_flux_perl_lib

            lib = get_flux_perl_lib(spec.get("flux_config") or {})
            if lib:
                os.environ["PERL5LIB"] = lib + (
                    os.pathsep + os.environ["PERL5LIB"] if os.environ.get("PERL5LIB") else ""
                )
        except Exception:  # noqa: BLE001 - without it FLUX may still be on Perl's own path
            pass
    runner = Runner(spec, fake, speed)

    def on_term(signum, frame):
        emit("stage.end", stage="relax", status="failed", detail="terminated")
        runner.say("terminated")
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    limit = float(os.environ.get("FLUXPYPE_TASK_TIMEOUT") or 0)
    if limit:  # the task timeout, as the pipeline runner applies it: run.end says timeout, exit 124

        def expire():
            emit("stage.end", stage="relax", status="failed", detail="timeout")
            emit("run.end", status="timeout", exit_code=124, elapsed_s=round(time.time() - runner.t0, 2))
            runner.say(f"timeout after {limit:g} s")
            os._exit(124)

        timer = threading.Timer(limit, expire)
        timer.daemon = True
        timer.start()
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
