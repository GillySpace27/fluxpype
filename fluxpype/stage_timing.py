"""Per-stage wall time for one run, read from its events.jsonl (audit A5-01).

Usage:  python -m fluxpype.stage_timing path/to/events.jsonl [--json] [--task ID]

Reads only the existing ``stage.start`` / ``stage.end`` / ``run.end`` events (schema in
docs/gui/EVENTS.md); it adds no instrumentation and changes no pipeline behavior.
A stage's wall time is its ``elapsed_s`` when present, else ``end t - start t``.
Skipped stages are listed with status ``skipped``. Shares are of ``run.end`` elapsed_s
when the run finished, else of the summed stage time.
"""

import argparse
import json
import sys
from pathlib import Path


def summarize(lines, task=None):
    """Return {"stages": [{stage, status, wall_s, share}], "total_s", "total_source", "bad_lines"}."""
    stages, order, starts = {}, [], {}
    run_start = run_end = None
    run_elapsed = None
    bad = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            kind = e["type"]
        except (ValueError, KeyError, TypeError):
            bad += 1
            continue
        if task is not None and e.get("task") != task:
            continue
        t = e.get("t")
        if kind == "run.start":
            run_start = t
        elif kind == "run.end":
            run_end = t
            run_elapsed = e.get("elapsed_s")
        elif kind == "stage.start":
            starts[e.get("stage")] = t
        elif kind == "stage.end":
            name = e.get("stage")
            wall = e.get("elapsed_s")
            if wall is None and t is not None and starts.get(name) is not None:
                wall = round(t - starts[name], 2)
            if name not in stages:
                order.append(name)
            stages[name] = {"stage": name, "status": e.get("status", "ok"), "wall_s": wall}
    rows = [stages[n] for n in order]
    if run_elapsed is not None:
        total, source = float(run_elapsed), "run.end elapsed_s"
    elif run_start is not None and run_end is not None:
        total, source = round(run_end - run_start, 2), "run.end t - run.start t"
    else:
        total, source = round(sum(r["wall_s"] or 0 for r in rows), 2), "sum of stages (run did not end)"
    for r in rows:
        r["share"] = round(r["wall_s"] / total, 3) if r["wall_s"] is not None and total else None
    return {"stages": rows, "total_s": total, "total_source": source, "bad_lines": bad}


def render(summary):
    out = [f"{'stage':<14}{'status':<9}{'wall_s':>9}{'share':>8}"]
    for r in summary["stages"]:
        wall = "-" if r["wall_s"] is None else f"{r['wall_s']:.2f}"
        share = "-" if r["share"] is None else f"{100 * r['share']:.0f}%"
        out.append(f"{r['stage']:<14}{r['status']:<9}{wall:>9}{share:>8}")
    out.append(f"{'total':<14}{'':<9}{summary['total_s']:>9.2f}  ({summary['total_source']})")
    if summary["bad_lines"]:
        out.append(f"{summary['bad_lines']} unreadable line(s) skipped")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("events", type=Path, help="events.jsonl written with FLUXPYPE_EVENTS")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--task", default=None, help="only events whose task id matches")
    args = ap.parse_args(argv)
    try:
        lines = args.events.read_text().splitlines()
    except OSError as exc:
        print(f"cannot read {args.events}: {exc}", file=sys.stderr)
        return 2
    summary = summarize(lines, task=args.task)
    print(json.dumps(summary, indent=2) if args.json else render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
