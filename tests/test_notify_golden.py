"""The push line, the transitions and the event API shape must match their golden files (see the *_golden.py scripts)."""

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import notify_line_golden as line_g  # noqa: E402
import notify_transitions_golden as trans_g  # noqa: E402

CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def moved(golden_path, now):
    golden = json.load(open(golden_path))
    diff = {
        k: {"golden": golden.get(k), "now": now.get(k)}
        for k in sorted(set(golden) | set(now))
        if golden.get(k) != now.get(k)
    }
    return diff


def test_push_lines_match_the_golden_master():
    diff = moved(line_g.GOLDEN, line_g.table())
    assert not diff, "format_line changed for: " + json.dumps(diff, indent=1)


def test_push_lines_carry_no_markup_control_characters_or_paths():
    for name, line in line_g.table().items():
        assert line.isascii() and "\n" not in line, name
        assert not CONTROL.search(line) and "<" not in line and ">" not in line and "/Users/" not in line, name
        assert len(line) < 400, name


def test_transitions_match_the_golden_master():
    diff = moved(trans_g.GOLDEN, trans_g.table())
    assert not diff, "task_transition/job_transition changed for: " + json.dumps(diff, indent=1)


def test_event_api_shape_matches_the_golden_master():
    pytest.importorskip("fastapi")
    import events_api_golden as api_g

    golden = dict(ln.split(" ", 1) for ln in open(api_g.GOLDEN).read().splitlines() if ln)
    now = api_g.shapes()
    gone = {k: v for k, v in golden.items() if now.get(k) != v}
    added = sorted(set(now) - set(golden))
    assert not gone, f"event API paths removed, renamed or retyped: {gone}"
    assert not added, f"new event API paths; run tests/events_api_golden.py --update if intended: {added}"
