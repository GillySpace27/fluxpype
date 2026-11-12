"""convert_value must return byte-identical results for the whole golden corpus (see convert_value_golden.py)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import convert_value_golden as g  # noqa: E402


def test_convert_value_matches_the_golden_master():
    golden = json.load(open(g.GOLDEN))
    now = g.table(g.current(), sorted(golden))
    moved = {v: (golden[v], now[v]) for v in golden if golden[v] != now[v]}
    assert not moved, "convert_value changed for: " + json.dumps(moved, indent=1)


def test_every_config_ini_value_is_in_the_golden_corpus():
    golden = json.load(open(g.GOLDEN))
    missing = sorted(set(g.corpus()) - set(golden))
    assert not missing, f"config.ini has values the golden file lacks; run convert_value_golden.py --update: {missing[:10]}"
