"""Golden master for pipe_helper.convert_value, the eval() sink for every config value.

    python tests/convert_value_golden.py --update              rewrite tests/golden/convert_value.json from today's code
    python tests/convert_value_golden.py --candidate           list every corpus value where ast.literal_eval-or-string differs
    python tests/convert_value_golden.py --candidate --home H  same, over the values in a Plumber home's presets.ini and job snapshots

Each corpus string maps to "type<TAB>repr" of what convert_value returns, or "raises<TAB>ExcName". The corpus is every raw
value in config.ini plus EDGE below. The golden file is the contract: if convert_value changes, test_convert_value_golden
names every string whose result moved, so the change can be reviewed or reverted line by line.
"""
import argparse
import ast
import configparser
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "golden", "convert_value.golden")
EDGE = [  # shapes real configs and pipeline output use, then values the Plumber validator lets through today
    "42", "-1", "0", "1e-3", "1.5", "0x10", "True", "False", "None", "nan", "inf", " 5", "5 ", "", "'a'", '"a"', "abc", "a b",
    "[1, 2]", "[1,2,3]", "(1, 2)", "(0,10,2)", "[]", "{}", "[a]", "[wsa]", "[parker, wsa]", "1/3", "2*3", "10**3", "-2**2", "1e3/7",
    "~/x", "${base_dir}/x", "/abs/path", "C:\\x", "1,2", "3.0.1", "2014-12-23", "1_000", "07", "1+", "(", "'", "True and False",
    "os", "os.environ", "sys.modules", "exit", "~os", "1/0", "1 if os else 2", "[k for k in os.pardir]", "os.pardir", "__import__",
    "[x for x in range(3)]", "range(3)", "float('x')", "len('abc')",
]


def describe(fn, raw):
    try:
        v = fn(raw)
    except Exception as e:  # noqa: BLE001  anything the sink lets out is part of the contract
        return f"raises\t{type(e).__name__}"
    # Only plain data is recorded in full. Anything else (modules, os.environ, sys.path, builtins) depends on the machine
    # and may hold secrets, so just its type is kept: the point is that a non-literal came out of the sink.
    if plain(v):
        return f"{type(v).__name__}\t{v!r}"
    return f"{type(v).__name__}\t<not plain data>"


def plain(v):
    if v is None or isinstance(v, (bool, int, float, str, range)):
        return True
    if isinstance(v, (list, tuple, set, frozenset)):
        return all(plain(x) for x in v)
    if isinstance(v, dict):
        return all(plain(k) and plain(x) for k, x in v.items())
    return False


def candidate(value):
    """What ast.literal_eval-with-string-fallback would do (not used anywhere; for the diff report only)."""
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def current():
    sys.path.insert(0, ROOT)
    from fluxpype.pipe_helper import convert_value

    return convert_value


def ini_values(paths):
    out = set()
    for p in paths:
        cp = configparser.RawConfigParser(interpolation=None, default_section="DEFAULT")
        cp.optionxform = str
        try:
            cp.read(p)
        except configparser.Error:
            continue
        for sect in [cp.default_section] + cp.sections():
            out.update(v for _, v in cp.items(sect))
    return out


def corpus(home=None):
    vals = set(EDGE) | ini_values([os.path.join(ROOT, "fluxpype", "config.ini")])
    if home:
        vals |= ini_values(glob.glob(os.path.join(home, "runs", "*", "config.snapshot.ini")) + [os.path.join(home, "presets.ini")])
    return sorted(vals)


def table(fn, vals):
    return {v: describe(fn, v) for v in vals}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--candidate", action="store_true")
    ap.add_argument("--home")
    a = ap.parse_args()
    cur = current()
    if a.update:
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(table(cur, corpus()), f, indent=1, sort_keys=True, ensure_ascii=True)
            f.write("\n")
        print("wrote", GOLDEN)
    if a.candidate:
        vals = corpus(a.home)
        now, new = table(cur, vals), table(candidate, vals)
        moved = [v for v in vals if now[v] != new[v]]
        print(f"{len(vals)} values, {len(moved)} differ under literal_eval-or-string")
        for v in moved:
            print(f"  {v!r:40} now: {now[v]!r}\n  {'':40} new: {new[v]!r}")
