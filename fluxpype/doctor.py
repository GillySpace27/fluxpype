"""flux-doctor: check that everything a FLUX run needs is in place, and say how to fix what isn't.

Exits 0 when every check passes, 1 otherwise, so install scripts and grid jobs can gate on it.
"""

import os
import shutil
import subprocess
import sys

PY_MIN, PY_MAX = (3, 10), (3, 12)  # numpy<2 (required) has no wheels past 3.12


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def checks():
    """Yield (name, ok, detail, fix) for each prerequisite; ok=None is a warning, not a failure."""
    v = sys.version_info[:2]
    yield ("python version", PY_MIN <= v <= PY_MAX, f"{sys.executable} ({v[0]}.{v[1]})",
           "use Python 3.10-3.12: rerun install.sh, or set PYTHON=python3.12")

    perl = shutil.which("perl")
    yield ("perl on PATH", perl is not None, perl or "not found",
           "install Perl (macOS: brew install perl)")

    code, out = _run(["perl", "-MPDL", "-e", "print $PDL::VERSION"]) if perl else (1, "")
    yield ("PDL", code == 0, out.splitlines()[-1] if out else "", "cpanm PDL (see fluxon-mhd/doc)")

    code, out = _run(["perl", "-MFlux", "-e", "print $INC{'Flux.pm'}"]) if perl else (1, "")
    yield ("Flux (FLUX core)", code == 0, (out.splitlines()[-1] if code == 0 else out.splitlines()[0]) if out else "",
           "build fluxon-mhd: cd fluxon-mhd && make everything (see fluxon-mhd/doc); "
           "if it says 'Permission denied' or names another user's path, remove stale PERL5LIB "
           "entries from your shell rc file")

    gp = shutil.which("gnuplot")
    yield ("gnuplot", gp is not None, gp or "not found", "brew install gnuplot (or conda install gnuplot)")

    try:
        from fluxpype.pipe_helper import configurations
        c = configurations(debug=False)
        name = c.get("config_name", "?")
        missing = [k for k in ("rotations", "fluxon_count", "adapts", "flow_method")
                   if not str(c.get(k, "")).strip("[] ")]
        yield (f"config profile [{name}]", not missing,
               "complete" if not missing else "missing " + ", ".join(missing),
               "set the missing keys in fluxpype/config.ini, or pick a profile with --config")
        email = os.environ.get("JSOC_EMAIL") or c.get("jsoc_email", "")
        ok = "@" in email and not email.strip().endswith("example.com")
        yield ("JSOC email", ok or None, email or "unset",
               "needed only to download magnetograms: set JSOC_EMAIL or jsoc_email in config.ini "
               "(register at http://jsoc.stanford.edu/ajax/register_email.html)")
    except Exception as e:  # report, don't crash: this is a diagnostic
        yield ("config.ini", False, str(e), "fix the error above in fluxpype/config.ini")


def main():
    failed = 0
    for name, ok, detail, fix in checks():
        print(f"[{'ok' if ok else 'warn' if ok is None else 'FAIL'}] {name}: {detail}")
        if not ok:
            failed += ok is not None
            print(f"       fix: {fix}")
    print("\nAll checks passed." if not failed else f"\n{failed} check(s) failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
