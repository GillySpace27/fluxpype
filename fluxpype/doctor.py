"""flux-doctor: check that everything a FLUX run needs is in place, and say how to fix what isn't.

Exits 0 when every check passes, 1 otherwise, so install scripts and grid jobs can gate on it.
"""

import os
import re
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


O2_MAKE = 'make everything CFLAGS="-I../include -g -fPIC -O2"'
O2_ADVICE = (
    f"libflux is built without optimization (-O0), which makes every relax about 1.5x slower with identical output. "
    f"Rebuild: cd fluxon-mhd && {O2_MAKE}. Link-order trap: Perl puts -L/usr/local/lib first, so an old "
    f"/usr/local/lib/libflux.a is linked silently instead of the new one; remove it first, or install to /usr/local"
)
_GCC_PRODUCER = re.compile(rb"GNU C[0-9A-Za-z+]* [\x20-\x7e]*")


def libflux_opt_level(data):
    """Optimization of a libflux.a image from its DWARF producer strings: 'O0', 'optimized' or None (cannot tell).

    GCC records its command line in DW_AT_producer, so a producer with no -O1/-O2/-O3/-Os/-Og/-Ofast flag means -O0.
    A build without -g, or one by clang (producer string carries no flags), cannot be read: return None.
    """
    producers = _GCC_PRODUCER.findall(data)
    if not producers:
        return None
    levels = [re.search(rb"(?:^|\s)-O(?:[1-3sg]|fast)\b", p) for p in producers]
    return "optimized" if any(levels) else "O0"


def _libflux_candidates():
    """libflux.a paths worth inspecting: $FL_PREFIX/lib first, then /usr/local/lib (Perl links it first)."""
    dirs = [os.path.join(os.path.expanduser(os.environ[k]), "lib") for k in ("FL_PREFIX",) if os.environ.get(k)]
    dirs.append("/usr/local/lib")
    seen, out = set(), []
    for d in dirs:
        f = os.path.join(d, "libflux.a")
        if f not in seen and os.path.isfile(f):
            seen.add(f)
            out.append(f)
    return out


def libflux_check(paths=None):
    """(ok, detail, fix) for the libflux optimization level; ok=None is a warning. Never a hard failure."""
    paths = _libflux_candidates() if paths is None else paths
    if not paths:
        return None, "libflux.a not found in $FL_PREFIX/lib or /usr/local/lib", f"hint: build with {O2_MAKE}"
    results = []
    for f in paths:
        try:
            with open(f, "rb") as fh:
                results.append((f, libflux_opt_level(fh.read())))
        except OSError:
            results.append((f, None))
    detail = "; ".join(f"{f}: {lvl or 'unknown'}" for f, lvl in results)
    if any(lvl == "O0" for _, lvl in results):
        return None, detail, O2_ADVICE
    if all(lvl == "optimized" for _, lvl in results):
        return True, detail, ""
    return None, detail, f"hint: optimization level cannot be read; for speed build with {O2_MAKE}"


def _flux_perl():
    """config.ini flux_perl (the Perl jobs run), or None when unset or the config cannot be read."""
    try:
        from fluxpype.pipe_helper import configurations

        return os.path.expanduser(configurations(debug=False).get("flux_perl") or "") or None
    except Exception:  # noqa: BLE001  the config check below reports a broken config.ini
        return None


def checks():
    """Yield (name, ok, detail, fix) for each prerequisite; ok=None is a warning, not a failure."""
    v = sys.version_info[:2]
    yield (
        "python version",
        PY_MIN <= v <= PY_MAX,
        f"{sys.executable} ({v[0]}.{v[1]})",
        "use Python 3.10-3.12: rerun install.sh, or set PYTHON=python3.12",
    )

    perl = _flux_perl() or shutil.which("perl")
    yield (
        "perl for jobs",
        perl is not None,
        perl or "not found",
        "install Perl (macOS: brew install perl), or set flux_perl in config.ini",
    )

    code, out = _run([perl, "-MPDL", "-e", "print $PDL::VERSION"]) if perl else (1, "")
    yield ("PDL", code == 0, out.splitlines()[-1] if out else "", "cpanm PDL (see fluxon-mhd/doc)")

    code, out = _run([perl, "-MFlux", "-e", "print $INC{'Flux.pm'}"]) if perl else (1, "")
    yield (
        "Flux (FLUX core)",
        code == 0,
        (out.splitlines()[-1] if code == 0 else out.splitlines()[0]) if out else "",
        "build fluxon-mhd: cd fluxon-mhd && make everything (see fluxon-mhd/doc); "
        "if it says 'Permission denied' or names another user's path, remove stale PERL5LIB "
        "entries from your shell rc file",
    )

    ok, detail, fix = libflux_check()
    yield ("libflux optimization", ok, detail, fix)

    # Inline::Pdlpp (used by the footpoint/wind code) loads silently without Inline::C and then dies
    # with 'Base class package "Inline::C" is empty', so test the pair the way the pipeline loads them.
    code, out = (
        _run([perl, "-e", "require Inline::C; require Inline::Pdlpp; print $Inline::C::VERSION"]) if perl else (1, "")
    )
    yield (
        "Inline::C + Inline::Pdlpp",
        code == 0,
        f"Inline::C {out.splitlines()[-1]}" if code == 0 else (out.split(" (@INC")[0].splitlines()[0] if out else ""),
        "cpanm Inline::C Inline::Pdlpp, using the cpanm of the same Perl that `perl` resolves to "
        "(e.g. `perl $(which cpanm) Inline::C`); a C compiler (Xcode CLT) is also required",
    )

    gp = shutil.which("gnuplot")
    yield ("gnuplot", gp is not None, gp or "not found", "brew install gnuplot (or conda install gnuplot)")

    try:
        from fluxpype.pipe_helper import configurations

        c = configurations(debug=False)
        name = c.get("config_name", "?")
        missing = [
            k for k in ("rotations", "fluxon_count", "adapts", "flow_method") if not str(c.get(k, "")).strip("[] ")
        ]
        yield (
            f"config profile [{name}]",
            not missing,
            "complete" if not missing else "missing " + ", ".join(missing),
            "set the missing keys in fluxpype/config.ini, or pick a profile with --config",
        )
        email = os.environ.get("JSOC_EMAIL") or c.get("jsoc_email", "")
        ok = "@" in email and not email.strip().endswith("example.com")
        yield (
            "JSOC email",
            ok or None,
            email or "unset",
            "needed only to download magnetograms: set JSOC_EMAIL or jsoc_email in config.ini "
            "(register at http://jsoc.stanford.edu/ajax/register_email.html)",
        )
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
