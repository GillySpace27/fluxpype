"""Readers for the raw CHIANTI database files GHOSTS used (``environment.cLoad*`` in coronasim.py)."""

import os

import numpy as np

ELEMENTS = ("h he li be b c n o f ne na mg al si p s cl ar k ca sc ti v cr mn fe co ni cu zn").split()


def chianti_dir(path=None):
    """The CHIANTI database root: ``path`` if given, else ``$XUVTOP``."""
    p = path or os.environ.get("XUVTOP")
    if not p or not os.path.isdir(p):
        raise FileNotFoundError("no CHIANTI database: pass chianti_dir or set XUVTOP")
    return os.path.abspath(os.path.expanduser(p))


def version(root):
    try:
        with open(os.path.join(root, "VERSION")) as f:
            return f.read().strip()
    except OSError:
        return "unknown"


def ion_name(z, ion):
    """CHIANTI file stem for element z and spectroscopic ion number (1 = neutral), e.g. (26, 12) -> 'fe_12'."""
    return f"{ELEMENTS[z - 1]}_{ion}"


def ion_file(root, z, ion, ext):
    stem = ion_name(z, ion)
    return os.path.join(root, ELEMENTS[z - 1], stem, f"{stem}.{ext}")


def _data_lines(path):
    """Lines before CHIANTI's ' -1' terminator."""
    out = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s == "-1" or s.startswith("-1 ") and len(s.split()) == 1:
                break
            if s:
                out.append(ln.rstrip("\n"))
    return out


def read_wgfa(root, z, ion):
    """Transitions: list of (lower, upper, wavelength A, gf, A_ul s^-1). Negative wavelengths are theoretical."""
    out = []
    for ln in _data_lines(ion_file(root, z, ion, "wgfa")):
        p = ln.split()
        try:
            out.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]), float(p[4])))
        except (ValueError, IndexError):
            continue
    return out


def find_line(root, z, ion, wavelength, tol=0.01):
    """The wgfa transition closest to ``wavelength`` within ``tol`` A. Observed (positive) wavelengths win over
    theoretical ones; ties go to the larger A value."""
    cand = [t for t in read_wgfa(root, z, ion) if abs(abs(t[2]) - wavelength) <= tol]
    if not cand:
        raise ValueError(f"{ion_name(z, ion)}: no transition within {tol} A of {wavelength}")
    return min(cand, key=lambda t: (t[2] < 0, round(abs(abs(t[2]) - wavelength), 4), -t[4]))


def read_elvlc_j(root, z, ion):
    """Level index -> J. GHOSTS took ``data[-3]`` of each row, which is right for CHIANTI 8; CHIANTI 9+ rows append
    comma-separated Rydberg energies, so only the text before the first comma is split."""
    out = {}
    for ln in _data_lines(ion_file(root, z, ion, "elvlc")):
        p = ln.split(",")[0].split()
        try:
            out[int(p[0])] = float(p[-3])
        except (ValueError, IndexError):
            continue
    return out


def read_elvlc_energy(root, z, ion):
    """Level index -> energy above ground in Rydberg: observed where CHIANTI has it, else theoretical (ChiantiPy's
    choice)."""
    out = {}
    for ln in _data_lines(ion_file(root, z, ion, "elvlc")):
        p = ln.split(",")[0].split()
        try:
            obs, th = float(p[-2]), float(p[-1])
            out[int(p[0])] = (obs if obs >= 0 else th) / 109737.31568
        except (ValueError, IndexError):
            continue
    return out


def read_scups(root, z, ion, lower, upper):
    """Burgess-Tully scaled collision strengths for one transition: dict with dE (Ryd), gf, lim, type, C, x, y."""
    path = ion_file(root, z, ion, "scups")
    with open(path) as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        p = ln.split()
        if len(p) >= 8 and p[0] == str(lower) and p[1] == str(upper):
            x = [float(v) for v in lines[i + 1].split()]
            y = [float(v) for v in lines[i + 2].split()]
            return {
                "dE": float(p[2]),
                "gf": float(p[3]),
                "lim": float(p[4]),
                "type": int(p[6]),
                "C": float(p[7]),
                "x": np.array(x),
                "y": np.array(y),
            }
    raise ValueError(f"{ion_name(z, ion)}: transition {lower}-{upper} not in scups")


def read_rrparams(root, z, ion):
    """Radiative recombination fit for ion -> ion-1: (type, params) as GHOSTS parsed them."""
    path = ion_file(root, z, ion, "rrparams")
    if not os.path.isfile(path):
        return None
    lines = _data_lines(path)
    typ = int(lines[0].split()[0])
    raw = lines[1].split()[2:]
    params = [float(v) for v in raw]
    if typ in (1, 2):
        params = params[1:]  # GHOSTS: params.pop(0)
    return typ, params


def read_drparams(root, z, ion):
    """Dielectronic recombination fit for ion -> ion-1: (1, E[], c[]) (Badnell) or (2, [A, B, T0, T1]) (Shull)."""
    path = ion_file(root, z, ion, "drparams")
    if not os.path.isfile(path):
        return None
    lines = _data_lines(path)
    typ = int(lines[0].split()[0])
    if typ == 1:
        e = np.array([float(v) for v in lines[1].split()[2:]])
        c = np.array([float(v) for v in lines[2].split()[2:]])
        return 1, e, c
    if typ == 2:
        return 2, [float(v) for v in lines[1].split()[2:6]]
    raise ValueError(f"{ion_name(z, ion)}: drparams type {typ} not supported")


def read_ip(root):
    """Ionization potentials, cm^-1: dict (z, ion) -> value."""
    out = {}
    for ln in _data_lines(os.path.join(root, "ip", "chianti.ip")):
        p = ln.split()
        try:
            out[(int(p[0]), int(p[1]))] = float(p[2])
        except (ValueError, IndexError):
            continue
    return out


def read_ioneq(root, name="chianti"):
    """CHIANTI equilibrium ion fractions: (log10 T array, dict (z, ion) -> fraction array)."""
    path = os.path.join(root, "ioneq", f"{name}.ioneq")
    with open(path) as f:
        nt = int(f.readline().split()[0])
        logt = []
        while len(logt) < nt:
            logt += [float(v) for v in f.readline().split()]
        frac = {}
        for ln in f:
            if not ln.strip() or ln.split()[0] == "-1":
                break
            # Fixed width (2i3, nT e10.3): adjacent values can touch, so split by column, not whitespace.
            body = ln.rstrip("\n")[6:]
            vals = [float(body[k : k + 10]) for k in range(0, 10 * nt, 10)]
            frac[(int(ln[:3]), int(ln[3:6]))] = np.array(vals)
    return np.array(logt), frac


def read_abund(root, name="sun_coronal_2012_schmelz"):
    """Abundances relative to H, A_Z = n_Z / n_H, from a CHIANTI .abund file (log scale with H = 12)."""
    out = {}
    for ln in _data_lines(os.path.join(root, "abundance", f"{name}.abund")):
        p = ln.split()
        try:
            out[int(p[0])] = 10 ** (float(p[1]) - 12.0)
        except (ValueError, IndexError):
            continue
    return out
