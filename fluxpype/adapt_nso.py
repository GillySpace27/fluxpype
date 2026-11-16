"""
Find and download ADAPT maps straight from the NSO archive listing.

sunpy's ADAPTClient (7.1) splits the archive at 2024-09-28 into an "old" file
name pattern (a letter for the version month) and a "new" one (a digit). NSO has
since renamed its older files to the new pattern, for example
``2015/adapt40311_044012_201502010000_i00005600n1.fts.gz``, so the client finds
nothing before late 2024. Reading the year's directory listing ourselves works
for every year.

File name fields (https://gong.nso.edu/adapt/maps/adapt_filename_notes.txt):
``adapt4 <lon type> <input source> <assimilation> <resolution> _ <version> _
<YYYYMMDDHHMM> _ <evolution mode> <time since last obs> <helio> <mag> .fts.gz``.
Lon type 0 is Carrington fixed, 1 central meridian, 2 east limb.
"""

import datetime as _dt
import os
import re
import urllib.request

BASE_URL = "https://gong.nso.edu/adapt/maps/gong"

_NAME = re.compile(
    r"(adapt4(?P<lon>\d)(?P<src>\d)(?P<da>\d)(?P<res>\d)_\w{6}_"
    r"(?P<stamp>\d{12})_\w\d{8}\w\d\.fts\.gz)"
)


def parse_listing(html):
    """Return [(datetime, lon_type, file_name)] for every ADAPT map in a listing page."""
    out = {}
    for m in _NAME.finditer(html):
        when = _dt.datetime.strptime(m.group("stamp"), "%Y%m%d%H%M")
        out[m.group(1)] = (when, int(m.group("lon")), m.group(1))
    return sorted(out.values())


def pick(entries, start, end, lon_type=0):
    """
    The map with the given lon type whose time is in [start, end], nearest start.

    Returns (datetime, file_name) or None.
    """
    hits = [
        (abs((when - start).total_seconds()), when, name)
        for when, lon, name in entries
        if lon == lon_type and start <= when <= end
    ]
    if not hits:
        return None
    _, when, name = min(hits)
    return when, name


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "fluxpype"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def find(start, end, lon_type=0, timeout=60):
    """
    URL of the ADAPT map in [start, end] (naive UTC datetimes) nearest start, or None.

    Reads one directory listing per calendar year the window touches.
    """
    entries = []
    for year in sorted({start.year, end.year}):
        html = _get(f"{BASE_URL}/{year}/", timeout).decode("utf-8", "replace")
        entries += [(w, l, f"{year}/{n}") for w, l, n in parse_listing(html)]
    hit = pick(entries, start, end, lon_type)
    return None if hit is None else f"{BASE_URL}/{hit[1]}"


def download(url, path, timeout=300):
    """Download url to path (written via a .part file so a failed fetch leaves nothing)."""
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(_get(url, timeout))
    os.replace(tmp, path)
    return path
