"""File names for the wind and FLUXlight products (audit A7-01).

A task is one (rotation, fluxon count, instrument), and the relaxed world already carries all three in its name
(``cr{CR}_f{N}_{inst}_relaxed_s{step}.flux``). The wind tables and the FLUXlight images used to carry only some of them,
so an HMI and an ADAPT task, or two fluxon counts, of one batch overwrote each other's files. They now carry the same
``cr{CR}_f{N}_{inst}`` stem. Batches written before this change keep the old names (``cr{CR}_f{N}_radial_*.dat``,
``cr{CR}_fluxlight_*.png``), so every reader tries the new name first and falls back to the old one.

Two runs of one preset (same rotation, count and instrument) still write the same names; telling those apart needs a run
id, which this module does not add.

No imports beyond the standard library, so Plumber can use it without the science stack.
"""

from __future__ import annotations

import os
import re
from typing import Optional

INSTRUMENTS = ("hmi", "adapt")
_UNKNOWN = (None, "", "unknown")
_TASK_IN_NAME = re.compile(r"_f(\d+)(?:_(hmi|adapt)(?=_|\.|$))?(?:_|\.|$)")


def inst_name(adapt) -> str:
    """'adapt' for a truthy adapt flag (0, '0', False and '' mean HMI), else 'hmi'."""
    return "adapt" if str(adapt).strip().lower() not in ("", "0", "false", "none") else "hmi"


def inst_in_name(path: str) -> Optional[str]:
    """The instrument a ``..._f<N>_<inst>_...`` or ``..._f<N>_<inst>.<ext>`` file name carries, else None (old name)."""
    m = _TASK_IN_NAME.search(os.path.basename(str(path)))
    return m.group(2) if m and m.group(2) else None


def stem(cr, nflux, inst: Optional[str]) -> str:
    """``cr{CR}_f{N}_{inst}``; the instrument is left out when it is not known (an old-style name)."""
    return f"cr{cr}_f{nflux}" + (f"_{inst}" if inst not in _UNKNOWN else "")


def wind_dat(cr, nflux, inst: Optional[str], kind: str, stiff: bool = False) -> str:
    """Name of a wind table: ``kind`` is bmag, bmag_all, fr or wind_<method>."""
    return f"{stem(cr, nflux, inst)}{'_stiff' if stiff else ''}_radial_{kind}.dat"


def light_png(cr, nflux, inst: Optional[str], tail: str) -> str:
    """Name of a FLUXlight image; ``tail`` is what follows ``fluxlight_`` (method, nz, ...)."""
    if nflux in _UNKNOWN or inst in _UNKNOWN:
        return f"cr{cr}_fluxlight_{tail}.png"  # a world with no count or instrument in its name: the old name
    return f"{stem(cr, nflux, inst)}_fluxlight_{tail}.png"


def light_fits(cr, nflux, inst: Optional[str], channel: str) -> str:
    """Name of a FLUXlight FITS image (audit A7-03): ``cr{CR}_f{N}_{inst}_{channel}.fits``, one file per channel."""
    return f"{stem(cr, nflux, inst)}_{channel}.fits"


def legacy_template(tmpl: str) -> Optional[str]:
    """The pre-A7-01 spelling of a wind/FLUXlight template or glob with ``{nflux}``/``{inst}`` fields, or None."""
    if "_f{nflux}_{inst}_fluxlight_" in tmpl:
        return tmpl.replace("_f{nflux}_{inst}_fluxlight_", "_fluxlight_")
    if "_{inst}_radial_" in tmpl or "_{inst}_stiff_radial_" in tmpl:
        return tmpl.replace("_{inst}_radial_", "_radial_").replace("_{inst}_stiff_radial_", "_stiff_radial_")
    return None


def existing(*paths: str) -> str:
    """The first path that exists, else the first (the new name), so a missing file reports the new name."""
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


def wind_dat_path(wind_dir: str, cr, nflux, inst: Optional[str], kind: str, stiff: bool = False) -> str:
    """Path of a wind table in ``wind_dir``: the new name if present, else the old one if present, else the new name."""
    new = os.path.join(wind_dir, wind_dat(cr, nflux, inst, kind, stiff))
    old = os.path.join(wind_dir, wind_dat(cr, nflux, None, kind, stiff))
    return existing(new, old)


def is_this_task(name: str, nflux, inst: Optional[str] = None) -> bool:
    """False only when the file name says it belongs to another task: a different fluxon count (``_f<N>_``) or,
    when the name carries one, a different instrument. A name with no count (or no instrument: an old batch) is kept."""
    m = _TASK_IN_NAME.search(name)
    if m is None:
        return True
    if m.group(1) != str(nflux):
        return False
    return not (m.group(2) and inst not in _UNKNOWN and m.group(2) != inst)
