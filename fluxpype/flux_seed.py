"""The `flux_seed` setting: which random seed the FLUX Perl processes start with (audit A1-05).

`config.ini` `flux_seed` (or the environment variable `FLUXPYPE_FLUX_SEED`, which wins) is one of:

    0 (default), fixed   a fixed seed, so a run repeats
    jd                   seed from the current Julian date (seconds resolution), so each run differs
    <integer>            that seed, to repeat an earlier `jd` run (the seed is logged for every run)

The seed goes to Perl as PERL_HASH_SEED (key order of every hash) and PERL_RAND_SEED (srand, Perl 5.28 or later),
so no FLUX or fluxpype Perl is changed. Perl's rand treats a seed of 0 as "not set", so the fixed mode seeds rand
with 1.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Mapping, Optional, Tuple

ENV_VAR = "FLUXPYPE_FLUX_SEED"
FIXED = 0
_UNIX_EPOCH_JD = 2440587.5


def julian_date(now: Optional[float] = None) -> float:
    """Julian date of a Unix time (default: now)."""
    return (time.time() if now is None else now) / 86400.0 + _UNIX_EPOCH_JD


def parse_seed(value, now: Optional[float] = None) -> int:
    """`0` / `fixed` / empty -> 0; `jd` -> Julian date in seconds, modulo 2**32; an integer -> itself."""
    text = "" if value is None else str(value).strip().lower()
    if text in ("", "fixed", "off"):
        return FIXED
    if text == "jd":
        return int(julian_date(now) * 86400) % 2**32
    try:
        seed = int(text, 10)
    except ValueError:
        raise ValueError(f"flux_seed must be 0, jd or an integer, not {value!r}") from None
    if not 0 <= seed < 2**32:
        raise ValueError(f"flux_seed must be between 0 and 2**32 - 1, not {seed}")
    return seed


def resolve(configs: Optional[Mapping] = None, env: Optional[Mapping] = None, now: Optional[float] = None) -> int:
    """The seed for one run: the environment variable if set, else config.ini `flux_seed`, else fixed."""
    env = os.environ if env is None else env
    raw = env.get(ENV_VAR)
    if raw is None or not str(raw).strip():
        raw = (configs or {}).get("flux_seed")
    return parse_seed(raw, now)


def apply(
    env: Dict[str, str], configs: Optional[Mapping] = None, now: Optional[float] = None
) -> Tuple[Dict[str, str], int]:
    """Set the Perl seed variables in `env` (modified in place and returned) and return (env, seed)."""
    seed = resolve(configs, env, now)
    env["PERL_HASH_SEED"] = str(seed)
    env["PERL_RAND_SEED"] = str(seed or 1)
    return env, seed


def describe(seed: int) -> str:
    """One log line that says how to repeat the run."""
    return f"flux_seed={seed} (PERL_HASH_SEED={seed}); repeat this run with flux_seed = {seed}"


def child_env(configs: Optional[Mapping] = None) -> Tuple[Dict[str, str], int]:
    """A copy of os.environ with the seed applied; config.ini is read here when `configs` is not given."""
    if configs is None:
        try:
            from fluxpype.config import configurations

            configs = configurations(debug=False)
        except Exception:  # noqa: BLE001 - no config.ini (a bare checkout, a test): the fixed default
            configs = {}
    return apply(dict(os.environ), configs)
