"""One answer to "which Perl, and which Flux.pm" for every entry point that launches FLUX (audit A7-05).

The pipeline runner (config_runner), Plumber scene tasks (plumber/scene_task.py), the scene CLI and the golden
driver (scenes/relax.py) all call `flux_runtime`, so the same config gives the same Perl and the same Flux.pm.

Perl, in order: an explicit argument (`--perl`); `$FLUX_PERL` when `honor_env` is set (scene CLI only; the pipeline
has never read it); config `flux_perl` (`~` expanded); else the first `perl` on PATH.

PERL5LIB: the fluxon-perl lib dir is added only when `use Flux` fails with the Perl and environment as they are,
so a conda Perl keeps the Flux.pm that matches its own XS build (a wrong .bundle gives the handshake failure).

Stdlib only, so importing it stays light.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

PROBE = "use Flux; print qq(ok ), $INC{q(Flux.pm)};"


@dataclass
class FluxRuntime:
    perl: str
    env: Dict[str, str]
    flux_loadable: bool  # `use Flux` worked before any PERL5LIB change
    flux_pm: Optional[str]  # the Flux.pm that loads, when known
    perl5lib_added: Optional[str]  # the fluxon-perl dir put on PERL5LIB, or None
    perl_source: str  # "arg", "env", "config" or "path"


def resolve_perl(configs: Optional[Mapping], perl: Optional[str] = None, honor_env: bool = False):
    """(perl, source) by the precedence in the module docstring."""
    if perl:
        return perl, "arg"
    if honor_env and os.environ.get("FLUX_PERL"):
        return os.environ["FLUX_PERL"], "env"
    cfg = (configs or {}).get("flux_perl")
    if cfg:
        return os.path.expanduser(cfg), "config"
    return "perl", "path"


def probe_flux(perl: str, env: Mapping[str, str]):
    """(loadable, Flux.pm path or None) for `use Flux` under `perl` and `env`."""
    try:
        r = subprocess.run([perl, "-e", PROBE], capture_output=True, text=True, timeout=10, env=dict(env))
    except (subprocess.TimeoutExpired, OSError):
        return False, None
    out = (r.stdout or "").strip()
    if r.returncode == 0 and out.startswith("ok"):
        return True, out[2:].strip() or None
    return False, None


def flux_runtime(
    configs: Optional[Mapping],
    perl: Optional[str] = None,
    honor_env: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> FluxRuntime:
    """Resolve the Perl and the environment FLUX runs in. `env` defaults to a copy of os.environ."""
    configs = configs or {}
    perl, source = resolve_perl(configs, perl, honor_env)
    env = dict(os.environ if env is None else env)
    loadable, flux_pm = probe_flux(perl, env)
    added = None
    if not loadable:
        from fluxpype.config import get_flux_perl_lib

        lib = get_flux_perl_lib(dict(configs))
        if lib:
            old = env.get("PERL5LIB")
            env["PERL5LIB"] = lib + (os.pathsep + old if old else "")
            added = lib
            _, flux_pm = probe_flux(perl, env)
    return FluxRuntime(perl, env, loadable, flux_pm, added, source)
