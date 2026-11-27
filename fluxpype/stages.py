"""The stage manifest (audit A2-09): one table of the pipeline stages and the files each one leaves.

Plumber's monitor (probes, log markers, labels), the results view, the fake FLUX backend and the ``stages`` list the
server sends to the web all read this table, so a new stage is one entry here (plus its producer) instead of edits
in five to eight files.

Path templates use ``/`` and these fields:

* ``data``       the data folder; ``batch_dir`` is ``{data}/batches/{batch}``
* ``cr``, ``nflux``, ``inst`` (hmi|adapt), ``rtag`` (``r{mag_reduce}`` for HMI, ``rf{adapt_select}`` for ADAPT)

``probes`` are glob templates (what the monitor and results look for; first pattern that matches wins where a
caller says so). ``files`` are the exact names one task writes, keyed by a short name; the fake backend writes them,
and ``tests/test_stages.py`` checks both against the spellings the Perl and Python drivers produce.
``perl`` names the Perl or Python sub that writes each path; it is documentation only (the drivers are not
generated from this table, and no output name is defined here: names come from the drivers, see ``names.py``).

Standard library only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DATA = "{data}"
BATCH = "{batch_dir}"
CR_DIR = BATCH + "/data/cr{cr}"


@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    # glob templates the monitor/results look for, new name first
    probes: Tuple[str, ...] = ()
    # exact file templates one task writes (fake backend, tests)
    files: Dict[str, str] = field(default_factory=dict)
    # how completion is seen: events (stage.end) always; these add the log and file evidence
    log_start: Optional[str] = None  # regex of the banner that starts the stage in run.log
    log_reused: Optional[str] = None  # regex of the line that says the stage found its product on disk
    log_failure: Optional[str] = None  # regex of the line that says the stage failed
    # results product (id, label, most files to list) when the stage's files are listed in Results; id may differ
    product: Optional[Tuple[str, str, int]] = None
    # the code that writes the files (documentation only)
    perl: str = ""


STAGES: Tuple[Stage, ...] = (
    Stage(
        "magnetogram",
        "Magnetogram",
        probes=(DATA + "/magnetograms/CR{cr}_{rtag}_{inst}.fits",),
        files={"fits": DATA + "/magnetograms/CR{cr}_{rtag}_{inst}.fits"},
        log_start=r"THE CR IS \d+|LIBS ARE LOADED",
        log_failure=r"get_magnetograms\.py failed.*",
        product=("magnetogram", "Magnetogram", 1),
        perl="science/get_magnetograms.py (called by magnetogram2wind.pdl)",
    ),
    Stage(
        "footpoints",
        "Footpoints",
        probes=(
            CR_DIR + "/floc/floc_cr{cr}_*_f{nflux}_{inst}.dat",
            BATCH + "/cr{cr}/floc/floc_cr{cr}_*_f{nflux}_{inst}.dat",
        ),
        files={
            "floc": CR_DIR + "/floc/floc_cr{cr}_{rtag}_f{nflux}_{inst}.dat",
            "open": CR_DIR + "/floc/floc_open_cr{cr}_{rtag}_f{nflux}_{inst}.dat",
            "closed": CR_DIR + "/floc/floc_closed_cr{cr}_{rtag}_f{nflux}_{inst}.dat",
        },
        log_start=r"\(pdl\) Tracing Magnetogram to get \d+ Footpoints",
        log_reused=r"Found a \d+ footpoint file on Disk",
        product=("footpoints", "Footpoints", 4),
        perl="pipe_helper.pm configs_update_magdir (name); get_hilbert_footpoints.pm",
    ),
    Stage(
        "pfss",
        "PFSS trace",
        probes=(DATA + "/pfss/pfss_cr{cr}_*{inst}*.pkl",),
        files={"pkl": DATA + "/pfss/pfss_cr{cr}_{rtag}_{inst}.pkl"},
        log_start=r"\(py\) Running PFSS Code to Trace Footpoints",
        log_reused=r"Fluxon Loc data files already exist",
        log_failure=r"get_pfss\.py failed.*",
        product=("pfss", "PFSS model", 4),
        perl="science/get_pfss.py (called by magnetogram2wind.pdl)",
    ),
    Stage(
        "world",
        "World",
        probes=(CR_DIR + "/world/cr{cr}_f{nflux}_{inst}.flux",),
        files={"flux": CR_DIR + "/world/cr{cr}_f{nflux}_{inst}.flux"},
        log_start=r"\(pdl\) Converting PFSS Fieldlines into FLUX Fluxons",
        log_reused=r"Skipped conversion! World already exists",
        product=("world", "Initial world", 2),
        perl="science/make_world_from_pfss.pm make_world_from_pfss",
    ),
    Stage(
        "relax",
        "Relaxation",
        probes=(CR_DIR + "/world/cr{cr}_f{nflux}_{inst}_relaxed_s*.flux",),
        files={"flux": CR_DIR + "/world/cr{cr}_f{nflux}_{inst}_relaxed_s{steps}.flux"},
        log_start=r"\(pdl\) Relax the Initial World State",
        log_reused=r"Skipped relaxation!",
        log_failure=r"TOO MANY FLUXONS.*",
        product=("relaxed_world", "Relaxed world", 6),
        perl="science/relax_pfss_world.pm relax_pfss_world; pipe_helper.pm find_highest_numbered_file",
    ),
    Stage(
        "world_plots",
        "World plots",
        files={"png": CR_DIR + "/world/cr{cr}_f{nflux}_{inst}_relaxed.png"},
        log_start=r"Relaxation Complete in \d+ Steps|Worlds Successfully loaded",
        perl="science/magnetogram2wind.pdl plot_worlds",
    ),
    Stage(
        "fieldmap",
        "Field map",
        files={"png": CR_DIR + "/fieldmap/cr{cr}_f{nflux}_{inst}_fieldmap.png"},
        log_start=r"\(py\) Plotting Magnetic Field Maps",
        log_reused=r"Skipped! Files already exist",
        log_failure=r"plot_fieldmap\.py failed.*",
        perl="plotting/plot_fieldmap.py (called by magnetogram2wind.pdl)",
    ),
    Stage(
        "wind",
        "Wind",
        # new names first (A7-01); the old ones keep a batch written before then found. The second pattern is the old
        # spellings cr<CR>_f<N>_radial_* and cr<CR>_f<N>_stiff_radial_* in one ([!ah]: not _hmi/_adapt)
        probes=(CR_DIR + "/wind/cr{cr}_f{nflux}_{inst}_*", CR_DIR + "/wind/cr{cr}_f{nflux}_[!ah]*"),
        files={
            "speed": CR_DIR + "/wind/cr{cr}_f{nflux}_{inst}_radial_wind_{method}.dat",
            "table": CR_DIR + "/wind/cr{cr}_f{nflux}_{inst}_radial_{kind}.dat",
            "full_velocity": CR_DIR + "/wind/full_velocity_profiles/results_{method}_full_velocity.dat",
        },
        log_start=r"\(pdl\) Calculating Coronal Plasma Parameters",
        product=("wind", "Wind", 40),
        perl="science/get_wind.pm get_wind; names.wind_dat",
    ),
    Stage(
        "wind_plots",
        "Wind plots",
        files={"png": CR_DIR + "/wind/cr{cr}_f{nflux}_{inst}_{method}_wind.png"},
        log_start=r"\(py\) Plotting Solar Wind Maps",
        perl="plotting/*.py (three commands in magnetogram2wind.pdl)",
    ),
    Stage(
        "fluxlight",
        "FLUXlight",
        probes=(
            BATCH + "/**/cr{cr}_f{nflux}_{inst}_fluxlight_*.png",
            BATCH + "/**/cr{cr}_fluxlight_*.png",
        ),
        files={"png": CR_DIR + "/light/cr{cr}_f{nflux}_{inst}_fluxlight_light_{method}_1.png"},
        product=("fluxlight", "FLUXlight", 40),
        perl="science/fluxlight.py; names.light_png",
    ),
    Stage(
        "euv",
        "EUV compare",
        files={"model_fits": CR_DIR + "/light/cr{cr}_f{nflux}_{inst}_aia{channel}.fits"},
        perl="science/fluxlight_euv.py; names.light_fits",
    ),
)

STAGE_IDS: List[str] = [s.id for s in STAGES]
LABELS: Dict[str, str] = {s.id: s.label for s in STAGES}
BY_ID: Dict[str, Stage] = {s.id: s for s in STAGES}


def _fmt(template: str, **fields) -> str:
    """Fill a template; ``/`` becomes the OS separator (the same result ``os.path.join`` gave)."""
    return template.replace("/", os.sep).format(**fields) if os.sep != "/" else template.format(**fields)


def fields_for(data_dir, batch, cr, nflux, inst, reduce="4", adapt_select="2", **extra) -> Dict[str, str]:
    """The template fields of one task. ``rtag`` is ``r{reduce}`` for HMI and ``rf{adapt_select}`` for ADAPT."""
    batch_dir = os.path.join(str(data_dir), "batches", str(batch))
    rtag = f"r{reduce}" if inst == "hmi" else f"rf{adapt_select}"
    return {
        "data": str(data_dir),
        "batch_dir": batch_dir,
        "cr": str(cr),
        "nflux": str(nflux),
        "inst": str(inst),
        "rtag": rtag,
        **extra,
    }


def probes(stage_id: str, fields: Dict[str, str]) -> List[str]:
    """The glob patterns that find a stage's output for one task."""
    return [_fmt(t, **fields) for t in BY_ID[stage_id].probes]


def path(stage_id: str, key: str, fields: Dict[str, str]) -> str:
    """The exact path of one named file of a stage (extra fields such as ``steps``, ``method`` go in ``fields``)."""
    return _fmt(BY_ID[stage_id].files[key], **fields)


def log_markers(attr: str):
    """``[(stage_id, compiled regex)]`` for one of ``log_start``, ``log_reused``, ``log_failure``, in stage order."""
    return [(s.id, re.compile(getattr(s, attr))) for s in STAGES if getattr(s, attr)]


def products() -> List[Tuple[str, str, str, int]]:
    """``(product id, label, stage id, most files)`` for the stages that list files in Results."""
    return [(s.product[0], s.product[1], s.id, s.product[2]) for s in STAGES if s.product]


def public() -> List[Dict[str, str]]:
    """What the server sends to the web: id and label in pipeline order."""
    return [{"id": s.id, "label": s.label} for s in STAGES]
