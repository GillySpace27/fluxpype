"""
Primary Library for PFSS (Potential Field Source Surface) Modeling.
=====================================================================

This module provides utility functions for:

- Loading and conditioning FITS files containing magnetogram data.
- Converting pixel coordinates to latitude and longitude.
- Computing the PFSS solution based on the conditioned magnetogram.
- Tracing magnetic field lines.
- Plotting fluxon locations on the magnetogram.
- Saving and loading PFSS solutions from pickle files.

Attributes:
    None

Functions:
    load_fits(file_path: str) -> object
    condition_data(data: object) -> object
    compute_pfss(data: object) -> object
    trace_field_lines(pfss_solution: object) -> list
    plot_fluxons(magnetogram: object, field_lines: list) -> None
    save_pfss_solution(pfss_solution: object, file_path: str) -> None
    load_pfss_solution(file_path: str) -> object

Dependencies:
    - astropy
    - matplotlib
    - numpy
    - pfsspy
    - sunpy
    - timeout_decorator
    - tqdm


Author:

    Gilly <gilly@swri.org> (and others!)

"""


import copy
import json
import os
import pickle
import re
import warnings
from concurrent import futures
from multiprocessing.shared_memory import SharedMemory
from os import path
from time import time

# Worker global: PFSS output loaded once per worker from shared memory (set by _init_trace_worker)
_TRACE_WORKER_PFSS_OUTPUT = None

# #region agent log
def _dlog(msg, **data):
    try:
        with open("/Users/gilly/vscode/fluxons/.cursor/debug.log", "a") as f:
            f.write(json.dumps({"location": "pfss_funcs", "message": msg, "data": data, "timestamp": int(time() * 1000)}, default=str) + "\n")
    except Exception:
        pass
# #endregion


def _init_trace_worker(shm_name, size):
    """Run once per worker: attach to shared memory, deserialize PFSS output, set global."""
    global _TRACE_WORKER_PFSS_OUTPUT
    # #region agent log
    _dlog("init_trace_worker entry", hypothesisId="H1", shm_name=shm_name, size=size)
    # #endregion
    shm = SharedMemory(name=shm_name, create=False)
    # #region agent log
    _dlog("init_trace_worker after attach", hypothesisId="H1")
    # #endregion
    try:
        payload = bytes(shm.buf[:size])
        # #region agent log
        _dlog("init_trace_worker before loads", hypothesisId="H1", payload_len=len(payload))
        # #endregion
        _TRACE_WORKER_PFSS_OUTPUT = pickle.loads(payload)
        # #region agent log
        _dlog("init_trace_worker after loads", hypothesisId="H1")
        # #endregion
    finally:
        shm.close()
    # #region agent log
    _dlog("init_trace_worker exit", hypothesisId="H1")
    # #endregion

import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
import timeout_decorator
from tqdm import tqdm

# Prefer sunkit-magex for PFSS and tracing (same API as pfsspy; forked from it).
try:
    import sunkit_magex.pfss as _magex_pfss
    _USE_MAGEX = True
except ImportError:
    _USE_MAGEX = False
    _magex_pfss = None

import pfsspy
from pfsspy import tracing as _pfsspy_tracing

from fluxpype.console import emit
from fluxpype.pipe_helper import load_fits_magnetogram, read_fits_data, shorten_path

try:
    mpl.use("qt5agg")
except ImportError:
    mpl.use("Agg")

# Suppress sunpy/astropy INFO and noisy warnings unless high debug.
# Call set_sunpy_log_level(verbose) with verbose >= 2 from entry points to show those messages.
def set_sunpy_log_level(verbose=0):
    """Set sunpy (and astropy) log level and warning filters: quiet by default; INFO + show warnings if verbose >= 2."""
    import logging
    show_all = verbose >= 2 or os.environ.get("FLUXPYPE_SUNPY_VERBOSE")
    level = logging.INFO if show_all else logging.WARNING
    try:
        from sunpy import log as sunpy_log
        sunpy_log.setLevel(level)
    except Exception:
        pass
    try:
        logging.getLogger("astropy").setLevel(level)
    except Exception:
        pass
    # pfsspy "Could not parse unit string ... Mx/cm^2 ..." UserWarning (suppress unless high debug)
    _pfsspy_unit_msg = r".*Could not parse unit string.*"
    if show_all:
        # Insert so it wins over any earlier "ignore" from import
        warnings.filters.insert(0, ("default", re.compile(_pfsspy_unit_msg, re.IGNORECASE), UserWarning, None, 0))
    else:
        warnings.filterwarnings("ignore", message=_pfsspy_unit_msg, category=UserWarning)


def _suppress_sunpy_info_default():
    """Run at import so worker processes and default runs suppress noisy INFO and warnings."""
    set_sunpy_log_level(0)


_suppress_sunpy_info_default()

# A note from the original author:
    # We can now use SunPy to load the HMI fits file, and extract the magnetic field data. Interpolate the data down to a more reasonable resolution.
    # The mean is subtracted to enforce div(B) = 0 on the solar surface: n.b. it is
    # not obvious this is the correct way to do this, so use the following lines
    # at your own risk!
    # # brmap = (sunpy.io.fits.read(datdir + fname)) # b radial map
    # # brdat = brmap[0].data[2,:,:] # b data itself



def load_and_condition_fits_file(fname, datdir, adapt):
    """
    Load and condition the FITS file containing the magnetogram data.
    This involves subtracting the mean from the data to enforce div(B) = 0 on the solar surface.
    The metadata is updated to reflect the correct coordinate system, the nans are removed,
    and the data is returned as a sunpy map object.

    Parameters
    ----------
    fname : str
        Filename path
    datdir : str
        Data directory path

    Returns
    -------
    br_safe: GenericMap
        Processed magnetogram data
    fits_path: str
        FITS file path
    """

    emit("\n\tLoading and conditioning fits file...")
    try:
        fits_path = fname
        hdulist = read_fits_data(fits_path)
    except FileNotFoundError:
        fits_path = os.path.join(datdir, fname)
        hdulist = read_fits_data(fits_path)
    emit(f"\t\t{shorten_path(fits_path)}")

    hdu_0 = hdulist[0]
    brdat = hdu_0.data
    header = hdu_0.header

    header["CUNIT2"] = "rad"

    # u.def_unit("sin", u.dimensionless_angles)
    # u.add_enabled_units("sin")

    brdat = brdat - np.mean(brdat)
    radial_magnetogram = sunpy.map.Map(brdat, header)



    radial_magnetogram.meta['ctype1'] = 'CRLN-CEA'
    radial_magnetogram.meta['ctype2'] = 'CRLT-CEA'
    radial_magnetogram.meta['naxis'] = 2
    # radial_magnetogram.meta['CUNIT2'] = 'Sine Latitude'

    radial_magnetogram.meta['crlt_obs'] = 0.0
    radial_magnetogram.meta['crln_obs'] = 0.0
    radial_magnetogram.meta['dsun_obs'] = 1.0

    br_safe = copy.copy(radial_magnetogram)
    br_safe.data[np.isnan(br_safe.data)] = 0

    return br_safe, fits_path


def fits_path_to_pickle_path(fits_path, reduce):
    """Returns the path to the pickle file, given a fits file path

    Parameters
    ----------
    fits_path : str
        FITS file path
    reduce : int
        Reduction factor

    Returns
    -------
    str
        Output file path
    """

    the_dir = path.dirname(path.dirname(fits_path))
    pickle_path = path.join(the_dir, f"pfss_output_{reduce}.pkl")
    return pickle_path


def load_pfss(pickle_path):
    """Loads the pickle file

    Parameters
    ----------
    pickle_path : str
        PFSS solution saved state path

    Returns
    -------
    output
        PFSS solution state function
    """

    emit("\n\tGetting PFSS...", end="")
    try:
        pickle_path = os.path.expanduser(pickle_path)
        with open(pickle_path, 'rb') as inp:
            output = pickle.load(inp)
            emit(f" Success! Loaded: {shorten_path(pickle_path)}")
        return output
    except (FileNotFoundError, TypeError):
        emit(" File not found.")
        return None


def compute_pfss(br_safe, pickle_path, nrho=60, rss=2.5):
    """The PFSS solution is calculated on a regular 3D grid in (phi, s, rho), where
        rho = ln(r), and r is the standard spherical radial coordinate. We need to
        define the number of rho grid points, and the source surface radius.

    Parameters
    ----------
    br_safe : HDU
        Magnetogram hdu
    pickle_path : str
        PFSS solution saved state path
    nrho : int, optional
        Number of points in the rho direction. Defaults to 60.
    rss : float, optional
        Source surface height in r/r_sun. Defaults to 2.5.

    Returns
    -------
    output
        PFSS solution state function
    elapsed : float
        Elapsed runtime in seconds
    """

    # Ingest inputs
    elapsed = 0
    shp = br_safe.data.shape
    emit(f"\t\tComputing PFSS on {shp} magnetogram...", end="")
    before = time()

    ###############################################################################
    # From the boundary condition, number of radial grid points, and source
    # surface, we now construct an Input object that stores this information.
    # Use sunkit-magex when available (same API; required for sunkit-magex tracing).
    if _USE_MAGEX:
        pfss_in = _magex_pfss.Input(br_safe, nrho, rss)
        output = _magex_pfss.pfss(pfss_in)
    else:
        pfss_in = pfsspy.Input(br_safe, nrho, rss)
        output = pfsspy.pfss(pfss_in)

    ###############################################################################
    # Report the time taken
    elapsed = time() - before
    emit(f" Done! ({elapsed:.2f}s)")

    ###############################################################################
    # Save the results
    with open(pickle_path, 'wb') as outp:
        emit("\t\tSaving pickled results...", end="")
        pickle.dump(output, outp, pickle.HIGHEST_PROTOCOL)
        emit(" Success!")
        # print(pickle_path)
    return output, elapsed


def pixel_to_latlon(mag_map, header, fluxon_location):
    """Converts magnetogram pixel index to lat/lon

    Parameters
    ----------
    mag_map : np.ndarray
        Magnetogram data array
    header : Header
        Magnetogram header object
    fluxon_location : Tuple
        Indices of the fluxon root locations

    Returns
    -------
    f_lat : float
        Fluxon root latitudes
    f_lon : float
        Fluxon root longitudes
    f_sgn : float
        Fluxon magnetic sign
    n_flux : int
        Fluxon count
    """

    radial_magnetogram = sunpy.map.Map(mag_map, header)
    shp = mag_map.shape
    n_lat = shp[0]
    lat_center = n_lat // 2
    f_lon = np.deg2rad(fluxon_location[:, 0]
                       * radial_magnetogram.meta['cdelt1'])
    f_lat = (fluxon_location[:, 1]-lat_center) * \
        radial_magnetogram.meta['cdelt2']
    f_sgn = fluxon_location[:, 2]
    n_flux = len(f_sgn)
    return f_lat, f_lon, f_sgn, n_flux


def get_fluxon_locations(floc_path, batch, configs=None, cr=None):
    """Loads the fluxon location file

    Parameters
    ----------
    floc_path : str
        Path to the fluxon location file
    batch : str
        Name of the current batch

    Returns
    -------
    f_lat : float
        Fluxon root latitudes
    f_lon : float
        Fluxon root longitudes
    f_sgn : float
        Fluxon magnetic sign
    n_flux : int
        Fluxon count
    """

    # print(f"{floc_path=}")
    if not configs and not cr:
        # Extract Carrington rotation number from file path if not provided
        if "cr" in floc_path:
            import re
            match = re.search(r"cr(\d{4})", floc_path)
            if match:
                cr = int(match.group(1))
            else:
                raise ValueError("Carrington rotation number not found in floc_path")
        else:
            raise ValueError("cr must be provided or inferable from floc_path")


    fluxon_location = np.genfromtxt(floc_path)
    if cr is not None:
        magnet, header = load_fits_magnetogram(batch=batch, ret_all=True, configs=configs, cr=cr)
    else:
        magnet, header = load_fits_magnetogram(batch=batch, ret_all=True, configs=configs)

    f_lat, f_lon, f_sgn, n_flux = pixel_to_latlon(
        magnet, header, fluxon_location)
    return f_lat, f_lon, f_sgn, n_flux


def plot_fluxon_locations(br_safe, cr, datdir, fits_path, reduce,
                          force_plot=False, batch='fluxon', nwant=0, do_plot=False):
    """Plot computed fluxon root locations

    Parameters
    ----------
    br_safe : GenericMap
        Magneogram map object
    cr : str
        Carrington rotation number
    datdir : str
        Data path directory
    fits_path : str
        FITS file path directory
    reduce : int
        Reduction factor
    force_plot : bool, optional
        Plot overwrite toggle. Defaults to False.
    batch : str, optional
        Output file descriptor label. Defaults to 'fluxon'.
    nwant : int, optional
        Output file descriptor label. Defaults to 0.
    do_plot : bool, optional
        Plot toggle. Defaults to False.

    Returns
    -------
    f_lat : float
        Fluxon root latitudes
    f_lon : float
        Fluxon root longitudes
    f_sgn : float
        Fluxon magnetic sign
    n_flux : int
        Fluxon count
    """

    floc_path = f"{datdir}/batches/{batch}/data/cr{cr}/floc/floc_cr{cr}_r{reduce}_f{nwant}.dat"
    f_lat, f_lon, f_sgn, n_flux = get_fluxon_locations(floc_path, batch, cr=cr)
    fluxon_location = np.genfromtxt(floc_path)

    if not do_plot:
        return f_lat, f_lon, f_sgn, n_flux

    the_dir = path.dirname(path.dirname(fits_path))

    top_dir = path.join(datdir, f"batches/{batch}/imgs/footpoints")
    fluxon_map_output_path_top = path.join(
        top_dir, f'cr{cr}_footpoints_r{reduce}_f{nwant}.pdf')

    fluxon_map_output_path_blank = path.join(
        the_dir, "magnetograms", f'footpoints_cr{cr}_r{reduce}_blank.png')
    fluxon_map_output_path_blank_top = path.join(
        top_dir, f'cr{cr}_footpoints_r{reduce}_blank.png')

    if not path.exists(top_dir):
        os.makedirs(top_dir)

    plot_paths = [
        fluxon_map_output_path_top,
        fluxon_map_output_path_blank_top,
        fluxon_map_output_path_blank,
    ]

    emit("\n\tPlotting fluxon footpoint locations...")

    need_plot = False
    for testpath in plot_paths:
        if not path.exists(testpath):
            need_plot = True
    if need_plot or force_plot:
        # Print the Fluxon Map
        fig, ax = plt.subplots()

        magnet = br_safe.data
        # find the max and min of the magnetogram plot for use in setting the colormap,
        sigma = 2
        mmean = np.nanmean(magnet)
        msig = np.nanstd(magnet)

        mvmin = mmean - sigma*msig
        mvmax = mmean + sigma*msig
        mvmin, mvmax = -500, 500

        magimg1 = ax.imshow(magnet, cmap='gray', interpolation=None,
                            origin="lower", aspect='auto', vmin=mvmin, vmax=mvmax)
        shp = magnet.shape

        plt.axis('off')
        sz0 = 6  # inches
        ratio = shp[1]/shp[0]
        sz1 = sz0*ratio  # inches
        DPI = shp[1] / sz1  # pixels/inch
        fig.set_size_inches((sz1, sz0))
        plt.tight_layout()

        # plot a blank version of the map
        plt.savefig(fluxon_map_output_path_blank,
                    bbox_inches='tight', dpi=4*DPI)

        # plot a blank version of the map
        magimg2 = ax.imshow(magnet, cmap='gray', interpolation=None,
                            origin="lower", aspect='auto', vmin=mvmin, vmax=mvmax)
        plt.savefig(fluxon_map_output_path_blank_top,
                    bbox_inches='tight', dpi=4*DPI)

        # scatter the fluxons on top of the map
        magimg3 = ax.imshow(magnet, cmap='gray', interpolation=None,
                            origin="lower", aspect='auto', vmin=mvmin, vmax=mvmax)

        x, y, sig = zip(*fluxon_location)
        colors = ['red' if s > 0 else 'teal' for s in sig]
        ax.scatter(x, y, c=colors, alpha=0.4)

        # plot the fluxons scattered on top of the map
        plt.savefig(fluxon_map_output_path_top, bbox_inches='tight', dpi=4*DPI)
        plt.close()
        emit("Success! Files saved to:")
    else:
        emit("Skipped! Files already exist:")

    for testpath in plot_paths:
        emit(f"\t\t{shorten_path(testpath)}")

    return f_lat, f_lon, f_sgn, n_flux


def _trace_chunk_in_process(chunk_items, r0, f_sgn, adapt, output, use_timeout=False, tracer=None):
    """Run a batch of fieldline traces in the current process. Used by serial path and by workers.
    If use_timeout is True, each trace has a 5s timeout (serial path only).
    If tracer is None, one is created (use a single shared tracer in serial for speed).
    Returns: [(i, msg), ...] with msg = ('ok', ...), ('timeout',), or ('err', e)."""
    if tracer is None:
        tracer = _pfsspy_tracing.PythonTracer()
    results = []
    trace_fn = trace_each_single if use_timeout else _trace_each_single_core
    for (i, coords) in chunk_items:
        fl_open = np.zeros([1, 5])
        fl_closed = np.zeros([1, 5])
        try:
            _, fl_open, fl_closed, no, nc, _ = trace_fn(
                coords, i, output, fl_open, fl_closed, 0, 0, tracer, r0, f_sgn, adapt)
            results.append((i, ("ok", fl_open[1:].copy(), fl_closed[1:].copy(), no, nc)))
        except timeout_decorator.TimeoutError:
            results.append((i, ("timeout",)))
        except Exception as e:
            results.append((i, ("err", e)))
    return results


def _trace_chunk_worker(args):
    """Run a batch of fieldline traces in one process. Uses PFSS output from shared memory (global).
    args: (chunk_items, r0, f_sgn, adapt), chunk_items = [(i, coords), ...].
    Returns: [(i, msg), ...] with msg = ('ok', fl_open[1:], fl_closed[1:], no, nc) or ('err', e)."""
    chunk_items, r0, f_sgn, adapt = args
    # #region agent log
    _dlog("trace_chunk_worker entry", hypothesisId="H3", n_chunk=len(chunk_items), first_i=chunk_items[0][0] if chunk_items else None)
    # #endregion
    output = _TRACE_WORKER_PFSS_OUTPUT
    if output is None:
        # #region agent log
        _dlog("trace_chunk_worker output is None", hypothesisId="H1")
        # #endregion
        return [(i, ("err", RuntimeError("PFSS output not loaded in worker"))) for i, _ in chunk_items]
    results = _trace_chunk_in_process(chunk_items, r0, f_sgn, adapt, output)
    # #region agent log
    _dlog("trace_chunk_worker exit", hypothesisId="H3", n_results=len(results))
    # #endregion
    return results


def _is_magex_output(output):
    """True if output is from sunkit_magex.pfss (required for sunkit-magex tracer)."""
    mod = getattr(type(output), "__module__", "") or ""
    return "sunkit_magex" in mod


def _trace_lines_magex(output, f_vars, open_path, closed_path, adapt):
    """Trace field lines using sunkit-magex (batch trace). Output format matches make_world_from_pfss.pm."""
    (f_lon, f_lat, f_sgn) = f_vars
    f_lon = np.atleast_1d(np.asarray(f_lon, dtype=float))
    f_lat = np.atleast_1d(np.asarray(f_lat, dtype=float))
    f_sgn = np.atleast_1d(np.asarray(f_sgn, dtype=float))
    n = len(f_lon)
    if len(f_lat) != n or len(f_sgn) != n:
        raise ValueError("f_lon, f_lat, f_sgn must have same length")

    r0 = 1.01 * const.R_sun
    coord_frame = output.coordinate_frame
    # Seed construction: match legacy convention (lon in rad; lat via arcsin(deg2rad(lat)) if adapt else arcsin(lat))
    flat = np.deg2rad(f_lat) if adapt else f_lat
    lat_rad = np.arcsin(np.clip(flat, -1.0, 1.0))
    seeds = SkyCoord(
        f_lon * u.rad,
        lat_rad * u.rad,
        r0,
        frame=coord_frame,
    )

    tracer = _magex_pfss.tracing.PerformanceTracer(max_steps=3000)
    emit("\n\t\tTracing field lines (sunkit-magex batch)...", end="")
    field_lines = tracer.trace(seeds, output)
    emit(" Done.")

    # FieldLines is iterable; each element is a FieldLine with .is_open, .polarity, .coords (SkyCoord)
    fl_open_rows = []
    fl_closed_rows = []
    flnum_open = 0
    flnum_closed = 0
    skip_num = 0

    try:
        fl_list = list(field_lines)
    except Exception:
        fl_list = []
    if len(fl_list) != n:
        emit(f"\n\t\tWarning: tracer returned {len(fl_list)} field lines, expected {n}. Filling missing with skip.", style="yellow")

    for i in range(n):
        if i >= len(fl_list):
            skip_num += 1
            continue
        fl = fl_list[i]
        try:
            is_open = fl.is_open
            polarity = fl.polarity
            sph = fl.coords.spherical
            fl_lats = np.atleast_1d(sph.lat.to_value(u.rad))
            fl_lons = np.atleast_1d(sph.lon.to_value(u.rad))
            fl_rads = np.atleast_1d((sph.distance / const.R_sun).to_value(u.dimensionless_unscaled))
        except Exception:
            skip_num += 1
            continue

        sgn_i = int(f_sgn[i]) if i < len(f_sgn) else 0

        if is_open:
            # Order so outward direction is consistent (reverse if polarity == -1)
            if polarity == -1:
                fl_lats, fl_lons, fl_rads = fl_lats[::-1], fl_lons[::-1], fl_rads[::-1]
            # Extend open lines to 200 R_sun (match legacy)
            extend_height = float(np.amax(fl_rads))
            extend_num2 = 40
            fl_lats = np.concatenate([fl_lats, np.full(extend_num2, fl_lats[-1])])
            fl_lons = np.concatenate([fl_lons, np.full(extend_num2, fl_lons[-1])])
            higher = np.logspace(np.log10(max(extend_height, 1.01)), np.log10(200.0), extend_num2 + 1)[1:]
            fl_rads = np.concatenate([fl_rads, higher])
            fl_rads = np.round(fl_rads, 8)
            unique_mask = np.concatenate(([True], fl_rads[1:] != fl_rads[:-1]))
            fl_rads = fl_rads[unique_mask]
            fl_lats = fl_lats[unique_mask]
            fl_lons = fl_lons[unique_mask]
            for j in range(len(fl_lats)):
                fl_open_rows.append([flnum_open, sgn_i, np.rad2deg(fl_lats[j]), np.rad2deg(fl_lons[j]), fl_rads[j]])
            flnum_open += 1
        else:
            # Closed: subsample by radius threshold (every other closed line)
            max_rad = float(np.max(fl_rads))
            rad_thresh = max((max_rad - 1.0) / 5.0, 0.01)
            prev_rad = 0.0
            if np.mod(flnum_closed, 2) == 0:
                for j in range(len(fl_rads)):
                    if np.abs(fl_rads[j] - prev_rad) > rad_thresh or fl_rads[j] == max_rad:
                        fl_closed_rows.append([
                            flnum_closed, sgn_i,
                            np.rad2deg(fl_lats[j]), np.rad2deg(fl_lons[j]),
                            fl_rads[j],
                        ])
                        prev_rad = fl_rads[j]
            flnum_closed += 1

    fl_open = np.array(fl_open_rows, dtype=float) if fl_open_rows else np.zeros((0, 5))
    fl_closed = np.array(fl_closed_rows, dtype=float) if fl_closed_rows else np.zeros((0, 5))

    emit(f"\n\tOpen: {flnum_open}, Closed: {flnum_closed}, Skipped: {skip_num}")

    emit("\n\tSaving fluxons...", end="")
    np.savetxt(open_path, fl_open)
    np.savetxt(closed_path, fl_closed)
    emit(" Success!")

    return fl_open, fl_closed, skip_num, 0, flnum_open, flnum_closed


def trace_lines(output, f_vars, open_path, closed_path, adapt, trace_serial=True):
    """Trace field lines using the PFSS solution. Uses sunkit-magex when available (batch tracing)."""

    (f_lon, f_lat, f_sgn) = f_vars

    # Use sunkit-magex batch tracing when we have sunkit_magex and its Output type
    if _USE_MAGEX and _is_magex_output(output):
        return _trace_lines_magex(output, f_vars, open_path, closed_path, adapt)

    # Fallback: pfsspy per-line tracing (legacy path)
    fl_open = np.zeros([1, 5])
    flnum_open = 0
    fl_closed = np.zeros([1, 5])
    flnum_closed = 0
    r0 = 1.01 * const.R_sun
    skip_num = 0
    timeout_num = 0
    items = list(enumerate(zip(f_lon, f_lat)))
    chunks = [[item] for item in items]
    results_by_i = {}

    _TRACE_SERIAL = bool(int(trace_serial)) if trace_serial is not None else (os.environ.get("FLUXPYPE_TRACE_SERIAL", "1") == "1")
    if _TRACE_SERIAL:
        _tracer = _pfsspy_tracing.PythonTracer()
        for chunk in tqdm(chunks, desc="Tracing Field Lines"):
            for i, msg in _trace_chunk_in_process(chunk, r0, f_sgn, adapt, output, use_timeout=True, tracer=_tracer):
                results_by_i[i] = msg
    else:
        _MAX_TRACE_WORKERS = min(16, max(1, (os.cpu_count() or 4) - 1))
        _TRACE_TIMEOUT_SEC = 5
        _BATCH_SIZE = max(_MAX_TRACE_WORKERS, 16)
        payload = pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL)
        shm = SharedMemory(create=True, size=len(payload))
        try:
            shm.buf[:len(payload)] = payload
            executor = futures.ProcessPoolExecutor(
                max_workers=_MAX_TRACE_WORKERS,
                initializer=_init_trace_worker,
                initargs=(shm.name, len(payload)),
            )
            with executor:
                pbar = tqdm(total=len(items), desc="Tracing Field Lines")
                for start in range(0, len(chunks), _BATCH_SIZE):
                    batch = chunks[start:start + _BATCH_SIZE]
                    future_to_chunk = {
                        executor.submit(_trace_chunk_worker, (chunk, r0, f_sgn, adapt)): chunk
                        for chunk in batch
                    }
                    done_count = 0
                    pending = set(future_to_chunk.keys())
                    batch_deadline = time() + min(30, len(batch) * (_TRACE_TIMEOUT_SEC + 2))
                    while pending and time() < batch_deadline:
                        done, pending = futures.wait(
                            pending, timeout=_TRACE_TIMEOUT_SEC, return_when=futures.FIRST_COMPLETED
                        )
                        for future in done:
                            chunk = future_to_chunk[future]
                            try:
                                for i, msg in future.result(timeout=0):
                                    results_by_i[i] = msg
                            except (futures.TimeoutError, Exception):
                                for i, _ in chunk:
                                    results_by_i[i] = ("timeout",)
                            done_count += 1
                            pbar.update(len(chunk))
                    for future in pending:
                        for i, _ in future_to_chunk[future]:
                            results_by_i[i] = ("timeout",)
                        pbar.update(len(future_to_chunk[future]))
                    batch_start = start
                executor.shutdown(wait=False)
                pbar.close()
        finally:
            shm.close()
            shm.unlink()

    for i, _ in items:
        msg = results_by_i.get(i, ("err", None))
        if msg[0] == "timeout":
            timeout_num += 1
            continue
        if msg[0] == "err":
            skip_num += 1
            continue
        _, new_open, new_closed, no, nc = msg
        if len(new_open) > 0:
            new_open = np.asarray(new_open)
            new_open[:, 0] += flnum_open
            fl_open = np.vstack((fl_open, new_open))
        if len(new_closed) > 0:
            new_closed = np.asarray(new_closed)
            new_closed[:, 0] += flnum_closed
            fl_closed = np.vstack((fl_closed, new_closed))
        flnum_open += no
        flnum_closed += nc

    if skip_num > 0 or timeout_num > 0:
        t_perc = 100 * timeout_num / len(f_lon)
        s_perc = 100 * skip_num / len(f_lon)
        emit(f"\n\t\tSome iterations failed. Timed-out: {timeout_num} ({t_perc:.2f}%), ValueError: {skip_num} ({s_perc:.2f}%)", style="yellow")

    emit(f"\n\tOpen: {flnum_open + 1}, Closed: {flnum_closed}, Failures: {skip_num + timeout_num}")

    fl_open = fl_open[1:]
    fl_closed = fl_closed[1:]

    emit("\n\tSaving fluxons...", end="")
    np.savetxt(open_path, fl_open)
    np.savetxt(closed_path, fl_closed)
    emit(" Success!")

    return fl_open, fl_closed, skip_num, timeout_num, flnum_open, flnum_closed


def _trace_each_single_impl(coords, i, output, fl_open, fl_closed, flnum_open, flnum_closed, tracer, r0, f_sgn, adapt):
    """Single-fieldline trace implementation (no timeout). Blocking call is output.trace() (pfsspy C/Fortran)."""
    (this_flon, this_flat) = coords
    if adapt:
        this_flat = np.deg2rad(this_flat)

    coord_frame = output.coordinate_frame

    x0 = SkyCoord(this_flon * u.rad, np.arcsin(this_flat)
                  * u.rad, r0, frame=coord_frame)
    first = True
    fl = output.trace(tracer, x0)
    fl = fl.field_lines[0]
    if fl.is_open:
        fl_pol = fl.polarity
        fl = fl.coords.spherical
        fl_lats = fl.lat.value
        fl_lons = fl.lon.value
        fl_rads = (fl.distance/const.R_sun).value

        if fl_pol == -1:
            (fl_lats, fl_lons, fl_rads) = (
                fl_lats[::-1], fl_lons[::-1], fl_rads[::-1])
        if False:
            # extend down to 1.001 R_sun
            extend_num0 = 0
            small_exponent = -4
            lat_cat = fl_lats[0]*np.ones(extend_num0)
            lon_cat = fl_lons[0]*np.ones(extend_num0)

            (fl_lats, fl_lons, fl_rads) = (
                fl_lats[extend_num0::], fl_lons[extend_num0::], fl_rads[extend_num0::])

            lower = np.logspace(small_exponent, np.log10(fl_rads[0]-1), extend_num0, endpoint=False) + 1
            fl_lats = np.concatenate((lat_cat, fl_lats), axis=0)
            fl_lons = np.concatenate((lon_cat, fl_lons), axis=0)
            fl_rads = np.concatenate((lower, fl_rads), axis=0)
            # assert False

        if True:
            #extend up to 21.5 R_sun logarithmically
            extend_height = np.amax(fl_rads)
            extend_height2 = 200.
            extend_num2 = 40
            fl_lats = np.concatenate((fl_lats, fl_lats[-1]*np.ones(extend_num2)), axis=0)
            fl_lons = np.concatenate((fl_lons, fl_lons[-1]*np.ones(extend_num2)), axis=0)
            higher2 = np.logspace(np.log10(extend_height), np.log10(extend_height2), extend_num2+1)[1:]
            fl_rads = np.concatenate((fl_rads, higher2), axis=0)

        fl_rads = np.round(fl_rads, 8)

        # Boolean array to mark unique elements
        unique_mask = np.concatenate(([True], fl_rads[1:] != fl_rads[:-1]))

        # Apply mask to filter out duplicates
        fl_rads = fl_rads[unique_mask]
        fl_lats = fl_lats[unique_mask]
        fl_lons = fl_lons[unique_mask]

        # print("Filtered fl_rads:", fl_rads.T)
        # print("Filtered fl_lats:", fl_lats.T)
        # print("Filtered fl_lons:", fl_lons.T)
        # import pdb; pdb.set_trace()

        # if first:
        #     # a=[print(x,"  \t",y,"\t",z) for x,y,z in zip(fl_rads, fl_lats, fl_lons)]
        #     first = False
            # 1/0


        # make_world_from_pfss.pm expects lat/lon in degrees (scales by pi/180 to get radians)
        prev_rad = 0.
        for j in np.arange(0, len(fl_lats)):
            # Output flnum, polarity, latitude (deg), longitude (deg), radius (R_sun)
            fl_open = np.append(fl_open, [[flnum_open, f_sgn[i], np.rad2deg(fl_lats[j]), np.rad2deg(fl_lons[j]), fl_rads[j]]], axis=0)
            prev_rad = fl_rads[j]
        flnum_open += 1
    else:
        fl = fl.coords.spherical
        fl_rads = (fl.distance/const.R_sun).value
        max_rad = fl_rads.max()
        rad_thresh = np.max([((max_rad-1.) / 5.), 0.01])
        prev_rad = 0.
        if np.mod(flnum_closed, 2) == 0:
            for j in np.arange(0, len(fl.lat)):
                # Output flnum, polarity, latitude (deg), longitude (deg), radius (R_sun)
                if (np.abs(fl_rads[j] - prev_rad) > rad_thresh) or (fl_rads[j] == max_rad):
                    fl_closed = np.append(fl_closed, [[flnum_closed, f_sgn[i], np.rad2deg(fl.lat[j].value),
                                    np.rad2deg(fl.lon[j].value), (fl.distance[j]/const.R_sun).value]], axis=0)
                    prev_rad = fl_rads[j]
        flnum_closed += 1
    return output, fl_open, fl_closed, flnum_open, flnum_closed, tracer


def _trace_each_single_core(coords, i, output, fl_open, fl_closed, flnum_open, flnum_closed, tracer, r0, f_sgn, adapt):
    """Core fieldline tracer (no timeout). Used by process-based timeout worker."""
    return _trace_each_single_impl(coords, i, output, fl_open, fl_closed, flnum_open, flnum_closed, tracer, r0, f_sgn, adapt)


@timeout_decorator.timeout(5)
def trace_each_single(coords, i, output, fl_open, fl_closed, flnum_open, flnum_closed, tracer, r0, f_sgn, adapt):
    """Fieldline tracer for each fieldline. Signal-based 5s timeout may not fire if trace runs in C/Fortran (GIL held)."""
    return _trace_each_single_impl(coords, i, output, fl_open, fl_closed, flnum_open, flnum_closed, tracer, r0, f_sgn, adapt)


import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import timeout_decorator
from functools import partial

# @timeout_decorator.timeout(5)
def trace_each_OLD(coords, r0, adapt, output):
    i, this_flon, this_flat, f_sgn = coords
    fl_open = []
    fl_closed = []
    flnum_closed = 0
    flnum_open = 0
    if adapt:
        this_flat = np.deg2rad(this_flat)
    coord_frame = output.coordinate_frame
    x0 = SkyCoord(this_flon * u.rad, np.arcsin(this_flat) * u.rad, r0, frame=coord_frame)
    tracer = _pfsspy_tracing.PythonTracer()

    fl = output.trace(tracer, x0)
    fl = fl.field_lines[0]
    if fl.is_open:
        fl_pol = fl.polarity
        fl = fl.coords.spherical
        fl_lats = fl.lat.value
        fl_lons = fl.lon.value
        fl_rads = (fl.distance / const.R_sun).value

        if fl_pol == -1:
            fl_lats, fl_lons, fl_rads = fl_lats[::-1], fl_lons[::-1], fl_rads[::-1]


        # Extend up to 21.5 R_sun
        extend_height = 22
        fl_lats = np.concatenate((fl_lats, fl_lats[-1]*np.ones(10)), axis=0)
        fl_lons = np.concatenate((fl_lons, fl_lons[-1]*np.ones(10)), axis=0)
        fl_rads = np.concatenate((fl_rads, np.linspace(2.5, extend_height, 10)), axis=0)

        prev_rad = 0.
        for j in np.arange(0, len(fl_lats)):
            if np.abs(fl_rads[j] - prev_rad) > 0.1:
                fl_open = np.append(
                    fl_open, [[flnum_open, f_sgn, fl_lats[j], fl_lons[j], fl_rads[j]]], axis=0)
                prev_rad = fl_rads[j]
        flnum_open += 1
    else:
        fl = fl.coords.spherical
        fl_rads = (fl.distance / const.R_sun).value
        # fl_pol = fl.polarity

        max_rad = fl_rads.max()
        rad_thresh = np.max([((max_rad - 1.) / 5.), 0.01])
        prev_rad = 0.
        if np.mod(flnum_closed, 2) == 0:
            for j in np.arange(0, len(fl.lat)):
                if (np.abs(fl_rads[j] - prev_rad) > rad_thresh) or (fl_rads[j] == max_rad):
                    fl_closed = np.append(fl_closed, [[flnum_closed, f_sgn, fl.lat[j].value,
                                                        fl.lon[j].value, (fl.distance[j] / const.R_sun).value]], axis=0)
                    prev_rad = fl_rads[j]
        flnum_closed += 1
    return fl_open, fl_closed, flnum_open, flnum_closed

def trace_all_OLD(field_lines):
    # i, this_flon, this_flat, f_sgn = coords
    fl_open = []
    fl_closed = []
    flnum_closed = 0
    flnum_open = 0
    # if adapt:
    #     this_flat = np.deg2rad(this_flat)
    # coord_frame = output.coordinate_frame
    # x0 = SkyCoord(this_flon * u.rad, np.arcsin(this_flat) * u.rad, r0, frame=coord_frame)
    # tracer = tracing.PythonTracer()

    for fl in field_lines:
        if fl.is_open:
            fl_pol = fl.polarity
            fl = fl.coords.spherical
            fl_lats = fl.lat.value
            fl_lons = fl.lon.value
            fl_rads = (fl.distance / const.R_sun).value

            if fl_pol == -1:
                fl_lats, fl_lons, fl_rads = fl_lats[::-1], fl_lons[::-1], fl_rads[::-1]

            fl_lats = np.concatenate((fl_lats, fl_lats[-1]*np.ones(10)), axis=0)
            fl_lons = np.concatenate((fl_lons, fl_lons[-1]*np.ones(10)), axis=0)
            fl_rads = np.concatenate((fl_rads, np.linspace(2.5, 22, 10)), axis=0)
            prev_rad = 0.
            for j in np.arange(0, len(fl_lats)):
                if np.abs(fl_rads[j] - prev_rad) > 0.1:
                    fl_open = np.append(
                        fl_open, [[flnum_open, fl_pol, fl_lats[j], fl_lons[j], fl_rads[j]]], axis=0)
                    prev_rad = fl_rads[j]
            flnum_open += 1
        else:
            fl = fl.coords.spherical
            fl_pol = 0
            fl_rads = (fl.distance / const.R_sun).value
            # fl_pol = fl.polarity

            max_rad = fl_rads.max()
            rad_thresh = np.max([((max_rad - 1.) / 5.), 0.01])
            prev_rad = 0.
            if np.mod(flnum_closed, 2) == 0:
                for j in np.arange(0, len(fl.lat)):
                    if (np.abs(fl_rads[j] - prev_rad) > rad_thresh) or (fl_rads[j] == max_rad):
                        fl_closed = np.append(fl_closed, [[flnum_closed, fl_pol, fl.lat[j].value,
                                                            fl.lon[j].value, (fl.distance[j] / const.R_sun).value]], axis=0)
                        prev_rad = fl_rads[j]
            flnum_closed += 1
    return fl_open, fl_closed, flnum_open, flnum_closed


def trace_lines_parallel_OLD(pfss_output, f_vars, open_path, closed_path, adapt):
    (f_lon, f_lat, f_sgn) = f_vars

    fl_open = []
    fl_closed = []
    flnum_open = 0
    flnum_closed = 0
    r0 = 1.01 * const.R_sun
    skip_num = 0
    timeout_num = 0


    seeds = SkyCoord(f_lon.ravel(), f_lat.ravel(), r0, frame=pfss_output.coordinate_frame)

    emit("\tTracing field lines...")
    tracer = _pfsspy_tracing.FortranTracer(max_steps=3000)
    field_lines = tracer.trace(seeds, pfss_output)
    emit("\tFinished tracing field lines")

    results = [] #trace_all(field_lines)
    # trace_each_partial = partial(trace_each, r0=r0, adapt=adapt, output=output)
    # ii = np.arange(len(f_lat))
    # with Pool() as pool:
    #     results = list(tqdm(pool.imap(trace_each_partial, zip(ii, f_lon, f_lat, f_sgn)),
    #                         desc="Tracing Field Lines", total=len(f_lat)))

    for result in results:
        fl_open.extend(result[0])
        fl_closed.extend(result[1])
        flnum_open += result[2]
        flnum_closed += result[3]

    if skip_num > 0 or timeout_num > 0:
        t_perc = 100 * timeout_num / len(f_lon)
        s_perc = 100 * skip_num / len(f_lon)
        emit(f"\n\t\tSome iterations failed. Timed-out: {timeout_num} ({t_perc:.2f}%), ValueError: {skip_num} ({s_perc:.2f}%)", style="yellow")

    emit(f"\n\tOpen: {flnum_open}, Closed: {flnum_closed}, Failures: {skip_num+timeout_num}")

    emit("\n\tSaving fluxons...", end="")
    np.savetxt(open_path, np.concatenate(fl_open))
    np.savetxt(closed_path, np.concatenate(fl_closed))
    emit(" Success!")

    return fl_open, fl_closed, skip_num, timeout_num, flnum_open, flnum_closed

