"""
Footpoint distances & coronal-hole boundary metrics
====================================================

This script computes and visualizes coronal-hole (CH) boundary geometry from PFSS maps,
treating longitude as periodic. It produces:
  * a great‑circle distance map (deg) to the nearest CH boundary,
  * red (morphed) vs white (raw) CH contours,
  * CSV/NPZ outputs for downstream analysis.

Usage:
    python footpoint_distances_wsa.py --cr 2220 --batch my_batch

Key functions:
    magnet_plot: orchestrates PFSS load/compute → morphology → distance → plots.

Author:
    Gilly <gilly@swri.org>

Dependencies:
    numpy, scipy.ndimage, matplotlib, sunpy, pfsspy, astropy, fluxpype.pipe_helper
"""

import os
import os.path as path
import argparse
import time

# import matplotlib as mpl; mpl.use("qt5agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    binary_closing,
    binary_fill_holes,
    generate_binary_structure,
    gaussian_filter,
)

# --- Morphology defaults (conservative) ---
_MORPH_DILATE_ITERS = 8  # light expansion to make boundaries more space-filling
_MORPH_ERODE_ITERS = 6  # no erosion by default
_MORPH_CLOSE_ITERS = 0  # close small gaps
_MORPH_OPEN_ITERS = 0  # do not open by default
_MORPH_MIN_SIZE = 16  # remove tiny speckles
_MORPH_SMOOTH_SIGMA = 0.5  # light Gaussian smoothing before thresholding

_REGION_ERODE = 8
_REGION_DIALATE = 6

# --- Polar cleanup (remove small purple islands near the poles) ---
_MORPH_POLAR_PURPLE_MIN_SIZE = 5000  # pixels; set 0 to disable
_MORPH_POLAR_ABS_LAT_DEG = 65.0      # operate where |lat| > this
_BOUND_POLAR_ABS_LAT_DEG =80.0      # operate where |lat| > this

# --- Morphology debug ---
_MORPH_DEBUG = True  # set True to save a panel of intermediate stages
_MORPH_DEBUG_SAVE = True  # save PNG next to your quicklook

# --- Distance map exclusion constants ---
_POLAR_CAP_EXCLUDE_ROWS = None  # rows excluded at poles in distance map
_PERIODIC_PADD_COLS = 20    # columns padded at periodic edges


def _morph_spacefill(boundary_mask: np.ndarray) -> np.ndarray:
    """
    Conservative morphological filtering to make boundaries more space-filling:
      1) optional light Gaussian smoothing then re-threshold
      2) closing (fill tiny gaps)
      3) dilation (expand one pixel)
      4) optional erosion (none by default)
      5) remove tiny components
    """
    B = boundary_mask.astype(bool)

    # 1) Light smoothing + threshold to clean salt-and-pepper
    if _MORPH_SMOOTH_SIGMA and _MORPH_SMOOTH_SIGMA > 0:
        sm = gaussian_filter(B.astype(float), sigma=_MORPH_SMOOTH_SIGMA)
        # threshold at 0.5 to get back to boolean
        B = sm >= 0.5

    # Use 4-neighborhood to preserve structure (no diagonals by default)
    st = generate_binary_structure(2, 1)

    # 2) Closing: fill tiny gaps
    for _ in range(int(_MORPH_CLOSE_ITERS)):
        B = binary_closing(B, st)

    # 3) Dilation: expand boundary slightly to be more space-filling
    for _ in range(int(_MORPH_DILATE_ITERS)):
        B = binary_dilation(B, st)

    # 4) Optional erosion
    for _ in range(int(_MORPH_ERODE_ITERS)):
        B = binary_erosion(B, st)

    # 5) Remove small speckles to keep contours clean
    if _MORPH_MIN_SIZE and _MORPH_MIN_SIZE > 0:
        lbl, n = ndimage.label(B)
        if n > 0:
            # keep only components with size >= MIN_SIZE
            counts = np.bincount(lbl.ravel())
            keep = counts >= _MORPH_MIN_SIZE
            keep[0] = False  # background
            B = keep[lbl]

    return B


# --- Helper: periodic padding and cropping for longitude ---
def _pad_periodic(arr: np.ndarray, pad_cols: int = _PERIODIC_PADD_COLS) -> np.ndarray:
    """Pad the array periodically in longitude."""
    if pad_cols and pad_cols > 0:
        return np.pad(arr, ((0, 0), (pad_cols, pad_cols)), mode="wrap")
    return arr


def _crop_periodic(arr: np.ndarray, pad_cols: int = _PERIODIC_PADD_COLS) -> np.ndarray:
    """Crop a previously periodic-padded array back to original shape."""
    if pad_cols and pad_cols > 0:
        return arr[:, pad_cols:-pad_cols]
    return arr


from matplotlib.colors import LinearSegmentedColormap

# Custom colormap: transparent to red
red_transparent_cmap = LinearSegmentedColormap.from_list(
    "red_transparent", [(0, (1, 0, 0, 0)), (1, (1, 0, 0, 0.25))], N=256
)

# --- Helper: single-pixel outline from thick boundary mask ---
def _singleline_outline(boundary_mask: np.ndarray) -> np.ndarray:
    """
    Build a continuous, one-pixel outline around the region indicated by a boundary mask.
    Modified so that the outline corresponds exactly to the geometry used for the distance map
    (post-morphology, pre–hole-fill), matching what the KDTree "sees".
    Steps:
      1) Expand + close using 8-neighborhood to connect diagonals.
      2) (Skip fill_holes, to avoid interior edges not present in the KDTree mask.)
      3) Remove tiny islands.
      4) Outline = boundary_mask & ~binary_erosion(boundary_mask, st8)
    """
    B = boundary_mask.astype(bool)

    # 8-neighborhood to prefer connectivity for the visual outline
    st8 = generate_binary_structure(2, 2)

    # 1) Connect diagonals and small gaps more aggressively than _morph_spacefill
    R = B.copy()
    R = binary_dilation(R, st8)  # one light dilation
    R = binary_closing(R, st8)  # close tiny gaps

    # 2) Fill interior to avoid "red foam"
    # R = binary_fill_holes(R)  # <-- REMOVE this step for consistency with KDTree mask

    # 3) Remove tiny islands for cleanliness
    if _MORPH_MIN_SIZE and _MORPH_MIN_SIZE > 0:
        lbl, n = ndimage.label(R)
        if n > 0:
            counts = np.bincount(lbl.ravel())
            keep = counts >= _MORPH_MIN_SIZE
            keep[0] = False
            R = keep[lbl]

    # 4) Ensure outline matches exactly the KDTree boundary mask:
    #    remove interior edges created by fill_holes, use boundary_mask itself.
    outline = boundary_mask.astype(bool) & ~binary_erosion(boundary_mask.astype(bool), st8)
    return outline


# --- Helper: reconstruct filled region from boundary mask for single-line contouring ---
def _region_from_boundary(boundary_mask: np.ndarray) -> np.ndarray:
    """
    Reconstruct a filled region from a thin/foamy boundary mask for clean contour plotting.
    Uses 8-neighborhood dilation + closing (iterated using _MORPH_DILATE_ITERS and _MORPH_CLOSE_ITERS),
    then fills holes and removes tiny islands. Contouring this region at level 0.5 yields a single closed
    curve per connected component.
    """
    B = boundary_mask.astype(bool)
    st8 = generate_binary_structure(2, 2)

    # Controlled dilation and closing for connectivity using global parameters
    R = B.copy()
    for _ in range(int(max(1, _MORPH_DILATE_ITERS))):
        R = binary_dilation(R, st8)
    for _ in range(int(max(1, _MORPH_CLOSE_ITERS))):
        R = binary_closing(R, st8)

    # Fill interiors and clean up
    R = binary_fill_holes(R)
    if _MORPH_MIN_SIZE and _MORPH_MIN_SIZE > 0:
        lbl, n = ndimage.label(R)
        if n > 0:
            counts = np.bincount(lbl.ravel())
            keep = counts >= _MORPH_MIN_SIZE
            keep[0] = False
            R = keep[lbl]

    return R


# --- Helper: mask horizontal boundary lines confined to polar caps, preserving vertical peninsulas ---
def _mask_polar_band_interior(boundary_mask: np.ndarray, lat_1d: np.ndarray,
                              lat_cut_deg: float = _BOUND_POLAR_ABS_LAT_DEG) -> np.ndarray:
    """Remove horizontal boundary lines confined to polar caps, preserving vertical peninsulas."""
    B = boundary_mask.copy()
    lat_cut = np.deg2rad(lat_cut_deg)
    polar_rows = np.abs(lat_1d) > lat_cut
    st8 = generate_binary_structure(2, 2)
    lbl, n = ndimage.label(B, structure=st8)
    for i in range(1, n + 1):
        coords = np.argwhere(lbl == i)
        lat_inds = coords[:, 0]
        # remove only components entirely within polar band
        if np.all(polar_rows[lat_inds]):
            B[lat_inds, coords[:, 1]] = False
            print("\n---------Removing Band!--------\n")
    return B

# --- Helper: treat all pixels above |lat| > lat_cut_deg as open field (True) ---
def _treat_poles_as_open_field(region_mask: np.ndarray, lat_1d: np.ndarray,
                               lat_cut_deg: float = _BOUND_POLAR_ABS_LAT_DEG) -> np.ndarray:
    """Force all pixels above |lat| > lat_cut_deg to open field (True)."""
    R = region_mask.copy()
    lat_cut = np.deg2rad(lat_cut_deg)
    polar_rows = np.abs(lat_1d) > lat_cut
    if np.any(polar_rows):
        R[polar_rows, :] = True
    return R


# --- Helper: remove small purple islands (False) at high latitudes only ---
def _clean_polar_islands(
    region_mask: np.ndarray,
    lat_1d: np.ndarray,
    min_size: int = _MORPH_POLAR_PURPLE_MIN_SIZE,
    lat_cut_deg: float = _MORPH_POLAR_ABS_LAT_DEG,
) -> np.ndarray:
    """
    Flip small purple (False) components to yellow (True) **only** in polar caps
    where |lat| > lat_cut_deg. Uses 8-connectivity. No-op if min_size <= 0.

    Parameters
    ----------
    region_mask : bool array (n_lat, n_lon)
        True = yellow region, False = purple.
    lat_1d : 1D array of latitudes in radians (size n_lat)
    min_size : int
        Minimum size in pixels for purple components to keep. Smaller ones are filled.
    lat_cut_deg : float
        Operate only on rows with |lat| > lat_cut_deg.
    """
    if min_size is None or min_size <= 0:
        return region_mask

    R = region_mask.copy()
    # Identify polar rows
    lat_cut = np.deg2rad(lat_cut_deg)
    polar_rows = np.abs(lat_1d) > lat_cut
    if not np.any(polar_rows):
        return R

    # Work on a local copy where non-polar rows are set to True so components
    # do not leak across the cap boundary.
    local = R.copy()
    local[~polar_rows, :] = True

    # Label purple (False) components with 8-connectivity
    st8 = generate_binary_structure(2, 2)
    lbl, n = ndimage.label(~local, structure=st8)
    if n == 0:
        return R

    counts = np.bincount(lbl.ravel())
    kill = counts < int(min_size)
    kill[0] = False  # background
    small_false = kill[lbl]

    # Apply only within polar rows
    to_flip = small_false & polar_rows[:, None]
    R[to_flip] = True
    return R


def _debug_morph_steps(
    boundary_mask_raw: np.ndarray,
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
    savepath: str | None = None,
    region_mask_morph: np.ndarray | None = None,
    region_mask_morph_pre_polar: np.ndarray | None = None,
    polar_cap_exclude_rows: int = _POLAR_CAP_EXCLUDE_ROWS,
    periodic_exclude_cols: int = _PERIODIC_PADD_COLS,
) -> None:
    """
    Save an 8-panel figure showing the morphological stages:
    raw, smoothed, closed, dilated, eroded, cleaned, region(raw), region(morphed)
    Now also overlays excluded polar rows and periodic longitude columns.
    """
    B0 = boundary_mask_raw.astype(bool)

    # 1) smoothed
    if _MORPH_SMOOTH_SIGMA and _MORPH_SMOOTH_SIGMA > 0:
        sm = gaussian_filter(B0.astype(float), sigma=_MORPH_SMOOTH_SIGMA)
        B1 = sm >= 0.5
    else:
        B1 = B0.copy()

    st4 = generate_binary_structure(2, 1)

    # 2) closed
    B2 = B1.copy()
    for _ in range(int(_MORPH_CLOSE_ITERS)):
        B2 = binary_closing(B2, st4)

    # 3) dilated
    B3 = B2.copy()
    for _ in range(int(_MORPH_DILATE_ITERS)):
        B3 = binary_dilation(B3, st4)

    # 4) eroded
    B4 = B3.copy()
    for _ in range(int(_MORPH_ERODE_ITERS)):
        B4 = binary_erosion(B4, st4)

    # 5) cleaned
    B5 = B4.copy()
    if _MORPH_MIN_SIZE and _MORPH_MIN_SIZE > 0:
        lbl, n = ndimage.label(B5)
        if n > 0:
            counts = np.bincount(lbl.ravel())
            keep = counts >= _MORPH_MIN_SIZE
            keep[0] = False
            B5 = keep[lbl]

    region_raw = _region_from_boundary(B0)
    region_morph = _region_from_boundary(B5)
    if region_mask_morph is not None:
        region_morph = region_mask_morph.astype(bool)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = [
        ("Raw", B0),
        (f"Smoothed σ={_MORPH_SMOOTH_SIGMA}", B1),
        (f"Closed ×{_MORPH_CLOSE_ITERS}", B2),
        (f"Dilated ×{_MORPH_DILATE_ITERS}", B3),
        (f"Eroded ×{_MORPH_ERODE_ITERS}", B4),
        (f"Cleaned ≥{_MORPH_MIN_SIZE}", B5),
        ("Region (raw)", region_raw),
        ("Region (morphed)", region_morph),
    ]
    extent = (lon_1d[0], lon_1d[-1], np.sin(lat_1d[0]), np.sin(lat_1d[-1]))
    for ax, (title, M) in zip(axes.ravel(), panels):
        im = ax.imshow(M.astype(int), origin="lower", interpolation="nearest", extent=extent, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("lon (rad)")
        ax.set_ylabel("sin(lat)")

    # Overlay polar cleanup changes if provided
    if region_mask_morph is not None and region_mask_morph_pre_polar is not None:
        try:
            flipped = region_mask_morph.astype(bool) & (~region_mask_morph_pre_polar.astype(bool))
            lat_cut = np.deg2rad(_MORPH_POLAR_ABS_LAT_DEG)
            polar_rows = np.abs(lat_1d) > lat_cut
            flipped &= polar_rows[:, None]
            if np.any(flipped):
                ax_morph = axes.ravel()[-1]
                ax_morph.imshow(flipped, aspect="auto", origin="lower", interpolation="nearest",
                                extent=extent, alpha=0.45, cmap="Reds")
                ax_morph.set_title("Region (morphed) + polar cleanup Δ")
        except Exception:
            pass


    if savepath and _MORPH_DEBUG_SAVE:
        plt.savefig(savepath, dpi=200)
    plt.close(fig)


def _boundary_mask_from_pols(P: np.ndarray) -> np.ndarray:
    """
    Build a boundary mask for coronal-hole interfaces using periodic longitude.

    We consider an interface wherever a zero-valued pixel has at least one non-zero
    neighbor. Longitude (axis=1) is treated as periodic via np.roll; latitude (axis=0)
    is *not* periodic and uses edge padding.
    """
    # Identify zero (CH) locations
    is_zero = (P == 0)

    # Longitude neighbors: periodic wrap
    nb_left  = is_zero & (np.roll(P,  1, axis=1) != 0)
    nb_right = is_zero & (np.roll(P, -1, axis=1) != 0)

    # Latitude neighbors: non-periodic (use edge padding)
    P_pad = np.pad(P, ((1, 1), (0, 0)), mode="edge")
    nb_up    = is_zero & (P_pad[:-2, :] != 0)
    nb_down  = is_zero & (P_pad[ 2:, :] != 0)

    boundary_mask = nb_up | nb_down | nb_left | nb_right

    return boundary_mask


def _smooth_regions(B,erode=2, dialate=2):
    # Use 8-neighborhood to preserve structure
    st = generate_binary_structure(2, 2)

    for _ in range(int(erode)):
        B = binary_erosion(B, st)
        # print("Eroding...")

    for _ in range(int(dialate)):
        B = binary_dilation(B, st)
        # print("Dialating...")

    return B


def _gc_distance_map_to_boundary(
    lon_1d: np.ndarray,
    lat_1d: np.ndarray,
    boundary_mask: np.ndarray,
    # polar_cap_exclude_rows: int = _POLAR_CAP_EXCLUDE_ROWS,
    periodic_exclude_cols: int = _PERIODIC_PADD_COLS,
) -> np.ndarray:
    """
    Return a (n_lat, n_lon) array of great-circle distances [deg] from each pixel
    (lon_1d, lat_1d grid) to the nearest TRUE pixel in `boundary_mask`.

    Parameters
    ----------
    lon_1d : 1D array (radians)
    lat_1d : 1D array (radians)
    boundary_mask : 2D bool array
        Mask of boundary pixels, shape (n_lat, n_lon). True indicates a boundary pixel.
    # polar_cap_exclude_rows : int, optional
    #     Number of *rows at each pole* (top and bottom) to blank out in `boundary_mask`
    #     before building the KDTree. This treats the north/south poles as poles (not edges)
    #     by preventing spurious boundary pixels at the map edges. Default is 4.
    periodic_exclude_cols: int, optional
        Number of columns at each longitude edge to blank out in `boundary_mask`.

    Notes
    -----
    * Distances are computed on the unit sphere using a KDTree in 3D:
        angle = 2 * arcsin( ||x - x_bnd|| / 2 ), in degrees.
    """
    # Treat the north/south poles as poles (not edges) by removing boundary pixels
    # from high-latitude rows (|lat| > threshold) before distance queries.
    BM = boundary_mask.astype(bool).copy()
    r2 = int(periodic_exclude_cols) if periodic_exclude_cols is not None else 0
    if r2 > 0:
        BM[:, :r2] = False
        BM[:, -r2:] = False

    # Treat poles as open field in distance calculation as well
    if _BOUND_POLAR_ABS_LAT_DEG and _BOUND_POLAR_ABS_LAT_DEG > 0:
        lat_cut = np.deg2rad(_BOUND_POLAR_ABS_LAT_DEG)
        polar_rows = np.abs(lat_1d) > lat_cut
        BM[polar_rows, :] = False

    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing="xy")  # (n_lat, n_lon)

    b_lon = lon_grid[BM]
    b_lat = lat_grid[BM]

    if b_lon.size == 0:
        return np.full((lat_1d.size, lon_1d.size), np.nan, dtype=float)

    def _sph2cart(lon, lat):
        cl = np.cos(lat)
        return np.column_stack((cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)))

    from scipy.spatial import cKDTree

    b_xyz = _sph2cart(b_lon, b_lat)
    tree = cKDTree(b_xyz)

    all_xyz = _sph2cart(lon_grid.ravel(), lat_grid.ravel())
    d_chord, _ = tree.query(all_xyz, k=1, workers=-1)

    angles_deg = 2.0 * np.arcsin(np.clip(d_chord * 0.5, 0.0, 1.0)) * (180.0 / np.pi)
    return angles_deg.reshape(lat_1d.size, lon_1d.size)


from fluxpype.pipe_helper import configurations, load_fits_magnetogram, load_magnetogram_params, shorten_path


def magnet_plot(
    get_cr=None,
    datdir=None,
    _batch=None,
    open_f=None,
    closed_f=None,
    force_pfss=False,
    reduce_amt=0,
    nact=0,
    nwant=None,
    do_print_top=True,
    ax=None,
    verb=True,
    ext="png",
    plot_all=True,
    plot_open=True,
    do_print=False,
    vmin=-500,
    vmax=500,
    configs=None,
    legend=False,
):
    """This function has been re-imagined into a

    Parameters
    ----------
    get_cr : int
        the carrington rotation number


    Returns
    -------

    """
    if True:
        print("\t\t(py) Determining Footpoint Distances")

    if configs is not None:
        datdir = datdir or configs.get("data_dir", None)
        _batch = _batch or configs.get("batch_name", None)
        get_cr = get_cr or configs.get("cr", None)
        nwant = nwant or int(configs.get("fluxon_count", None)[0])
        reduce_amt = reduce_amt or configs.get("mag_reduce", None)
        if configs.get("adapt", False):
            inst = "adapt"
            reduce_amt = "f" + str(configs.get("adapt_select"))
        else:
            inst = "hmi"
    else:
        print("No configs given!")
        raise ValueError

    # Define the directory paths for the files
    floc_path = f"{datdir}/batches/{_batch}/data/cr{get_cr}/floc/"
    top_dir = f"{datdir}/batches/{_batch}/imgs/footpoints/"
    if not path.exists(top_dir):
        os.makedirs(top_dir)

    magnet_file = f"{datdir}/magnetograms/CR{get_cr}_r{reduce_amt}_{inst}.fits"
    fname = magnet_file
    # Load the magnetogram (always available for PFSS path)
    magnet, header = load_fits_magnetogram(batch=_batch, ret_all=True, configs=configs, fname=fname)
    crs = [get_cr]
    # roll_cols = 0  # default; only set nonzero when rolling SunPy map for PFSS compute path
    nsteps = 360

    # ofmap = np.zeros((nsteps, len(crs)))
    # efmap = np.zeros((nsteps, len(crs)))

    # with tqdm(total=len(crs)) as pbar:
    for i, cr in enumerate(crs):
        output_file = floc_path + f"pfss_ofmap_cr{cr}.npz"
        if os.path.exists(output_file) and not force_pfss:
            pols, expfs, lon_1d, lat_1d = _load_pfss_results(output_file)
        else:
            pols, expfs, lon_1d, lat_1d = _compute_pfss_results(magnet_file, floc_path, get_cr, nsteps, do_print_top)
        _process_pfss_results(pols, lon_1d, lat_1d, floc_path, top_dir, do_print_top, cr=get_cr)

    if do_print_top:
        print("\t\t    Success!\n\t\t\t```````````````````````````````\n")
    return 0, 0, 0, 0, 0


# --- Helper: load PFSS results from .npz file ---
def _load_pfss_results(output_file):
    data = np.load(output_file)
    pols = data["ofmap"]
    expfs = data["efmap"]
    lon_1d = data["lon"]
    lat_1d = data["lat"]
    print("LOADED POLS, EXPFS")
    return pols, expfs, lon_1d, lat_1d


# --- Helper: compute PFSS results and return pols, expfs, lon_1d, lat_1d ---
def _compute_pfss_results(magnet_file, floc_path, get_cr, nsteps, do_print_top):
    import sunpy
    from astropy import units as u, constants as const
    from astropy.coordinates import SkyCoord
    import pfsspy
    from pfsspy import tracing
    from fluxpype.pipe_helper import shorten_path
    cr = get_cr
    output_file = floc_path + f"pfss_ofmap_cr{cr}.npz"
    if do_print_top:
        print("\t[pfss] Preparing HMI map and resampling...", flush=True)
    hmi_map = sunpy.map.Map(magnet_file)
    roll_cols = int(hmi_map.data.shape[1] * (9 / 3) / (2 * np.pi))
    hmi_map = sunpy.map.Map(np.roll(hmi_map.data, shift=roll_cols, axis=1), hmi_map.meta)
    print(f"[diag] Rolled SunPy map data by {roll_cols} columns (~2/3π radians).")
    if do_print_top:
        print(f"\t[pfss] Loaded {shorten_path(magnet_file)}; resampling to {(2 * nsteps)}x{nsteps} pixels...", flush=True)
    hmi_map = hmi_map.resample([2 * nsteps, nsteps] * u.pix)
    if do_print_top:
        print("\t[pfss] Building PFSS input and running potential-field solver (pfsspy.pfss)... this can take a bit.", flush=True)
    nrho = 40
    rss = 2.5
    pfss_in = pfsspy.Input(hmi_map, nrho, rss)
    if do_print_top:
        print(f"\t[pfss] Input ready (nrho={nrho}, rss={rss}); launching solver...", flush=True)
    pfss_out = pfsspy.pfss(pfss_in)
    if do_print_top:
        print("\t[pfss] PFSS solution computed. Building seed grid and tracing field lines...", flush=True)
    r = const.R_sun
    lon_1d = np.linspace(0, 2 * np.pi, nsteps * 2)
    lat_1d = np.arcsin(np.linspace(-0.999, 0.999, nsteps))
    lon, lat = np.meshgrid(lon_1d, lat_1d, indexing="ij")
    lon, lat = lon * u.rad, lat * u.rad
    if do_print_top:
        print("\t[pfss] Seed grid defined on lon/lat; constructing SkyCoord seeds...", flush=True)
    seeds = SkyCoord(lon.ravel(), lat.ravel(), r, frame=pfss_out.coordinate_frame)
    if do_print_top:
        seed_n = (2 * nsteps) * nsteps
        print(f"\t[pfss] Using FortranTracer; tracing a {2*nsteps}x{nsteps} seed grid ({seed_n} field lines)... this can be slow.", flush=True)
    tracer = tracing.FortranTracer(max_steps=2500)
    field_lines = tracer.trace(seeds, pfss_out)
    if do_print_top:
        print("\t[pfss] Tracing complete. Reshaping polarities and expansion factors...", flush=True)
    pols = field_lines.polarities.reshape(2 * nsteps, nsteps).T
    expfs = field_lines.expansion_factors.reshape(2 * nsteps, nsteps).T
    expfs[np.where(np.isnan(expfs))] = 0
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    np.savez_compressed(output_file, ofmap=pols, efmap=expfs, brmap=hmi_map.data, lon=lon_1d, lat=lat_1d)
    if do_print_top:
        print(f"\t[pfss] Saved PFSS maps to {shorten_path(output_file)}", flush=True)
    return pols, expfs, lon_1d, lat_1d


# --- Helper: process PFSS results and perform morphology, distance, and plotting ---
def _process_pfss_results(pols, lon_1d, lat_1d, floc_path, top_dir, do_print_top, cr: int | None = None):
    from fluxpype.pipe_helper import shorten_path
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    # pols shape is (nsteps, 2*nsteps); build matching lon/lat grids
    # Identify boundary pixels: ONLY 0 ↔ non-zero (±1) interfaces
    P = pols
    boundary_mask = _boundary_mask_from_pols(P)
    boundary_mask_raw = boundary_mask.copy()
    # Pad periodically before morph
    boundary_mask = _pad_periodic(boundary_mask)
    boundary_mask = _morph_spacefill(boundary_mask)
    boundary_mask = _crop_periodic(boundary_mask)
    boundary_mask = _mask_polar_band_interior(boundary_mask, lat_1d)
    region_mask_raw = _region_from_boundary(boundary_mask_raw)

    # Apply periodic padding and cropping during region reconstruction for continuity in red contour
    boundary_mask_padded = _pad_periodic(boundary_mask)
    region_mask_morph = _region_from_boundary(boundary_mask_padded)
    region_mask_morph_pre_polar = region_mask_morph.copy()
    # Treat all pixels above |lat| > lat_cut_deg as open field (True)
    region_mask_morph = _smooth_regions(region_mask_morph, erode=_REGION_ERODE, dialate=_REGION_DIALATE)
    region_mask_morph = _treat_poles_as_open_field(region_mask_morph, lat_1d)
    # Polar cleanup: remove small purple islands at |lat| > threshold
    region_mask_morph = _clean_polar_islands(region_mask_morph, lat_1d)
    # Build a clean 1-pixel perimeter for distance queries
    region_mask_morph = _crop_periodic(region_mask_morph)
    outline_mask = _singleline_outline(region_mask_morph)
    if _MORPH_DEBUG:
        dbg_path = top_dir + "morph_debug.png"
        _debug_morph_steps(
            boundary_mask_raw,
            lon_1d,
            lat_1d,
            dbg_path,
            region_mask_morph=region_mask_morph,
            region_mask_morph_pre_polar=region_mask_morph_pre_polar,
            polar_cap_exclude_rows=_POLAR_CAP_EXCLUDE_ROWS,
            periodic_exclude_cols=_PERIODIC_PADD_COLS,
        )
    # Distance from every pixel to the nearest boundary pixel (deg)
    ch_distance_deg = _gc_distance_map_to_boundary(lon_1d, lat_1d, outline_mask)
    # Simple diagnostics: report interior vs exterior distance percentiles
    try:
        interior = region_mask_morph  # filled CH regions (zeros)
        exterior = ~region_mask_morph
        intr_pct = np.nanpercentile(ch_distance_deg[interior], [5, 50, 95])
        extr_pct = np.nanpercentile(ch_distance_deg[exterior], [5, 50, 95])
        print(f"[diag] interior deg p5/50/95: {intr_pct}")
        print(f"[diag] exterior deg p5/50/95: {extr_pct}")
    except Exception as _e:
        print(f"[diag] percentile check skipped: {_e}")
    # Persist results alongside lon/lat for later reuse/interpolation
    ch_out_npz = floc_path + f"pfss_distances.npz"
    np.savez_compressed(
        ch_out_npz, ch_distance_deg=ch_distance_deg, lon=lon_1d, lat=lat_1d, pols=P.astype(np.int8)
    )
    ch_out_csv = floc_path + f"pfss_distances.csv"
    np.savetxt(ch_out_csv, ch_distance_deg, delimiter=", ")
    # Back-compat: preserve old downstream expectation of floc/distances.csv
    compat_csv = os.path.join(floc_path, "distances.csv")
    np.savetxt(compat_csv, ch_distance_deg, delimiter=", ")
    if do_print_top:
        print(f"\t\tWrote CSV: {shorten_path(compat_csv)}")
    from scipy import interpolate as _interp
    ch_distance_interp = _interp.RectBivariateSpline(lat_1d, lon_1d, ch_distance_deg)
    # Plotting
    fig_ch, ax_ch = plt.subplots(figsize=(2 * 6.4, 2 * 4.8))
    im = ax_ch.imshow(
        ch_distance_deg,
        cmap="viridis",
        origin="lower",
        interpolation="none",
        extent=(lon_1d[0], lon_1d[-1], np.sin(lat_1d[0]), np.sin(lat_1d[-1])),
        aspect="auto",
    )
    # sanity contours to check symmetry of distances inside vs outside
    ax_ch.contour(lon_1d, np.sin(lat_1d), ch_distance_deg, levels=[5, 10, 20, 30], colors='k', linewidths=0.6, alpha=0.4)
    im.figure.colorbar(im, ax=ax_ch, label="Distance to CH Boundary (deg)")
    title = f"Distance to Nearest Coronal-Hole Boundary for CR {cr}" if cr is not None else "Distance to Nearest Coronal-Hole Boundary"
    ax_ch.set_title(title)
    cs_raw = ax_ch.contour(
        lon_1d,
        np.sin(lat_1d),
        region_mask_raw.astype(int),
        levels=[0.5],
        colors="white",
        linewidths=0.5,
        linestyles="dashed",
    )
    cs_morph = ax_ch.contour(
        lon_1d,
        np.sin(lat_1d),
        region_mask_morph.astype(int),
        levels=[0.5],
        colors="red",
        linewidths=1.5,
    )
    # Overlay closed-field mask after red contour, before legend
    closed_field_mask = ~region_mask_morph
    ax_ch.imshow(
        closed_field_mask,
        extent=(lon_1d[0], lon_1d[-1], np.sin(lat_1d[0]), np.sin(lat_1d[-1])),
        origin="lower",
        cmap=red_transparent_cmap,
        alpha=1.0,
        interpolation="none",
        aspect="auto",
    )
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="white", lw=0.5, ls="--", label="raw"),
        Line2D([0], [0], color="red", lw=1.5, ls="-", label="morphed"),
    ]
    ax_ch.legend(handles=handles, loc="upper right", fontsize="small", framealpha=0.6)
    fname_base = f"distances_{cr}" if cr is not None else "distances"

    plt.ylabel("Solar Latitude in sin(Radians)")
    plt.xlabel("Solar Longitude in Radians")

    plt.tight_layout()
    plt.savefig(top_dir + f"{fname_base}.png", dpi=300)
    plt.savefig(top_dir + f"t_{fname_base}_{time.time():0.0f}.png", dpi=300)
    plt.show()


########################################################################
# Main Code
# ----------------------------------------------------------------------
#

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="This script determines coronal hole boundary distances from a magnetogram")
    parser.add_argument("--cr", type=int, default=2230, help="Carrington Rotation")
    args = parser.parse_args()
    configs = configurations(debug=False, args=args)

    magnet_plot(configs=configs)
