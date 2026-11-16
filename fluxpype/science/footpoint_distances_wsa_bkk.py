"""
Changed
==========================================================

This script is designed to plot the magnetogram along with footpoints. It provides
options to specify the Carrington Rotation (CR), batch name, reduction factor, data directory,
and other parameters.

Usage:
    python plot_fieldmap.py [--cr CARRINGTON_ROTATION] [--nwant NUMBER_WANTED]
                             [--open OPEN_FLUXONS] [--closed CLOSED_FLUXONS]
                             [--dat_dir DATA_DIRECTORY] [--batch BATCH_NAME]

Arguments:
    --cr:           The Carrington Rotation for which the magnetogram is to be plotted. Default is None.
    --nwant:        The number of fluxons wanted. Default is None.
    --open:         The number of open fluxons. Default is None.
    --closed:       The number of closed fluxons. Default is None.
    --dat_dir:      The directory where the data will be stored. Default is defined in the config.ini file.
    --batch:        The batch name for the operation. Default is 'default_batch'.

Functions:
    magnet_plot:    A primary function for plotting the magnetogram with footpoints.

Example:
    python plot_fieldmap.py --cr 2220 --nwant 100 --open 50 --closed 50 --dat_dir '/path/to/data' --batch 'my_batch'

Author:
    Gilly <gilly@swri.org> (and others!)

Dependencies:
    os, os.path, argparse, matplotlib.pyplot, numpy, pfss_funcs, pipe_helper
"""

import os
import os.path as path
import argparse

# import matplotlib as mpl; mpl.use("qt5agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    binary_closing,
    binary_fill_holes,
    generate_binary_structure,
    gaussian_filter,
    label,
)

# --- Morphology defaults (conservative) ---
_MORPH_DILATE_ITERS = 8  # light expansion to make boundaries more space-filling
_MORPH_ERODE_ITERS = 6  # no erosion by default
_MORPH_CLOSE_ITERS = 0  # close small gaps
_MORPH_OPEN_ITERS = 0  # do not open by default
_MORPH_MIN_SIZE = 16  # remove tiny speckles
_MORPH_SMOOTH_SIGMA = 0.4  # light Gaussian smoothing before thresholding

# --- Morphology debug ---
_MORPH_DEBUG = True  # set True to save a panel of intermediate stages
_MORPH_DEBUG_SAVE = True  # save PNG next to your quicklook


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
        lbl, n = label(B)
        if n > 0:
            # keep only components with size >= MIN_SIZE
            counts = np.bincount(lbl.ravel())
            keep = counts >= _MORPH_MIN_SIZE
            keep[0] = False  # background
            B = keep[lbl]

    return B


# --- Helper: single-pixel outline from thick boundary mask ---
def _singleline_outline(boundary_mask: np.ndarray) -> np.ndarray:
    """
    Build a continuous, one-pixel outline around the region indicated by a boundary mask.
    Steps:
      1) Expand + close using 8-neighborhood to connect diagonals.
      2) Fill interior holes to get a solid region.
      3) Remove tiny islands.
      4) Outline = region XOR erode(region)  -> ~1px perimeter.
    """
    B = boundary_mask.astype(bool)

    # 8-neighborhood to prefer connectivity for the visual outline
    st8 = generate_binary_structure(2, 2)

    # 1) Connect diagonals and small gaps more aggressively than _morph_spacefill
    R = B.copy()
    R = binary_dilation(R, st8)  # one light dilation
    R = binary_closing(R, st8)  # close tiny gaps

    # 2) Fill interior to avoid "red foam"
    R = binary_fill_holes(R)

    # 3) Remove tiny islands for cleanliness
    if _MORPH_MIN_SIZE and _MORPH_MIN_SIZE > 0:
        lbl, n = label(R)
        if n > 0:
            counts = np.bincount(lbl.ravel())
            keep = counts >= _MORPH_MIN_SIZE
            keep[0] = False
            R = keep[lbl]

    # 4) 1-pixel perimeter via morphological gradient (region minus eroded region)
    er = binary_erosion(R, st8)
    outline = R & ~er
    return outline


# --- Helper: outline for distance queries, omitting polar map-edge rims ---
def _outline_for_distance(region_mask: np.ndarray,
                          lat_1d: np.ndarray) -> np.ndarray:
    """
    Return a 1‑px perimeter to use for distance queries that
    *excludes* artificial rims along the polar map edges.

    Rationale
    ---------
    On a sphere, the pole is the *center* of a polar coronal hole (PCH),
    not a boundary. When a PCH touches the top/bottom row of the map grid,
    a naive perimeter contains a spurious ring along that edge, which
    erroneously makes the distance-to-boundary *small* at the pole.

    We remove any outline pixels on the first/last latitude rows so that
    distance increases monotonically towards the pole inside a PCH.
    """
    outline = _singleline_outline(region_mask).copy()
    # Drop artificial boundary pixels on the polar rows
    if outline.shape[0] >= 2:
        outline[0, :] = False
        outline[-1, :] = False
    return outline


# --- Helper: clean mask of open (pols == 0) regions from PFSS topology ---
def _open_region_mask_from_pols(P: np.ndarray,
                                min_size: int = 16,
                                close_iters: int = 1) -> np.ndarray:
    """
    Build a clean boolean mask of OPEN (pols == 0) regions using only the topology
    in `P`. Lightly close tiny gaps (8-connected) to seal cracks, then drop tiny islands.
    """
    st8 = generate_binary_structure(2, 2)
    R = (P == 0).astype(bool)
    for _ in range(int(max(0, close_iters))):
        R = binary_closing(R, st8)
    if min_size and min_size > 0:
        lbl, n = label(R, st8)
        if n > 0:
            counts = np.bincount(lbl.ravel())
            keep = counts >= min_size
            keep[0] = False
            R = keep[lbl]
    return R


# --- Helper: extract 1-px interface outline directly from pols (open/closed interface) ---

def _interface_outline_from_pols(P: np.ndarray) -> np.ndarray:
    """
    Return a 1-pixel outline that lies exactly on the open/closed interface.
    Robust to polar caps: the outline follows the *equatorward* rim of polar holes
    and excludes spurious rims on the map border.

    Steps
    -----
    1) Build open/closed masks from `P`.
    2) Take the morphological gradient of the open mask (~1px ring).
    3) Keep only pixels that are adjacent to CLOSED (true interface).
    4) Drop first/last latitude rows so poles cannot act like boundaries.
    """
    st8 = generate_binary_structure(2, 2)
    open_mask = (P == 0)
    closed_mask = ~open_mask

    # 2) morphological gradient (open minus eroded-open) -> ~1 px ring
    grad = open_mask & ~binary_erosion(open_mask, st8)

    # 3) true interface: only where that ring touches CLOSED when dilated
    adj_closed = binary_dilation(closed_mask, st8)
    outline = grad & adj_closed

    # 4) never let polar rows serve as boundaries
    if outline.shape[0] >= 2:
        outline[0, :] = False
        outline[-1, :] = False

    return outline


# --- Helper: interface outline treating polar caps as interiors and ignoring small closed islands inside them ---
def _interface_outline_polar_aware(P: np.ndarray) -> np.ndarray:
    """
    Interface outline that:
      * Uses open/closed topology from `P`
      * Treats polar-cap components (touching first/last latitude rows) as interiors
        and fills their interior holes so only the equatorward rim remains
      * Keeps the standard open/closed interface for non-polar regions
      * Drops first/last latitude rows so poles can't be boundaries
    """
    st8 = generate_binary_structure(2, 2)
    open_mask = (P == 0).astype(bool)
    closed_mask = ~open_mask

    # Label open components; identify polar-connected ones (touch top/bottom rows)
    lbl, n = label(open_mask, structure=st8)
    polar_ids = set()
    if n > 0:
        if lbl.shape[0] > 0:
            top_ids = np.unique(lbl[0, :])
            bot_ids = np.unique(lbl[-1, :])
            for _id in np.concatenate([top_ids, bot_ids]):
                if _id != 0:
                    polar_ids.add(int(_id))

    polar_comp = np.isin(lbl, list(polar_ids)) if polar_ids else np.zeros_like(open_mask, dtype=bool)
    nonpolar_open = open_mask & ~polar_comp

    # Fill holes inside polar cap so tiny closed islands don't create internal boundaries
    if polar_comp.any():
        polar_filled = binary_fill_holes(polar_comp)
    else:
        polar_filled = polar_comp

    # Build interface for non-polar opens (standard)
    grad_nonpolar = nonpolar_open & ~binary_erosion(nonpolar_open, st8)
    adj_closed = binary_dilation(closed_mask, st8)
    outline_nonpolar = grad_nonpolar & adj_closed

    # Build equatorward rim for polar caps from the filled component
    grad_polar = polar_filled & ~binary_erosion(polar_filled, st8)
    outline_polar = grad_polar & adj_closed

    outline = outline_nonpolar | outline_polar

    # Never let polar rows act as boundary pixels
    if outline.shape[0] >= 2:
        outline[0, :] = False
        outline[-1, :] = False

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
        lbl, n = label(R)
        if n > 0:
            counts = np.bincount(lbl.ravel())
            keep = counts >= _MORPH_MIN_SIZE
            keep[0] = False
            R = keep[lbl]

    return R


# --- Helper: Save 8-panel figure of intermediate morphology steps ---
def _debug_morph_steps(
    boundary_mask_raw: np.ndarray, lon_1d: np.ndarray, lat_1d: np.ndarray, savepath: str | None = None
) -> None:
    """
    Save an 8-panel figure showing the morphological stages:
    raw, smoothed, closed, dilated, eroded, cleaned, region(raw), region(morphed)
    """
    B0 = boundary_mask_raw.astype(bool)

    # 1) smoothed
    if _MORPH_SMOOTH_SIGMA and _MORPH_SMOOTH_SIGMA > 0:
        sm = gaussian_filter(B0.astype(float), sigma=_MORPH_SMOOTH_SIGMA)
        B1 = sm >= 0.5
    else:
        B1 = B0.copy()

    # 4-neighborhood for the “spacefill” sequence (conservative, preserves shapes)
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

    # 5) cleaned (remove tiny components)
    B5 = B4.copy()
    if _MORPH_MIN_SIZE and _MORPH_MIN_SIZE > 0:
        lbl, n = label(B5)
        if n > 0:
            counts = np.bincount(lbl.ravel())
            keep = counts >= _MORPH_MIN_SIZE
            keep[0] = False
            B5 = keep[lbl]

    # Regions for single-line closed contours
    region_raw = _region_from_boundary(B0)
    region_morph = _region_from_boundary(B5)

    # Assemble figure
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

    if savepath and _MORPH_DEBUG_SAVE:
        plt.savefig(savepath, dpi=200)
    plt.close(fig)


def _gc_distance_map_to_boundary(lon_1d: np.ndarray, lat_1d: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    """
    Return a (n_lat, n_lon) array of great-circle distances [deg] from each pixel
    (lon_1d, lat_1d grid) to the nearest TRUE pixel in `boundary_mask`.

    Notes
    -----
    * `boundary_mask` must be shape (n_lat, n_lon) and mark boundary pixels (True).
    * Distances are computed on the unit sphere using a KDTree in 3D:
        angle = 2 * arcsin( ||x - x_bnd|| / 2 ), in degrees.
    """
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing="xy")  # (n_lat, n_lon)

    b_lon = lon_grid[boundary_mask]
    b_lat = lat_grid[boundary_mask]

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


from fluxpype.science.pfss_funcs import pixel_to_latlon
from fluxpype.pipe_helper import configurations, load_fits_magnetogram, load_magnetogram_params, shorten_path, get_ax


def magnet_plot(
    get_cr=None,
    datdir=None,
    _batch=None,
    open_f=None,
    closed_f=None,
    force=False,
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
    figbox = []
    fig, ax0 = get_ax(ax)
    figbox.append(fig)

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

    # Define the file names with their complete paths
    open_file = open_f or f"{floc_path}floc_open_cr{get_cr}_r{reduce_amt}_f{nwant}_{inst}.dat"
    closed_file = closed_f or f"{floc_path}floc_closed_cr{get_cr}_r{reduce_amt}_f{nwant}_{inst}.dat"
    magnet_file = f"{datdir}/magnetograms/CR{get_cr}_r{reduce_amt}_{inst}.fits"
    all_file = closed_file.replace("closed_", "")
    fname = magnet_file

    # Load the data
    if do_print_top:
        print(f"\t\tOpening {shorten_path(all_file)}...")
    fluxon_location = np.genfromtxt(all_file)
    # import pdb; pdb.set_trace()
    magnet, header = load_fits_magnetogram(batch=_batch, ret_all=True, configs=configs, fname=fname)
    f_lat, f_lon, f_sgn, _fnum = pixel_to_latlon(magnet, header, fluxon_location)

    if do_print_top:
        print(f"\t\tOpening {shorten_path(open_file)}...")
    oflnum, oflx, olat, olon, orad = np.loadtxt(open_file, unpack=True)

    if do_print_top:
        print(f"\t\tOpening {shorten_path(closed_file)}...\n")
    cflnum, cflx, clat, clon, crad = np.loadtxt(closed_file, unpack=True)

    ## Keep only the values where the radius is 1.0
    rtol = 0.001
    get_r = 1.0

    # Open fields
    oflnum_low = oflnum[np.isclose(orad, get_r, rtol)]
    oflx_low = oflx[np.isclose(orad, get_r, rtol)]
    olat_low = olat[np.isclose(orad, get_r, rtol)]
    olon_low = olon[np.isclose(orad, get_r, rtol)]

    # Closed fields
    cflnum_low = cflnum[np.isclose(crad, get_r, rtol)]
    cflx_low = cflx[np.isclose(crad, get_r, rtol)]
    clat_low = clat[np.isclose(crad, get_r, rtol)]
    clon_low = clon[np.isclose(crad, get_r, rtol)]

    # Convert to radians
    ph_olow, th_olow = np.sin(np.deg2rad(olat_low)), np.deg2rad(olon_low)
    ph_clow, th_clow = np.sin(np.deg2rad(clat_low)), np.deg2rad(clon_low)

    # Report the number of open and closed fluxons
    _n_open = int(np.max(oflnum_low))
    _n_closed = int(np.max(cflnum_low))
    _n_flux = _n_open + _n_closed
    _n_outliers = np.abs(_fnum - _n_flux)
    print(f"\t\t\tOpen: {_n_open}, Closed: {_n_closed}, Total: {_n_flux}, outliers: {_n_outliers}")

    # Define the file name for the plot
    pic_name = f"distance_cr{get_cr}_f{nwant}_ou{_n_open}_footpoints.{ext}"
    fluxon_map_histput_path = path.join(floc_path, pic_name)
    fluxon_map_histput_path_top = path.join(top_dir, pic_name)
    fluxon_csv_histput_path_top = path.join(floc_path, "distances.csv")

    # Check if the plot already exists
    do_plot = False
    pic_paths = [fluxon_map_histput_path, fluxon_map_histput_path_top]
    # pic_paths = [fluxon_map_histput_path_top]
    for testpath in pic_paths:
        if not path.exists(testpath):
            do_plot = True
            break

    force = True

    if do_print:
        print("\tPlotting...", end="")
    if do_plot or force or (ax is not None):
        # Plot the magnetogram

        # ax0.imshow(magnet, cmap='gray', interpolation=None, origin="lower",
        #         extent=(0,2*np.pi,-1,1), aspect='auto', vmin=vmin, vmax=vmax, zorder=5, alpha=0.8)
        ax0.imshow(
            magnet,
            cmap="gray",
            interpolation=None,
            origin="lower",
            extent=(0, 2 * np.pi, -1, 1),
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            zorder=-5,
            alpha=1,
        )

        # # Plot all the fluxons
        # Filter positive and negative cases for closed fluxons
        positive_indices_closed = [i for i, s in enumerate(cflx_low) if s > 0]
        negative_indices_closed = [i for i, s in enumerate(cflx_low) if s < 0]

        # Data for positive and negative cases
        f_lon_positive_closed = [th_clow[i] for i in positive_indices_closed]
        f_lat_positive_closed = [ph_clow[i] for i in positive_indices_closed]
        f_lon_negative_closed = [th_clow[i] for i in negative_indices_closed]
        f_lat_negative_closed = [ph_clow[i] for i in negative_indices_closed]

        if plot_all:
            # Plot positive cases with labels
            ax0.scatter(f_lon_positive_closed, f_lat_positive_closed, s=6**2, c="orange", alpha=0.8, label="Positive")

            # Plot negative cases with labels
            ax0.scatter(f_lon_negative_closed, f_lat_negative_closed, s=6**2, c="teal", alpha=0.8, label="Negative")

        # Filter positive and negative cases for open fluxons
        positive_indices_open = [i for i, s in enumerate(oflx_low) if s > 0]
        negative_indices_open = [i for i, s in enumerate(oflx_low) if s <= 0]

        # Data for positive and negative cases
        f_lon_positive_open = [th_olow[i] for i in positive_indices_open]
        f_lat_positive_open = [ph_olow[i] for i in positive_indices_open]
        f_lon_negative_open = [th_olow[i] for i in negative_indices_open]
        f_lat_negative_open = [ph_olow[i] for i in negative_indices_open]

        if plot_open:
            ax0.scatter(
                f_lon_positive_open,
                f_lat_positive_open,
                s=5**2,
                c="red",
                alpha=1.0,
                label="Positive (Open)",
                edgecolors="k",
            )
            ax0.scatter(
                f_lon_negative_open,
                f_lat_negative_open,
                s=5**2,
                c="blue",
                alpha=1.0,
                label="Negative (Open)",
                edgecolors="k",
            )
        # print("A")
        # plt.show(block=True)

        # Convert to radians
        ph_olow, th_olow = np.sin(np.deg2rad(olat_low)), np.deg2rad(olon_low)
        ph_clow, th_clow = np.sin(np.deg2rad(clat_low)), np.deg2rad(clon_low)

        plt.savefig(fluxon_map_histput_path, dpi=200)
        print(fluxon_map_histput_path)
        # plt.show(block=True)

        ########## FIGURE 2 ###########
        # fig, new_ax = plt.subplots()
        # figbox.append(fig)

        # plt.scatter(th_olow, ph_olow, c='r', s=5, label='Open')
        # plt.scatter(th_clow, ph_clow, c='b', s=5, label='Closed')

        if True:
            # # Provided points (longitude, latitude)
            points = np.array([th_olow, ph_olow]).T  # These are in radians and sin(radians)
            kind = "open"
        else:
            points = np.array([th_clow, ph_clow]).T  # These are in radians and sin(radians)
            kind = "closed"

        # # Image dimensions
        img_width, img_height = magnet.T.shape
        print(img_height, img_width)
        # Plotting the histogram
        print("B")
        # plt.show(block=True)

        fig, ax = plt.subplots()
        figbox.append(fig)

        hist, xedges, yedges = np.histogram2d(th_olow, ph_olow, bins=(36, 18))
        hist = hist.T  # Transpose for correct orientation
        ax.imshow(
            hist,
            cmap="viridis",
            origin="lower",
            interpolation="none",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect=2,
        )

        # Original scatter plot
        ax.scatter(th_olow, ph_olow, c="orange", s=50)

        # Find non-zero bins in the histogram
        open_points_inds = np.where(hist > 0)

        # Calculate the midpoints of bins for correct plotting
        # Note: Convert bin indices to the midpoint values in the original data space
        open_points_lons = xedges[open_points_inds[1]] + np.mean(np.diff(xedges)) / 2
        open_points_lats = yedges[open_points_inds[0]] + np.mean(np.diff(yedges)) / 2

        # Scatter plot on non-zero bins
        ax.scatter(open_points_lons, open_points_lats, c="red", s=50, alpha=0.6)

        from scipy.spatial import cKDTree

        # # Normalize and scale points to image dimensions
        # Longitude: 0 to 2pi maps to 0 to img_width
        # Latitude: -1 to 1 maps to 0 to img_height
        scaled_points = np.empty_like(points)
        scaled_points[:, 0] = (points[:, 0] / (2 * np.pi)) * img_width
        scaled_points[:, 1] = ((points[:, 1] + 1) / 2) * img_height

        # Generate grid points based on the image shape
        y_indices, x_indices = np.indices((img_height, img_width))
        grid_points = np.column_stack((x_indices.ravel(), y_indices.ravel()))

        # # Build a KDTree for efficient nearest-neighbor query
        tree = cKDTree(scaled_points)

        # # Calculate degrees per pixel
        deg_per_pixel_x = 360 / img_width  # For longitude
        deg_per_pixel_y = 180 / img_height  # For latitude

        # # Query the nearest distance for each grid point, requesting separate x and y components
        distances, _ = tree.query(grid_points, k=1, p=2, workers=-1, eps=0)

        # # Convert pixel distances to degrees
        distances_x = (distances // img_width) * deg_per_pixel_x
        distances_y = (distances % img_width) * deg_per_pixel_y
        distances_in_degrees = np.sqrt(distances_x**2 + distances_y**2)

        # Calculate the distance in degrees directly from pixel distances
        distances_in_degrees = distances * np.sqrt(deg_per_pixel_x**2 + deg_per_pixel_y**2)

        # # Reshape and display
        distance_array_degrees = distances_in_degrees.reshape((img_height, img_width))

        # # Determine the range for consistent color scaling
        vmin = 0
        vmax = distance_array_degrees.max()

        print("C")
        # plt.show(block=True)

        ######## FIGURE 3 #########
        fig, new_ax = plt.subplots()
        figbox.append(fig)

        # import numpy as np
        # import matplotlib.pyplot as plt
        from scipy import interpolate

        # Assuming distance_array_degrees, img_height, img_width, vmin, vmax are defined elsewhere

        # Create a meshgrid for longitude and latitude
        latitude = np.linspace(-1, 1, img_height)  # Corresponds to -90 to 90 degrees if scaled properly
        longitude = np.linspace(0, 2 * np.pi, img_width)  # 0 to 360 degrees
        longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)

        # Initialize the plot
        # fig, ax0 = plt.subplots()
        im = new_ax.imshow(
            distance_array_degrees,
            cmap="viridis",
            origin="lower",
            interpolation="none",
            alpha=1,
            extent=(0, 2 * np.pi, -1, 1),
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        im.figure.colorbar(im, ax=new_ax, label=f"Distance to Nearest {kind} Footpoint (Degrees)")
        plt.title(f"Distance to Nearest {kind} Footpoint (Degrees)")

        # Create the interpolator function
        dist_interp = interpolate.RectBivariateSpline(latitude, longitude, distance_array_degrees)

        # points = np.array([th_olow, ph_olow]).T #These are in radians and sin(radians)
        # Use the interpolator to get distances for the grid points
        distances = dist_interp.ev(latitude_grid.ravel(), longitude_grid.ravel())
        img_distances = distances.reshape(img_height, img_width)
        # Scatter plot on the grid points
        # Note: Adjust the sizes (s=), alpha, and edgecolors as needed
        # new_ax.scatter(longitude_grid.ravel(), latitude_grid.ravel(), c=distances, s=20,
        #             cmap='viridis', alpha=1, edgecolors='none', zorder=10, vmin=vmin, vmax=vmax)

        distances_points = dist_interp.ev(ph_olow.ravel(), th_olow.ravel())
        new_ax.scatter(
            th_olow,
            ph_olow,
            c=distances_points,
            s=100,
            cmap="viridis",
            alpha=1,
            edgecolors="k",
            zorder=100000,
            vmin=vmin,
            vmax=vmax,
        )

        # Draw a contour and collect its vertex paths robustly using `allsegs`
        CS = new_ax.contour(longitude, latitude, img_distances, levels=[10])
        # Extract contour segments for every level (here only one level is used)
        contour_paths = []
        for lvl in CS.allsegs:
            for seg in lvl:
                contour_paths.append(np.asarray(seg))

        # plt.show(block=True)
        print("D")
        # Prepare a new figure/axes for the contour-derived distance map
        fig, ax = plt.subplots()
        figbox.append(fig)

        for pth in contour_paths:
            ax.plot(pth[:, 0], pth[:, 1], "r.", lw=0)  # Plotting contour lines in red
        # plt.show(block=True)

        points = np.array([pth[:, 0], pth[:, 1]]).T  # These are in radians and sin(radians)

        # # Normalize and scale points to image dimensions
        # Longitude: 0 to 2pi maps to 0 to img_width
        # Latitude: -1 to 1 maps to 0 to img_height
        scaled_points = np.empty_like(points)
        scaled_points[:, 0] = (points[:, 0] / (2 * np.pi)) * img_width
        scaled_points[:, 1] = ((points[:, 1] + 1) / 2) * img_height

        # Generate grid points based on the image shape
        y_indices, x_indices = np.indices((img_height, img_width))
        grid_points = np.column_stack((x_indices.ravel(), y_indices.ravel()))

        # # Build a KDTree for efficient nearest-neighbor query
        tree = cKDTree(scaled_points)

        # # Calculate degrees per pixel
        deg_per_pixel_x = 360 / img_width  # For longitude
        deg_per_pixel_y = 180 / img_height  # For latitude

        # # Query the nearest distance for each grid point, requesting separate x and y components
        distances, _ = tree.query(grid_points, k=1, p=2, workers=-1, eps=0)

        # Calculate the distance in degrees directly from pixel distances
        distances_in_degrees = distances * np.sqrt(deg_per_pixel_x**2 + deg_per_pixel_y**2)

        # # Reshape and display
        distance_array_degrees = distances_in_degrees.reshape((img_height, img_width))

        im = ax.imshow(
            distance_array_degrees,
            cmap="viridis",
            origin="lower",
            interpolation="none",
            alpha=1,
            extent=(0, 2 * np.pi, -1, 1),
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        im.figure.colorbar(im, ax=ax, label=f"Distance to Nearest {kind} Footpoint (Degrees)")
        plt.title(f"Distance to Nearest {kind} Footpoint (Degrees)")

        # Create the interpolator function
        dist_interp_2 = interpolate.RectBivariateSpline(latitude, longitude, distance_array_degrees)
        distances_points = dist_interp_2.ev(ph_olow.ravel(), th_olow.ravel())
        ax.scatter(
            th_olow,
            ph_olow,
            c=distances_points,
            s=100,
            cmap="viridis",
            alpha=1,
            edgecolors="k",
            zorder=100000,
            vmin=vmin,
            vmax=vmax,
        )

        # plt.show()
        np.savetxt(fluxon_csv_histput_path_top, distance_array_degrees, delimiter=", ")

        do_legend = False

        if do_legend:
            ax0.legend(fontsize="small", loc="upper left", framealpha=0.75)

        if ax is None:
            shp = magnet.shape  # pixels
            plt.axis("off")
            sz0 = 6  # inches
            ratio = shp[1] / shp[0]
            sz1 = sz0 * ratio  # inches
            DPI = shp[1] / sz1  # pixels/inch
            fig.set_size_inches((sz1, sz0))
            plt.tight_layhist()
            plt.savefig(fluxon_map_histput_path_top, bbox_inches="tight", dpi=4 * DPI)
            # plt.show()
            plt.close(fig)

        nsteps = 360

        crs = [get_cr]
        ofmap = np.zeros((nsteps, len(crs)))
        efmap = np.zeros((nsteps, len(crs)))

        from tqdm import tqdm
        import sunpy
        from astropy import units as u, constants as const
        from astropy.coordinates import SkyCoord
        from fluxpype.science.pfss_funcs import load_pfss
        import pfsspy
        from pfsspy import tracing

        # Define the directory paths for the files
        floc_path = f"{datdir}/batches/{_batch}/data/cr{get_cr}/floc/"
        top_dir = f"{datdir}/batches/{_batch}/imgs/footpoints/"

        # with tqdm(total=len(crs)) as pbar:
        for i, cr in enumerate(crs):
            output_file = floc_path + f"pfss_ofmap_cr{cr}.npz"

            if os.path.exists(output_file):
                data = np.load(output_file)
                pols, expfs = data["ofmap"], data["efmap"]
                print("LOADED POLS, EXPFS")
                # === Coronal-hole boundary distance map (degrees) for pols == 0 ===
                # Ensure longitude/latitude vectors are present whether we loaded or computed
                try:
                    lon_1d
                    lat_1d
                except NameError:
                    # When loading from disk, fetch lon/lat saved earlier
                    if "data" in locals():
                        lon_1d = data["lon"]
                        lat_1d = data["lat"]
                    else:
                        raise

                # pols shape is (nsteps, 2*nsteps); build matching lon/lat grids
                lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing="xy")  # (nsteps, 2*nsteps)

                # Identify boundary pixels: ONLY 0 ↔ non-zero (±1) interfaces
                P = pols
                P_pad = np.pad(P, ((1, 1), (1, 1)), mode="edge")

                is_zero = P == 0
                nb_up = is_zero & (P_pad[:-2, 1:-1] != 0)
                nb_down = is_zero & (P_pad[2:, 1:-1] != 0)
                nb_left = is_zero & (P_pad[1:-1, :-2] != 0)
                nb_right = is_zero & (P_pad[1:-1, 2:] != 0)
                boundary_mask = nb_up | nb_down | nb_left | nb_right
                boundary_mask_raw = boundary_mask.copy()
                boundary_mask = _morph_spacefill(boundary_mask)
                region_mask_raw = _region_from_boundary(boundary_mask_raw)
                region_mask_morph = _open_region_mask_from_pols(P, min_size=_MORPH_MIN_SIZE, close_iters=1)

                # Build a 1-pixel open/closed interface outline for distance queries
                outline_mask = _interface_outline_polar_aware(P)

                if _MORPH_DEBUG:
                    dbg_path = top_dir + "morph_debug.png"
                    _debug_morph_steps(boundary_mask_raw, lon_1d, lat_1d, dbg_path)

                # Distance from every pixel to the nearest boundary pixel (deg)
                # This replaces the two-pass approach and works uniformly inside and outside.
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

                from scipy import interpolate as _interp

                ch_distance_interp = _interp.RectBivariateSpline(lat_1d, lon_1d, ch_distance_deg)

                if True:
                    fig_ch, ax_ch = plt.subplots()
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
                    ax_ch.set_title("Distance to Nearest Coronal-Hole Boundary")
                    cs_raw = ax_ch.contour(
                        lon_1d,
                        np.sin(lat_1d),
                        region_mask_raw.astype(int),
                        levels=[0.5],
                        colors="white",
                        linewidths=1.2,
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
                    from matplotlib.lines import Line2D

                    handles = [
                        Line2D([0], [0], color="white", lw=1.2, ls="--", label="raw"),
                        Line2D([0], [0], color="red", lw=1.5, ls="-", label="morphed"),
                    ]
                    ax_ch.legend(handles=handles, loc="upper right", fontsize="small", framealpha=0.6)
                    plt.tight_layout()
                    plt.savefig(top_dir + "distances.png")
                    plt.show(block=True)
                # === end boundary distance map ===
            else:
                hmi_map = sunpy.map.Map(magnet_file)
                hmi_map = hmi_map.resample([2 * nsteps, nsteps] * u.pix)

                nrho = 40
                rss = 2.5
                pfss_in = pfsspy.Input(hmi_map, nrho, rss)
                pfss_out = pfsspy.pfss(pfss_in)

                r = const.R_sun
                lon_1d = np.linspace(0, 2 * np.pi, nsteps * 2)
                lat_1d = np.arcsin(np.linspace(-0.999, 0.999, nsteps))
                lon, lat = np.meshgrid(lon_1d, lat_1d, indexing="ij")
                lon, lat = lon * u.rad, lat * u.rad
                seeds = SkyCoord(lon.ravel(), lat.ravel(), r, frame=pfss_out.coordinate_frame)

                tracer = tracing.FortranTracer(max_steps=2000)
                field_lines = tracer.trace(seeds, pfss_out)

                pols = field_lines.polarities.reshape(2 * nsteps, nsteps).T
                expfs = field_lines.expansion_factors.reshape(2 * nsteps, nsteps).T
                expfs[np.where(np.isnan(expfs))] = 0

                np.savez_compressed(output_file, ofmap=pols, efmap=expfs, brmap=hmi_map.data, lon=lon_1d, lat=lat_1d)

                print(output_file)
                # === Coronal-hole boundary distance map (degrees)
                try:
                    lon_1d
                    lat_1d
                except NameError:
                    if "data" in locals():
                        lon_1d = data["lon"]
                        lat_1d = data["lat"]
                    else:
                        raise

                lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing="xy")

                P = pols
                P_pad = np.pad(P, ((1, 1)), mode="edge") if P.ndim == 2 else np.pad(P, ((1, 1), (1, 1)), mode="edge")
                is_zero = P == 0
                nb_up = is_zero & (P_pad[:-2, 1:-1] != 0)
                nb_down = is_zero & (P_pad[2:, 1:-1] != 0)
                nb_left = is_zero & (P_pad[1:-1, :-2] != 0)
                nb_right = is_zero & (P_pad[1:-1, 2:] != 0)
                boundary_mask = nb_up | nb_down | nb_left | nb_right
                boundary_mask_raw = boundary_mask.copy()
                boundary_mask = _morph_spacefill(boundary_mask)
                region_mask_raw = _region_from_boundary(boundary_mask_raw)
                region_mask_morph = _open_region_mask_from_pols(P, min_size=_MORPH_MIN_SIZE, close_iters=1)

                # Build a 1-pixel open/closed interface outline for distance queries
                outline_mask = _interface_outline_polar_aware(P)

                if _MORPH_DEBUG:
                    dbg_path = top_dir + "morph_debug.png"
                    _debug_morph_steps(boundary_mask_raw, lon_1d, lat_1d, dbg_path)

                # Distance from every pixel to the nearest boundary pixel (deg)
                # This replaces the two-pass approach and works uniformly inside and outside.
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

                ch_out_npz = floc_path + f"pfss_ch_distance_cr{cr}.npz"
                np.savez_compressed(
                    ch_out_npz, ch_distance_deg=ch_distance_deg, lon=lon_1d, lat=lat_1d, pols=P.astype(np.int8)
                )
                ch_out_csv = floc_path + f"pfss_ch_distance_cr{cr}.csv"
                np.savetxt(ch_out_csv, ch_distance_deg, delimiter=", ")

                from scipy import interpolate as _interp

                ch_distance_interp = _interp.RectBivariateSpline(lat_1d, lon_1d, ch_distance_deg)

                if True:
                    fig_ch, ax_ch = plt.subplots()
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
                    ax_ch.set_title("Distance to Nearest Coronal-Hole Boundary")
                    cs_raw = ax_ch.contour(
                        lon_1d,
                        np.sin(lat_1d),
                        region_mask_raw.astype(int),
                        levels=[0.5],
                        colors="white",
                        linewidths=1.2,
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
                    from matplotlib.lines import Line2D

                    handles = [
                        Line2D([0], [0], color="white", lw=1.2, ls="--", label="raw"),
                        Line2D([0], [0], color="red", lw=1.5, ls="-", label="morphed"),
                    ]
                    ax_ch.legend(handles=handles, loc="upper right", fontsize="small", framealpha=0.6)
                    plt.savefig(top_dir + "distances.png")
                    plt.show(block=True)
                # === end boundary distance map ===

    else:
        if do_print:
            print("\tSkipped! Files already exist:")
            print(f"\t\t{shorten_path(fluxon_map_histput_path)}")
            print(f"\t\t{shorten_path(fluxon_map_histput_path_top)}")
    if do_print:
        print(
            f"\n\t    n_open: {_n_open}, n_closed: {_n_closed}, \
                n_total: {_n_flux}, n_all: {_fnum}, n_outliers: {_n_outliers}"
        )

    if do_print_top:
        print("\t\t    Success!")
        print("\t\t\t```````````````````````````````\n\n")

    for fig in figbox:
        plt.close(fig)
    return _n_open, _n_closed, _n_flux, _fnum, _n_outliers


########################################################################
# Main Code
# ----------------------------------------------------------------------
#

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="This script plots the expansion factor of the given radial_fr.dat")
    parser.add_argument("--cr", type=int, default=2160, help="Carrington Rotation")
    parser.add_argument("--file", type=str, default=None, help="Data File Name")
    parser.add_argument("--nwant", type=int, default=None, help="Number of Fluxons")
    parser.add_argument("--open", type=str, default=None)
    parser.add_argument("--closed", type=str, default=None)
    parser.add_argument("--adapt", type=str, default=None)

    args = parser.parse_args()
    configs = configurations(debug=False, args=args)

    magnet_plot(configs=configs)
