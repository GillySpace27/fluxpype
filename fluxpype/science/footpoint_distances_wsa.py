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

from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, binary_closing, binary_fill_holes, generate_binary_structure, gaussian_filter, label

# --- Morphology defaults (conservative) ---
_MORPH_DILATE_ITERS = 1      # light expansion to make boundaries more space-filling
_MORPH_ERODE_ITERS  = 0      # no erosion by default
_MORPH_CLOSE_ITERS  = 1      # close small gaps
_MORPH_OPEN_ITERS   = 0      # do not open by default
_MORPH_MIN_SIZE     = 32     # remove tiny speckles
_MORPH_SMOOTH_SIGMA = 0.6    # light Gaussian smoothing before thresholding

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
    Build a continuous, one‑pixel outline around the region indicated by a boundary mask.
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
    R = binary_closing(R, st8)   # close tiny gaps

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

    # 4) 1‑pixel perimeter via morphological gradient (region minus eroded region)
    er = binary_erosion(R, st8)
    outline = R & ~er
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


from fluxpype.science.pfss_funcs import pixel_to_latlon
from fluxpype.pipe_helper import (configurations, load_fits_magnetogram, load_magnetogram_params,
                            shorten_path, get_ax)

def magnet_plot(
        get_cr=None, datdir=None, _batch=None, open_f=None, closed_f=None, force=False, reduce_amt=0,
        nact=0, nwant=None, do_print_top=True, ax=None, verb=True, ext="png",
        plot_all=True, plot_open=True, do_print=False, vmin=-500, vmax=500, configs=None, legend=False):

    """ This function has been re-imagined into a

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
        nwant = nwant   or int(configs.get("fluxon_count", None)[0])
        reduce_amt = reduce_amt or configs.get("mag_reduce", None)
        if configs.get('adapt', False):
            inst = "adapt"
            reduce_amt = "f" + str(configs.get("adapt_select"))
        else:
            inst = "hmi"
    else:
        print("No configs given!")
        raise ValueError

    # Define the directory paths for the files
    floc_path = f"{datdir}/batches/{_batch}/data/cr{get_cr}/floc/"
    top_dir   = f"{datdir}/batches/{_batch}/imgs/footpoints/"
    if not path.exists(top_dir):
        os.makedirs(top_dir)

    # Define the file names with their complete paths
    open_file   = open_f     or   f"{floc_path}floc_open_cr{get_cr}_r{reduce_amt}_f{nwant}_{inst}.dat"
    closed_file = closed_f   or   f"{floc_path}floc_closed_cr{get_cr}_r{reduce_amt}_f{nwant}_{inst}.dat"
    magnet_file = f"{datdir}/magnetograms/CR{get_cr}_r{reduce_amt}_{inst}.fits"
    all_file    = closed_file.replace("closed_", "")
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
    oflx_low =     oflx[np.isclose(orad, get_r, rtol)]
    olat_low =     olat[np.isclose(orad, get_r, rtol)]
    olon_low =     olon[np.isclose(orad, get_r, rtol)]

    # Closed fields
    cflnum_low = cflnum[np.isclose(crad, get_r, rtol)]
    cflx_low =     cflx[np.isclose(crad, get_r, rtol)]
    clat_low =     clat[np.isclose(crad, get_r, rtol)]
    clon_low =     clon[np.isclose(crad, get_r, rtol)]

    # Convert to radians
    ph_olow, th_olow = np.sin(np.deg2rad(olat_low)), np.deg2rad(olon_low)
    ph_clow, th_clow = np.sin(np.deg2rad(clat_low)), np.deg2rad(clon_low)

    # Report the number of open and closed fluxons
    _n_open = int(np.max(oflnum_low))
    _n_closed = int(np.max(cflnum_low))
    _n_flux = _n_open + _n_closed
    _n_outliers = np.abs(_fnum-_n_flux)
    print(f"\t\t\tOpen: {_n_open}, Closed: {_n_closed}, Total: {_n_flux}, outliers: {_n_outliers}")

    # Define the file name for the plot
    pic_name = f'distance_cr{get_cr}_f{nwant}_ou{_n_open}_footpoints.{ext}'
    fluxon_map_histput_path =   path.join(floc_path, pic_name)
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

    force=True

    if do_print:
        print("\tPlotting...", end="")
    if do_plot or force or (ax is not None):
        # Plot the magnetogram

        # ax0.imshow(magnet, cmap='gray', interpolation=None, origin="lower",
        #         extent=(0,2*np.pi,-1,1), aspect='auto', vmin=vmin, vmax=vmax, zorder=5, alpha=0.8)
        ax0.imshow(magnet, cmap='gray', interpolation=None, origin="lower",
                extent=(0,2*np.pi,-1,1), aspect='auto', vmin=vmin, vmax=vmax, zorder=-5, alpha=1)

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
            ax0.scatter(f_lon_positive_closed, f_lat_positive_closed, s=6**2, c='orange', alpha=0.8, label='Positive')

            # Plot negative cases with labels
            ax0.scatter(f_lon_negative_closed, f_lat_negative_closed, s=6**2, c='teal', alpha=0.8, label='Negative')

        # Filter positive and negative cases for open fluxons
        positive_indices_open = [i for i, s in enumerate(oflx_low) if s > 0]
        negative_indices_open = [i for i, s in enumerate(oflx_low) if s <= 0]

        # Data for positive and negative cases
        f_lon_positive_open = [th_olow[i] for i in positive_indices_open]
        f_lat_positive_open = [ph_olow[i] for i in positive_indices_open]
        f_lon_negative_open = [th_olow[i] for i in negative_indices_open]
        f_lat_negative_open = [ph_olow[i] for i in negative_indices_open]

        if plot_open:
            ax0.scatter(f_lon_positive_open, f_lat_positive_open, s=5**2, c='red', alpha=1.0, label='Positive (Open)', edgecolors='k')
            ax0.scatter(f_lon_negative_open, f_lat_negative_open, s=5**2, c='blue', alpha=1.0, label='Negative (Open)', edgecolors='k')
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
            points = np.array([th_olow, ph_olow]).T #These are in radians and sin(radians)
            kind = "open"
        else:
            points = np.array([th_clow, ph_clow]).T #These are in radians and sin(radians)
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
        ax.imshow(hist, cmap='viridis', origin='lower', interpolation='none',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect=2)

        # Original scatter plot
        ax.scatter(th_olow, ph_olow, c='orange', s=50)

        # Find non-zero bins in the histogram
        open_points_inds = np.where(hist > 0)

        # Calculate the midpoints of bins for correct plotting
        # Note: Convert bin indices to the midpoint values in the original data space
        open_points_lons = xedges[open_points_inds[1]] + np.mean(np.diff(xedges))/2
        open_points_lats = yedges[open_points_inds[0]] + np.mean(np.diff(yedges))/2

        # Scatter plot on non-zero bins
        ax.scatter(open_points_lons, open_points_lats, c='red', s=50, alpha=0.6)

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
        longitude = np.linspace(0, 2*np.pi, img_width)  # 0 to 360 degrees
        longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)

        # Initialize the plot
        # fig, ax0 = plt.subplots()
        im = new_ax.imshow(distance_array_degrees, cmap='viridis', origin='lower', interpolation='none',
                        alpha=1, extent=(0, 2*np.pi, -1, 1), aspect='auto', vmin=vmin, vmax=vmax)
        im.figure.colorbar(im, ax=new_ax, label=f'Distance to Nearest {kind} Footpoint (Degrees)')
        plt.title(f'Distance to Nearest {kind} Footpoint (Degrees)')

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
        new_ax.scatter(th_olow, ph_olow, c=distances_points, s=100, cmap='viridis',
                    alpha=1, edgecolors='k', zorder=100000, vmin=vmin, vmax=vmax)

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
            ax.plot(pth[:, 0], pth[:, 1], 'r.', lw=0)  # Plotting contour lines in red
        # plt.show(block=True)

        points = np.array([pth[:, 0], pth[:, 1]]).T #These are in radians and sin(radians)

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

        im = ax.imshow(distance_array_degrees, cmap='viridis', origin='lower', interpolation='none',
                        alpha=1, extent=(0, 2*np.pi, -1, 1), aspect='auto', vmin=vmin, vmax=vmax)
        im.figure.colorbar(im, ax=ax, label=f'Distance to Nearest {kind} Footpoint (Degrees)')
        plt.title(f'Distance to Nearest {kind} Footpoint (Degrees)')

        # Create the interpolator function
        dist_interp_2 = interpolate.RectBivariateSpline(latitude, longitude, distance_array_degrees)
        distances_points = dist_interp_2.ev(ph_olow.ravel(), th_olow.ravel())
        ax.scatter(th_olow, ph_olow, c=distances_points, s=100, cmap='viridis',
                    alpha=1, edgecolors='k', zorder=100000, vmin=vmin, vmax=vmax)

        # plt.show()
        np.savetxt(fluxon_csv_histput_path_top, distance_array_degrees, delimiter=", ")

        do_legend=False

        if do_legend:
            ax0.legend(fontsize="small", loc="upper left", framealpha=0.75)

        if ax is None:
            shp = magnet.shape #pixels
            plt.axis('off')
            sz0=6 #inches
            ratio = shp[1]/shp[0]
            sz1=sz0*ratio #inches
            DPI = shp[1] / sz1 #pixels/inch
            fig.set_size_inches((sz1, sz0))
            plt.tight_layhist()
            plt.savefig(fluxon_map_histput_path_top, bbox_inches='tight', dpi=4*DPI)
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
        top_dir   = f"{datdir}/batches/{_batch}/imgs/footpoints/"

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
                    if 'data' in locals():
                        lon_1d = data["lon"]
                        lat_1d = data["lat"]
                    else:
                        raise

                # pols shape is (nsteps, 2*nsteps); build matching lon/lat grids
                # lon varies along axis=1 (size 2*nsteps), lat along axis=0 (size nsteps)
                lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing="xy")  # shapes (nsteps, 2*nsteps)

                # Identify boundary pixels: ONLY 0 ↔ non-zero (±1) interfaces
                # Define boundary as ZERO cells that have at least one non-zero 4-neighbor.
                P = pols
                # Pad with edge values to keep shapes consistent during neighbor checks
                P_pad = np.pad(P, ((1, 1), (1, 1)), mode='edge')

                is_zero = (P == 0)
                nb_up    = is_zero & (P_pad[:-2, 1:-1] != 0)
                nb_down  = is_zero & (P_pad[ 2:, 1:-1] != 0)
                nb_left  = is_zero & (P_pad[1:-1, :-2] != 0)
                nb_right = is_zero & (P_pad[1:-1,  2:] != 0)
                boundary_mask = nb_up | nb_down | nb_left | nb_right
                # Morphologically expand/clean boundary to be more space-filling (conservative defaults)
                boundary_mask = _morph_spacefill(boundary_mask)

                # Distances are defined only for ZERO-valued pixels
                zero_mask = is_zero

                # Coordinates (lon, lat) of boundary pixel centers (radians)
                b_lon = lon_grid[boundary_mask]
                b_lat = lat_grid[boundary_mask]

                # If there are no boundary pixels (pathological), fill with NaN and continue
                ch_distance_deg = np.full(P.shape, np.nan, dtype=float)
                if b_lon.size == 0 or np.count_nonzero(zero_mask) == 0:
                    # Save and continue with NaNs
                    pass
                else:
                    # Build 3D unit vectors for spherical KDTree: x=cos(lat)cos(lon), y=cos(lat)sin(lon), z=sin(lat)
                    def sph2cart(lon, lat):
                        cl = np.cos(lat)
                        return np.column_stack((cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)))

                    # Boundary set in 3D
                    from scipy.spatial import cKDTree
                    b_xyz = sph2cart(b_lon, b_lat)
                    tree = cKDTree(b_xyz)

                    # Query all zero pixels in one shot
                    z_lon = lon_grid[zero_mask]
                    z_lat = lat_grid[zero_mask]
                    z_xyz = sph2cart(z_lon, z_lat)

                    # Euclidean chord distance in 3D; convert to great‑circle angle
                    d_chord, _ = tree.query(z_xyz, k=1, workers=-1)
                    # angle = 2 * arcsin(d/2); return in degrees
                    angles = 2.0 * np.arcsin(np.clip(d_chord * 0.5, 0.0, 1.0)) * (180.0 / np.pi)

                # Populate map only at zero locations; others remain NaN
                ch_distance_deg[zero_mask] = angles
                # --- Second pass: fill internal (non-zero) pixels with distance to nearest ZERO cell ---
                nonzero_mask = (P != 0)
                if np.any(nonzero_mask) and np.any(zero_mask):
                    # KDTree built over ALL zero cells (original open/closed boundary side)
                    z2_lon = lon_grid[zero_mask]
                    z2_lat = lat_grid[zero_mask]
                    z2_xyz = sph2cart(z2_lon, z2_lat)
                    tree_zero = cKDTree(z2_xyz)

                    n_lon = lon_grid[nonzero_mask]
                    n_lat = lat_grid[nonzero_mask]
                    n_xyz = sph2cart(n_lon, n_lat)

                    d_chord_nz, _ = tree_zero.query(n_xyz, k=1, workers=-1)
                    angles_nz = 2.0 * np.arcsin(np.clip(d_chord_nz * 0.5, 0.0, 1.0)) * (180.0 / np.pi)
                    ch_distance_deg[nonzero_mask] = angles_nz
                # --- end second pass ---

                # Persist results alongside lon/lat for later reuse/interpolation
                ch_out_npz = floc_path + f"pfss_distances.npz"
                np.savez_compressed(ch_out_npz,
                                    ch_distance_deg=ch_distance_deg,
                                    lon=lon_1d, lat=lat_1d,
                                    pols=P.astype(np.int8))
                # Also provide a CSV in the same shape for quick inspection
                ch_out_csv = floc_path + f"pfss_distances.csv"
                np.savetxt(ch_out_csv, ch_distance_deg, delimiter=", ")

                # Build an interpolator consistent with prior figure usage
                from scipy import interpolate as _interp
                # Note: RectBivariateSpline expects increasing coordinates (lat first, then lon)
                ch_distance_interp = _interp.RectBivariateSpline(lat_1d, lon_1d, ch_distance_deg)

                # Optional quicklook plot (comment out to disable)
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
                    im.figure.colorbar(im, ax=ax_ch, label='Distance to CH Boundary (deg)')
                    ax_ch.set_title('Distance to Nearest Coronal-Hole Boundary')
                    # Overplot the open/closed boundary in red (sin-lat on Y) using filled region for single-line contour
                    region_mask = _region_from_boundary(boundary_mask)
                    ax_ch.contour(
                        lon_1d,
                        np.sin(lat_1d),
                        region_mask.astype(int),
                        levels=[0.5],
                        colors='red',
                        linewidths=1.5,
                    )
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
                # === Coronal-hole boundary distance map (degrees) for pols == 0 ===
                # Ensure longitude/latitude vectors are present whether we loaded or computed
                try:
                    lon_1d
                    lat_1d
                except NameError:
                    # When loading from disk, fetch lon/lat saved earlier
                    if 'data' in locals():
                        lon_1d = data["lon"]
                        lat_1d = data["lat"]
                    else:
                        raise

                # pols shape is (nsteps, 2*nsteps); build matching lon/lat grids
                # lon varies along axis=1 (size 2*nsteps), lat along axis=0 (size nsteps)
                lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d, indexing="xy")  # shapes (nsteps, 2*nsteps)

                # Identify boundary pixels: ONLY 0 ↔ non-zero (±1) interfaces
                # Define boundary as ZERO cells that have at least one non-zero 4-neighbor.
                P = pols
                # Pad with edge values to keep shapes consistent during neighbor checks
                P_pad = np.pad(P, ((1, 1), (1, 1)), mode='edge')

                is_zero = (P == 0)
                nb_up    = is_zero & (P_pad[:-2, 1:-1] != 0)
                nb_down  = is_zero & (P_pad[ 2:, 1:-1] != 0)
                nb_left  = is_zero & (P_pad[1:-1, :-2] != 0)
                nb_right = is_zero & (P_pad[1:-1,  2:] != 0)
                boundary_mask = nb_up | nb_down | nb_left | nb_right
                # Morphologically expand/clean boundary to be more space-filling (conservative defaults)
                boundary_mask = _morph_spacefill(boundary_mask)

                # Distances are defined only for ZERO-valued pixels
                zero_mask = is_zero

                # Coordinates (lon, lat) of boundary pixel centers (radians)
                b_lon = lon_grid[boundary_mask]
                b_lat = lat_grid[boundary_mask]

                # If there are no boundary pixels (pathological), fill with NaN and continue
                ch_distance_deg = np.full(P.shape, np.nan, dtype=float)
                if b_lon.size == 0 or np.count_nonzero(zero_mask) == 0:
                    # Save and continue with NaNs
                    pass
                else:
                    # Build 3D unit vectors for spherical KDTree: x=cos(lat)cos(lon), y=cos(lat)sin(lon), z=sin(lat)
                    def sph2cart(lon, lat):
                        cl = np.cos(lat)
                        return np.column_stack((cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)))

                    # Boundary set in 3D
                    from scipy.spatial import cKDTree
                    b_xyz = sph2cart(b_lon, b_lat)
                    tree = cKDTree(b_xyz)

                    # Query all zero pixels in one shot
                    z_lon = lon_grid[zero_mask]
                    z_lat = lat_grid[zero_mask]
                    z_xyz = sph2cart(z_lon, z_lat)

                    # Euclidean chord distance in 3D; convert to great‑circle angle
                    d_chord, _ = tree.query(z_xyz, k=1, workers=-1)
                    # angle = 2 * arcsin(d/2); return in degrees
                    angles = 2.0 * np.arcsin(np.clip(d_chord * 0.5, 0.0, 1.0)) * (180.0 / np.pi)

                # Populate map only at zero locations; others remain NaN
                ch_distance_deg[zero_mask] = angles
                # --- Second pass: fill internal (non-zero) pixels with distance to nearest ZERO cell ---
                nonzero_mask = (P != 0)
                if np.any(nonzero_mask) and np.any(zero_mask):
                    # KDTree built over ALL zero cells (original open/closed boundary side)
                    z2_lon = lon_grid[zero_mask]
                    z2_lat = lat_grid[zero_mask]
                    z2_xyz = sph2cart(z2_lon, z2_lat)
                    tree_zero = cKDTree(z2_xyz)

                    n_lon = lon_grid[nonzero_mask]
                    n_lat = lat_grid[nonzero_mask]
                    n_xyz = sph2cart(n_lon, n_lat)

                    d_chord_nz, _ = tree_zero.query(n_xyz, k=1, workers=-1)
                    angles_nz = 2.0 * np.arcsin(np.clip(d_chord_nz * 0.5, 0.0, 1.0)) * (180.0 / np.pi)
                    ch_distance_deg[nonzero_mask] = angles_nz
                # --- end second pass ---

                # Persist results alongside lon/lat for later reuse/interpolation
                ch_out_npz = floc_path + f"pfss_ch_distance_cr{cr}.npz"
                np.savez_compressed(ch_out_npz,
                                    ch_distance_deg=ch_distance_deg,
                                    lon=lon_1d, lat=lat_1d,
                                    pols=P.astype(np.int8))
                # Also provide a CSV in the same shape for quick inspection
                ch_out_csv = floc_path + f"pfss_ch_distance_cr{cr}.csv"
                np.savetxt(ch_out_csv, ch_distance_deg, delimiter=", ")

                # Build an interpolator consistent with prior figure usage
                from scipy import interpolate as _interp
                # Note: RectBivariateSpline expects increasing coordinates (lat first, then lon)
                ch_distance_interp = _interp.RectBivariateSpline(lat_1d, lon_1d, ch_distance_deg)

                # Optional quicklook plot (comment out to disable)
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
                    im.figure.colorbar(im, ax=ax_ch, label='Distance to CH Boundary (deg)')
                    ax_ch.set_title('Distance to Nearest Coronal-Hole Boundary')
                    # Overplot the open/closed boundary in red (sin-lat on Y) using filled region for single-line contour
                    region_mask = _region_from_boundary(boundary_mask)
                    ax_ch.contour(
                        lon_1d,
                        np.sin(lat_1d),
                        region_mask.astype(int),
                        levels=[0.5],
                        colors='red',
                        linewidths=1.5,
                    )
                    plt.savefig(top_dir + "distances.png")
                    plt.show(block=True)
                # === end boundary distance map ===

    else:
        if do_print:
            print("\tSkipped! Files already exist:")
            print(f"\t\t{shorten_path(fluxon_map_histput_path)}")
            print(f"\t\t{shorten_path(fluxon_map_histput_path_top)}")
    if do_print:
        print(f"\n\t    n_open: {_n_open}, n_closed: {_n_closed}, \
                n_total: {_n_flux}, n_all: {_fnum}, n_outliers: {_n_outliers}")

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
    parser = argparse.ArgumentParser(description=
            'This script plots the expansion factor of the given radial_fr.dat')
    parser.add_argument('--cr', type=int, default=2160, help='Carrington Rotation')
    parser.add_argument('--file', type=str, default=None, help='Data File Name')
    parser.add_argument('--nwant', type=int, default=None, help='Number of Fluxons')
    parser.add_argument('--open', type=str, default=None)
    parser.add_argument('--closed', type=str, default=None)
    parser.add_argument('--adapt', type=str, default=None)

    args = parser.parse_args()
    configs = configurations(debug=False, args=args)

    magnet_plot(configs=configs)
