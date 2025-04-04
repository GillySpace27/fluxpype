"""
Primary Script for Plotting the Magnetogram with Footpoints
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


from fluxpype.science.pfss_funcs import pixel_to_latlon
from fluxpype.pipe_helper import configurations, load_fits_magnetogram, load_magnetogram_params, shorten_path, get_ax

def magnet_plot(get_cr=None, datdir=None, _batch=None, open_f=None, closed_f=None, force=False, reduce_amt=0,
                nact=0, nwant=None, do_print_top=False, ax=None, verb=True, ext="png",
                plot_all=True, plot_open=True, do_print=False, vmin=-500, vmax=500, configs=None, legend=False):
    """ The primary function for plotting the magnetogram with footpoints

    Parameters
    ----------
    get_cr : int
        the carrington rotation number
    datdir : str
        the data directory
    _batch : str
        the name of the batch
    open_f : int, optional
        number of open fluxons, by default None
    closed_f : int, optional
        number of closed fluxons, by default None
    force : bool, optional
        force the plot to occur, by default False
    reduce_amt : int, optional
        the factor by which to reduce the magnetogram, by default 0
    nact : int, optional
        the actual number of fluxons, by default 0
    nwant : int, optional
        the desired number of fluxons, by default None
    do_print_top : bool, optional
        print the top, by default False
    ax : pyplot axis, optional
        the axis to plot upon, by default None
    verb : bool, optional
        whether to be verbose in output, by default True
    ext : str, optional
        the type of plot to save, by default "pdf"
    plot_all : bool, optional
        whether to plot all the fluxons, by default True
    plot_open : bool, optional
        whether to plot the open fields, by default True
    do_print : bool, optional
        even more verbosity, by default False
    vmin : int, optional
        minimum value of the colortable, by default -500
    vmax : int, optional
        maximum value of the colortable, by default 500

    Returns
    -------
    _n_open, _n_closed, _n_flux, _fnum, _n_outliers
        integers describing the number of open fluxons, closed fluxons, total fluxons,
        actual fluxons, and outliers, respectively
    """

    fig, ax0 = get_ax(ax)
    if do_print:
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("(py) Plotting Magnetic Field Maps")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    if do_print_top:
        print("\tMaking Magnetogram with Footpoints...")

    if configs is not None:
        datdir = datdir or configs.get("data_dir", None)
        _batch = _batch or configs.get("batch_name", None)
        get_cr = get_cr or configs.get("cr", None)
        nwant = nwant   or configs.get("nwant", None)
        reduce_amt = reduce_amt or configs.get("mag_reduce", None)
        if configs['adapt']:
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

    if do_print_top:
        print(f"\t\tOpening {shorten_path(open_file)}...")
    oflnum, oflx, olat, olon, orad = np.loadtxt(open_file, unpack=True)

    if do_print_top:
        print(f"\t\tOpening {shorten_path(closed_file)}...\n")
    cflnum, _, _, _, crad = np.loadtxt(closed_file, unpack=True)

    magnet, header = load_fits_magnetogram(batch=_batch, ret_all=True, configs=configs, fname=fname)
    f_lat, f_lon, f_sgn, _fnum = pixel_to_latlon(magnet, header, fluxon_location)

    ## Keep only the values where the radius is 1.0
    rtol = 0.01
    get_r = 1.0

    #Open fields
    oflnum_low = oflnum[np.isclose(orad, get_r, rtol)]
    oflx_low =     oflx[np.isclose(orad, get_r, rtol)]
    olat_low =     olat[np.isclose(orad, get_r, rtol)]
    olon_low =     olon[np.isclose(orad, get_r, rtol)]

    # Closed fields
    cflnum_low = cflnum[np.isclose(crad, get_r, rtol)]


    # Convert to radians
    ph_olow, th_olow = np.sin(np.deg2rad(olat_low)), np.deg2rad(olon_low)

    # Report the number of open and closed fluxons
    _n_open = int(np.max(oflnum_low))
    _n_closed = int(np.max(cflnum_low))
    _n_flux = _n_open + _n_closed
    _n_outliers = np.abs(_fnum-_n_flux)

    # Define the file name for the plot
    pic_name = f'cr{get_cr}_f{nwant}_ou{_n_open}_footpoints_topology.{ext}'
    fluxon_map_output_path =   path.join(floc_path, pic_name)
    fluxon_map_output_path_top = path.join(top_dir, pic_name)

    # Check if the plot already exists
    do_plot = False
    pic_paths = [fluxon_map_output_path, fluxon_map_output_path_top]
    # pic_paths = [fluxon_map_output_path_top]
    for testpath in pic_paths:
        if not path.exists(testpath):
            do_plot = True
            break

    if do_print:
        print("\tPlotting...", end="")
    if do_plot or force or (ax is not None):
        # Plot the magnetogram

        # ax0.imshow(magnet, cmap='gray', interpolation=None, origin="lower",
        #         extent=(0,2*np.pi,-1,1), aspect='auto', vmin=vmin, vmax=vmax, zorder=5, alpha=0.8)
        ax0.imshow(magnet, cmap='gray', interpolation=None, origin="lower",
                extent=(0,2*np.pi,-1,1), aspect='auto', vmin=vmin, vmax=vmax, zorder=-5, alpha=1)
        # # Plot all the fluxons
        # Filter positive and negative cases
        positive_indices = [i for i, s in enumerate(f_sgn) if s > 0]
        negative_indices = [i for i, s in enumerate(f_sgn) if s <= 0]

        # Data for positive and negative cases
        f_lon_positive = [f_lon[i] for i in positive_indices]
        f_lat_positive = [f_lat[i] for i in positive_indices]
        f_lon_negative = [f_lon[i] for i in negative_indices]
        f_lat_negative = [f_lat[i] for i in negative_indices]

        if plot_all:
            # Plot positive cases with labels
            ax0.scatter(f_lon_positive, f_lat_positive, s=3**2, c='orange', alpha=0.6, label='Positive')

            # Plot negative cases with labels
            ax0.scatter(f_lon_negative, f_lat_negative, s=3**2, c='teal', alpha=0.6, label='Negative')

        # Filter positive and negative cases for open fluxons
        positive_indices = [i for i, s in enumerate(oflx_low) if s > 0]
        negative_indices = [i for i, s in enumerate(oflx_low) if s <= 0]

        # Data for positive and negative cases
        th_olow_positive = [th_olow[i] for i in positive_indices]
        ph_olow_positive = [ph_olow[i] for i in positive_indices]
        th_olow_negative = [th_olow[i] for i in negative_indices]
        ph_olow_negative = [ph_olow[i] for i in negative_indices]

        if plot_open:
            ax0.scatter(th_olow_positive, ph_olow_positive, s=5**2, c='red', alpha=1.0, label='Positive (Open)', edgecolors='k')
            ax0.scatter(th_olow_negative, ph_olow_negative, s=5**2, c='blue', alpha=1.0, label='Negative (Open)', edgecolors='k')

        if legend:
            ax0.legend(fontsize="small", loc="upper left", framealpha=0.75)

        if ax is None:
            shp = magnet.shape #pixels
            plt.axis('off')
            sz0=6 #inches
            ratio = shp[1]/shp[0]
            sz1=sz0*ratio #inches
            DPI = shp[1] / sz1 #pixels/inch
            fig.set_size_inches((sz1, sz0))
            plt.tight_layout()
            plt.savefig(fluxon_map_output_path_top, bbox_inches='tight', dpi=4*DPI)
            # plt.show()
            plt.close(fig)
    else:
        if do_print:
            print("\tSkipped! Files already exist:")
            print(f"\t\t{shorten_path(fluxon_map_output_path)}")
            print(f"\t\t{shorten_path(fluxon_map_output_path_top)}")
    if do_print:
        print(f"\n\t    n_open: {_n_open}, n_closed: {_n_closed}, \
                n_total: {_n_flux}, n_all: {_fnum}, n_outliers: {_n_outliers}")

    if do_print_top:
        print("\t\t    Success!")
        print("\t\t\t```````````````````````````````\n\n")
    return _n_open, _n_closed, _n_flux, _fnum, _n_outliers


########################################################################
# Main Code
# ----------------------------------------------------------------------
#

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description=
            'This script plots the expansion factor of the given radial_fr.dat')
    parser.add_argument('--cr', type=int, default=2205, help='Carrington Rotation')
    parser.add_argument('--file', type=str, default=None, help='Data File Name')
    parser.add_argument('--nwant', type=int, default=None, help='Number of Fluxons')
    parser.add_argument('--open', type=str, default=None)
    parser.add_argument('--closed', type=str, default=None)
    parser.add_argument('--adapt', type=str, default=None)

    args = parser.parse_args()
    configs = configurations(debug=False, args=args)


    # Run the code
    # (hdr, cr, fname, adapt, doplot, reduce) = load_magnetogram_params(args.dat_dir)

    # magnet_plot(CR,              configs['data_dir'],  configs['batch_name'],
    #             configs['open'], configs['closed'],    do_print=configs['verbose'],
    #             reduce_amt=configs['mag_reduce'],      nwant=nwant,
    #             do_print_top=True)

    magnet_plot(configs=configs, do_print_top=True)
