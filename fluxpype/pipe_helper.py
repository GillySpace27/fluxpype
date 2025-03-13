"""
pipe_helper: Comprehensive Library for FLUXpype Algorithm and Fluxon Simulations
===============================================================================

This library provides a collection of utility functions to assist with the
FLUXpype algorithm and Fluxon simulations. It offers functionalities for managing directories,
handling FITS files, manipulating magnetogram data, parsing and plotting data generated from fluxon simulations.

Modules:
--------
- os, os.path, sys, pathlib.PosixPath, Path
- numpy as np
- matplotlib.pyplot as plt
- astropy.io.fits, astropy.nddata.block_reduce
- sunpy, sunpy.coordinates, sunpy.io, sunpy.net.Fido, attrs as a
- pandas

Functions:
----------
### General Utilities
- `configurations`: Reads and sanitizes configuration settings from a specified config file.
- `convert_value`: Converts a string to an int or float if possible.
- `calculate_directories`: Helper function to calculate directories.
- `add_dir_to_path`: Adds a directory and all subdirectories to the PATH environment variable.
- `add_top_level_dirs_to_path`: Adds the top-level directories under a root directory to the PATH.
- `add_paths`: Adds various paths to the system path.
- `find_file_with_string`: Searches a directory for a file containing a given string.
- `shorten_path`: Removes the DATAPATH environment variable from a string.

### Magnetogram Utilities
- `make_mag_dir`: Creates a directory for magnetogram data.
- `get_magnetogram_file`: Grabs HMI data.
- `reduce_mag_file`: Reduces the size of a magnetogram FITS file by a given factor.
- `reduce_fits_image`: Reduces the size of a FITS image.
- `plot_raw_magnetogram`: Plots the magnetogram.
- `load_fits_magnetogram`: Loads a magnetogram from a FITS file.
- `write_magnetogram_params`: Writes the magnetic_target.params file for a given CR and reduction amount.
- `load_magnetogram_params`: Reads the magnetic_target.params file and returns the parameters.
- `read_fits_data`: Reads FITS data and fixes/ignores any non-standard FITS keywords.
- `get_fixed_coords`: Corrects input coordinates.

### Fluxon Simulation Utilities
- `parse_line`: Parse a line of the output file into a dictionary.
- `load_data`: Load the data from the file into a pandas DataFrame.
- `get_ax`: Get or create a pyplot figure and axis pair.
- `add_fluxon_dirs_to_path`: Add the fluxon directories to the system path.
- `list_directories`: List the directories in the given path.
- `path_add`: Add directories to the system path.

Usage Example:
--------------
```python
# Example usage of convert_value function
import pipe_helper as ph
result = ph.convert_value("42")

# Example usage of the configurations module
from pipe_helper import configurations
configs = configurations()


Author:
-------
    Gilly <gilly@swri.org> (and others!)

Dependencies:
-------------
    os, os.path, sys, pathlib.PosixPath, Path, numpy as np,
    matplotlib.pyplot as plt, astropy.io.fits, astropy.nddata.block_reduce,
    sunpy, sunpy.coordinates, sunpy.io, sunpy.net.Fido, attrs as a, pandas
"""

# Import libraries
import os
import os.path
import sys
import ast
from pathlib import PosixPath, Path

import pandas as pd
import re

# from pipe_helper import convert_value
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

# from astropy.nddata import block_reduce

import sunpy
import sunpy.coordinates
import sunpy.io
from sunpy.net import Fido, attrs as a
import configparser


# CONFIGURATION MANAGEMENT #######################################################

import ast

import configparser
import os


def configurations(
    config_name=None, config_filename="config.ini", args=None, debug=False
):
    """
    Reads and sanitizes configuration settings from a specified config file.

    Args:
        config_name (str, optional): The specific configuration section to read.
                                     Defaults to the 'DEFAULT' section.
        config_filename (str, optional): The filename of the configuration file.
                                         Defaults to 'config.ini'.
        args (Namespace, optional): Command-line arguments to override config values.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        dict: Configuration settings as key-value pairs.
    """
    # Load configuration file
    config_path = find_config_file(config_filename)
    config_obj = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )

    with open(config_path, "r") as f:
        clean_content = clean_config_content(f.readlines())
    config_obj.read_string(clean_content)

    # Determine the configuration section to use
    config_name = config_name or config_obj["DEFAULT"].get("config_name", "DEFAULT")
    if config_name not in config_obj:
        available_sections = ", ".join(config_obj.sections())
        raise ValueError(
            f"Configuration section '{config_name}' not found in {config_filename}. "
            f"Available sections: {available_sections}"
        )

    # Create the configuration dictionary
    the_config = dict(config_obj[config_name])

    # Override with command-line arguments
    assimilate_args(the_config, args)

    # Resolve and compute configuration settings
    base_dir = resolve_base_dir(config_filename)
    the_config.update({"base_dir": base_dir})
    the_config = resolve_placeholders(the_config, {"base_dir": base_dir})

    # Process directories and derived values
    the_config.update(calculate_directories(the_config))
    if "batch_dir" not in the_config:
        raise KeyError("batch_dir is missing after calculate_directories.")
    update_magdir_paths(the_config)
    compute_configs(the_config)

    # Final type conversion
    the_config = {key: convert_value(value) for key, value in the_config.items()}

    # Debugging output
    if debug:
        print_debug_info(the_config)

    return the_config


def find_config_file(config_filename):
    """Find the configuration file in the current directory or subdirectories."""
    config_path = os.path.expanduser(config_filename)
    if os.path.exists(config_path):
        return config_path

    for root, _, files in os.walk(os.getcwd()):
        if config_filename in files:
            return os.path.join(root, config_filename)

    raise FileNotFoundError(f"Configuration file '{config_filename}' not found.")


def clean_config_content(lines):
    """Remove comments and trailing whitespace from configuration file content."""
    return "\n".join(line.split("#", 1)[0].strip() for line in lines if line.strip())


def resolve_base_dir(config_path=None):
    """Determine the base directory dynamically."""
    return os.path.abspath(os.path.dirname(config_path)) if config_path else os.getcwd()


def assimilate_args(config, args=None):
    """Override configuration values with command-line arguments, if provided."""
    if args:
        for arg, value in vars(args).items():
            if value is not None:
                config[arg] = value


def resolve_placeholders(config, placeholders):
    """Resolve placeholders like ${base_dir} in the configuration dictionary."""
    for key, value in config.items():
        if isinstance(value, str):
            for placeholder, replacement in placeholders.items():
                value = value.replace(f"${{{placeholder}}}", replacement)
        config[key] = value
    return config


def calculate_directories(config):
    """Calculate and update directory paths in the configuration."""
    basedir = os.path.expanduser(config.get("base_dir", "").strip())
    batch_name = config.get("batch_name", "").strip()
    dat_dir = config.get("data_dir", os.path.join(basedir, "fluxpype", "data"))

    batch_dir = os.path.join(dat_dir, "batches", batch_name)
    CR = config.get("cr", "")

    return {
        "batch_dir": os.path.expanduser(batch_dir),
        "flocdir": os.path.join(batch_dir, f"cr{CR}/floc"),
        "pipe_dir": os.path.join(basedir, "fluxpype", "fluxpype"),
        "pdl_dir": os.path.join(basedir, "pdl", "PDL"),
        "datdir": os.path.expanduser(dat_dir),
        "data_dir": os.path.expanduser(dat_dir),
        "mag_dir": os.path.join(dat_dir, "magnetograms"),
        "logfile": os.path.join(batch_dir, "pipe_log.txt"),
    }


def update_magdir_paths(config):
    """Update paths related to magnetograms and fluxons."""
    CR = config.get("cr", config.get("rotations", [2100])[0])
    n_fluxons_wanted = config.get("fluxon_count")
    if not CR or not n_fluxons_wanted:
        raise ValueError(
            "Instance values 'cr' or 'fluxon_count' not found in configuration."
        )

    adapt_select = config.get("adapt_select", 0)
    reduction = config.get("mag_reduce", 1)

    if config.get("adapt", False):
        magfile = f"CR{CR}_rf{adapt_select}_adapt.fits"
        flocfile = f"floc_cr{CR}_rf{adapt_select}_f{n_fluxons_wanted}_adapt.dat"
    else:
        magfile = f"CR{CR}_r{reduction}_hmi.fits"
        flocfile = f"floc_cr{CR}_r{reduction}_f{n_fluxons_wanted}_hmi.dat"

    config.update(
        {
            "magfile": magfile,
            "flocfile": flocfile,
            "magpath": os.path.join(config["mag_dir"], magfile),
            "flocpath": os.path.join(config["flocdir"], flocfile),
        }
    )
    return config


def compute_configs(config):
    """Process and compute additional configuration settings."""
    config["abs_rc_path"] = os.path.expanduser(config.get("rc_path", ""))
    config["abs_fl_mhdlib"] = os.path.expanduser(config.get("fl_mhdlib", ""))

    # Parse lists or ranges
    config["rotations"] = parse_list_or_range(config.get("rotations", "[]"))
    config["flow_method"] = parse_list(config.get("flow_method", "[]"))
    config["fluxon_count"] = parse_list(config.get("fluxon_count", "[]"))
    config["adapts"] = parse_list(config.get("adapts", "[]"))

    # Compute derived values
    config["cr"] = config["rotations"][0]
    config["nwant"] = config["fluxon_count"][0]
    config["n_jobs"] = str(
        len(config["rotations"])
        * len(config["fluxon_count"])
        * len(config["adapts"])
        * len(config["flow_method"])
    )


def parse_list_or_range(value):
    """Parse a list or range-like string into a Python list."""
    if value.startswith("[") and value.endswith("]"):  # Parse list
        value = value.strip("[]")
        return [item.strip() for item in value.split(",")]
    elif value.startswith("(") and value.endswith(")"):  # Parse range
        start, stop, step = map(int, value.strip("()").split(","))
        return list(range(start, stop, step))
    return [value.strip()]  # Single value as list


def parse_list(value):
    """Parse a string into a list."""
    return value.strip("[]").split(",") if value else []


def convert_value(value):
    """Convert configuration values to appropriate types."""
    # If the value is already a type other than string, return it as-is
    if not isinstance(value, str):
        return value

    # Try to evaluate the value as Python code
    try:
        return eval(value)
    except (SyntaxError, NameError):
        # If eval fails, return the value as a string
        return value


def print_debug_info(config):
    """Print debug information for the configuration."""
    print("\nDebug Configuration Values:")
    for key, value in sorted(config.items()):
        print(f"{key}: {value}")


# PATH MANAGEMENT ################################################################


def add_dir_to_path(root_dir=None):
    """Adds a directory and all subdirectories to the PATH environment variable.

    Parameters
    ----------
    root_dir : str, optional
        Root directory path
    """

    if root_dir is None:
        root_dir = os.environ("FL_PREFIX", None) or "fluxon-mhd/"

    # Get the current PATH
    current_path = os.environ.get("PATH", "")

    # Initialize a set with the current PATH elements to avoid duplicates
    path_set = set(current_path.split(os.pathsep))

    # Walk through the directory tree
    for dirpath, _, _ in os.walk(root_dir):
        # Add each directory to the set
        path_set.add(dirpath)

    # Convert the set back to a string
    new_path = os.pathsep.join(path_set)

    # Update the PATH
    os.environ["PATH"] = new_path


def add_top_level_dirs_to_path(root_dir):
    """Adds the top-level directories under a root directory to the PATH environment variable.

    Parameters
    ----------
    root_dir : str
        Root directory path
    """

    # Get the current PATH
    current_path = os.environ.get("PATH", "")

    # Initialize a set with the current PATH elements to avoid duplicates
    path_set = set(current_path.split(os.pathsep))

    # List the top-level directories under the root directory
    top_level_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    top_level_dirs.append(f"{root_dir}:")

    # Add each top-level directory to the set
    path_set.update(top_level_dirs)

    # Convert the set back to a string
    new_path = os.pathsep.join(path_set)

    # Update the PATH
    os.environ["PATH"] = new_path

    # news = os.environ.get('PATH', '')
    # news_list = news.split(os.pathsep)
    # news_list.sort()
    # a=[print(x) for x in news_list]
    return new_path


def set_paths(do_plot=False):
    """
    Checks if the environment variable FL_PREFIX is set and optionally prints the paths.

    Parameters:
        do_plot (bool): Whether to print the paths or not.

    Returns:
        None
    """

    fl_prefix = os.environ.get("FL_PREFIX")

    if fl_prefix:
        envpath = os.path.join(fl_prefix, "python_paths.py")

        # Check if the file exists and is readable
        if os.path.exists(envpath) and os.access(envpath, os.R_OK):
            exec(open(envpath).read())
        else:
            print(f"File does not exist or is not readable: {envpath}")
    else:
        print("Environment variable FL_PREFIX is not set.")

    # Optionally print the lists of directories
    if do_plot:
        print("\n\nsys.path has:")
        for path in sys.path:
            print(f" {path}")
        print("--------------------------\n")

        # Add any additional paths you want to print here
        # For example, if you have a list similar to Python's sys.path
        # print("\nYour_Path has:")
        # for path in Your_Path:
        #     print(f" {path}")
        # print("--------------------------\n")


def add_paths(flux_pipe_dir):
    """Adds various paths to the system path.

    Parameters
    ----------
    flux_pipe_dir : str
        FLUXpype directory path

    Returns
    -------
    str
        PDL script directory path
    """

    # Path to the PDL script
    pdl_script_path = flux_pipe_dir + "magnetogram2wind.pdl"
    os.chdir(flux_pipe_dir)
    # Get the plotscript directory path
    plot_dir = os.path.abspath(os.path.join(flux_pipe_dir, "plotting"))
    sys.path.append(plot_dir)
    return pdl_script_path


def find_file_with_string(directory, search_string):
    """Searches a directory for a file containing a given string.

    Parameters
    ----------
    directory : str
        Search directory path
    search_string : str
        Search string

    Returns
    -------
    str
        Search result file path

    """
    for file_name in os.listdir(directory):
        if search_string in file_name:
            return os.path.join(directory, file_name)
    return None


def shorten_path(string, do=False):
    """Removes the DATAPATH environment variable from a string.
    This makes it much more readable when printing paths.

    Parameters
    ----------
    string : str
        String to shorten

    Returns
    -------
    str
        Shortened string
    """
    datapath = os.getenv("DATAPATH")
    if datapath and do:
        return string.replace(datapath, "$DATAPATH")
    else:
        return string


# MAGNETOGRAM MANAGEMENT


def make_mag_dir(datdir):
    """Creates a directory for magnetogram data.

    Parameters
    ----------
    datdir : str
        Data directory path

    Returns
    -------
    str
        Magnetogram directory path
    """

    mag_dir = os.path.join(datdir, "magnetograms")
    if not os.path.exists(mag_dir):
        os.makedirs(mag_dir)
    return mag_dir


# def get_magnetogram_file(cr=None, date=None, datdir=None, email=None,
#                          force_download=False, reduce = False):
#     """
#     Grabs HMI data.

#     Parameters
#     ----------
#     cr : int
#         Carrington rotation number.
#     date : str
#         Date in YYYY-MM-DD format.
#     data_dir : str, optional
#         Directory where data will be stored. If not specified, default directories will be used.

#     Returns
#     -------
#     big_path : str
#         Path to the full resolution magnetogram file.
#     small_path : str
#         Path to the reduced resolution magnetogram file.
#     """

#     # Set the download account
#     try:
#         jsoc_email = email or os.environ["JSOC_EMAIL"]
#     except KeyError:
#         jsoc_email = default_email

#     # Set the Carrington rotation number
#     if cr is not None:
#         CR = cr
#         date = sunpy.coordinates.sun.carrington_rotation_time(CR)
#     elif date is not None:
#         CR = int(sunpy.coordinates.sun.carrington_rotation_number(date))
#     else:
#         raise ValueError("Must specify either cr or date!")
#     mag_dir = make_mag_dir(datdir)

#     print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(f"(py) Getting Magnetogram for CR{CR}, from {date}...")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

#     hmi_object = Path(mag_dir)
#     file_list = list(hmi_object.iterdir())
#     print("\tSearching for file...")
#     found_file = False
#     file = None
#     for file in file_list:
#         if str(CR)+"_r1_" in str(file):
#             print(f"\t\tFound '{os.path.basename(file)}' in '{shorten_path(mag_dir)}'")
#             found_file = True
#             break

#     if found_file:
#         if force_download:
#             print("\tForcing redownload...")
#         else:
#             small_path = reduce_mag_file(file, reduce, force=force_download)
#             return file, small_path

#     # c = drms.Client()
#     # Generate a search
#     crot = a.jsoc.PrimeKey('CAR_ROT', str(CR))
#     res = Fido.search(a.jsoc.Series('hmi.Synoptic_Mr_polfil_720s'), crot,
#                     a.jsoc.Notify(jsoc_email))

#     # Once the query is made and trimmed down...
#     big_path = os.path.join(mag_dir, f"CR{CR}_r1_hmi.fits")
#     # hmi_path = hmidat+'/{file}.fits'

#     print("\tDownloading HMI from JSOC...")
#     out = Fido.fetch(res, path=mag_dir)
#     hmi_path_out = out[0]
#     os.rename(hmi_path_out, big_path)
#     print(f"\n\tSaved to {big_path}\n")

#     small_path = reduce_mag_file(big_path, reduce, force=force_download)
#     return big_path, small_path


def reduce_mag_file(mag_file, reduction=3, force=False):
    """Reduces the size of a magnetogram FITS file by a given factor.

    Parameters
    ----------
    mag_file : str
        Path to input magnetogram file
    reduction : int, optional
        Reduction factor. Defaults to 3.
    force : bool, optional
        Overwrite toggle. Defaults to False.

    Returns
    -------
    small_file: str
        Path to the reduced resolution magnetogram file.
    """

    small_file = PosixPath(str(mag_file).replace("_r1_", f"_r{reduction}_"))
    # reduce the FITS image
    print(f"\tReducing image size by a factor of {reduction}...", end="")

    if not os.path.exists(small_file) or force:
        small_file = reduce_fits_image(
            mag_file, small_file, target_resolution=None, reduction_amount=reduction
        )
        # print("Success!\n")
    else:
        print("Skipped! Reduced file already exists:")
        # print("\t\t", shorten_path(str(small_file), 2))
        ### WORKING HERE
        print(
            f"\t\tFound '{os.path.basename(small_file)}' in '\
              {shorten_path(os.path.dirname(small_file))}'"
        )
        print("\n\t\t\t```````````````````````````````\n \n\n")

    return small_file


def reduce_fits_image(
    fits_path, small_file, target_resolution=None, reduction_amount=None, func=np.nansum
):
    """
    Open a FITS file, reduce the size of the image using astropy's block_reduce
    function, and save a new copy of the FITS file with the smaller image in the
    same directory as the original.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file.
    target_resolution : int, optional
        Target resolution in pixels, if specified reduction_amount is ignored.
    reduction_amount : int, optional
        Amount to reduce the size of the image, if target_resolution is not specified.
    func : numpy function, optional
        Function to use for the reduction, defaults to np.nanmean.
    """

    print(f"\n\tReducing {fits_path}...")
    # Open the FITS file and read the data
    with fits.open(fits_path, ignore_missing_simple=True) as hdul:
        hdul.verify("silentfix")
        data = hdul[0].data
        if data is None:
            data = hdul[1].data

        current_resolution = max(data.shape)
        print("\t\tOriginal Shape:\t", data.shape)

        # Calculate the reduction amount if target resolution is specified
        if target_resolution is not None:
            reduction_amount = int(np.ceil(current_resolution / target_resolution))

        # Raise an error if neither target_resolution nor reduction_amount is specified
        elif reduction_amount is None:
            raise ValueError(
                "Either target_resolution or reduction_amount must be specified."
            )

        before_sum = np.sum(data)
        small_image = block_reduce(data, reduction_amount, func)
        after_sum = np.sum(small_image)
        if not np.isclose(before_sum, after_sum):
            print(
                "\tREDUCTION WARNING: \n\tSum before:    ",
                before_sum,
                "\n\tSum after:     ",
                after_sum,
            )
        try:
            date_check = hdul[0].header["DATE"]
            useheader = hdul[0].header
        except KeyError:
            useheader = hdul[1].header

        del useheader["BLANK"]
        useheader["DATAMIN"] = np.min(small_image)
        useheader["DATAMAX"] = np.max(small_image)
        useheader["BZERO"] = 0
        useheader["BSCALE"] = 1

        useheader["CDELT1"] = 360 / small_image.shape[1]  ## DEGREES
        useheader["CDELT2"] = np.deg2rad(
            360 / (small_image.shape[0] * np.pi)
        )  # RADIANS

        print("\t\tFinal Shape:    ", small_image.shape)

        print("\tSaving  ", small_file)
        fits.writeto(small_file, small_image, useheader, overwrite=True)

        # plot_raw_magnetogram(fits_path, data, small_image)

        print("    Reduction Complete!\n")
    print("```````````````````````````\n")

    return small_file


def plot_raw_magnetogram(fits_path, data, small_image):
    """Plot the magnetogram

    Parameters
    ----------
    fits_path : str
        Magnetogram FITS file path
    data : np.ndarray
        Magnetogram data array
    small_image : np.ndarray
        Small magnetogram data array
    """

    # Save the high resolution image as a grayscale PNG
    plt.axis("off")
    high_res_output_path = fits_path.replace(".fits", ".png")
    fig = plt.gcf()
    shp = data.shape
    dmean = np.nanmean(data)
    dsig = np.nanstd(data)
    thresh = 3
    vmin = dmean - thresh * dsig
    vmax = dmean + thresh * dsig
    plt.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)

    ratio = shp[1] / shp[0]
    sz0 = 6  # inches
    sz1 = sz0 * ratio  # inches
    DPI = shp[1] / sz1  # pixels/inch
    fig.set_size_inches((sz1, sz0))
    plt.savefig(high_res_output_path, bbox_inches="tight", dpi=4 * DPI)
    plt.close()

    # Save the low resolution image as a grayscale PNG
    plt.imshow(small_image, cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")
    low_res_output_path = fits_path.replace(".fits", "_small.png")
    fig = plt.gcf()
    shp = small_image.shape
    ratio = shp[1] / shp[0]
    sz0 = 6  # inches
    sz1 = sz0 * ratio  # inches
    DPI = shp[1] / sz1  # pixels/inch
    fig.set_size_inches((sz1, sz0))
    plt.savefig(low_res_output_path, bbox_inches="tight", dpi=4 * DPI)
    plt.close()


def write_magnetogram_params(datdir, cr, file_path, reduction):
    """Writes the magnetic_target.params file for a given CR and reduction amount.

    Parameters
    ----------
    datdir : str
        Data directory path
    cr : str
        Carrington rotation number
    file_path : str
        File path
    reduction : int
        Reduction factor
    """

    # write the parameter file
    params_path = os.path.join(datdir, "magnetic_target.params")
    with open(params_path, "w", encoding="utf-8") as fp:
        fp.write("## CR_int, Filename_str, Adapt_bool, Doplot_bool, reduction ##\n")
        fp.write(str(cr) + "\n")
        fp.write(str(file_path) + "\n")
        fp.write(str(0) + "\n")
        fp.write(str(0) + "\n")
        fp.write(str(reduction))


def load_magnetogram_params(datdir):
    """Reads the magnetic_target.params file and returns the parameters.

    Parameters
    ----------
    datdir : str
        Data directory path

    Returns
    -------
    hdr : str
        Header information
    cr : str
        Carrington rotation number
    fname : str
        Filename path
    adapt : int
        Adapt specification
    doplot : int
        Plotting toggle
    reduce : int
        Reduction factor
    """

    params_path = os.path.join(datdir, "magnetic_target.params")
    with open(params_path, "r", encoding="utf-8") as fp:
        hdr = fp.readline().rstrip()
        cr = fp.readline().rstrip()
        fname = fp.readline().rstrip()
        adapt = int(fp.readline().rstrip())
        doplot = int(fp.readline().rstrip())
        reduce = int(fp.readline().rstrip())
    return (hdr, cr, fname, adapt, doplot, reduce)


def read_fits_data(fname):
    """Reads FITS data and fixes/ignores any non-standard FITS keywords.

    Parameters
    ----------
    fname : str
        FITS file path

    Returns
    -------
    HDUList
        HDU list read from FITS file
    """
    hdulist = fits.open(fname, ignore_missing_simple=True)
    hdulist.verify("silentfix+warn")
    return hdulist


def get_fixed_coords(phi0, theta0):
    """Corrects input coordinates

    Parameters
    ----------
    phi0 : np.ndarray
        Array of phi coordinates
    theta0 : np.ndarray
        Array of theta coordinates

    Returns
    -------
    phi0: np.ndarray
        Corrected array of phi coordinates
    theta0: np.ndarray
        Corrected array of theta coordinates
    """
    ph0, th0 = phi0 + np.pi, np.sin(-(theta0 - (np.pi / 2)))
    return ph0, th0


## Helper functions to parse the output file


def parse_line(line):
    """Parse a line of the output file into a dictionary.

    Parameters
    ----------
    line : str
        one line of the output file, with key:value pairs separated by commas

    Returns
    -------
    dict
        a dictionary of the key:value pairs
    """

    key_values = line.strip().split(",")
    # print(key_values)
    parsed_dict = {}
    for key_value in key_values:
        if ":" in key_value:
            key, value = key_value.split(":")
            parsed_dict[key.strip()] = convert_value(value)
    return parsed_dict


def load_data(the_path):
    """Load the data from the file into a pandas DataFrame.

    Parameters
    ----------
    the_path : str
        the path to the file to load

    Returns
    -------
    dataframe
        the data from the file given by the_path
    """

    data = []
    print("\n", the_path, "\n")
    with open(the_path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            data.append(parse_line(line))
    return pd.DataFrame(data)


def get_ax(ax=None):
    """Get the fig and ax. If None, create a new fig and ax.
    Otherwise, return the given ax.

    Parameters
    ----------
    ax : pyplot axis or None, optional
        Either an axis or None, by default None

    Returns
    -------
    figure, axis
        a pyplot figure and axis pair
    """
    if ax is not None:
        fig, ax0 = ax.get_figure(), ax
    else:
        fig, ax0 = plt.subplots(1)
    return fig, ax0


def add_fluxon_dirs_to_path(do_print=False):
    """Add the fluxon directories to the system path.

    Parameters
    ----------
    do_print : bool, optional
        print the paths added, by default False
    """

    # Get the current directory path
    this_cwd = os.getcwd()
    # print(f"WE ARE AT {this_cwd}")

    # Get the list of directories in the current directory
    dirs = list_directories(this_cwd)
    dirlist = [os.path.join(this_cwd, x) for x in dirs if "fluxon" in x]

    # Add the pipe and plotting directories to the path
    for thepath in dirlist:
        if "mhd" in thepath:
            dirlist.append(os.path.join(thepath, "fluxpype"))
            dirlist.append(os.path.join(thepath, "fluxpype", "plotting"))
            dirlist.append(os.path.join(thepath, "fluxpype", "science"))
            break

    # Get the pipedir environment variable and add it to the path
    pipedir = os.environ.get("PIPEPATH")
    if pipedir is not None:
        dirlist.append(pipedir)

    path_add(dirlist, do_print=do_print)

    if do_print:
        print("Added fluxon directories to path.\n")

    return dirlist


def path_add(
    dirlist, do_print=False
):  # Add the parent directory to the module search path
    for path in dirlist:
        sys.path.append(path)
        if do_print:
            print(path)


def list_directories(path):
    """List the directories in the given path.

    Parameters
    ----------
    path : str
        the directory to list the subdirectories of

    Returns
    -------
    list
        a list of the subdirectories of the given path
    """

    dirs = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir():
                dirs.append(entry.name)
    return dirs


import sunpy
import sunpy.io
import sunpy.coordinates

# import sunpy.net
from sunpy.net import Fido, attrs as a

# Fido = sunpy.net.Fido
import drms
import os
import glob
import sunpy.coordinates.frames as frames
import astropy.units as u

default_email = "chris.gilly@colorado.edu"
from pathlib import PosixPath

# import subprocess
from astropy.nddata import block_reduce
from astropy.io import fits
import numpy as np
import os
import os.path
import sys

# import ADAPTClient
import matplotlib as mpl

# mpl.use('qt5agg')
import matplotlib.pyplot as plt  # Import libraries
from rich import print

# from sunpy.net.dataretriever import GenericClient


def add_paths(flux_pipe_dir):
    # Path to the PDL script
    pdl_script_path = flux_pipe_dir + "magnetogram2wind.pdl"
    os.chdir(flux_pipe_dir)
    # Get the plotscript directory path
    plot_dir = os.path.abspath(os.path.join(flux_pipe_dir, "plotting"))
    sys.path.append(plot_dir)
    return pdl_script_path


# Magnetogram things


def make_mag_dir(datdir):
    mag_dir = os.path.expanduser(os.path.join(datdir, "magnetograms"))

    if not os.path.exists(mag_dir):
        os.makedirs(mag_dir)
    return mag_dir


def get_magnetogram_file(
    cr=None,
    date=None,
    datdir=None,
    email=None,
    force_download=False,
    reduce=False,
    args=None,
):
    """
    Function to grab HMI data.

    Args:
        cr (int): Carrington rotation number.
        date (str): Date in YYYY-MM-DD format.
        data_dir (str): Optional directory where data will be stored. If not specified,
            default directories will be used.

    Returns:
        None
    """

    # Set the download account
    try:
        jsoc_email = email or os.environ["JSOC_EMAIL"]
    except KeyError:
        jsoc_email = default_email

    # Set the Carrington rotation number
    if cr is not None:
        CR = cr
        date = sunpy.coordinates.sun.carrington_rotation_time(CR)
    elif date is not None:
        CR = int(sunpy.coordinates.sun.carrington_rotation_number(date))
    else:
        raise ValueError("Must specify either cr or date!")

    mag_dir = make_mag_dir(datdir)

    print(
        "\n[cyan]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[/cyan]"
    )
    print(f"[cyan](py) Getting Magnetogram for CR{CR}, from {date}...[/cyan]")
    print(
        "[cyan]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~[/cyan]\n"
    )

    import pathlib

    hmi_object = pathlib.Path(mag_dir)
    file_list = list(hmi_object.iterdir())
    print("\tSearching for file...")
    found_file = False
    inst = "hmi"
    for file in file_list:
        file = str(file.expanduser())
        # print("\t\t\t", file)
        if str(CR) + "_r1_" in str(file) and inst in str(file).casefold():
            print(f"\t\tFound '{os.path.basename(file)}' in '{shorten_path(mag_dir)}'")
            found_file = True
            break

    if found_file:
        if force_download:
            print("\tForcing redownload...")
        else:
            small_path = reduce_mag_file(file, reduce, force=force_download)
            return file, small_path
    print("\t\tFile not found...")
    c = drms.Client()
    # Generate a search
    crot = a.jsoc.PrimeKey("CAR_ROT", str(CR))
    res = Fido.search(
        a.jsoc.Series("hmi.Synoptic_Mr_polfil_720s"), crot, a.jsoc.Notify(jsoc_email)
    )

    # Once the query is made and trimmed down...
    big_path = os.path.join(mag_dir, f"CR{CR}_r1_hmi.fits")
    # hmi_path = hmidat+'/{file}.fits'

    print("\tDownloading HMI from JSOC...")
    out = Fido.fetch(res, path=mag_dir)
    hmi_path_out = out[0]
    os.rename(hmi_path_out, big_path)
    print(f"\n\tSaved to {big_path}\n")

    small_path = reduce_mag_file(big_path, reduce, force=force_download)
    return big_path, small_path


def get_ADAPT_file(
    cr=None,
    date=None,
    datdir=None,
    email=None,
    force_download=False,
    reduce=False,
    method=2,
):
    """
    Function to grab ADAPT data.

    Args:
        cr (int): Carrington rotation number.
        date (str): Date in YYYY-MM-DD format.
        data_dir (str): Optional directory where data will be stored. If not specified,
            default directories will be used.

    Returns:
        None
    """

    ## Parse the Dates
    # Set the Carrington rotation number
    if cr is not None:
        CR = cr
        date = sunpy.coordinates.sun.carrington_rotation_time(CR)
        # date_end = sunpy.coordinates.sun.carrington_rotation_time(CR+1)
    elif date is not None:
        CR = int(sunpy.coordinates.sun.carrington_rotation_number(date))
        # date_end = sunpy.coordinates.sun.carrington_rotation_time(CR+1)
    else:
        raise ValueError("Must specify either cr or date!")

    date_end = date + (1.9999999 * u.hour)

    # Format the Display Dates
    tstring_display = r"%H:%M:%S %m-%d-%Y"
    display_date = date.strftime(tstring_display)
    display_date_end = date_end.strftime(tstring_display)

    # Format the Search Dates
    tstring = r"%Y-%m-%dT%H:%M:%S"
    get_date = date.strftime(tstring)
    get_date_end = date_end.strftime(tstring)

    # Make the directory
    mag_dir = make_mag_dir(datdir)

    print(
        "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    )
    print(
        f"(py) Getting ADAPT Magnetogram(s) for CR{CR}, from {display_date} to {display_date_end}..."
    )
    print(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    )

    print("\tChecking for file...")
    found_file = False

    date_obs = [date, date + (1.9999999 * u.hour), date - (1.9999999 * u.hour)]

    dates = [xx.strftime("%Y%m%d") for xx in date_obs]

    import pathlib

    path_obj = pathlib.Path(mag_dir)
    file_list = list(path_obj.iterdir())

    for file in file_list:
        look = str(CR) + "_r1_"
        file_string = str(file)
        if look in file_string and "adapt" in file_string:
            found_file = file
            break
            for ii, dt in enumerate(dates):
                if dt in str(file):
                    print(
                        f"\t\tFound '{os.path.basename(file)}' in '{shorten_path(mag_dir)}'"
                    )
                    print("\t\tDate: ", date_obs[ii].strftime(tstring_display))
                    found_file = True
                    break
            if found_file:
                break

    if found_file:
        if force_download:
            print("\tForcing redownload...")
        else:
            print("\tSkipping Download!\n")
            # small_path = None # reduce_mag_file(file, reduce, force=force_download)
            # mean_path = format_ADAPT_file(file, reduce, force=force_download)
            the_path = format_ADAPT_file(file, method=method, force=force_download)
            return file, the_path
    else:
        print("\t\tNo file found!")

    print("\n\tSearching FIDO for ADAPT Map...\n")
    from fluxpype.fidoclients.ADAPTClient import ADAPTLngType

    LngType = "0"  # 0 is carrington, 1 is central meridian
    print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
    res = Fido.search(
        a.Instrument("adapt"), a.Time(get_date, get_date_end), ADAPTLngType(LngType)
    )
    print(res)
    print("\tDownloading ADAPT map...\n")
    out = Fido.fetch(res, path=mag_dir)
    assert len(out) == 1, f"More than one file found! {out}"
    if len(out) == 1:
        print("\tSuccess!")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    file_all = str(out[0])
    # file_end = file_all.split('adapt')[1].split('.')
    # file_end = file_all.split('adapt')[1].split('.')[1:]
    big_path = os.path.join(mag_dir, f"CR{CR}_r1_adapt.fts.gz")

    os.rename(file_all, big_path)
    print(f"\n\t\tSaved to {big_path}")

    # mean_path = format_ADAPT_file(big_path, reduce, force=force_download)
    the_path = format_ADAPT_file(big_path, method=method, force=force_download)

    return big_path, the_path


def expose_adapt_fits(adapt_fits):
    return [
        print(x, " : ", adapt_fits[0].header[x])
        for x in adapt_fits[0].header
        if not "KEYCOMMENTS" in x
    ]


def print_adapt_maps(adapt_maps):
    for map in adapt_maps:
        print(map)


def plot_all_adapt(adapt_maps):
    from matplotlib import gridspec

    fig = plt.figure(figsize=(7, 8))
    gs = gridspec.GridSpec(4, 3, figure=fig)
    for i, a_map in enumerate(adapt_maps):
        ax = fig.add_subplot(gs[i], projection=a_map)
        a_map.plot(axes=ax, cmap="bwr", vmin=-2, vmax=2, title=f"Realization {1+i:02d}")

    # adapt_maps.plot()
    plt.tight_layout(pad=5, h_pad=2)
    plt.show(block=True)


def plot_mean_adapt(mean_adapt):
    map_mean = np.nanmean(mean_adapt)
    map_std = np.std(mean_adapt)
    plt.imshow(mean_adapt, vmin=map_mean - 2 * map_std, vmax=map_mean + 2 * map_std)
    plt.show(block=True)


def format_ADAPT_file(filename, method="mean", force=False, configs=None):
    import sunpy.io
    import sunpy.map

    print(filename)

    if method == "mean":
        out_file_name = str(filename).replace("_r1_", "_rmean_")
    else:
        out_file_name = str(filename).replace("_r1_", f"_rf{method}_")

    out_file_name = out_file_name.replace(".fts.gz", ".fits")
    # print(out_file_name)

    if os.path.exists(out_file_name) and not force:
        print("\tFile already formatted!")
        print("\t\t", shorten_path(out_file_name), "\n")
        print("\t\t\t```````````````````````````````\n\n")

        return out_file_name

    print("\n\tFormatting ADAPT file...", end="")
    adapt_fits = sunpy.io.read_file(filename)
    main_data = adapt_fits[0].data
    main_header = adapt_fits[0].header

    if not "CRVAL1" in main_header.keys():
        main_header = adapt_fits[1].header
        main_data = adapt_fits[1].data

    # a = [print(x, main_header[x]) for x in main_header.keys()]
    # a = [print(x, main_header["KEYCOMMENTS"][x]) for x in main_header["KEYCOMMENTS"].keys()]

    # main_header['DATE-AVG'] = main_header['MAPTIME']
    with fits.open(filename, ignore_missing_simple=True) as hdul:
        hdul.verify("silentfix")
        header2 = hdul[0].header

    if method == "mean":
        data_header_pairs = [(map_slice, main_header) for map_slice in main_data]
        adapt_maps = sunpy.map.Map(data_header_pairs, sequence=True)
        adapt_cube = np.asarray([the_map.data for the_map in adapt_maps])
        output_map = np.nanmean(adapt_cube, axis=0)

        # if False:
        #     # Lots of Plots
        #     plot_all_adapt(adapt_maps)
        #     expose_adapt_fits(adapt_fits)
        #     print_adapt_maps(adapt_maps)
        #     plot_mean_adapt(mean_adapt)

    elif isinstance(method, int):
        adapt_map = sunpy.map.Map((main_data[method], main_header))
        output_map = np.asarray(adapt_map.data)
        print(output_map.shape)
    else:
        assert False, "Method not recognized!"

    useheader = fix_header_ADAPT(header2, output_map)
    fits.writeto(out_file_name, output_map, useheader, overwrite=True)

    print("Success!")
    print("\t\tSaved to", out_file_name, "\n")

    return out_file_name


def reduce_mag_file(mag_file, reduction=3, force=False):
    """Reduces the size of a magnetogram FITS file by a given factor."""
    small_file = PosixPath(str(mag_file).replace("_r1_", f"_r{reduction}_"))
    # reduce the FITS image
    print(f"\tReducing image size by a factor of {reduction}...", end="")
    if not os.path.exists(small_file) or force:
        small_file = reduce_fits_image(
            mag_file, small_file, target_resolution=None, reduction_amount=reduction
        )
        # print("Success!\n")
    else:
        print("Skipped! Reduced file already exists:")
        # print("\t\t", shorten_path(str(small_file), 2))
        ### WORKING HERE
        print(
            f"\t\tFound '{os.path.basename(small_file)}' in '{os.path.dirname(small_file)}'"
        )
        print("\n\t\t\t```````````````````````````````\n \n\n")

    return small_file


def read_fits_data(fname):
    """Reads FITS data and fixes/ignores any non-standard FITS keywords."""
    hdulist = fits.open(fname, ignore_missing_simple=True)
    hdulist.verify("silentfix+warn")
    return hdulist


def reduce_fits_image(
    fits_path, small_file, target_resolution=None, reduction_amount=None, func=np.nansum
):
    """
    Open a FITS file, reduce the size of the image using astropy's block_reduce
    function, and save a new copy of the FITS file with the smaller image in the
    same directory as the original.

    :param fits_path: str, path to the FITS file
    :param target_resolution: int, optional, target resolution in pixels, if
                              specified reduction_amount is ignored
    :param reduction_amount: int, optional, amount to reduce the size of the
                              image, if target_resolution is not specified
    :param func: numpy function, optional, function to use for the reduction
                              defaults to np.nanmean
    """
    print(f"\n\tReducing {fits_path}...")
    # Open the FITS file and read the data
    with fits.open(fits_path, ignore_missing_simple=True) as hdul:
        hdul.verify("silentfix")
        data = hdul[0].data
        if data is None:
            data = hdul[1].data

        current_resolution = max(data.shape)
        print("\tOriginal Shape: ", data.shape)

        # Calculate the reduction amount if target resolution is specified
        if target_resolution is not None:
            reduction_amount = int(np.ceil(current_resolution / target_resolution))

        # Raise an error if neither target_resolution nor reduction_amount is specified
        elif reduction_amount is None:
            raise ValueError(
                "Either target_resolution or reduction_amount must be specified."
            )

        before_sum = np.sum(data)
        small_image = block_reduce(data, reduction_amount, func)
        after_sum = np.sum(small_image)
        if not np.isclose(before_sum, after_sum):
            print(
                "\tREDUCTION WARNING: \n\tSum before:    ",
                before_sum,
                "\n\tSum after:     ",
                after_sum,
            )

        try:
            hdul[0].header["DATE"]
            useheader = hdul[0].header
        except KeyError:
            useheader = hdul[1].header

        try:
            del useheader["BLANK"]
        except KeyError:
            print("No BLANK keyword found!")

        useheader = fix_header(useheader, small_image)
        # small_file = fits_path.replace('_r1_', f'_r{reduction_amount}_')

        # del useheader['BLANK']
        # useheader['DATAMIN'] = np.min(small_image)
        # useheader['DATAMAX'] = np.max(small_image)
        # useheader['BZERO'] = 0
        # useheader['BSCALE'] = 1

        # useheader['CDELT1'] = 360 / small_image.shape[1]  ## DEGREES
        # useheader['CDELT2'] = np.deg2rad(360 / (small_image.shape[0] * np.pi)) #RADIANS

        print("\tFinal Shape: ", small_image.shape)

        print("\tSaving  ", small_file)
        fits.writeto(small_file, small_image, useheader, overwrite=True)

        # plot_raw_magnetogram(fits_path, data, small_image)

        print("    Reduction Complete!\n")
    print("```````````````````````````\n")

    return small_file


def fix_header(useheader, image):

    useheader["DATAMIN"] = np.min(image)
    useheader["DATAMAX"] = np.max(image)
    useheader["BZERO"] = 0
    useheader["BSCALE"] = 1

    useheader["CDELT1"] = 360 / image.shape[1]  ## DEGREES
    useheader["CDELT2"] = np.deg2rad(360 / (image.shape[0] * np.pi))  # RADIANS
    return useheader


def fix_header_ADAPT(useheader, image):
    useheader["DATAMIN"] = np.min(image)
    useheader["DATAMAX"] = np.max(image)
    useheader["BZERO"] = 0
    useheader["BSCALE"] = 1

    # import pdb; pdb.set_trace()
    useheader["CDELT1"] = 360 / image.shape[1]  ## DEGREES
    useheader["CDELT2"] = 360 / (image.shape[0] * np.pi)  ## DEGREES
    return useheader


def plot_raw_magnetogram(fits_path, data, small_image):
    # Save the high resolution image as a grayscale PNG
    plt.axis("off")
    high_res_output_path = fits_path.replace(".fits", ".png")
    fig = plt.gcf()
    shp = data.shape
    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    dmean = np.nanmean(data)
    dsig = np.nanstd(data)
    thresh = 3
    vmin = dmean - thresh * dsig
    vmax = dmean + thresh * dsig
    plt.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)

    ratio = shp[1] / shp[0]
    sz0 = 6  # inches
    sz1 = sz0 * ratio  # inches
    DPI = shp[1] / sz1  # pixels/inch
    fig.set_size_inches((sz1, sz0))
    plt.savefig(high_res_output_path, bbox_inches="tight", dpi=4 * DPI)
    plt.close()

    # Save the low resolution image as a grayscale PNG
    plt.imshow(small_image, cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")
    low_res_output_path = fits_path.replace(".fits", "_small.png")
    fig = plt.gcf()
    shp = small_image.shape
    ratio = shp[1] / shp[0]
    sz0 = 6  # inches
    sz1 = sz0 * ratio  # inches
    DPI = shp[1] / sz1  # pixels/inch
    fig.set_size_inches((sz1, sz0))
    plt.savefig(low_res_output_path, bbox_inches="tight", dpi=4 * DPI)
    plt.close()


# def load_fits_magnetogram(datdir = "~/vscode/fluxon-data/", batch="fluxon", bo=2, bn=2, ret_all=False):
#     """Loads a magnetogram from a FITS file."""
#     fname = load_magnetogram_params(datdir)[2].replace("/fluxon/", f"/{batch}/").replace(f"_{bo}_", f"_{bn}_")
#     fits_path = datdir + fname
#     try:
#         hdulist = read_fits_data(fits_path)
#     except FileNotFoundError as e:
#         hdulist = read_fits_data(fname)
#     brdat = hdulist[0].data
#     header= hdulist[0].header
#     brdat = brdat - np.mean(brdat)
#     if ret_all:
#         return brdat, header
#     else:
#         return brdat


def load_fits_magnetogram(
    datdir=None,
    batch=None,
    bo=2,
    bn=2,
    ret_all=False,
    fname=None,
    configs=None,
    cr=None,
):
    """Loads a magnetogram from a FITS file.

    Parameters
    ----------
    datdir : str, optional
        Data directory path. Defaults to None.
    batch : str, optional
        Output file descriptor label. Defaults to "fluxon".
    bo : int, optional
        Output file descriptor label. Defaults to 2.
    bn : int, optional
        Output file descriptor label. Defaults to 2.
    ret_all : bool, optional
        Toggle to return both data and header. Defaults to False.

    Returns
    -------
    np.ndarray
        Output magnetogram data array
    Header
        Magnetogram header object
    """
    configs = configs or configurations()
    if cr is None:
        cr = configs.get("cr", None)
    else:
        # TODO : This is a hack to get around the fact that the configs are not being updated
        configs["cr"] = cr

    assert cr is not None, "Must specify a Carrington rotation number!"
    update_magdir_paths(configs)
    fname = fname or configs["magpath"].format(cr)
    batch = batch or configs["batch_name"]
    datdir = datdir or configs["data_dir"]

    # fname = load_magnetogram_params(datdir)[2]
    # fname = fname.replace("/fluxon/", f"/{batch}/").replace(f"_{bo}_", f"_{bn}_")
    # fits_path = datdir + fname
    hdulist = read_fits_data(fname)
    # try:
    #     hdulist = read_fits_data(fits_path)
    # except FileNotFoundError:
    brdat = hdulist[0].data
    header = hdulist[0].header
    brdat = brdat - np.mean(brdat)
    header["CUNIT2"] = "rad"

    if ret_all:
        return brdat, header
    else:
        return brdat


# File I/O and pathing
def find_file_with_string(directory, search_string):
    """Searches a directory for a file containing a given string."""
    for file_name in os.listdir(directory):
        if search_string in file_name:
            return os.path.join(directory, file_name)
    return None


def shorten_path(string, __=None, do=False):
    datapath = os.getenv("DATAPATH")
    if datapath and do:
        return string.replace(datapath, "$DATAPATH ")
    else:
        return string


def get_fixed_coords(phi0, theta0, do=True):
    if do:
        ph0, th0 = phi0 + np.pi, np.sin(-(theta0 - (np.pi / 2)))
    else:
        ph0, th0 = phi0, theta0
    return ph0, th0


fields = ["ph0", "th0", "fr0", "vel0", "ph1", "th1", "fr1", "vel1", "polarity"]


def load_wind_files(directory):
    """
    Load wind files into a nested dictionary based on CR and data fields.

    Parameters:
    - directory: Path to the directory containing the files.

    Returns:
    A nested dictionary {CR: {field: data, ...}, ...}.
    """
    # Define the fields based on the order in the saved array
    big_dict = {}

    # Regex to extract details from filename
    pattern = re.compile(r"cr(\d+)_f(\d+)_op(\d+)_radial_wind_(\w+).npy")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            CR, nwant, n_open, method = match.groups()
            CR = int(CR)  # Convert CR to integer for use as a dictionary key

            # Ensure the CR key exists in the dictionary
            if CR not in big_dict:
                big_dict[CR] = {}

            file_path = os.path.join(directory, filename)
            data = np.load(file_path)

            # Assume data is saved in the order specified above
            for i, field in enumerate(fields):
                big_dict[CR][field] = data[i]

    return big_dict


from datetime import datetime, timedelta


def decimal_years_to_datetimes(decimal_years):
    def convert(decimal_year):
        year = int(decimal_year)
        remainder = decimal_year - year
        start_of_year = datetime(year, 1, 1)
        # Check if it's a leap year
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            days_in_year = 366
        else:
            days_in_year = 365
        days = remainder * days_in_year
        return start_of_year + timedelta(days=days)

    return [convert(year) for year in decimal_years]


def sunspotplot(carr_ax, cr=None, use_years=False):
    # Plot the Sunspot Number
    carrington = np.loadtxt("fluxpype/SN_m_tot_V2.0.tsv").T
    ## https://sidc.be/SILSO/datafiles#total ##
    from sunpy.coordinates.sun import (
        carrington_rotation_time as crt,
        carrington_rotation_number as crn,
    )

    # import pdb; pdb.set_trace()
    date = carrington[2]
    sunspots = carrington[3]

    if cr is not None:
        this_date = crt(cr)
        if use_years:
            carr_ax.axvline(this_date.decimalyear, ls=":", c="k", zorder=1000000)
        else:
            carr_ax.axvline(cr, ls=":", c="k", zorder=1000000)
    # fig, ax = plt.subplots()
    if use_years:
        carr_ax.plot(date, sunspots, label="Sunspots", color="b", lw=2)
        carr_ax.set_xlim(crt(2095).decimalyear, crt(2282).decimalyear)

    else:
        datetimes = decimal_years_to_datetimes(date)
        CR = crn(datetimes)
        carr_ax.plot(CR, sunspots, label="Sunspots", color="b", lw=2)
        carr_ax.set_xlim(2095, 2282)
    # carr_ax.set_xlabel("Year")
    carr_ax.set_ylabel("Sunspots")
    carr_ax.set_title("Solar Cycle Phase")
    # carr_ax.axhline(100, c='k', ls="--")
    # set the major tick formatter to display integers
    from matplotlib.ticker import MaxNLocator

    carr_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # carr_ax.set_ylim(0, 200)


VMIN, VMAX = 450, 700


def parse_big_dict(big_dict, field="vel1", vmin=VMIN, vmax=VMAX):
    # Prepare data for interpolation and plotting
    all_phi = []
    all_theta = []
    all_vel = []
    all_hist = []
    all_cr = []
    all_mean = []
    all_std = []
    all_count = []
    the_phi = "ph1" if "1" in field else "ph0"
    the_theta = "th1" if "1" in field else "th0"

    # Collect data from all CRs
    for ii, CR in enumerate(sorted(big_dict.keys())):

        phi1 = big_dict[CR][the_phi] / (2 * np.pi) + ((CR - 0.5))
        theta1 = big_dict[CR][the_theta]
        vel1_data = big_dict[CR][field]
        hist, bins = np.histogram(vel1_data, range=(vmin, vmax), bins=20, density=True)

        all_phi.extend(phi1)
        all_theta.extend(theta1)
        all_vel.extend(vel1_data)

        all_hist.append(hist)
        all_mean.append(np.mean(vel1_data))
        all_std.append(np.std(vel1_data))
        all_cr.append(CR)
        all_count.append(len(vel1_data))

    # Convert lists to numpy arrays for griddata
    all_phi = np.array(all_phi)
    all_theta = np.array(all_theta)
    all_vel = np.array(all_vel)
    all_hist = np.array(all_hist)
    all_cr = np.array(all_cr)
    all_mean = np.array(all_mean)
    all_std = np.array(all_std)

    total_mean = np.mean(all_mean)
    total_std = np.mean(all_std)
    all_count = np.array(all_count)

    clip = total_mean + 2 * total_std
    all_hist[all_hist > clip] = clip
    return (
        all_phi,
        all_theta,
        all_vel,
        all_hist,
        all_cr,
        all_mean,
        all_std,
        total_mean,
        total_std,
        all_count,
    )



# A quick function to read output FLUX world data

# Import libraries
import numpy

def rdworld(filename):

    # Some quick testing on storing this data in a class object
    class world(object):

        def __init__(self):
            self.fc = self.fc()
            self.fx = self.fx()

        # Flux concentrations
        class fc:
            def __init__(self):
                self.id = []
                self.x = []
                self.y = []
                self.z = []
                self.fl = []

        def add_fc(self, id, x, y, z, fl):
            self.fc.id.append(id)
            self.fc.x.append(x)
            self.fc.y.append(y)
            self.fc.z.append(z)
            self.fc.fl.append(fl)

        # Fluxons
        class fx:
            def __init__(self):
                self.id = []
                self.fc0 = []
                self.fc1 = []
                self.fl = []
                self.x = []
                self.y = []
                self.z = []

        def add_fx(self, id, fc0, fc1, fl, r0, r1, x, y, z):
            self.fx.id.append(id)
            self.fx.fc0.append(fc0)
            self.fx.fc1.append(fc1)
            self.fx.fl.append(fl)
            self.fx.x.append(numpy.array([r0[0]] + x + [r1[0]]))
            self.fx.y.append(numpy.array([r0[1]] + y + [r1[1]]))
            self.fx.z.append(numpy.array([r0[2]] + z + [r1[2]]))

    # Let there be light
    w = world()

    # Now actually go about reading the data
    df = open(filename, 'r')

    # Initialize data storage
    ln_id = []
    ln_fc0 = []
    ln_fc1 = []
    ln_fl = []
    ln_r0 = []
    ln_r1 = []

    vx_lid = []
    vx_id = []
    vx_pos = []
    vx_x = []
    vx_y = []
    vx_z = []

    for rl in df:
        sl = rl.strip().split()

        # Skip ahead for blank lines
        if len(sl) == 0: continue

        # Read out flux concentrations
        if sl[0] == 'NEW':
            w.add_fc(numpy.int(sl[1]), numpy.double(sl[2]), numpy.double(sl[3]), numpy.double(sl[4]), numpy.double(sl[5]))

        # Read out fluxon data
        if sl[0] == 'LINE':
            if numpy.int(sl[1])>0:
                ln_id.append(numpy.int(sl[1]))
                ln_fc0.append(numpy.int(sl[4]))
                ln_fc1.append(numpy.int(sl[5]))
                ln_fl.append(numpy.double(sl[6]))
                ln_r0.append([numpy.double(sl[7]), numpy.double(sl[8]), numpy.double(sl[9])])
                ln_r1.append([numpy.double(sl[10]), numpy.double(sl[11]), numpy.double(sl[12])])

        # Read out vertex points
        if sl[0] == 'VERTEX':
            vx_lid.append(numpy.int(sl[1]))
            vx_id.append(numpy.int(sl[2]))
            vx_pos.append(numpy.int(sl[3]))
            vx_x.append(numpy.double(sl[4]))
            vx_y.append(numpy.double(sl[5]))
            vx_z.append(numpy.double(sl[6]))

        # Exit on neighbor information
        if 'VNEIGHBOR' in sl[0]:
            break

    df.close()

    # Convert lists into numpy arrays
    ln_id  = numpy.array(ln_id)
    ln_fc0 = numpy.array(ln_fc0)
    ln_fc1 = numpy.array(ln_fc1)
    ln_fl = numpy.array(ln_fl)
    ln_r0 = numpy.array(ln_r0)
    ln_r1 = numpy.array(ln_r1)

    vx_lid = numpy.array(vx_lid)
    vx_id = numpy.array(vx_id)
    vx_pos = numpy.array(vx_pos)
    vx_x = numpy.array(vx_x)
    vx_y = numpy.array(vx_y)
    vx_z = numpy.array(vx_z)

    # Parse the line and vertex lists and create fluxon objects
    for lid in ln_id:
        wl = numpy.where(ln_id == lid)[0]
        wv = numpy.where(vx_lid == lid)[0]

        w.add_fx(lid, ln_fc0[wl][0], ln_fc1[wl][0], ln_fl[wl][0], ln_r0[wl,:][0].tolist(), ln_r1[wl,:][0].tolist(), vx_x[wv].tolist(), vx_y[wv].tolist(), vx_z[wv].tolist())

    return w

# A quick comment on reading FLUX output world files:

# Flux concentrations
# NEW	100	0.314000	-0.059000	-0.948000	-1	(null)	0
# FC    ID      X               Y               Z               Flux    BND PT  Rad

# Line
# LINE	101	-302	-301	-1	100	1            7.61951 -2.51477 -7.52456   0.314 -0.059 -0.948
# Line  ID      Start   End     FC0     FC1     Flux        StX     StY     StZ     EndX    EndY    EndZ

# Vertex
# VERTEX	101	10650	1	7.203456	-2.377456	-7.113698
# Vertex        LineID  VtxID   Pos     X               Y               Z