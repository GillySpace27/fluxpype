"""Download a Magnetogram for a Particular Carrington Rotation
===========================================================
import argparse


This module automates the process of downloading a magnetogram for a specific Carrington Rotation (CR).
It provides options to specify the CR, data directory, reduction factor, and whether to force download the files.

Attributes
----------
configs : dict
    The configuration parameters loaded from 'config.ini'.

Functions
---------
get_magnetogram_file : function
    Fetches the magnetogram file based on the provided arguments.

Examples
--------
Run the script with the following command:
    python get_magnetograms.py --cr 2220  --reduce 4

Authors
-------
Gilly <gilly@swri.org> (and others!)
"""


import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from importlib import import_module
# pipe_helper = import_module("pipe_helper")
# from . import pipe_helper
# import fluxpype
# pipe_helper = fluxpype.pipe_helper
from fluxpype.pipe_helper import configurations, get_magnetogram_file, get_ADAPT_file


# configurations = pipe_helper.configurations
# get_magnetogram_file = pipe_helper.get_magnetogram_file
# get_ADAPT_file = pipe_helper.get_ADAPT_file

# from fluxpype.pipe_helper import (configurations, get_magnetogram_file, get_ADAPT_file)
configs = configurations()
# Create the argument parser
parser = argparse.ArgumentParser(description='This script downloads a magnetogram for a particular Carrington Rotation')
parser.add_argument('--cr', type=int, default=2100, help='Carrington Rotation')
parser.add_argument('--datdir', type=str, default=configs["data_dir"], help='data directory')
parser.add_argument('--reduce', type=int, default=5, help='factor by which the magnetogram is reduced')
parser.add_argument('--do_download', type=int, default=0, help='download the files')
parser.add_argument('--adapt', type=int, default=1, help='download the files')
args = parser.parse_args()

# get the magnetogram files
# get_adapt = True
if args.adapt:
    big_path, processed_path = get_ADAPT_file(cr=args.cr, datdir=args.datdir, force_download=args.do_download, reduce=args.reduce)
else:
    big_path, small_path = get_magnetogram_file(cr=args.cr, datdir=args.datdir, force_download=args.do_download, reduce=args.reduce, args=args)
