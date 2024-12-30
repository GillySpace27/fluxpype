# FLUXpype

## Preamble
This package is in active development, and acts as a wrapper around the Field Line Universal relaXer (FLUX), a solar magnetic field model. This means that it requires the FLUX perl/C software located [on Github](https://github.com/lowderchris/fluxon-mhd) to be installed separately. Automating the install and integration of these two tools is beyond the scope of this release, but please contact the authors if you struggle with the FLUX installation provided on its page or the integration with FLUXpype here.

## Installation

Open a terminal, navigate to your desired repo library directory, then type the following to download and install it in a local virtual environment:

> git clone [https://github.com/GillySpace27/fluxpype.git](https://github.com/GillySpace27/fluxpype.git)

> cd fluxpype

> python3 -m venv .venv

> source .venv/bin/activate

> pip install .

Once the package has been locally installed, test it out by typing this:

> python3 fluxpype \
or \
> flux_config_run

A simple test you can run is to type

> perl -e "use Flux"

If this throws an exception, your FLUX code isn't correctly compiled and linked. Please [see the repo for the FLUX project](https://github.com/lowderchris/fluxon-mhd) for installation instructions.

## Configuration
Editing the configuration file by running the flux_config_edit command is one of the primary ways to interface with the code. The file can also be edited directly. The second line of the config file selects which profile that fluxpype will use. Any settings not redefined in that profile will use the default settings from the first block ("DEFAULT").

To see the details of the current job configuration:
> flux_config_view

To edit the details of the configuration, modify config.ini. You can find that by typing:
> flux_config_edit

## Running
The pipe can be invoked in four ways.

> \% flux_config_run

> \% python3 fluxpype/config_runner.py

> \% python3 fluxpype

> \% python3 \
>\>\>\> from fluxpype.config_runner import run \
>\>\>\> run()


## Contributing

This project welcomes contributions and suggestions. For details, visit the repository's [Contributor License Agreement (CLA)](https://cla.opensource.microsoft.com) and [Code of Conduct](https://opensource.microsoft.com/codeofconduct/) pages.
