# FLUXpype

## Preamble
This package is in active development, and acts as a wrapper around the Field Line Universal relaXer (FLUX), a solar magnetic field model. This means that it requires the FLUX perl/C software located [on Github](https://github.com/lowderchris/fluxon-mhd).

## Installation

Instructions for installation on Linux or on Macos can be found in the [main FLUX repo here](https://github.com/lowderchris/fluxon-mhd/blob/7ef558f047bd74a8be0f5d38807152000b9adc66/doc).


# (Accuracy below this point is dubious)

## Configuration
Editing the configuration file by running the flux-config-edit command is one of the primary ways to interface with the code. The file can also be edited directly. The second line of the config file selects which profile that fluxpype will use. Any settings not redefined in that profile will use the default settings from the first block ("DEFAULT").

To see the details of the current job configuration:
> flux-config-view

To edit the details of the configuration, modify config.ini. You can find that by typing:
> flux-config-edit

## Running
The pipe can be invoked in four ways.

> \% flux-config-run

> \% python3 fluxpype/config_runner.py

> \% python3 fluxpype

> \% python3 \
>\>\>\> from fluxpype.config_runner import run \
>\>\>\> run()


## Contributing

This project welcomes contributions and suggestions. For details, visit the repository's [Contributor License Agreement (CLA)](https://cla.opensource.microsoft.com) and [Code of Conduct](https://opensource.microsoft.com/codeofconduct/) pages.
