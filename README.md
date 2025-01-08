# FLUXpype

## Preamble
This package is in active development, and acts as a wrapper around the Field Line Universal relaXer (FLUX), a solar magnetic field model. This means that it requires the FLUX perl/C software located [on Github](https://github.com/lowderchris/fluxon-mhd) to be installed separately. Please contact the authors if you struggle with the FLUX installation provided on its page or the integration with FLUXpype here.

## Installation

Open a terminal, navigate to your desired repo library directory, then type the following commands to download and install it in a local virtual environment:

> git clone [https://github.com/GillySpace27/fluxpype.git](https://github.com/GillySpace27/fluxpype.git)

This may ask you to install the developer tools, which you will need. Run the command again, then run:

> cd fluxpype

> python3 -m venv .venv

> source .venv/bin/activate

> python3 comprehensive_fluxpype_installer.py

This will likely fail the first time, and you'll just restart the process by opening a new terminal, reactivating the python environment, and running this command again

Once the package has been locally installed, test it out by typing this from within the fluxpype directory:

> python3 fluxpype \
or \
> flux-config-run

If python doesn't see the fluxpype module, try running:

> python3 -m pip3 install -e .

Another simple test you can run is to type

> perl -e "use Flux"

If this throws an exception, your FLUX code isn't correctly compiled and linked. Please [see the repo for the FLUX project](https://github.com/lowderchris/fluxon-mhd) for installation instructions.

## Configuration
Editing the configuration file by running the flux-config-edit command is one of the primary ways to interface with the code. The file can also be edited directly. The second line of the config file selects which profile that fluxpype will use. Any settings not redefined in that profile will use the default settings from the first block ("DEFAULT").

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
