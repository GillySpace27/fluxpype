# FLUXpype

## Preamble
This package acts as a wrapper around the Field Line Universal relaXer (FLUX), a solar magnetic field model. This means that it requires the FLUX perl/C software located on Github.

## Installation

FLUX has two halves: the **core** (a C library with Perl/PDL bindings, in
[fluxon-mhd](https://github.com/lowderchris/fluxon-mhd)) and this **Python wrapper**.

1. **Build the core** by following the guide for your platform in
   [fluxon-mhd/doc](https://github.com/lowderchris/fluxon-mhd/tree/main/doc).
   It is done when, in a *new* terminal, `perl -MPDL -MFlux -e1` prints nothing and returns.
2. **Install fluxpype** (Python 3.10 to 3.12):

   ```sh
   git clone https://github.com/GillySpace27/fluxpype.git
   cd fluxpype
   ./install.sh
   ```

   This creates `./.venv`, installs fluxpype into it, and runs `flux-doctor`,
   which checks Python, Perl, PDL, Flux, gnuplot and the config, and prints
   the fix for anything missing. Run `flux-doctor` again any time.
3. **Set your JSOC email** in `fluxpype/config.ini` (`jsoc_email`) or as
   `JSOC_EMAIL`. JSOC only sends data to registered addresses
   ([register here](http://jsoc.stanford.edu/ajax/register_email.html)).
4. **First run** (one rotation, 100 fluxons, about 10 minutes):

   ```sh
   source .venv/bin/activate
   flux-config-run --config quickstart
   ```

   `flux-config-run` exits nonzero if any job fails, and prints which ones.

## Configuration
Once the code is all installed correctly, you'll be able to edit the configuration file and run it in the following manner.

Editing the configuration file by running the flux-config-edit command is one of the primary ways to interface with the code. The file can also be edited directly. The second line of the config file (`config_name`) selects the default profile; `flux-config-run --config NAME` overrides it for one run. Any settings not redefined in that profile will use the default settings from the first block ("DEFAULT").

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
