"""
Config Runner: Automate the FLUXpype Workflow for Multiple Carrington Rotations
===============================================================================

This module serves as a workflow automation tool for the FLUXpype project. It runs
the `magnetogram2wind.pdl` script for various Carrington Rotations (CRs) specified
in an external 'config.ini' file. The script leverages Python's subprocess module to
invoke the PDL script and uses the tqdm library for tracking progress.

Attributes
----------
configs : dict
    The configuration parameters loaded from 'config.ini'.

Examples
--------
1. Edit the 'config.ini' file to configure your desired settings.
2. Run the script using `python config_runner.py`.

Authors
-------
Gilly <gilly@swri.org> (and others!)

"""

print(__file__)

import subprocess
from tqdm import tqdm
from pipe_helper import configurations
import timeout_decorator
import os
from rich import print
from rich.panel import Panel
import logging

# Setting up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

configs = configurations(debug=False)
timeout_num = 0


def validate_configurations(configs):
    """Validate the configuration parameters."""
    required_keys = [
        "rotations",
        "fluxon_count",
        "adapts",
        "flow_method",
        "run_script",
        "n_jobs",
    ]
    for key in required_keys:
        if key not in configs:
            logging.error(f"Missing configuration parameter: {key}")
            raise KeyError(f"Configuration parameter '{key}' is missing.")


def display_splash_screen():
    """Displays a splash screen with configuration details."""
    splash_text = f"""
    [bold blue]Config Runner: Automate the FLUXpype Workflow for Multiple Carrington Rotations[/bold blue]
    [green]===============================================================================[/green]

    [bold]Configurations:[/bold]
    Rotations: {configs["rotations"]}
    Fluxon Count: {configs["fluxon_count"]}
    Adaptations: {configs["adapts"]}
    Methods: {configs["flow_method"]}
    Run Script: {configs["run_script"]}
    Number of Jobs: {configs["n_jobs"]}

    Authors: Gilly <gilly@swri.org> (and others!)

    [green]===============================================================================[/green]
    """
    print(
        Panel(
            splash_text,
            title="Welcome to FLUXpype Config Runner",
            border_style="bold green",
        )
    )


@timeout_decorator.timeout(1000)
def run_pdl_script(rot, nflux, adapt, method):
    """
    Runs the specified PDL script with provided parameters.

    Parameters:
    rot (int): Rotation number.
    nflux (int): Fluxon count.
    adapt (int): Adaptation parameter.
    method (str): Method to use.

    Raises:
    FileNotFoundError: If the Perl script specified in the configurations is not found.
    subprocess.CalledProcessError: If the subprocess running the PDL script fails.
    """
    run_script_path = os.path.abspath(os.path.expanduser(configs["run_script"]))

    if not os.path.isfile(run_script_path):
        logging.error(f"Perl script not found at {run_script_path}")
        raise FileNotFoundError(f"Perl script not found at {run_script_path}")

    if 'PERL5LIB' not in os.environ:
        logging.error("PERL5LIB environment variable is not set. Please configure it before running the script")
        raise EnvironmentError("PERL5LIB environment variable is not set")

    try:
        logging.info(f"Running PDL script: {run_script_path} with parameters: rot={rot}, nflux={nflux}, adapt={adapt}, method={method}")
        subprocess.run(
            ["bash", "fluxpype/run_pdl.sh", run_script_path, str(rot), str(nflux), str(adapt), str(method)],
            check=True,
        )
        logging.info("PDL script executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing PDL script: {e}")
        raise


def run():
    """Runs the PDL script for each configuration combination."""
    global timeout_num
    validate_configurations(configs)

    with tqdm(total=int(configs["n_jobs"]), unit="runs") as pbar:
        for adapt in configs["adapts"]:
            for rot in configs["rotations"]:
                for nflux in configs["fluxon_count"]:
                    for method in configs["flow_method"]:
                        pbar.set_description(
                            f"Job:: Rotation {rot}, n_fluxon {nflux}, flow_method {method}"
                        )
                        try:
                            run_pdl_script(rot, nflux, adapt, method)
                        except timeout_decorator.TimeoutError:
                            timeout_num += 1
                        except Exception as e:
                            logging.error(
                                f"Error: Rotation {rot}, n_fluxon {nflux}, flow_method {method}: {e}"
                            )
                        pbar.update(1)
    logging.info(f"Total timeouts: {timeout_num}")


if __name__ == "__main__":
    display_splash_screen()
    run()
