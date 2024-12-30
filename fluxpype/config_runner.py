"""
Config Runner: Automate the FLUXpype Workflow for Multiple Carrington Rotations
===============================================================================

This module automates the FLUXpype workflow, executing the `magnetogram2wind.pdl`
script for various Carrington Rotations (CRs) defined in a `config.ini` file.
It uses Python's `subprocess` module for script execution and `tqdm` for progress
tracking.

Authors
-------
Gilly <gilly@swri.org> (and others!)

Examples
--------
1. Edit the 'config.ini' file to configure your desired settings.
2. Run the script using `python config_runner.py`.
"""

import subprocess
import os
import logging
from importlib import import_module
from tqdm import tqdm
from rich import print
from rich.panel import Panel
from rich.console import Console
import timeout_decorator

# Initialize Rich console and logger
console = Console()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import configurations
from pipe_helper import configurations

# pipe_helper = import_module("pipe_helper")
configs = configurations(debug=False)

# Global counter for timeouts
timeout_count = 0


def validate_configurations(configurations):
    """
    Validates the configuration parameters.

    Parameters
    ----------
    configurations : dict
        The configuration dictionary to validate.

    Raises
    ------
    KeyError
        If a required configuration key is missing.
    """
    required_keys = [
        "rotations",
        "fluxon_count",
        "adapts",
        "flow_method",
        "run_script",
        "n_jobs",
    ]
    missing_keys = [key for key in required_keys if key not in configurations]
    if missing_keys:
        raise KeyError(f"Missing configuration parameters: {', '.join(missing_keys)}")


def display_splash_screen():
    """
    Displays a splash screen with configuration details using Rich.
    """
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
    console.print(
        Panel(
            splash_text,
            title="Welcome to FLUXpype Config Runner",
            border_style="bold green",
        )
    )


@timeout_decorator.timeout(1000)
def run_pdl_script(rotation, fluxon_count, adaptation, method):
    """
    Executes the specified PDL script with given parameters.

    Parameters
    ----------
    rotation : int
        Carrington rotation number.
    fluxon_count : int
        Number of fluxons.
    adaptation : int
        Adaptation parameter.
    method : str
        Flow method.

    Raises
    ------
    FileNotFoundError
        If the PDL script path is invalid.
    subprocess.CalledProcessError
        If the script execution fails.
    """
    run_script_path = os.path.abspath(os.path.expanduser(configs["run_script"]))

    if not os.path.isfile(run_script_path):
        logging.error(f"PDL script not found: {run_script_path}")
        raise FileNotFoundError(f"PDL script not found: {run_script_path}")

    if "PERL5LIB" not in os.environ:
        logging.warning("PERL5LIB environment variable is not set. Execution may fail.")

    command = [
        "bash",
        "fluxpype/run_pdl.sh",
        run_script_path,
        str(rotation),
        str(fluxon_count),
        str(adaptation),
        method,
    ]

    try:
        logging.info(f"Executing PDL script with parameters: {command}")
        subprocess.run(command, check=True)
        logging.info("PDL script executed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing PDL script: {e}")
        raise

def view():
    display_splash_screen()

# def open():


import os
import subprocess
import platform


def open_file(file_path="fluxpype/config.ini"):
    """
    Open a file in the default text editor based on the operating system.
    """
    os_name = platform.system()  # Determine the OS name

    if os_name == "Windows":
        open_file_windows(file_path)
    elif os_name == "Darwin":  # macOS
        open_file_mac(file_path)
    elif os_name == "Linux":
        open_file_linux(file_path)
    else:
        raise OSError(f"Unsupported operating system: {os_name}")


def open_file_windows(file_path):
    """
    Open a file using the default editor on Windows.
    """
    os.startfile(file_path)

def open_file_mac(file_path):
    """
    Open a file using the default editor on macOS.
    """
    subprocess.call(["open", file_path])

def open_file_linux(file_path):
    """
    Open a file using the default editor on Linux.
    """
    subprocess.call(["xdg-open", file_path])


def run():
    """
    Executes the PDL script for all combinations of configurations.
    """
    global timeout_count
    display_splash_screen()
    validate_configurations(configs)

    total_jobs = (
        len(configs["rotations"])
        * len(configs["fluxon_count"])
        * len(configs["adapts"])
        * len(configs["flow_method"])
    )
    with tqdm(
        total=total_jobs, unit="runs", colour="green", desc="Processing"
    ) as progress_bar:
        for adaptation in configs["adapts"]:
            for rotation in configs["rotations"]:
                for fluxon_count in configs["fluxon_count"]:
                    for method in configs["flow_method"]:
                        progress_bar.set_description(
                            f"Rotation: {rotation}, Fluxon: {fluxon_count}, Method: {method}"
                        )
                        try:
                            run_pdl_script(rotation, fluxon_count, adaptation, method)
                        except timeout_decorator.TimeoutError:
                            timeout_count += 1
                            logging.warning(
                                f"Timeout: Rotation {rotation}, Fluxon {fluxon_count}, Method {method}"
                            )
                        except Exception as e:
                            logging.error(
                                f"Error: Rotation {rotation}, Fluxon {fluxon_count}, Method {method}: {e}"
                            )
                        progress_bar.update(1)
    logging.info(f"Total timeouts: {timeout_count}")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.critical(f"Critical failure: {e}")
