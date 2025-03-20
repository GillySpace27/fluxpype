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

os.nice(16)

# Initialize Rich console and logger
console = Console()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def display_splash_screen(configs):
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
    # Import configurations
    from fluxpype.pipe_helper import configurations
    configs = configurations(debug=False)

    run_script_path = os.path.abspath(os.path.expanduser(configs["run_script"]))

    if not os.path.isfile(run_script_path):
        logging.error(f"PDL script not found: {run_script_path}")
        raise FileNotFoundError(f"PDL script not found: {run_script_path}")

    if "PERL5LIB" not in os.environ:
        logging.warning("PERL5LIB environment variable is not set. Execution may fail.")

    command = [
        "perl",
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
    # Import configurations
    from fluxpype.pipe_helper import configurations
    configs = configurations(debug=False)

    display_splash_screen(configs)

# def open():


import os
import subprocess
import platform

def open_config(path="fluxpype/config.ini"):
    return open_path(path)


def gallery():
    """
    Opens the default directory containing the batches in the file explorer.
    """
    # Define the path to the batches directory
    directory_path = "fluxpype/data/batches"

    # Ensure the directory exists before attempting to open it
    if not os.path.isdir(directory_path):
        logging.error(f"Directory not found: {directory_path}")
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Open the directory
    open_path(directory_path)


def open_path(path="fluxpype/config.ini"):
    """
    Open a file or directory in the default application or file explorer
    based on the operating system.
    """
    os_name = platform.system()  # Determine the OS name

    if os_name == "Windows":
        open_path_windows(path)
    elif os_name == "Darwin":  # macOS
        open_path_mac(path)
    elif os_name == "Linux":
        open_path_linux(path)
    else:
        raise OSError(f"Unsupported operating system: {os_name}")


def open_path_windows(path):
    """
    Open a file or directory using the default application or file explorer on Windows.
    """
    os.startfile(path)


import subprocess
from pathlib import Path


def open_path_mac(path):
    """
    Open a file or directory using the default application or file explorer on macOS.
    If the file type does not have a default application, open it using TextEdit.

    Args:
        path (str or Path): The path to the file or directory to open.
    """
    p = Path(path)

    # Check if the path is a directory or has a defined extension
    if p.is_dir():
        subprocess.call(["open", path])
        return

    # Check if the file extension has an associated default application
    try:
        result = subprocess.run(
            ["mdls", "-name", "kMDItemContentType", "-raw", path], check=True, capture_output=True, text=True
        )
        content_type = result.stdout.strip()

        if not content_type or result.returncode:
            raise ValueError("No default application found")

        # Try opening the file with the default application
        returncode = subprocess.call(["open", path])
        if not returncode ==0:
            raise ValueError("No default application found")

    except (subprocess.CalledProcessError, ValueError):
        # Revert to opening as .txt if the default application does not exist
        print(f"Default application not found for {path}. Opening with TextEdit.")
        subprocess.call(["open", "-a", "TextEdit", path])


def open_path_linux(path):
    """
    Open a file or directory using the default application or file explorer on Linux.
    """
    subprocess.call(["xdg-open", path])


def run():
    """
    Executes the PDL script for all combinations of configurations.
    """
    # Import configurations
    from fluxpype.pipe_helper import configurations

    # pipe_helper = import_module("pipe_helper")
    configs = configurations(debug=False)

    # Global counter for timeouts
    global timeout_count
    timeout_count = 0
    display_splash_screen(configs)
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
