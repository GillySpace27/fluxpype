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

Functions
---------
None

Examples
--------
1. Edit the 'config.ini' file to configure your desired settings.
2. Run the script using `python config_runner.py`.

Authors
-------
Gilly <gilly@swri.org> (and others!)

"""
import subprocess
from tqdm import tqdm
import fluxpype
from fluxpype.helpers.pipe_helper import configurations
import timeout_decorator

configs = configurations(debug=False)
# print(fluxpype.helpers.pipe_helper.__file__)

# Initialize timeout_num
timeout_num = 0

import os


@timeout_decorator.timeout(1000)  # Set a timeout for each subprocess call
def run_pdl_script(rot, nflux, adapt, method):
    """
    Executes the PDL script with a timeout.

    Parameters:
    - rot: Carrington Rotation
    - nflux: Fluxon count
    - adapt: Adaptation parameter
    """
    # Resolve the absolute path to the script
    run_script_path = os.path.expanduser(configs["run_script"])
    run_script_path = os.path.abspath(run_script_path)

    # print(f"Resolved RUN SCRIPT = {run_script_path}")

    if not os.path.isfile(run_script_path):
        raise FileNotFoundError(f"Perl script not found at {run_script_path}")

    subprocess.run(
        [
            "perl",
            run_script_path,
            str(rot),
            str(nflux),
            str(adapt),
            str(method),
        ],
        check=False,
    )


def run():
    global timeout_num
    with tqdm(total=int(configs["n_jobs"]), unit="runs") as pbar:
        for adapt in configs["adapts"]:
            for rot in configs["rotations"]:
                for nflux in configs["fluxon_count"]:
                    for method in configs["flow_method"]:
                        pbar.set_description(f"Job:: Rotation {rot}, n_fluxon {nflux}, flow_method {method}")
                        try:
                            print("Running PDL...")
                            run_pdl_script(rot, nflux, adapt, method)
                        except timeout_decorator.TimeoutError:
                            print(f"Timeout for Rotation {rot}, n_fluxon {nflux}, flow_method {method}")
                            timeout_num += 1
                        except Exception as e:
                            print(f"Error for Rotation {rot}, n_fluxon {nflux}, flow_method {method}: {e}")
                        pbar.update(1)

    print(f"Total timeouts: {timeout_num}")


if __name__ == "__main__":
    run()
