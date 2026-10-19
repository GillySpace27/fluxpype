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
import sys
import argparse
import logging
from tqdm import tqdm
from rich.panel import Panel
from rich.console import Console
import timeout_decorator

from fluxpype.console import (
    init_log,
    close_log,
    emit,
    emit_raw,
    write_to_log,
)

os.nice(16)

# Repo checkout holding the fluxpype package; children run from here so relative paths resolve.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    # list-valued keys that parse to [] would silently schedule zero jobs
    missing_keys += [
        key for key in ("rotations", "fluxon_count", "adapts", "flow_method")
        if key in configurations and not [v for v in configurations[key] if str(v).strip()]
    ]
    if missing_keys:
        raise KeyError(
            f"Profile [{configurations.get('config_name', '?')}] must set: {', '.join(missing_keys)}"
        )


def display_splash_screen(configs, log_path=None):
    """
    Displays a splash screen with configuration details using Rich.
    Also writes a plain-text summary to the log file if one is active.
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
    if log_path:
        write_to_log(
            f"FLUXpype Config Runner | Log: {log_path}\n"
            f"Rotations: {configs['rotations']} | Fluxons: {configs['fluxon_count']} | "
            f"Adapts: {configs['adapts']} | Methods: {configs['flow_method']}\n"
        )


def _check_pdl_available():
    """Return (True, None) if PDL is loadable, else (False, error_message). Cached per process."""
    if getattr(_check_pdl_available, "_ok", None) is True:
        return True, None
    try:
        r = subprocess.run(
            ["perl", "-e", "use PDL; use PDL::AutoLoader; print qq(ok);"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and "ok" in (r.stdout or ""):
            _check_pdl_available._ok = True
            return True, None
        err = (r.stderr or r.stdout or "").strip() or f"exit code {r.returncode}"
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        err = str(e)
    msg = (
        "PDL (Perl Data Language) is not available. The magnetogram2wind pipeline requires PDL.\n"
        f"Reason: {err}\n\n"
        "Install PDL with one of:\n"
        "  • Conda (recommended if you use conda):  conda install -c conda-forge perl-pdl\n"
        "  • System Perl:  cpanm PDL   or  cpan PDL\n"
        "  • If cpanm fails on Module::Build:  cpanm --force Module::Build  then  cpanm PDL\n"
        "  • Or use a Perl where PDL is already installed and set PERL5LIB if needed."
    )
    return False, msg


@timeout_decorator.timeout(2000)
def run_pdl_script(rotation, fluxon_count, adaptation, method):
    """
    Executes the specified PDL script with given parameters.
    Streams all output (Perl and Python subprocesses) to console and log.

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
    RuntimeError
        If PDL is not installed.
    subprocess.CalledProcessError
        If the script execution fails.
    """
    from fluxpype.pipe_helper import configurations, get_flux_perl_lib
    configs = configurations(debug=False)

    ok, pdl_msg = _check_pdl_available()
    if not ok:
        emit(pdl_msg, style="red")
        raise RuntimeError("PDL is not available. Install PDL (see message above).")

    run_script_path = os.path.expanduser(configs["run_script"])
    if not os.path.isabs(run_script_path):
        run_script_path = os.path.join(REPO_ROOT, run_script_path)

    if not os.path.isfile(run_script_path):
        emit(f"PDL script not found: {run_script_path}", style="red")
        raise FileNotFoundError(f"PDL script not found: {run_script_path}")

    env = os.environ.copy()
    # The Perl stage calls bare `python`/`python3`; make those resolve to this interpreter
    # (venv or conda bin dirs hold both names), so helpers see the same packages we do.
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    # Prefer Flux from current Perl's default @INC (e.g. conda site_perl) so C extension handshake matches.
    # Only add fluxon-perl to PERL5LIB when Flux is not already loadable (avoids loading wrong .bundle from fluxon-perl).
    try:
        r = subprocess.run(
            ["perl", "-e", "use Flux; print qq(ok);"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        flux_ok = r.returncode == 0 and "ok" in (r.stdout or "")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        flux_ok = False
    if not flux_ok:
        flux_perl_lib = get_flux_perl_lib(configs)
        if flux_perl_lib:
            env["PERL5LIB"] = flux_perl_lib + (os.pathsep + env.get("PERL5LIB", "")) if env.get("PERL5LIB") else flux_perl_lib
        else:
            emit("Flux not loadable and flux_perl_lib not found. Execution may fail.", style="yellow")

    command = [
        "perl",
        run_script_path,
        str(rotation),
        str(fluxon_count),
        str(adaptation),
        method,
    ]

    emit(f"\nExecuting: {' '.join(command)}\n", style="dim")

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=REPO_ROOT,
    )
    out_lines = []
    for line in proc.stdout:
        out_lines.append(line)
        emit_raw(line)
    proc.wait()
    if proc.returncode != 0:
        emit(f"PDL script failed with exit code {proc.returncode}", style="red")
        out_text = "".join(out_lines)
        if "loadable library and perl binaries are mismatched" in out_text or "handshake key" in out_text:
            emit(
                "The Flux C extensions (e.g. Core.bundle) were built with a different Perl than the one running.\n"
                "Rebuild fluxon-perl with your current Perl (e.g. conda's perl), or run using the same Perl that built fluxon-perl.",
                style="yellow",
            )
        raise subprocess.CalledProcessError(proc.returncode, command)

def view():
    # Import configurations
    from fluxpype.pipe_helper import configurations
    configs = configurations(debug=False)

    display_splash_screen(configs)

# def open():


import os
import subprocess
import platform

def open_config(path=None):
    return open_path(path or os.path.join(REPO_ROOT, "fluxpype", "config.ini"))


def gallery():
    """
    Opens the default directory containing the batches in the file explorer.
    """
    # Define the path to the batches directory
    from fluxpype.pipe_helper import configurations
    directory_path = os.path.join(configurations(debug=False)["data_dir"], "batches")

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


def run(argv=None):
    """
    Executes the PDL script for all combinations of configurations.
    Initializes a log file and tees all output (Python + Perl) to it.

    Exits nonzero if any job failed or timed out, so shells, batch scripts
    and grid schedulers can tell a failed run from a good one.
    """
    parser = argparse.ArgumentParser(prog="flux-config-run")
    parser.add_argument(
        "--config", help="config.ini profile to run (overrides config_name in the file)"
    )
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.config:
        # every subprocess (Python and Perl) re-reads config.ini; this is how they agree
        os.environ["FLUXPYPE_CONFIG"] = args.config

    from fluxpype.pipe_helper import configurations
    configs = configurations(debug=False)

    batch_name = configs.get("batch_name", "run")
    log_path = init_log(batch_name)
    emit(f"Log file: {log_path}\n", style="dim")

    global timeout_count
    timeout_count = 0
    failures = []
    display_splash_screen(configs, log_path=log_path)
    validate_configurations(configs)

    total_jobs = (
        len(configs["rotations"])
        * len(configs["fluxon_count"])
        * len(configs["adapts"])
        * len(configs["flow_method"])
    )
    try:
        with tqdm(
            total=total_jobs, unit="runs", colour="green", desc="Processing"
        ) as progress_bar:
            for adaptation in configs["adapts"]:
                for rotation in configs["rotations"]:
                    for fluxon_count in configs["fluxon_count"]:
                        for method in configs["flow_method"]:
                            job = f"Rotation {rotation}, Fluxon {fluxon_count}, Adapt {adaptation}, Method {method}"
                            progress_bar.set_description(job)
                            try:
                                run_pdl_script(rotation, fluxon_count, adaptation, method)
                            except timeout_decorator.TimeoutError:
                                timeout_count += 1
                                failures.append(f"{job}: timeout")
                                emit(f"Timeout: {job}", style="yellow")
                            except Exception as e:
                                failures.append(f"{job}: {e}")
                                emit(f"Error: {job}: {e}", style="red")
                            progress_bar.update(1)
        emit(f"Total timeouts: {timeout_count}", style="bold")
        if failures:
            emit(f"FAILED {len(failures)} of {total_jobs} jobs:", style="bold red")
            for f in failures:
                emit(f"  {f}", style="red")
        else:
            emit(f"All {total_jobs} jobs succeeded.", style="bold green")
    finally:
        close_log()
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    run()
