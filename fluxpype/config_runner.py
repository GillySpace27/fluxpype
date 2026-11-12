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
import json
import logging
import queue
import signal
import socket
import threading
import time
from tqdm import tqdm
from rich.panel import Panel
from rich.console import Console

from fluxpype.events import emit as emit_event
from fluxpype.console import (
    init_log,
    close_log,
    emit,
    emit_raw,
    write_to_log,
)

# Capabilities advertised by `flux-config-run --capabilities` (the plumber probes this
# before submitting, so an older install cannot silently run a whole profile).
FEATURES = ["only", "events", "config_file", "log_file", "task_timeout"]

# Default per-task wall-clock limit in seconds; override with FLUXPYPE_TASK_TIMEOUT.
DEFAULT_TASK_TIMEOUT = 2000

# Exit code used when a task hit its time limit (same as coreutils `timeout`).
EXIT_TIMEOUT = 124

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


def _perl(configs):
    """The Perl every job stage uses: config.ini `flux_perl` if set, else the first `perl` on PATH."""
    return os.path.expanduser(configs.get("flux_perl") or "perl")


def _check_pdl_available(perl="perl"):
    """Return (True, None) if PDL is loadable, else (False, error_message). Cached per process."""
    if getattr(_check_pdl_available, "_ok", None) is True:
        return True, None
    try:
        r = subprocess.run(
            [perl, "-e", "use PDL; use PDL::AutoLoader; print qq(ok);"],
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


class TaskTimeout(Exception):
    """A pipeline task exceeded its wall-clock limit and was killed."""


def _task_timeout():
    """Per-task time limit in seconds (FLUXPYPE_TASK_TIMEOUT, default 2000)."""
    try:
        value = float(os.environ.get("FLUXPYPE_TASK_TIMEOUT", DEFAULT_TASK_TIMEOUT))
    except ValueError:
        return float(DEFAULT_TASK_TIMEOUT)
    return value if value > 0 else float(DEFAULT_TASK_TIMEOUT)


def _fluxpype_sha():
    """Best-effort git sha of the checkout, None on any failure."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def _kill_group(pgid, grace=10.0, proc=None):
    """SIGTERM a process group, wait up to `grace` seconds, then SIGKILL it."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if proc is not None:
            if proc.poll() is not None:
                break
        else:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_pdl_script(rotation, fluxon_count, adaptation, method):
    """
    Run one pipeline task and report it as `run.start` / `run.end` events.

    Thin wrapper around :func:`_run_pdl_script`; the events are no-ops unless
    FLUXPYPE_EVENTS is set.
    """
    from fluxpype.pipe_helper import configurations

    profile = os.environ.get("FLUXPYPE_CONFIG")
    if profile is None and os.environ.get("FLUXPYPE_EVENTS"):
        profile = configurations(debug=False).get("config_name")
    emit_event(
        "run.start",
        cr=str(rotation),
        nflux=str(fluxon_count),
        adapt=str(adaptation),
        method=str(method),
        profile=profile,
        host=socket.gethostname(),
        fluxpype_sha=_fluxpype_sha() if os.environ.get("FLUXPYPE_EVENTS") else None,
    )
    t0 = time.time()
    try:
        _run_pdl_script(rotation, fluxon_count, adaptation, method)
    except TaskTimeout:
        emit_event("run.end", status="timeout", exit_code=EXIT_TIMEOUT, elapsed_s=round(time.time() - t0, 2))
        raise
    except Exception as e:
        code = getattr(e, "returncode", None)
        emit_event(
            "run.end", status="failed", exit_code=code if code is not None else 1, elapsed_s=round(time.time() - t0, 2)
        )
        raise
    emit_event("run.end", status="ok", exit_code=0, elapsed_s=round(time.time() - t0, 2))


def _run_pdl_script(rotation, fluxon_count, adaptation, method):
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

    perl = _perl(configs)
    ok, pdl_msg = _check_pdl_available(perl)
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
            [perl, "-e", "use Flux; print qq(ok);"],
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
        perl,
        run_script_path,
        str(rotation),
        str(fluxon_count),
        str(adaptation),
        method,
    ]

    emit(f"\nExecuting: {' '.join(command)}\n", style="dim")

    # Own session so a timeout (or a cancel from the plumber) can kill Perl and every
    # forked Flux daughter together.
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=REPO_ROOT,
        start_new_session=True,
    )
    pgid = proc.pid
    lines = queue.Queue()

    def _reader():
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    out_lines = []
    deadline = time.monotonic() + _task_timeout()
    timed_out = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                line = lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                break
            out_lines.append(line)
            emit_raw(line)
        if timed_out:
            _kill_group(pgid, proc=proc)
            proc.wait()
            raise TaskTimeout(f"timed out after {_task_timeout():g} s")
        proc.wait()
    finally:
        # also reached on Ctrl-C or SIGTERM of this process: never orphan Flux daughters
        _kill_group(pgid, grace=0.5, proc=proc) if proc.poll() is None else _reap_group(pgid)
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


def _reap_group(pgid):
    """Kill stragglers left in a finished task's process group (forked daughters)."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

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


def _capabilities():
    """The JSON document printed by `--capabilities`."""
    try:
        from importlib.metadata import version

        ver = version("fluxpype")
    except Exception:
        ver = "unknown"
    return {"version": ver, "features": list(FEATURES)}


def _parse_only(text):
    """Split `CR,NFLUX,ADAPT,METHOD` into four stripped strings."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4 or not all(parts):
        raise ValueError(f"--only expects CR,NFLUX,ADAPT,METHOD (got {text!r})")
    return parts


def select_tasks(configs, only=None):
    """
    Expand a profile into its (adapt, rotation, fluxon_count, method) task list,
    in the order the runner executes them. With `only`, return just that task
    (validated against the profile's lists).
    """
    tasks = [
        (adaptation, rotation, fluxon_count, method)
        for adaptation in configs["adapts"]
        for rotation in configs["rotations"]
        for fluxon_count in configs["fluxon_count"]
        for method in configs["flow_method"]
    ]
    if only is None:
        return tasks
    cr, nflux, adapt, method = _parse_only(only)
    wanted = (adapt.casefold(), cr, nflux, method.casefold())
    for task in tasks:
        if (str(task[0]).casefold(), str(task[1]), str(task[2]), str(task[3]).casefold()) == wanted:
            return [task]
    raise ValueError(f"--only {only!r} is not in this profile's task list")


def _raise_on_sigterm(signum, frame):
    # lets the `finally` blocks run, so a cancelled task never orphans Perl or Flux daughters
    raise SystemExit(128 + signum)


def run(argv=None):
    """
    Executes the PDL script for all combinations of configurations.
    Initializes a log file and tees all output (Python + Perl) to it.

    Exits nonzero if any job failed or timed out, so shells, batch scripts
    and grid schedulers can tell a failed run from a good one: 1 for failures,
    124 when the only failures were timeouts.
    """
    parser = argparse.ArgumentParser(prog="flux-config-run")
    parser.add_argument(
        "--config", help="config.ini profile to run (overrides config_name in the file)"
    )
    parser.add_argument(
        "--only",
        metavar="CR,NFLUX,ADAPT,METHOD",
        help="run exactly this one task of the profile",
    )
    parser.add_argument(
        "--capabilities", action="store_true", help="print supported features as JSON and exit"
    )
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if args.capabilities:
        print(json.dumps(_capabilities()))
        return
    if args.config:
        # every subprocess (Python and Perl) re-reads config.ini; this is how they agree
        os.environ["FLUXPYPE_CONFIG"] = args.config

    try:
        os.nice(16)
    except OSError:
        pass
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _raise_on_sigterm)

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

    try:
        tasks = select_tasks(configs, args.only)
    except ValueError as e:
        emit(f"Error: {e}", style="red")
        close_log()
        sys.exit(2)
    total_jobs = len(tasks)
    try:
        with tqdm(
            total=total_jobs, unit="runs", colour="green", desc="Processing"
        ) as progress_bar:
            for adaptation, rotation, fluxon_count, method in tasks:
                job = f"Rotation {rotation}, Fluxon {fluxon_count}, Adapt {adaptation}, Method {method}"
                progress_bar.set_description(job)
                try:
                    run_pdl_script(rotation, fluxon_count, adaptation, method)
                except TaskTimeout:
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
        sys.exit(EXIT_TIMEOUT if timeout_count == len(failures) else 1)


if __name__ == "__main__":
    run()
