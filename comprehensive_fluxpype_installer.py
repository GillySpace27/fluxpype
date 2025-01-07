#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
import shutil


def log(message, level="INFO"):
    """
    Logs a message with a specified level.

    Args:
        message (str): The message to log.
        level (str): The log level (e.g., INFO, ERROR). Defaults to "INFO".
    """
    print(f"[{level}] {message}")


def run_command(command, shell=False, check=True, capture_output=False):
    """
    Runs a system command using subprocess.

    Args:
        command (str or list): The command to run.
        shell (bool): Whether to use the shell as the program to execute. Defaults to False.
        check (bool): Whether to raise an exception on a non-zero exit status. Defaults to True.
        capture_output (bool): Whether to capture the output. Defaults to False.

    Returns:
        str: The captured output if `capture_output` is True.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit status.
    """
    log(f"Running command: {' '.join(command) if isinstance(command, list) else command}")
    try:
        result = subprocess.run(command, shell=shell, check=check, text=True, capture_output=capture_output)
        if capture_output:
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        log(f"Command failed with error: {e}", level="ERROR")
        raise


def check_and_install_homebrew():
    """
    Checks if Homebrew is installed, and installs it if it's not. Adds Homebrew to the PATH
    by updating the ~/.zshrc file idempotently.
    """
    homebrew_path_1 = "/opt/homebrew/bin"
    homebrew_path_2 = "/usr/local/bin"
    shell_rc = Path.home() / ".zshrc"

    if not shutil.which("brew"):
        log("Homebrew not found. Installing...")
        homebrew_install_command = (
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        )
        run_command(homebrew_install_command, shell=True)
        log("Homebrew installed successfully.")
    else:
        log("Homebrew already installed.")

    # Add Homebrew PATH to ~/.zshrc if not already present
    log("Ensuring Homebrew paths are in the shell RC file...")
    append_to_file_if_not_exists(shell_rc, f'export PATH="{homebrew_path_1}:$PATH"')
    append_to_file_if_not_exists(shell_rc, f'export PATH="{homebrew_path_2}:$PATH"')

    # Source the updated ~/.zshrc file
    run_command(f"source {shell_rc}", shell=True)


def install_homebrew_packages():
    """
    Installs the required Homebrew packages.
    """
    log("Installing required Homebrew packages...")
    packages = [
        "cpanminus",
        "perl",
        "gnuplot",
        "fftw",
        "qt",
        "make",
        "gcc",
        "git",
    ]
    run_command(["brew", "install"] + packages)


def append_to_file_if_not_exists(file_path, content):
    """
    Appends content to a file if it doesn't already exist in the file.

    Args:
        file_path (Path): The file to modify.
        content (str): The content to append.
    """
    with open(file_path, "a+") as file:
        file.seek(0)
        if content not in file.read():
            file.write(content)
            file.write("\n")


def install_perlbrew(perl_version="perl-5.32.0"):
    """
    Checks if Perlbrew is installed, and installs it if it's not. Also installs and switches to
    the specified Perl version.

    Args:
        perl_version (str): The Perl version to install and switch to. Defaults to "perl-5.32.0".
    """
    if not shutil.which("perlbrew"):
        log("Perlbrew not found. Installing...")
        perlbrew_install_command = "curl -L https://install.perlbrew.pl | bash"
        run_command(perlbrew_install_command, shell=True)

        shell_rc = Path.home() / ".zshrc"
        perlbrew_bashrc = Path.home() / "perl5" / "perlbrew" / "etc" / "bashrc"

        if perlbrew_bashrc.exists():
            log("Adding Perlbrew initialization to shell RC file...")
            append_to_file_if_not_exists(shell_rc, f"\n# Initialize Perlbrew\nsource {perlbrew_bashrc}")
            run_command(f"source {perlbrew_bashrc}", shell=True)
        else:
            log("Perlbrew initialization script not found.", level="ERROR")
            sys.exit(1)
    else:
        log("Perlbrew already installed.")

    log(f"Installing Perl version {perl_version}...")
    run_command(f"perlbrew --notest install {perl_version}", shell=True, check=False)
    run_command(f"perlbrew switch {perl_version}", shell=True, check=False)


def setup_local_lib(pl_prefix):
    """
    Sets up local::lib for Perl module management.

    Args:
        pl_prefix (Path): The prefix path for the local::lib installation.
    """
    log(f"Setting up local::lib with PL_PREFIX={pl_prefix} ...")
    os.environ["PERL_MM_OPT"] = f"INSTALL_BASE={pl_prefix}"
    run_command(["cpanm", "--local-lib-contained", str(pl_prefix), "local::lib"])


def install_perl_modules(pl_prefix):
    """
    Installs the required Perl modules.

    Args:
        pl_prefix (Path): The prefix path for the Perl modules installation.
    """
    log("Installing Perl modules...")
    modules = [
        "Test::Builder",
        "File::ShareDir::Install",
        "File::ShareDir",
        "File::HomeDir",
        "PDL::Graphics::Gnuplot",
        "Math::RungeKutta",
        "Moo::Role",
        "Chart::Gnuplot",
        "Text::CSV",
        "Math::Interpolate",
        "Math::GSL",
        "Math::GSL::Alien",
        "Config::IniFiles",
        "Inline::C",
        "Parallel::ForkManager",
        "Inline",
        "Inline::Python",
        "Capture::Tiny",
        "Devel::CheckLib",
    ]
    try:
        run_command(["cpanm", "-L", str(pl_prefix)] + modules, check=True)
    except Exception as e:
        log("Reverting to Individual Perl Dependency Installation", level="")
        for module in modules:
            run_command(["cpanm", "-L", str(pl_prefix), module], check=False)

    eval_command = f"eval `perl -I {pl_prefix}/lib/perl5 -Mlocal::lib={pl_prefix}`"
    log(f"Evaluating local::lib environment with: {eval_command}")
    run_command(eval_command, shell=True)


def clone_and_build_flux(fl_prefix, pl_prefix):
    """
    Clones the fluxon-mhd repository and builds the project.

    Args:
        fl_prefix (Path): The prefix path for the FLUX installation.
        pl_prefix (Path): The prefix path for the Perl modules installation.
    """
    log("Cloning and building the fluxon-mhd repository...")
    repo_url = "https://github.com/lowderchris/fluxon-mhd.git"
    repo_dir = Path.home() / "fluxon-mhd"

    if not repo_dir.exists():
        run_command(["git", "clone", repo_url, str(repo_dir)])
    else:
        log("Repository already cloned. Pulling latest changes...")
        run_command(["git", "-C", str(repo_dir), "pull"])

    os.chdir(repo_dir)
    os.environ["FL_PREFIX"] = str(fl_prefix)
    os.environ["PL_PREFIX"] = str(pl_prefix)
    run_command(["make", "everything"], check=False)


def setup_python_virtualenv():
    """
    Sets up a Python virtual environment.
    """
    log("Setting up Python virtual environment...")
    venv_dir = ".venv"
    if not venv_dir.exists():
        run_command(["python3", "-m", "venv", str(venv_dir)])
    activate_script = venv_dir / "bin" / "activate"
    run_command(["source", str(activate_script)], shell=True)
    log("Python virtual environment activated.")


def install_fluxpype():
    """
    Installs Python dependencies for FluxPype.
    """
    log("Installing FluxPype Python dependencies...")
    requirements_file = Path.cwd() / "requirements-pip.txt"
    if requirements_file.exists():
        run_command(["pip", "install", "-r", str(requirements_file)])
        run_command(["pip", "install", "-e", "."])
    else:
        log("Requirements file not found. Skipping Python dependency installation.", level="WARNING")


def append_flux_env_vars_to_rc(fl_prefix, pl_prefix, shell_rc):
    """
    Adds FLUX-related environment variables to the shell RC file if they are not already present.

    Args:
        fl_prefix (Path): The prefix path for the FLUX installation.
        pl_prefix (Path): The prefix path for the Perl modules installation.
        shell_rc (Path): The path to the shell RC file (e.g., ~/.zshrc).
    """
    log("Adding environment variables to shell RC file...")
    append_to_file_if_not_exists(
        shell_rc, f'\n# FLUX environment variables\nexport FL_PREFIX="{fl_prefix}"\nexport PL_PREFIX="{pl_prefix}"'
    )


def main():
    """
    Main function that orchestrates the complete installation process for the FluxPype project.
    """
    if sys.platform != "darwin":
        log("This installer is designed for macOS only.", level="ERROR")
        sys.exit(1)

    log("Starting comprehensive FluxPype installer...")

    fl_prefix = Path.home() / "Library" / "flux"
    pl_prefix = Path.home() / "Library" / "perl5"
    shell_rc = Path.home() / ".zshrc"

    try:
        check_and_install_homebrew()
        install_homebrew_packages()
        install_perlbrew()
        setup_local_lib(pl_prefix)
        install_perl_modules(pl_prefix)
        clone_and_build_flux(fl_prefix, pl_prefix)
        setup_python_virtualenv()
        install_fluxpype()
        append_flux_env_vars_to_rc(fl_prefix, pl_prefix, shell_rc)
        log("Installation completed successfully!")
    except Exception as e:
        log(f"An error occurred: {e}", level="ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
