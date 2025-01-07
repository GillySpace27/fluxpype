#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
import shutil

def log(message, level="INFO"):
    print(f"[{level}] {message}")

def run_command(command, shell=False, check=True, capture_output=False):
    log(f"Running command: {' '.join(command) if isinstance(command, list) else command}")
    try:
        result = subprocess.run(
            command,
            shell=shell,
            check=check,
            text=True,
            capture_output=capture_output
        )
        if capture_output:
            return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        log(f"Command failed with error: {e}", level="ERROR")
        raise e
        # sys.exit(1)


def check_and_install_homebrew():
    if not shutil.which("brew"):
        log("Homebrew not found. Installing...")
        homebrew_install_command = (
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        )
        os.system(homebrew_install_command)
        # Ensure Homebrew is added to the PATH
        os.environ["PATH"] += ":/opt/homebrew/bin:/usr/local/bin"
    else:
        log("Homebrew already installed.")


def install_homebrew_dependencies():
    log("Installing required Homebrew packages...")
    packages = ["gnuplot", "fftw", "qt"]
    run_command(["brew", "install"] + packages)

def install_perlbrew(perl_version="perl-5.32.0"):
    if not shutil.which("perlbrew"):
        log("Installing Perlbrew...")
        run_command(["curl", "-L", "https://install.perlbrew.pl", "|", "bash"], shell=True)
        run_command(["perlbrew", "init"])
        perlbrew_bashrc = Path.home() / "perl5" / "perlbrew" / "etc" / "bashrc"
        if perlbrew_bashrc.exists():
            log("Sourcing Perlbrew environment...")
            run_command(["source", str(perlbrew_bashrc)], shell=True)
        else:
            log("Perlbrew initialization script not found.", level="ERROR")
            sys.exit(1)
    else:
        log("Perlbrew already installed.")

    log(f"Installing Perl version {perl_version}...")
    try:
        run_command(["perlbrew", "install", perl_version])
    except Exception as e:
        print("", e)

    run_command(["perlbrew", "switch", perl_version])

def setup_local_lib(pl_prefix):
    log(f"Setting up local::lib with PL_PREFIX={pl_prefix} ...")
    os.environ["PERL_MM_OPT"] = f"INSTALL_BASE={pl_prefix}"
    run_command(["cpanm", "--local-lib-contained", pl_prefix._str, "local::lib"])


def install_perl_modules(pl_prefix):
    log("Installing Perl modules...")
    modules = [
        "Test::Builder",
        "File::ShareDir",
        "File::ShareDir::Install",
        "PDL::Graphics::Gnuplot",
        "Math::RungeKutta",
        "Moo::Role",
        "Chart::Gnuplot",
        "Text::CSV",
        "Math::Interpolate",
        "Math::GSL",
        "Config::IniFiles",
        "File::HomeDir",
        "Inline::C",
        "Parallel::ForkManager",
        "Inline",
        "Inline::Python",
        "Capture::Tiny",
        "Devel::CheckLib",
    ]
    run_command(["cpanm", "-L", pl_prefix._str] + modules)

    eval_command = f"eval `perl -I {pl_prefix}/lib/perl5 -Mlocal::lib={pl_prefix}`"
    log(f"Evaluating local::lib environment with: {eval_command}")
    run_command(eval_command, shell=True)


def clone_and_build_flux(fl_prefix, pl_prefix):
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
    run_command(["make", "everything"])

def setup_python_virtualenv():
    log("Setting up Python virtual environment...")
    venv_dir = Path.cwd() / "fluxenv_pip"
    if not venv_dir.exists():
        run_command(["python3", "-m", "venv", str(venv_dir)])
    activate_script = venv_dir / "bin" / "activate"
    run_command(["source", str(activate_script)], shell=True)
    log("Python virtual environment activated.")

def install_fluxpype():
    log("Installing FluxPype Python dependencies...")
    requirements_file = Path.cwd() / "requirements-pip.txt"
    if requirements_file.exists():
        run_command(["pip", "install", "-r", str(requirements_file)])
        run_command(["pip", "install", "-e", "."])
    else:
        log("Requirements file not found. Skipping Python dependency installation.", level="WARNING")

def main():
    if sys.platform != "darwin":
        log("This installer is designed for macOS only.", level="ERROR")
        sys.exit(1)

    log("Starting comprehensive FluxPype installer...")

    fl_prefix = Path.home() / "Library" / "flux"
    pl_prefix = Path.home() / "Library" / "perl5"

    try:
        check_and_install_homebrew()
        install_homebrew_dependencies()
        install_perlbrew()
        setup_local_lib(pl_prefix)
        install_perl_modules(pl_prefix)
        clone_and_build_flux(fl_prefix, pl_prefix)
        setup_python_virtualenv()
        install_fluxpype()
        log("Installation completed successfully!")
    except Exception as e:
        log(f"An error occurred: {e}", level="ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
