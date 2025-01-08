import os
import subprocess
import sys
from pathlib import Path

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
        sys.exit(1)

def check_and_install_homebrew():
    if not shutil.which("brew"):
        log("Homebrew not found. Installing...")
        run_command(
            ["/bin/bash", "-c", "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"]
        )
        os.environ["PATH"] += ":/opt/homebrew/bin"
    else:
        log("Homebrew already installed.")

def install_homebrew_dependencies():
    log("Installing required Homebrew packages...")
    packages = ["gnuplot", "fftw", "qt"]
    run_command(["brew", "install"] + packages)

def install_perlbrew():
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

def setup_python_virtualenv():
    log("Setting up Python virtual environment...")
    venv_dir = Path.cwd() / "fluxenv_pip"
    if not venv_dir.exists():
        run_command(["python3", "-m", "venv", str(venv_dir)])
    log("Activating virtual environment...")
    activate_script = venv_dir / "bin" / "activate"
    run_command(["source", str(activate_script)], shell=True)

def install_fluxpype():
    log("Installing FluxPype Python dependencies...")
    requirements_file = Path.cwd() / "requirements-pip.txt"
    if not requirements_file.exists():
        log("Requirements file not found.", level="ERROR")
        sys.exit(1)
    run_command(["pip", "install", "-r", str(requirements_file)])
    run_command(["pip", "install", "-e", "."])

def main():
    if sys.platform != "darwin":
        log("This installer is designed for macOS only.", level="ERROR")
        sys.exit(1)

    log("Starting FluxPype unified installer...")

    try:
        check_and_install_homebrew()
        install_homebrew_dependencies()
        install_perlbrew()
        setup_python_virtualenv()
        install_fluxpype()
        log("Installation completed successfully!")
    except Exception as e:
        log(f"An error occurred: {e}", level="ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
