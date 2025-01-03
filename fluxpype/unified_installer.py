import os
import subprocess
import sys
import shutil
import logging
import re
from pathlib import Path

# import rich
# from rich.console import Console
# from rich.logging import RichHandler

# Initialize Rich console
# console = Console()

# Set up Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    # handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("rich")


def log(message):
    """Log informational messages."""
    log_info(message)

def log_info(message):
    """Log informational messages."""
    logger.info(f"{message}")

def log_warning(message):
    """Log warning messages."""
    logger.warning(f"[bold yellow]{message}[/bold yellow]")


def log_error(message):
    """Log error messages."""
    logger.error(f"[bold red]{message}[/bold red]")


def log_debug(message):
    """Log debug messages."""
    logger.debug(f"[bold blue]{message}[/bold blue]")


def check_command(command):
    """Check if a command exists on the system."""
    return shutil.which(command) is not None


def check_perl(package):
    """Check if a Perl package is installed."""
    answer = subprocess.run(["perl", "-e", f"use {package}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if answer.returncode != 0:
        log(f"Package '{package}' is not installed in Perl!")
    return answer.returncode == 0


def install_homebrew():
    """Install Homebrew if not already installed."""
    if not check_command("brew"):
        log("Homebrew not found. Installing...")
        subprocess.run(
            ["/bin/zsh", "-c", "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"],
            check=True,
        )
        log("Homebrew installed successfully.")
    else:
        log("Homebrew is already installed.")


def install_dependencies():
    """Install required dependencies using Homebrew and cpanm."""
    brew_dependencies = ["make", "perl", "gcc", "cpanm", "gnuplot", "fftw"]
    for dep in brew_dependencies:
        log(f"Checking for {dep}...")
        if subprocess.run(["brew", "list", dep], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0:
            log(f"\t{dep} not found. Installing...")
            subprocess.run(["brew", "install", dep], check=True)
        else:
            log(f"\t{dep} is already installed.")

    if not check_command("cpanm"):
        log("Installing cpanminus...")
        subprocess.run(["brew", "install", "cpanminus"], check=True)
    else:
        log("cpanminus is already installed.")


def setup_perl():
    """Set up Perl environment and install required modules."""
    perl_lib_dir = os.path.expanduser("~/Library/perl5")
    os.environ["PERL_MM_OPT"] = f"INSTALL_BASE={perl_lib_dir}"
    perl_modules = [
        "File::ShareDir",
        "File::Map",
        "PDL",
        "PDL::Graphics::Gnuplot",
        "Math::RungeKutta",
        "Term::ReadKey",
    ]

    log("Installing Perl modules...")
    for module in perl_modules:
        log(f"Installing {module}...")
        if not check_perl(module):
            subprocess.run(["cpanm", "--local-lib", perl_lib_dir, module], check=True)
        if not check_perl(module):
            raise ModuleNotFoundError(f"Failed to install Perl module: {module}")

    # Update .zprofile for Perl environment
    zprofile_path = os.path.expanduser("~/.zprofile")
    perl_setup_line = f"export PERL5LIB={perl_lib_dir}/lib/perl5:$PERL5LIB\n"

    # Check and add the line if it doesn't exist
    if update_profile(zprofile_path, perl_setup_line, "Perl library path"):
        log("Perl setup added to .zprofile.")
    else:
        log("Perl setup already exists in .zprofile.")


def update_profile(profile_path, entry, description):
    """Ensure a specific entry exists in the given profile file."""
    try:
        if os.path.exists(profile_path):
            with open(profile_path, "r") as profile:
                lines = profile.readlines()
        else:
            lines = []

        if entry not in lines:
            with open(profile_path, "a") as profile:
                profile.write(f"\n# {description}\n")
                profile.write(entry)
            return True
    except Exception as e:
        log(f"Error updating {profile_path}: {e}")
    return False


def capture_and_tee(command):
    """Run a command, print its output live, and return captured output."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = []

    for line in process.stdout:
        print(line, end="")
        output.append(line.strip())

    process.stdout.close()
    process.wait()

    if process.returncode != 0:
        error_output = process.stderr.read()
        raise RuntimeError(f"Command failed with error: {error_output}")

    return output


def extract_flux_paths(output):
    """Extract paths from the Flux build output."""
    perl_inc_path = None
    pdllib_path = None

    for line in output:
        if "Will install Flux perl modules into" in line:
            perl_inc_path = line.split("into")[-1].strip().split(". ")[0]
        elif line.startswith("+"):
            pdllib_path = line.strip()[1:]
    print(f"{perl_inc_path=} \n{pdllib_path=}")
    return perl_inc_path, pdllib_path


def update_perldlrc(pdllib_path):
    """Ensure PDLLIB and autoload settings are in ~/.perldlrc."""
    perldlrc_path = Path.home() / ".perldlrc"
    required_lines = ["require(q|PDL/default.perldlrc|);", "use PDL::AutoLoader;", "$PDL::AutoLoader::Rescan=1;", "1;"]

    if not pdllib_path:
        log("No PDLLIB path found in the Flux build output.")
        return

    pdllib_line = f"export PDLLIB=+{pdllib_path}:$PDLLIB\n"

    existing_lines = []
    if perldlrc_path.exists():
        with open(perldlrc_path, "r") as f:
            existing_lines = f.readlines()

    missing_lines = [line for line in [pdllib_line] + required_lines if line not in existing_lines]
    if missing_lines:
        with open(perldlrc_path, "a") as f:
            f.writelines(missing_lines)
        log("Updated ~/.perldlrc with required settings.")
    else:
        log("All required settings already exist in ~/.perldlrc.")


def add_to_inc(perl_inc_path):
    """Add Perl module path to @INC."""
    zprofile_path = Path.home() / ".zprofile"
    inc_line = f"export PERL5LIB={perl_inc_path}:$PERL5LIB\n"
    if update_profile(zprofile_path, inc_line, "Perl @INC path"):
        log("Updated ~/.zprofile with @INC path.")


def clone_and_build_flux():
    """Clone the fluxon-mhd repository and build it."""
    repo_url = "https://github.com/lowderchris/fluxon-mhd.git"
    repo_dir = os.path.expanduser("~/fluxon-mhd")

    if not os.path.exists(repo_dir):
        log("Cloning fluxon-mhd repository...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    else:
        log("fluxon-mhd repository already exists. Pulling latest changes...")
        subprocess.run(["git", "-C", repo_dir, "pull"], check=True)

    log("Building fluxon-mhd...")
    os.chdir(repo_dir)
    build_output = capture_and_tee(["make", "everything"])
    perl_inc_path, pdllib_path = extract_flux_paths(build_output)
    add_to_inc(perl_inc_path)
    update_perldlrc(pdllib_path)
    if check_perl("Flux"):
        log("Flux build complete.")
    else:
        log("Flux failed to build")
        raise Exception()

def main():
    if sys.platform != "darwin":
        log("This installer is designed for macOS only.")
        sys.exit(1)

    log("Starting unified FLUX installer...")

    try:
        install_homebrew()
        install_dependencies()
        setup_perl()
        clone_and_build_flux()
        log("Installation completed successfully!")
        sys.exit(0)
    except Exception as e:
        log_error(f"Error during installation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
