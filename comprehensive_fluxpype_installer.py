#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
import shutil

############################
# 1) NEW LOGIC YOU PROVIDED
############################


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
        print(line, end="")  # print live
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
        # Example line: "Will install Flux perl modules into /Users/tester/Library/perl5. Make sure..."
        if "Will install Flux perl modules into" in line:
            # Everything after 'into' until the period
            # e.g. "into /Users/tester/Library/perl5. Make sure ..."
            # We'll split on 'into' then take the piece after that, strip, and split on '.'
            perl_inc_path = line.split("into")[-1].strip().split(".")[0]
            perl_inc_path = perl_inc_path.strip()
        elif line.startswith("+"):
            # For example: "+/some/path/PDL"
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

    # We'll add a line to 'export PDLLIB=+...' to append to PDLLIB
    pdllib_line = f"export PDLLIB=+{pdllib_path}:$PDLLIB\n"

    existing_lines = []
    if perldlrc_path.exists():
        with open(perldlrc_path, "r") as f:
            existing_lines = f.readlines()

    # We only append lines that aren't already present
    missing_lines = [line for line in ([pdllib_line] + required_lines) if line not in existing_lines]
    if missing_lines:
        with open(perldlrc_path, "a") as f:
            f.writelines(missing_lines)
        log("Updated ~/.perldlrc with required settings.")
    else:
        log("All required settings already exist in ~/.perldlrc.")


############################
# 2) YOUR ORIGINAL SCRIPT CODE
############################


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


def check_python_version(min_version=(3, 7)):
    """
    Ensures this script is running on at least Python 3.7.

    Args:
        min_version (tuple): Minimum (major, minor) version required.
    """
    if sys.version_info < min_version:
        log(f"This script requires Python {min_version[0]}.{min_version[1]} or higher.", level="ERROR")
        sys.exit(1)


def enable_perl5lib(pl_prefix):
    """
    Immediately set PERL5LIB in this process and persist the setting in ~/.zprofile
    so that new shells also pick it up.
    """
    new_path = f"{pl_prefix}/lib/perl5"
    existing_value = os.environ.get("PERL5LIB", "")

    # 1) Update the script's environment so subsequent commands can see local::lib
    if new_path not in existing_value.split(":"):
        if existing_value:
            os.environ["PERL5LIB"] = f"{new_path}:{existing_value}"
        else:
            os.environ["PERL5LIB"] = new_path
        log(f"PERL5LIB updated in the current environment: {os.environ['PERL5LIB']}")

    # 2) Persist this in ~/.zprofile
    shell_rc = Path.home() / ".zprofile"
    perl5lib_line = f'export PERL5LIB="{new_path}:$PERL5LIB"'
    append_to_file_if_not_exists(shell_rc, perl5lib_line)
    log(f"Ensured PERL5LIB is set in {shell_rc}")


def check_and_install_homebrew():
    homebrew_path_1 = "/opt/homebrew/bin"
    homebrew_path_2 = "/usr/local/bin"
    shell_rc = Path.home() / ".zprofile"

    # 1) Check if brew exists at all:
    if not shutil.which("brew"):
        log("Homebrew not found. Installing...")
        homebrew_install_command = (
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        )
        run_command(homebrew_install_command, shell=True)
        log("Homebrew installed successfully.")
    else:
        log("Homebrew already installed.")

    # 2) Update ~/.zprofile so future shells have these paths
    log("Ensuring Homebrew paths are in the shell RC file...")
    append_to_file_if_not_exists(shell_rc, f'export PATH="{homebrew_path_1}:$PATH"')
    append_to_file_if_not_exists(shell_rc, f'export PATH="{homebrew_path_2}:$PATH"')

    # 3) Make sure this *current* Python process sees brew in PATH as well:
    current_path = os.environ.get("PATH", "")
    # Prepend /opt/homebrew/bin if it's not already in PATH
    if homebrew_path_1 not in current_path:
        os.environ["PATH"] = f"{homebrew_path_1}:{os.environ['PATH']}"
        log(f"Added {homebrew_path_1} to PATH for this session.")
    # Similarly for /usr/local/bin
    if homebrew_path_2 not in os.environ["PATH"]:
        os.environ["PATH"] = f"{homebrew_path_2}:{os.environ['PATH']}"
        log(f"Added {homebrew_path_2} to PATH for this session.")

    # 4) Now that 'brew' is definitely on PATH, we can safely run brew commands:
    log("Updating Homebrew (brew update)...")
    run_command(["brew", "update"], check=False)

    log("Done installing Homebrew. No need to restart the shell for this script to continue.")


def install_homebrew_packages():
    """
    Installs the required Homebrew packages.
    """
    log("Installing required Homebrew packages...")
    packages = [
        "perl",
        "cpanminus",
        "pkg-config",
        "gnuplot",
        "fftw",
        "qt",
        "make",
        "gcc",
        "gsl",
        "cmake",
        "git",
        "openssl",
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


def install_perl():
    """
    Installs the perl package from Homebrew if not already installed, and then pins it.
    """
    if not shutil.which("perl"):
        run_command("brew install perl", shell=True)
        run_command("brew pin perl", shell=True)


def setup_local_lib(pl_prefix):
    """
    Sets up local::lib for Perl module management on macOS.

    Args:
        pl_prefix (Path): The prefix path for the local::lib installation.
    """
    log(f"Setting up local::lib with PL_PREFIX={pl_prefix} ...")

    # 2) Set up environment variable so cpanm will put modules under pl_prefix
    os.environ["PERL_MM_OPT"] = f"INSTALL_BASE={pl_prefix}"

    # 3) Install local::lib via cpanm
    run_command(
        [
            "cpanm",
            "--notest",  # skip tests if they're problematic on mac
            "--local-lib-contained",
            str(pl_prefix),
            "local::lib",
        ],
        check=True,
    )

    # 4) Update ~/.perldlrc so that local::lib is recognized
    add_perl5lib_to_perldlrc(pl_prefix)


def install_perl_modules(pl_prefix):
    """
    Installs the required Perl modules on macOS.

    Args:
        pl_prefix (Path): The prefix path for the Perl modules installation.
    """
    log(f"Installing Perl modules into {pl_prefix} ...")

    # Potentially relevant Homebrew libs for some of these Perl modules
    brew_deps = ["gnuplot", "cfitsio", "swig"]
    for pkg in brew_deps:
        run_command(["brew", "install", pkg], check=False)

    # If you want to guarantee GSL is installed:
    run_command(["brew", "install", "gsl"], check=False)

    modules = [
        # Core
        "local::lib",
        "Devel::CheckLib",
        "List::MoreUtils",
        "Capture::Tiny",
        # Config
        "Config::IniFiles",
        # File Handling
        "File::HomeDir",
        "File::ShareDir",
        "File::ShareDir::Install",
        # Testing
        "Test::Builder",
        # Parallel
        "Parallel::ForkManager",
        # Math
        "Math::GSL::Alien",
        "Math::GSL",
        "Math::Interpolate",
        "Math::Interpolator",
        "Math::RungeKutta",
        # PDL
        "PDL",
        "PDL::GSL::INTEG",
        "PDL::Graphics::Gnuplot",
        "PDL::Graphics::Simple",
        # Inline
        "Inline",
        "Inline::C",
        "Inline::Python",
        # Web
        "Net::SSLeay",
        # Charts
        "Chart::Gnuplot",
        "Alien::Build::Plugin::Gather::Dino",
        # CSV
        "Text::CSV",
        # Moose-like
        "Moo::Role",
    ]

    try:
        run_command(["cpanm", "-l", str(pl_prefix), "--notest"] + modules, check=False)
    except Exception as e:
        log(e)
        log("Bulk install failed. Reverting to individual Perl Dependency Installation.")
        for module in modules:
            try:
                run_command(["cpanm", "-l", str(pl_prefix), "--notest", module], check=True)
            except Exception:
                run_command(["cpanm", "--force", module], check=True)

    # Evaluate local::lib env
    eval_command = f"eval `perl -I {pl_prefix}/lib/perl5 -Mlocal::lib={pl_prefix}`"
    log(f"Evaluating local::lib environment with: {eval_command}")
    run_command(eval_command, shell=True)


############################
# 3) MODIFIED clone_and_build_flux
############################


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

    # Instead of run_command(..., capture_output=True), we capture the build output with capture_and_tee:
    make_output = capture_and_tee(["make", "everything"])

    # Use the new logic to extract the relevant paths:
    perl_inc_path, pdllib_path = extract_flux_paths(make_output)

    # If we found a new Perl inc path, we can update .zprofile accordingly
    if perl_inc_path:
        zprofile_path = Path.home() / ".zprofile"
        # e.g. 'export PERL5LIB="/Users/tester/Library/perl5:$PERL5LIB"'
        # We'll call your update_profile:
        line_to_add = f'export PERL5LIB="{perl_inc_path}:$PERL5LIB"\n'
        updated = update_profile(zprofile_path, line_to_add, "Flux Perl Inc Path")
        if updated:
            log(f"Added new Perl inc path to {zprofile_path}")

    # Also ensure .perldlrc is updated with the PDLLIB lines if we found one:
    update_perldlrc(pdllib_path)


############################
# 4) REMAINDER OF SCRIPT UNCHANGED
############################


def setup_python_virtualenv():
    """
    Sets up a Python virtual environment without 'sourcing' it.

    Note:
        We do NOT activate the environment in this function. Instead, any
        pip installs will be done by explicitly calling the Python executable
        inside .venv. The user can still manually source .venv/bin/activate
        later if they want an interactive shell in that venv.
    """
    log("Setting up Python virtual environment...")
    venv_dir = Path(".venv")
    if not venv_dir.exists():
        run_command(["python3", "-m", "venv", str(venv_dir)])
    else:
        log("Python virtual environment already exists.")
    log("Virtual environment setup complete (not activated).")


def install_fluxpype():
    """
    Installs Python dependencies for FluxPype by calling the .venv Python interpreter directly.
    """
    log("Installing FluxPype Python dependencies...")
    venv_python = Path(".venv") / "bin" / "python3"
    script_dir = Path(__file__).parent
    requirements_file = script_dir / "requirements.txt"

    # Ensure the .venv was created:
    if not venv_python.exists():
        log("Python virtual environment not found. Please run setup_python_virtualenv first.", level="ERROR")
        return

    # If the requirements file exists, install from it using the venv python
    if requirements_file.exists():
        run_command([str(venv_python), "-m", "pip", "install", "-r", str(requirements_file)])
        log("FluxPype Python dependencies installed successfully.")
    else:
        log("Requirements file not found. Skipping Python dependency installation.", level="WARNING")

    log("Attempting to install FluxPype Package")
    run_command([str(venv_python), "-m", "pip", "install", "-e", str(script_dir)])

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


def add_perl5lib_to_perldlrc(pl_prefix):
    """
    Appends pl_prefix/lib/perl5 to PERL5LIB in ~/.perldlrc, without overwriting existing paths.
    """
    perldlrc_path = Path.home() / ".perldlrc"
    perl5_append_snippet = f"""
# Append PL_PREFIX/lib/perl5 to PERL5LIB if it's not already in there
if (defined $ENV{{PERL5LIB}}) {{
    if ($ENV{{PERL5LIB}} !~ /{pl_prefix}\\/lib\\/perl5/) {{
        $ENV{{PERL5LIB}} .= ":{pl_prefix}/lib/perl5";
    }}
}} else {{
    $ENV{{PERL5LIB}} = "{pl_prefix}/lib/perl5";
}}
"""
    append_to_file_if_not_exists(perldlrc_path, perl5_append_snippet)


def main():
    """
    Main function that orchestrates the complete installation process for the FluxPype project on macOS.
    """
    # 1) Ensure Python is a suitable version
    check_python_version()

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
        setup_local_lib(pl_prefix)
        enable_perl5lib(pl_prefix)
        try:
            install_perl_modules(pl_prefix)
        except Exception as e:
            log(e, level="")
            log("Perl Modules may need to be addressed individually", level="")
        clone_and_build_flux(fl_prefix, pl_prefix)
        setup_python_virtualenv()
        install_fluxpype()
        append_flux_env_vars_to_rc(fl_prefix, pl_prefix, shell_rc)
        add_perl5lib_to_perldlrc(pl_prefix)
        log("Installation completed successfully!")
        log("Open a new shell, navigate to fluxpype, reactivate the environment, test 'flux-config-run'")
        log("If that doesn't work, try 'pip install -e .' too.")
    except Exception as e:
        log(f"An error occurred: {e}", level="ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
