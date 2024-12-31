
import os
import subprocess
import sys
import shutil


def log(message):
    print(f"[INFO] {message}")


def check_command(command):
    """Check if a command exists on the system."""
    return shutil.which(command) is not None


def install_homebrew():
    """Install Homebrew if not already installed."""
    if not check_command("brew"):
        log("Homebrew not found. Installing...")
        subprocess.run(
            ["/bin/bash", "-c", "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"],
            check=True,
        )
        log("Homebrew installed successfully.")
    else:
        log("Homebrew is already installed.")


def install_dependencies():
    """Install required dependencies using Homebrew and cpanm."""
    brew_dependencies = ["gnuplot", "fftw", "perl"]
    for dep in brew_dependencies:
        log(f"Checking for {dep}...")
        if subprocess.run(["brew", "list", dep], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0:
            log(f"{dep} not found. Installing...")
            subprocess.run(["brew", "install", dep], check=True)
        else:
            log(f"{dep} is already installed.")

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
        "PDL",
        "File::ShareDir",
        "PDL::Graphics::Gnuplot",
        "Math::RungeKutta",
        "Term::ReadKey",
    ]

    log("Installing Perl modules...")
    for module in perl_modules:
        log(f"Installing {module}...")
        subprocess.run(["cpanm", "--local-lib", perl_lib_dir, module], check=True)

    # Update .zprofile for Perl environment
    zprofile_path = os.path.expanduser("~/.zprofile")
    with open(zprofile_path, "a") as zprofile:
        zprofile.write(f"\n# Perl environment setup\n")
        zprofile.write(f"export PERL5LIB={perl_lib_dir}/lib/perl5:$PERL5LIB\n")
    log("Perl setup complete.")


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
    subprocess.run(["make", "everything"], check=True)
    log("Flux build complete.")


def main():
    if sys.platform != "darwin":
        log("This installer is designed for macOS only.")
        sys.exit(1)

    log("Starting unified FLUX installer...")
    install_homebrew()
    install_dependencies()
    setup_perl()
    clone_and_build_flux()
    log("Installation completed successfully!")


if __name__ == "__main__":
    main()
