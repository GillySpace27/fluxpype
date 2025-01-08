import os
import shutil
import subprocess
from pathlib import Path


def is_flux_installed():
    """Check if FLUX is installed."""
    flux_path = Path(os.environ.get("FL_PREFIX", Path.home() / "Library" / "flux"))
    flux_files = ["lib/libflux.a", "bin/flux"]
    return all((flux_path / file).exists() for file in flux_files)


def check_dependencies():
    """Check for required system tools."""
    tools = ["perl", "cpanm", "make", "gcc", ""]
    missing = [tool for tool in tools if not shutil.which(tool)]
    if missing:
        print(f"The following dependencies are missing: {', '.join(missing)}")
        return False
    return True


def install_flux():
    """Install FLUX and dependencies."""
    try:
        print("Installing FLUX...")
        repo_url = "https://github.com/lowderchris/fluxon-mhd.git"
        repo_dir = Path.home() / "fluxon-mhd"
        if not repo_dir.exists():
            subprocess.check_call(["git", "clone", repo_url, str(repo_dir)])
        else:
            print("FLUX repository found. Updating...")
            subprocess.check_call(["git", "-C", str(repo_dir), "pull"])
        os.chdir(repo_dir)
        subprocess.check_call(["make", "everything"])
        print("FLUX installation completed successfully!")
    except Exception as e:
        print(f"FLUX installation failed: {e}")


def install():
    return post_build_check()

def post_build_check():
    """Run after fluxpype is built to check for FLUX."""
    # Check if FLUXcore was explicitly requested
    if "FLUXcore" in os.environ.get("PIP_OPTIONAL_ARGS", ""):
        print("FLUXcore extra detected. Proceeding with FLUX installation...")
        if check_dependencies():
            install_flux()
        else:
            print("Please install missing dependencies manually before proceeding.")
        return

    # Default behavior for normal installation
    if is_flux_installed():
        print("FLUX is already installed. You're ready to go!")
    else:
        print("FLUX installation not detected.")
        if check_dependencies() and prompt_user():
            install_flux()
        else:
            print("Please install FLUX manually or rerun the installer.")


def prompt_user():
    """Prompt user for installation options."""
    response = input("Flux isn't installed yet. Shall we proceed to install it? (yes/no): ").lower()
    return response in ["yes", "y", ""]


if __name__ == "__main__":
    install()
