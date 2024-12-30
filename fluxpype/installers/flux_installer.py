import os
import shutil
import subprocess
from pathlib import Path


def check_dependencies():
    """Check for required system tools."""
    tools = ["perl", "cpanm", "make", "gcc"]
    missing = [tool for tool in tools if not shutil.which(tool)]
    if missing:
        print(f"The following dependencies are missing: {', '.join(missing)}")
        return False
    return True


def prompt_user():
    """Prompt user for installation options."""
    response = input("FLUX is not installed. Would you like to install it now? (yes/no): ").lower()
    return response in ["yes", "y", ""]


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


def main():
    """Main function to manage FLUX installation."""
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return

    if prompt_user():
        install_flux()
    else:
        print("Skipping FLUX installation.")


if __name__ == "__main__":
    main()
