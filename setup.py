import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
import sys
import os


class CustomInstallCommand(install):
    """Customized setuptools install command to integrate FLUX installation."""

    def run(self):
        # Run standard installation
        install.run(self)

        # Run the unified installer script in a subprocess
        installer_path = os.path.join(os.path.dirname(__file__), "fluxpype", "unified_installer.py")
        if os.path.exists(installer_path):
            try:
                print("Running unified installer...")
                subprocess.run([sys.executable, installer_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"FLUX installation failed: {e}")
                raise RuntimeError("Failed to complete FLUX installation.")
        else:
            print("Unified installer script not found. Skipping FLUX installation.")


setup(
    name="fluxpype",
    version="0.1.8",
    description="A Python wrapper and installer for the FLUX model",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "matplotlib",
        "numpy<2.0",
        "sunpy",
        "pandas",
        "tqdm",
        "astropy<7.0",
        "bs4",
        "lxml",
        "zeep",
        "drms",
        "timeout-decorator",
        "rich",
    ],
    cmdclass={"install": CustomInstallCommand},
)
