import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
import os
from install_flux import install_flux, validate_flux_installation


class CustomInstallCommand(install):
    """Customized setuptools install command to integrate FLUX installation."""

    def run(self):
        # Run standard installation
        install.run(self)

        # Install FLUX using custom logic
        try:
            install_flux()
            validate_flux_installation()
        except Exception as e:
            print(f"FLUX installation failed: {e}")
            raise


setup(
    name="fluxpype",
    version="0.2.0",
    description="A Python wrapper for the FLUX model",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "matplotlib",
        "sunpy",
        "pandas",
        "tqdm",
        "astropy",
        "bs4",
        "lxml",
        "zeep",
        "drms",
        "timeout_decorator",
    ],
    cmdclass={
        "install": CustomInstallCommand,
    },
)
