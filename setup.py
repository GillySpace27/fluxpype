"""Minimal setup.py for fluxpype; metadata and dependencies are in pyproject.toml."""
import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Customized setuptools install command to integrate FLUX installation."""

    def run(self):
        install.run(self)
        try:
            installer_path = os.path.join(
                os.path.dirname(__file__),
                "fluxpype",
                "comprehensive_fluxpype_installer.py",
            )
            if os.path.exists(installer_path):
                print("Running unified installer...")
                subprocess.run([sys.executable, installer_path], check=True)
            else:
                print("Unified installer script not found. Skipping FLUX installation.")
        except subprocess.CalledProcessError as e:
            print(f"FLUX installation failed: {e}")
            raise RuntimeError("Failed to complete FLUX installation.") from e


setup(cmdclass={"install": CustomInstallCommand})
