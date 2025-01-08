import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Customized setuptools install command to integrate FLUX installation."""

    def run(self):
        # Run standard installation
        install.run(self)

        # Run the unified installer script after dependencies are installed
        try:
            installer_path = os.path.join(os.path.dirname(__file__), "fluxpype", "unified_installer.py")
            if os.path.exists(installer_path):
                print("Running unified installer...")
                subprocess.run([sys.executable, installer_path], check=True)
            else:
                print("Unified installer script not found. Skipping FLUX installation.")
        except subprocess.CalledProcessError as e:
            print(f"FLUX installation failed: {e}")
            raise RuntimeError("Failed to complete FLUX installation.")


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
        "opencv-python",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "flux-install = fluxpype.unified_installer:main",
            "flux-config-run = fluxpype.config_runner:run",
            "flux-config-view = fluxpype.config_runner:view",
            "flux-config-edit = fluxpype.config_runner:open_config",
            "flux-config-gallery = fluxpype.config_runner:gallery",
        ]
    },
    cmdclass={"install": CustomInstallCommand},
)
