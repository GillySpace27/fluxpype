
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
import os

class CustomInstallCommand(install):
    """Customized setuptools install command - installs FLUX."""
    def run(self):
        # Run the standard installation process first
        install.run(self)

        # FLUX installation script path
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "install-fluxpipe-macos.sh")

        # Ensure the script is executable
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Installation script not found: {script_path}")

        print("Installing FLUX using the provided script...")
        try:
            subprocess.check_call(["bash", script_path], shell=False)
            print("FLUX installation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"FLUX installation failed with error: {e}")
            raise

setup(
    name="fluxpype",
    version="0.1.2",
    description="The python wrapper used to run FLUX",
    packages=find_packages(),
    install_requires=[
        # Add Python dependencies here if required
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
)
