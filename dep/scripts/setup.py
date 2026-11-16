from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from pathlib import Path
import os


class CustomInstallCommand(install):
    def run(self):
        super().run()
        self.save_module_root()

    def save_module_root(self):
        install_path = Path(self.install_lib)
        os.environ["FLUXPYPE_ROOT"] = str(install_path)
        with open(install_path / "fluxpype/__init__.py", "a") as f:
            f.write(f"\nos.environ['FLUXPYPE_ROOT'] = '{install_path}'")


class CustomDevelopCommand(develop):
    def run(self):
        super().run()
        self.save_module_root()

    def save_module_root(self):
        develop_path = Path(self.egg_base) / "fluxpype"
        os.environ["FLUXPYPE_ROOT"] = str(develop_path)
        with open(develop_path / "fluxpype/__init__.py", "a") as f:
            f.write(f"\nos.environ['FLUXPYPE_ROOT'] = '{develop_path}'")


def load_flux_installer():
    import importlib.util

    installer_path = Path(__file__).parent / "flux_installer.py"
    if not installer_path.exists():
        raise FileNotFoundError(f"flux_installer.py not found at {installer_path}")
    spec = importlib.util.spec_from_file_location("flux_installer", installer_path)
    flux_installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flux_installer)
    return flux_installer


flux_installer = load_flux_installer()

setup(
    name="fluxpype",
    version="0.1.4",
    description="A wrapper and installer for the FLUX model",
    long_description=(
        Path("README.md").read_text() if Path("README.md").exists() else ""
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/gillyspace27/fluxpype",
    packages=find_packages(where="fluxpype"),
    package_dir={"": "fluxpype"},
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
        "rich",
    ],
    extras_require={
        "FLUXcore": [],
    },
    package_data={
        "fluxpype": [
            "bootstrap_pdl.pl",
            "science/*",
            "plotting/*",
        ],
    },
    include_package_data=True,
    # cmdclass={
    #     "install": CustomInstallCommand,
    #     "develop": CustomDevelopCommand,
    # },
)
