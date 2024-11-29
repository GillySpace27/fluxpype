
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import shutil

class CustomInstallCommand(install):
    """Customized setuptools install command - installs prerequisites and FLUX."""

    def run(self):
        # Run the standard installation process first
        install.run(self)

        # Paths for FLUX installation
        user_home = os.path.expanduser("~")
        fl_prefix = os.environ.get("FL_PREFIX", os.path.join(user_home, "Library", "flux"))
        pl_prefix = os.environ.get("PL_PREFIX", os.path.join(user_home, "Library", "perl5"))

        print(f"Checking FLUX installation with FL_PREFIX={fl_prefix} and PL_PREFIX={pl_prefix}...")

        try:
            # Perform necessary installation steps
            self.install_perlbrew()
            self.install_homebrew()
            self.setup_virtualenv(pl_prefix)
            self.install_prerequisites(pl_prefix)
            self.install_flux(fl_prefix, pl_prefix)
            print("Installation completed successfully.")
        except Exception as e:
            print(f"Installation failed: {e}")
            raise

    def install_perlbrew(self):
        """Ensure Perlbrew is installed and a specific version of Perl is available."""
        perl_version = "perl-5.36.0"  # Define the Perl version you need

        # Install Perlbrew if not installed
        if not shutil.which("perlbrew"):
            print("Installing Perlbrew...")
            subprocess.check_call(["curl", "-L", "https://install.perlbrew.pl", "|", "bash"], shell=True)
        else:
            print("Perlbrew is already installed.")

        # Initialize Perlbrew environment
        perlbrew_env = os.path.expanduser("~/.perlbrew/etc/bashrc")
        if os.path.exists(perlbrew_env):
            subprocess.check_call(["bash", "-c", f"source {perlbrew_env}"])

        # Check if the desired Perl version is installed
        installed_versions = subprocess.check_output(["perlbrew", "list"]).decode()
        if perl_version not in installed_versions:
            print(f"Installing Perl version {perl_version} using Perlbrew...")
            subprocess.check_call(["perlbrew", "install", perl_version])
        else:
            print(f"Perl version {perl_version} is already installed.")

        # Use the installed Perl version
        print(f"Switching to Perl version {perl_version}...")
        subprocess.check_call(["perlbrew", "use", perl_version])

    def install_homebrew(self):
        """Ensure Homebrew is installed (macOS only) and required packages are available."""
        required_packages = ["make", "perl", "gcc", "cpanm", "gnuplot", "fftw"]

        # Check if Homebrew is installed
        if shutil.which("brew") is None:
            print("Installing Homebrew...")
            subprocess.check_call([
                "/bin/bash",
                "-c",
                "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            ], shell=True)
        else:
            print("Homebrew is already installed.")

        # Install required packages
        for package in required_packages:
            try:
                print(f"Checking for {package}...")
                subprocess.check_call(["brew", "info", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"{package} is already installed.")
            except subprocess.CalledProcessError:
                print(f"{package} not found. Installing...")
                subprocess.check_call(["brew", "install", package])

    def setup_virtualenv(self, pl_prefix):
        """Set up a virtual Perl environment using local::lib."""
        print(f"Setting up local::lib with PL_PREFIX={pl_prefix}...")
        os.environ["PERL_MM_OPT"] = f"INSTALL_BASE={pl_prefix}"
        subprocess.check_call(["cpanm", "--local-lib-contained", pl_prefix, "local::lib"])

    def install_prerequisites(self, pl_prefix):
        """Install Perl dependencies."""
        perl_dependencies = [
            "local::lib", "File::ShareDir", "File::ShareDir::Install",
            "PDL::Graphics::Gnuplot", "Math::RungeKutta", "Moo::Role",
            "Chart::Gnuplot", "Text::CSV", "Math::Interpolate", "Math::GSL",
            "Config::IniFiles", "File::HomeDir", "Inline::C",
            "Parallel::ForkManager"
        ]
        print("Installing Perl dependencies...")
        subprocess.check_call(["cpanm"] + perl_dependencies)

    def install_flux(self, fl_prefix, pl_prefix):
        """Check for or install FLUX."""
        if self.check_existing_flux(fl_prefix, pl_prefix):
            print("FLUX installation detected. Linking the Python package to the existing installation.")
            return

        print("No existing FLUX installation detected. Installing FLUX...")
        self.clone_flux_repo(fl_prefix)
        self.build_and_install_flux(fl_prefix, pl_prefix)

    def check_existing_flux(self, fl_prefix, pl_prefix):
        """Check if FLUX is already installed."""
        libflux_path = os.path.join(fl_prefix, "lib", "libflux.a")
        flux_pm_path = os.path.join(pl_prefix, "lib", "perl5", "Flux.pm")
        if os.path.exists(libflux_path) and os.path.exists(flux_pm_path):
            print(f"Found existing FLUX installation: \
                    - {libflux_path} \
                    - {flux_pm_path}")
            return True
        return False

    def clone_flux_repo(self, fl_prefix):
        """Clone the FLUX repository."""
        repo_url = "https://github.com/lowderchris/fluxon-mhd.git"
        repo_dir = os.path.join(fl_prefix, "fluxon-mhd")
        if not os.path.exists(repo_dir):
            print(f"Cloning FLUX repository from {repo_url}...")
            subprocess.check_call(["git", "clone", repo_url, repo_dir])
        else:
            print("FLUX repository already exists. Pulling latest changes...")
            subprocess.check_call(["git", "-C", repo_dir, "pull"])

    def build_and_install_flux(self, fl_prefix, pl_prefix):
        """Build and install FLUX."""
        repo_dir = os.path.join(fl_prefix, "fluxon-mhd")
        os.chdir(repo_dir)
        os.environ["FL_PREFIX"] = fl_prefix
        os.environ["PL_PREFIX"] = pl_prefix
        subprocess.check_call(["make", "libbuild"])
        subprocess.check_call(["make", "libinstall"])
        subprocess.check_call(["make", "pdlbuild"])
        subprocess.check_call(["make", "pdltest"])
        subprocess.check_call(["make", "pdlinstall"])
        self.update_shell_config(fl_prefix, pl_prefix)

    def update_shell_config(self, fl_prefix, pl_prefix):
        """Update user's shell configuration to include FLUX paths."""
        shell_config = os.path.expanduser("~/.zprofile")
        if os.path.exists(shell_config):
            with open(shell_config, "r") as file:
                existing_config = file.read()
        else:
            existing_config = ""

        if f"export FL_PREFIX={fl_prefix}" not in existing_config:
            with open(shell_config, "a") as file:
                file.write(f"\n# FLUX environment setup\n")
                file.write(f"export FL_PREFIX={fl_prefix}\n")
                file.write(f"export PL_PREFIX={pl_prefix}\n")
                perl_lib_path = os.path.join(pl_prefix, "lib", "perl5")
                file.write(f"eval `perl -I {perl_lib_path} -Mlocal::lib={perl_lib_path}`\n")
            print(f"Shell configuration updated in {shell_config}.")
        else:
            print("FLUX environment variables already set in shell configuration.")

# Check if `src` directory exists and use it, otherwise use current directory
package_dir = {"": "src"} if os.path.isdir("src") else {"": "."}
packages = find_packages(where=package_dir[""])
setup(
    name="fluxpype",
    version="0.1.3",
    description="A wrapper and installer for the FLUX Model",
    long_description=open("README.md").read() if os.path.exists("README.md") else "A wrapper and installer for the FLUX Model",
    long_description_content_type="text/markdown",
    url="https://github.com/gillyspace27/fluxpype",
    packages=packages,
    package_dir=package_dir,
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
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    cmdclass={
        "install": CustomInstallCommand,
    },
)
