import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import sys
import shutil

class CustomInstallCommand(install):
    """Customized setuptools install command - installs FLUX directly."""

    def run(self):
        # Run the standard installation process first
        install.run(self)

        # Paths for FLUX installation
        user_home = os.path.expanduser("~")
        fl_prefix = os.environ.get("FL_PREFIX", os.path.join(user_home, "Library", "flux"))
        pl_prefix = os.environ.get("PL_PREFIX", os.path.join(user_home, "Library", "perl5"))

        print(f"Installing FLUX with FL_PREFIX={fl_prefix} and PL_PREFIX={pl_prefix}...")

        try:
            # Install prerequisites (example for macOS)
            self.install_prerequisites()

            # Clone the FLUX repository
            repo_url = "https://github.com/lowderchris/fluxon-mhd.git"
            repo_dir = os.path.join(user_home, "fluxon-mhd")
            self.clone_flux_repo(repo_url, repo_dir)

            # Build and install FLUX
            self.build_and_install_flux(repo_dir, fl_prefix, pl_prefix)

            print("FLUX installation completed successfully.")
        except Exception as e:
            print(f"FLUX installation failed: {e}")
            sys.exit(1)

    def install_prerequisites(self):
        """Ensure required tools are installed."""
        required_tools = ["make", "perl", "gcc", "cpanm", "gnuplot", "fftw"]
        for tool in required_tools:
            if not shutil.which(tool):
                raise EnvironmentError(f"Required tool {tool} is not installed. Please install it first.")

        # Install Perl dependencies using cpanm
        perl_dependencies = ["PDL", "File::ShareDir", "PDL::Graphics::Gnuplot", "Math::RungeKutta", "Term::ReadKey"]
        for dep in perl_dependencies:
            print(f"Installing Perl dependency: {dep}")
            subprocess.check_call(["cpanm", dep])

    def clone_flux_repo(self, repo_url, repo_dir):
        """Clone the FLUX repository."""
        if not os.path.exists(repo_dir):
            print(f"Cloning FLUX repository from {repo_url}...")
            subprocess.check_call(["git", "clone", repo_url, repo_dir])
        else:
            print("FLUX repository already exists. Pulling latest changes...")
            subprocess.check_call(["git", "-C", repo_dir, "pull"])

    def build_and_install_flux(self, repo_dir, fl_prefix, pl_prefix):
        """Build and install FLUX from source."""
        os.chdir(repo_dir)

        # Set environment variables
        os.environ["FL_PREFIX"] = fl_prefix
        os.environ["PL_PREFIX"] = pl_prefix

        # Run build commands
        subprocess.check_call(["make", "libbuild"])
        subprocess.check_call(["make", "libinstall"])
        subprocess.check_call(["make", "pdlbuild"])
        subprocess.check_call(["make", "pdltest"])
        subprocess.check_call(["make", "pdlinstall"])

        # Append to shell configuration for persistence
        self.update_shell_config(fl_prefix, pl_prefix)

    def update_shell_config(self, fl_prefix, pl_prefix):
        """Update user's shell configuration to include FLUX paths."""
        shell_config = os.path.expanduser("~/.zprofile")
        with open(shell_config, "a") as file:
            file.write(f" # FLUX environment setup")
            file.write(f"export FL_PREFIX={fl_prefix}")
            file.write(f"export PL_PREFIX={pl_prefix}")
            perl_lib_path = os.path.join(pl_prefix, "lib", "perl5")
            file.write(f"eval `perl -I {perl_lib_path} -Mlocal::lib={perl_lib_path}`")
        print(f"Shell configuration updated in {shell_config}.")


# Check if `src` directory exists and use it, otherwise use current directory
package_dir = {"": "src"} if os.path.isdir("src") else {"": "."}
packages = find_packages(where=package_dir[""])
setup(
    name="fluxpype",
    version="0.1.2",
    description="A basic Python project",
    long_description=open("README.md").read(),
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
        "shutil",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    cmdclass={
        "install": CustomInstallCommand,
    },
)
