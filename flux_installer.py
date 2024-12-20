import subprocess
import os
import shutil
from pathlib import Path
from contextlib import contextmanager


class FluxInstaller:
    def __init__(self, fl_prefix, pl_prefix):
        self.fl_prefix = fl_prefix
        self.pl_prefix = pl_prefix

    def ensure_dependencies(self):
        print("Ensuring dependencies...")
        if not shutil.which("cpanm"):
            raise EnvironmentError(
                "cpanm (App::cpanminus) not found. Please install it to proceed."
            )
        print("cpanm found.")

    def install_perlbrew(self):
        print("Installing Perlbrew if not already installed...")
        perl_version = "perl-5.36.0"

        if not shutil.which("perlbrew"):
            print("Perlbrew not found, installing...")
            subprocess.check_call(
                "/bin/bash -c 'curl -L https://install.perlbrew.pl | bash'", shell=True
            )
            perlbrew_env = Path.home() / "perl5" / "perlbrew" / "etc" / "bashrc"
            if perlbrew_env.exists():
                print(f"Initializing Perlbrew from {perlbrew_env}...")
                self.initialize_perlbrew(perlbrew_env)
            else:
                raise FileNotFoundError(
                    f"Perlbrew initialization script not found: {perlbrew_env}"
                )
        else:
            print("Perlbrew is already installed.")

        if not shutil.which("perlbrew"):
            raise EnvironmentError(
                "Perlbrew command not found after installation. Ensure your environment is set up correctly."
            )

        installed_versions = subprocess.check_output(["perlbrew", "list"]).decode()
        if perl_version not in installed_versions:
            print(f"Installing Perl version {perl_version} using Perlbrew...")
            subprocess.check_call(["perlbrew", "install", perl_version])
        else:
            print(f"Perl version {perl_version} is already installed.")

        print(f"Switching to Perl version {perl_version}...")
        subprocess.check_call(["perlbrew", "use", perl_version])

    def initialize_perlbrew(self, perlbrew_env):
        print("Initializing Perlbrew environment...")
        command = f"source {perlbrew_env} && env"
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, shell=True, executable="/bin/bash"
        )
        output, _ = proc.communicate()

        for line in output.decode().splitlines():
            key, _, value = line.partition("=")
            os.environ[key] = value

    @contextmanager
    def temporary_env(self, env):
        print(f"Setting temporary environment variables: {env}")
        old_env = os.environ.copy()
        os.environ.update(env)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def setup_perl_environment(self):
        print(f"Setting up local::lib with PL_PREFIX={self.pl_prefix}...")
        env = {"PERL_MM_OPT": f"INSTALL_BASE={self.pl_prefix}"}
        with self.temporary_env(env):
            subprocess.check_call(
                ["cpanm", "--local-lib-contained", self.pl_prefix, "local::lib"]
            )

    def install_prerequisites(self):
        print("Installing Perl dependencies...")
        perl_dependencies = [
            "local::lib",
            "File::ShareDir",
            "File::ShareDir::Install",
            "PDL::Graphics::Gnuplot",
            "Math::RungeKutta",
            "Moo::Role",
            "Chart::Gnuplot",
            "Text::CSV",
            "Math::Interpolate",
            "Math::GSL",
            "Config::IniFiles",
            "File::HomeDir",
            "Inline::C",
            "Parallel::ForkManager",
        ]
        subprocess.check_call(["cpanm"] + perl_dependencies)

    def install_flux(self):
        if self.check_existing_flux():
            print(
                "FLUX installation detected. Linking the Python package to the existing installation."
            )
            return

        print("No existing FLUX installation detected. Installing FLUX...")
        self.clone_flux_repo()
        self.build_and_install_flux()

    def check_existing_flux(self):
        libflux_path = self.fl_prefix / "lib" / "libflux.a"
        flux_pm_path = self.pl_prefix / "lib" / "perl5" / "Flux.pm"
        if libflux_path.exists() and flux_pm_path.exists():
            print(
                f"Found existing FLUX installation:\n- {libflux_path}\n- {flux_pm_path}"
            )
            return True
        return False

    def clone_flux_repo(self):
        repo_url = "https://github.com/lowderchris/fluxon-mhd.git"
        repo_dir = self.fl_prefix / "fluxon-mhd"
        if not repo_dir.exists():
            print(f"Cloning FLUX repository from {repo_url}...")
            subprocess.check_call(["git", "clone", repo_url, str(repo_dir)])
        else:
            print("FLUX repository already exists. Pulling latest changes...")
            subprocess.check_call(["git", "-C", str(repo_dir), "pull"])

    def build_and_install_flux(self):
        repo_dir = self.fl_prefix / "fluxon-mhd"
        os.chdir(repo_dir)
        env = {"FL_PREFIX": str(self.fl_prefix), "PL_PREFIX": str(self.pl_prefix)}
        with self.temporary_env(env):
            subprocess.check_call(["make", "libbuild"])
            subprocess.check_call(["make", "libinstall"])
            subprocess.check_call(["make", "pdlbuild"])
            subprocess.check_call(["make", "pdltest"])
            subprocess.check_call(["make", "pdlinstall"])
        self.update_shell_config()

    def update_shell_config(self):
        shell_config = Path.home() / ".zprofile"
        existing_config = shell_config.read_text() if shell_config.exists() else ""

        if f"export FL_PREFIX={self.fl_prefix}" not in existing_config:
            with shell_config.open("a") as file:
                file.write(f"\n# FLUX environment setup\n")
                file.write(f"export FL_PREFIX={self.fl_prefix}\n")
                file.write(f"export PL_PREFIX={self.pl_prefix}\n")
                perl_lib_path = self.pl_prefix / "lib" / "perl5"
                file.write(
                    f"eval `perl -I {perl_lib_path} -Mlocal::lib={perl_lib_path}`\n"
                )
            print(f"Shell configuration updated in {shell_config}.")
        else:
            print("FLUX environment variables already set in shell configuration.")


def run_flux_installer(fl_prefix=None, pl_prefix=None):
    """Helper to run the FluxInstaller."""
    user_home = Path.home()
    fl_prefix = fl_prefix or Path(
        os.environ.get("FL_PREFIX", user_home / "Library" / "flux")
    )
    pl_prefix = pl_prefix or Path(
        os.environ.get("PL_PREFIX", user_home / "Library" / "perl5")
    )
    print("THIS IS RUNNIGN AND IT IS A BIG DEAL")
    installer = FluxInstaller(fl_prefix, pl_prefix)
    installer.ensure_dependencies()
    installer.install_perlbrew()
    installer.setup_perl_environment()
    installer.install_prerequisites()
    installer.install_flux()
