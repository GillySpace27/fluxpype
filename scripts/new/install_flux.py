import os
import pexpect
import subprocess
import sys
import shutil


def run_flux_install(fl_prefix=None, pl_prefix=None):
    # Default paths
    user_home = os.path.expanduser("~")
    fl_prefix = fl_prefix or os.path.join(user_home, "Library", "flux")
    pl_prefix = pl_prefix or os.path.join(user_home, "Library", "perl5")

    # Export environment variables
    os.environ["FL_PREFIX"] = fl_prefix
    os.environ["PL_PREFIX"] = pl_prefix

    # Check dependencies
    required_tools = ["make", "perl", "gcc", "cpanm", "gnuplot", "fftw"]
    for tool in required_tools:
        if not shutil.which(tool):
            print(f"Error: {tool} is not installed. Please install it first.")
            sys.exit(1)

    # Clone the repository
    repo_url = "https://github.com/lowderchris/fluxon-mhd.git"
    repo_dir = os.path.join(user_home, "fluxon-mhd")

    if not os.path.exists(repo_dir):
        print("Cloning FLUX repository...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    else:
        print("FLUX repository already exists. Pulling latest changes...")
        subprocess.run(["git", "-C", repo_dir, "pull"], check=True)

    # Change to the repository directory
    os.chdir(repo_dir)

    # Automate `make everything`
    print("Running `make everything`...")
    try:
        child = pexpect.spawn("make everything", encoding="utf-8")
        child.logfile = sys.stdout

        while True:
            index = child.expect(
                [
                    pexpect.EOF,
                    pexpect.TIMEOUT,
                    "password for",  # Handle sudo password prompts
                    pexpect.re.compile(r"(error|failed)", pexpect.re.IGNORECASE),
                ]
            )
            if index == 0:  # Process completed
                break
            elif index == 1:  # Timeout
                print("Timeout occurred. Please check your setup.")
                sys.exit(1)
            elif index == 2:  # Sudo password prompt
                child.sendline(input("Enter sudo password: "))
            elif index == 3:  # Error detected
                print("Error encountered during installation.")
                sys.exit(1)
    except Exception as e:
        print(f"Installation failed: {e}")
        sys.exit(1)

    # Append to environment setup files
    update_env_files(fl_prefix, pl_prefix)

    print("FLUX installation completed successfully!")


def update_env_files(fl_prefix, pl_prefix):
    """Update shell configuration files for environment variables."""
    zprofile = os.path.expanduser("~/.zprofile")

    # Construct the dynamic Perl include path
    perl_lib_path = os.path.join(pl_prefix, "lib", "perl5")

    with open(zprofile, "a") as file:
        file.write(f"\n# FLUX environment setup\n")
        file.write(f"export FL_PREFIX={fl_prefix}\n")
        file.write(f"export PL_PREFIX={pl_prefix}\n")
        file.write(f"eval `perl -I {perl_lib_path} -Mlocal::lib={perl_lib_path}`\n")

    print(f"Environment variables added to {zprofile}.")
    subprocess.run(["source", zprofile], shell=True, check=False)


if __name__ == "__main__":
    run_flux_install()
