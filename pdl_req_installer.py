#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def log(message):
    """Simple logger function."""
    print(f"[PDL Installer] {message}")

def run_command(command, check=True, shell=False):
    """Runs a system command and handles errors."""
    try:
        result = subprocess.run(command, shell=shell, check=check, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        log(f"Command failed: {command}\n{e.stderr}")
        if check:
            sys.exit(1)

def install_perl_modules():
    """Installs the required Perl modules on macOS or Linux."""
    # pl_prefix = Path.home() / ".local" / "perl5"  # Default Perl installation prefix
    log(f"Installing Perl modules...")

    modules = [
        # Core
        "local::lib", "Devel::CheckLib", "List::MoreUtils", "Capture::Tiny", "PDL",
        # Config
        "Config::IniFiles",
        # File Handling
        "File::HomeDir", "File::ShareDir", "File::ShareDir::Install",
        # Testing
        "Test::Builder",
        # Parallel
        "Parallel::ForkManager",
        # Math
        "Math::GSL::Alien", "Math::GSL", "Math::Interpolate", "Math::Interpolator", "Math::RungeKutta",
        # PDL
        "PDL::GSL::INTEG", "PDL::Graphics::Gnuplot", "PDL::Graphics::Simple",
        # Inline
        "Inline", "Inline::C", "Inline::Python",
        # Web
        "Net::SSLeay",
        # Charts
        "Chart::Gnuplot", "Alien::Build::Plugin::Gather::Dino",
        # CSV
        "Text::CSV",
        # Moose-like
        "Moo::Role",
    ]

    # Try bulk installation first
    log("Attempting bulk installation...")
    try:
        run_command(["cpanm", "--notest"] + modules, check=False)
    except Exception as e:
        log(f"Bulk install failed: {e}")
        log("Reverting to individual module installation.")

        for module in modules:
            try:
                run_command(["cpanm", "--notest", module], check=True)
            except Exception:
                run_command(["cpanm", "--force", module], check=True)

    # # Set local::lib environment variables
    # eval_command = f"eval $(perl -I {pl_prefix}/lib/perl5 -Mlocal::lib={pl_prefix})"
    # log(f"Setting up Perl environment with: {eval_command}")
    # run_command(eval_command, shell=True)

if __name__ == "__main__":
    install_perl_modules()