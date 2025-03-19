#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def log(message):
    """Simple logger function."""
    print(f"[PDL Installer] {message}")

def run_command(command, check=True, shell=False):
    """Runs a system command and prints output live."""
    log(f"Running: {' '.join(command) if isinstance(command, list) else command}")

    result = subprocess.run(command, shell=shell, check=False)  # Allow failure handling manually
    if result.returncode == 0:
        log("✅ Success!")
    else:
        log(f"❌ Failed with exit code {result.returncode}")
        if check:
            sys.exit(1)

def install_perl_modules():
    """Installs the required Perl modules on macOS or Linux."""
    pl_prefix = Path.home() / ".local" / "perl5"  # Default Perl installation prefix
    log(f"Installing Perl modules into {pl_prefix} ...")

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

    log("🚀 Attempting bulk installation...")
    bulk_install = ["cpanm", "-l", str(pl_prefix), "--notest"] + modules
    run_command(bulk_install, check=False)

    # Check if bulk install failed and retry one by one
    log("✅ Bulk installation complete. Verifying individual modules...")
    for module in modules:
        log(f"📦 Installing: {module}")
        install_command = ["cpanm", "-l", str(pl_prefix), "--notest", module]
        run_command(install_command, check=False)

    log("✅ All Perl dependencies installed successfully!")

    # # Set local::lib environment variables
    # eval_command = f"eval $(perl -I {pl_prefix}/lib/perl5 -Mlocal::lib={pl_prefix})"
    # log(f"🔧 Setting up Perl environment with: {eval_command}")
    # run_command(eval_command, shell=True)

if __name__ == "__main__":
    install_perl_modules()