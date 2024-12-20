#!/bin/zsh

# Ensure the script stops on the first error
set -e

PERL_VERSION="perl-5.32.0"
PL_PREFIX="./perl_lib"

LOG() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") [INFO] $1"
}

CHECK_ROOT() {
    if [[ "$EUID" -ne 0 ]]; then
        LOG "Root access not detected. Running non-root installation."
        SUDO_PREFIX=""
    else
        LOG "Root access detected."
        SUDO_PREFIX="sudo"
    fi
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1;
}

INSTALL_PERLBREW() {
    if ! command_exists perlbrew; then
        LOG "Installing perlbrew..."
        curl -L https://install.perlbrew.pl | bash || { LOG "Failed to download and install perlbrew"; exit 1; }
        perlbrew init
        source ~/perl5/perlbrew/etc/bashrc
        perlbrew install $PERL_VERSION
        perlbrew switch $PERL_VERSION
        curl -L https://cpanmin.us | perl - App::cpanminus || { LOG "Failed to install cpanminus"; exit 1; }
    else
        LOG "perlbrew already installed!"
    fi
}

INSTALL_HOMEBREW() {
    if ! command_exists brew; then
        LOG "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || { LOG "Failed to install Homebrew"; exit 1; }
        eval "$($SUDO_PREFIX /opt/homebrew/bin/brew shellenv)"
        $SUDO_PREFIX brew install gnuplot fftw qt || { LOG "Failed to install Homebrew packages"; exit 1; }
    else
        LOG "Homebrew already installed!"
    fi
}

SETUP_VIRTUALENV() {
    if ! [ -d "fluxenv_pip" ]; then
        LOG "Setting up Python virtual environment..."
        if ! command_exists virtualenv; then
            LOG "Installing virtualenv..."
            pip install virtualenv || { LOG "Failed to install virtualenv"; exit 1; }
        fi
        LOG "Creating virtual environment..."
        virtualenv fluxenv_pip || { LOG "Failed to create virtual environment"; exit 1; }
        LOG "Activating virtual environment..."
        source fluxenv_pip/bin/activate
        LOG "Installing Python dependencies..."
        pip install -r requirements-pip.txt || { LOG "Failed to install Python dependencies"; exit 1; }
        pip install -e . || { LOG "Failed to install local Python package"; exit 1; }
    else
        LOG "Virtual environment already exists!"
    fi
}

INSTALL_PERL_MODULES() {
    LOG "Installing Perl modules..."
    mkdir -p $PL_PREFIX/lib/perl5

    export PERL_MM_OPT="INSTALL_BASE=$PL_PREFIX/lib/perl5"

    local modules=(
        Test::Builder local::lib File::ShareDir File::ShareDir::Install
        PDL::Graphics::Gnuplot Math::RungeKutta Moo::Role Chart::Gnuplot
        Text::CSV Math::Interpolate Math::GSL Config::IniFiles
        File::HomeDir Inline::C Parallel::ForkManager Inline Inline::Python
        Capture::Tiny Devel::CheckLib
    )
    cpanm -L $PL_PREFIX PDL || { LOG "Failed to install PDL"; exit 1; }

    for module in "${modules[@]}"; do
        cpanm -L $PL_PREFIX "$module" || { LOG "Failed to install $module"; exit 1; }
    done

    eval "$(perl -I$PL_PREFIX/lib/perl5 -Mlocal::lib=$PL_PREFIX/lib/perl5)"

    LOG "Perl modules installed in $PL_PREFIX"
}

main() {
    LOG "Start install-fluxpype-macos.sh"

    # Ensure the script runs on macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        LOG "This script is intended to be run on macOS."
        exit 1
    fi

    CHECK_ROOT
    # [[ ! $(pwd) =~ fluxpype ]] && cd fluxpype || { LOG "Failed to change directory to fluxpype"; exit 1; }
    LOG "Installing fluxpype..."

    INSTALL_PERLBREW
    INSTALL_HOMEBREW
    SETUP_VIRTUALENV
    INSTALL_PERL_MODULES

    LOG "fluxpype installation complete!"
    LOG "To activate the virtual environment, run 'source fluxenv_pip/bin/activate' before running 'python fluxpype/fluxpype/runners/config_runner.py'"

    echo -n "Do you want to test the config_runner? ([yes]/no): "
    read response
    if [[ "$response" == "yes" || "$response" == "y" || "$response" == "" ]]; then
        LOG "Running the file..."
        source fluxenv_pip/bin/activate
        python fluxpype/fluxpype/runners/config_runner.py || { LOG "Failed to run config_runner"; exit 1; }
    fi
}

main
