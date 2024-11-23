#!/bin/zsh

# Ensure the script stops on the first error
set -e

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
        \curl -L https://install.perlbrew.pl | bash
        perlbrew init
        source ~/perl5/perlbrew/etc/bashrc
        perlbrew install perl-5.32.0
        perlbrew switch perl-5.32.0
        curl -L https://cpanmin.us | perl - App::cpanminus
    else
        LOG "perlbrew already installed!"
    fi
}

INSTALL_HOMEBREW() {
    if ! command_exists brew; then
        LOG "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$($SUDO_PREFIX /opt/homebrew/bin/brew shellenv)"
        $SUDO_PREFIX brew install gnuplot fftw qt
    else
        LOG "Homebrew already installed!"
    fi
}

SETUP_VIRTUALENV() {
    if ! [ -d "fluxenv_pip" ]; then
        LOG "Setting up Python virtual environment..."
        if ! command_exists virtualenv; then
            LOG "Installing virtualenv..."
            pip install virtualenv
        fi
        LOG "Creating virtual environment..."
        virtualenv fluxenv_pip
        LOG "Activating virtual environment..."
        source fluxenv_pip/bin/activate
        LOG "Installing Python dependencies..."
        pip install -r requirements-pip.txt
        pip install -e .
    else
        LOG "Virtual environment already exists!"
    fi
}

INSTALL_PERL_MODULES() {
    LOG "Installing Perl modules..."
    local PL_PREFIX="./perl_lib"
    mkdir -p $PL_PREFIX/lib/perl5

    export PERL_MM_OPT="INSTALL_BASE=$PL_PREFIX/lib/perl5"
    cpanm -L $PL_PREFIX PDL
    cpanm -L $PL_PREFIX local::lib File::ShareDir File::ShareDir::Install \
      PDL::Graphics::Gnuplot Math::RungeKutta Moo::Role Chart::Gnuplot

    eval "$(perl -I$PL_PREFIX/lib/perl5 -Mlocal::lib=$PL_PREFIX/lib/perl5)"

    LOG "Perl modules installed in $PL_PREFIX"
}

main() {
    LOG "Start install-fluxpype-macos.sh"

    CHECK_ROOT
    [[ ! $(pwd) =~ fluxpype ]] && cd fluxpype
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
        python fluxpype/fluxpype/runners/config_runner.py
    fi
}

main
