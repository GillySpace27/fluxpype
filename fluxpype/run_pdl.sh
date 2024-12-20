#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setting the PERL5LIB environment variable
export PERL5LIB="$PERL5LIB:$SCRIPT_DIR:$SCRIPT_DIR/science:$SCRIPT_DIR/plotting:$SCRIPT_DIR/fluxpype/science:$SCRIPT_DIR/fluxpype/plotting"
echo "Library:"
awk -v RS=':' '{print}' <<< "$PERL5LIB"

# Running the specified PDL file with any provided arguments
perl "$@"
