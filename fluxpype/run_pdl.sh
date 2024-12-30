#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#!/bin/bash


# Setting the PERL5LIB environment variable
export PERL5LIB="$PERL5LIB:$SCRIPT_DIR:$SCRIPT_DIR/science:$SCRIPT_DIR/plotting"

# Print the word "Library:" in blue using printf
printf "\n\033[34mPERL5LIB:\033[0m\n"

# Loop through each directory path in PERL5LIB and print in blue
awk -v RS=':' '
{
    printf "\033[34m%s\033[0m\n", $0  # Print each directory in blue
}' <<< "$PERL5LIB"


# Running the specified PDL file with any provided arguments
perl "$@"
