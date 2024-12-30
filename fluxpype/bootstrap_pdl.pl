#!/usr/bin/perl
use strict;
use warnings;
use FindBin qw($Bin);

# Add necessary directories to @INC
BEGIN {
    my @subdirs = qw(science plotting);
    my $base_path = ".";

    unshift @INC, $base_path;               # Add the base directory
    foreach my $dir (@subdirs) {
        unshift @INC, "$base_path/$dir";    # Add each subdirectory
    }

    # Optional: Print the @INC paths for debugging
    foreach my $path (@INC) {
        print "Added to \@INC: $path\n";
    }
}

1; # End of script
