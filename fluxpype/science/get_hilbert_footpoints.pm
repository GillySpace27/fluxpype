=head1 NAME

get_hilbert_footpoints - Traces Magnetogram to get Footpoints using a Hilbert Curve

=cut

=head1 SYNOPSIS

    use pipe_helper;
    my ($floc, $N_actual) = get_hilbert_footpoints($flocdir, $flocpath, $magfile, $n_fluxons_wanted, $process_magnetogram);

=head1 DESCRIPTION

This subroutine is designed to trace a magnetogram and place footpoints using a Hilbert curve.
It accepts several parameters for customization and performs tasks such as data loading,
footpoint placement, and data writing.

=head1 FUNCTIONS

=head2 get_hilbert_footpoints

       get_hilbert_footpoints($flocdir, $flocpath, $magfile, $n_fluxons_wanted, $process_magnetogram);

This function does the following:

=over

=item * Checks for the existence of the footpoint file and directory.

=item * Reads the magnetogram and runs the Hilbert tracer if necessary.

=item * Places footpoints based on Hilbert curves.

=item * Writes the footpoints to disk if they were generated anew.

=item * Reads the footpoints from the existing file if applicable.

=back

=head3 PARAMETERS

=over

=item * C<$flocdir>: Directory where the footpoint file is located or will be saved.

=item * C<$flocpath>: File name for the footpoints.

=item * C<$magfile>: Path to magnetogram file to be traced.

=item * C<$n_fluxons_wanted>: Number of footpoints desired.

=item * C<$process_magnetogram>: Top level flag to short-circuit processing the magnetogram if it already exists.

=back

=head3 OUTPUT

Returns an array C<$floc> containing the footpoint locations and an integer C<$N_actual> representing the actual number of footpoints placed.

=head3 EXCEPTIONS

Dies if it fails to create the directory for footpoints.

=head1 AUTHOR

Gilly <gilly@swri.org> (and others!)

=head1 SEE ALSO

L<pipe_helper>, L<fluxon_placement_hilbert>, L<rfits>, L<wcols>, L<rcols>

=cut

package get_hilbert_footpoints;
use strict;
use warnings;
use Exporter qw(import);
our @EXPORT_OK = qw(get_hilbert_footpoints);
use lib "fluxpype/fluxpype/science";
use lib "fluxpype/science";
use lib ".";
use File::Path qw(mkpath);
use File::Spec::Functions qw(catfile catdir);
use fluxon_placement_hilbert qw(fluxon_placement_hilbert);
use PDL;
use PDL::NiceSlice;
use PDL::IO::FITS;   # For rfits
use PDL::IO::Misc;   # For wcols and rcols
use pipe_helper qw(shorten_path);  # Ensure that you actually need this

sub get_hilbert_footpoints {
    my (%configs) = @_;

    # Retrieve required configurations
    my $process_magnetogram = $configs{'force_process_magnetogram'};
    my $magfile             = $configs{'magfile'};
    my $magpath             = $configs{'magpath'};
    my $adapt               = $configs{'adapt'};
    my $n_fluxons_wanted    = $configs{'n_fluxons_wanted'};
    my $datdir              = $configs{'datdir'};       # Data directory
    my $batch_name          = $configs{'batch_name'};   # Batch name
    my $CR                  = $configs{'CR'};           # Carrington Rotation

    # Compute 'flocdir' and 'flocpath' if not provided
    my $flocdir = $configs{'flocdir'} // catdir($datdir, "batches", $batch_name, "data", "cr${CR}", "footpoints");
    my $flocfile = $configs{'flocfile'} // "footpoints_${n_fluxons_wanted}.dat";
    my $flocpath = catfile($flocdir, $flocfile);

    # print "dir:  $flocdir\n";
    # print "file: $flocfile\n";
    # print "path: $flocpath\n";


    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    print "(pdl) Tracing Magnetogram to get $n_fluxons_wanted Footpoints\n";
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

    my $no_floc  = 0;
    my $floc     = 0;
    my $N_actual = 0;

    if ( !-e $flocpath ) {
        $no_floc = 1;
        if ( !-d $flocdir ) {
            print "Footpoint directory not found. Creating: $flocdir\n";
            mkpath($flocdir) or die "Failed to create directory: $flocdir !\n";
        }
    }

    print "\n\tSearching for file...";
    if ( $process_magnetogram || $no_floc ) {
        # Magnetogram not found or reprocessing forced
        print "Not found.\n\n";
        print "\tRunning Hilbert Tracer...\n";
        print "\t\tFile: " . $magpath . "\n\n";

    # Read the magnetogram
    my $smag;
    eval {
        $smag = rfits($magpath);
        1; # Success flag
    } or do {
        die "Failed to read magnetogram: $magpath!\n";
    };

    # Check if $smag is defined and valid
    unless (defined $smag) {
        die "Magnetogram read operation returned undefined result: $magpath!\n";
    }

        print "\n\t\tPlacing $n_fluxons_wanted footpoints using Hilbert Curves...\n";

        # Define tolerance and initial step size
        my $tolerance = $n_fluxons_wanted * 0.05;    # 5% tolerance
        my $step      = $n_fluxons_wanted * 0.1;     # Initial step size as 10% of $n_fluxons_wanted

        # Initialize variables
        my $width       = 100;
        my $most        = $n_fluxons_wanted + $width;
        my $least       = $n_fluxons_wanted - $width;
        my $iterate     = 1;
        my $try_fluxons = int( $n_fluxons_wanted * 1.5 );
        my $count       = 0;
        my $maxreps     = 30;

        while ( $iterate && $count < $maxreps ) {
            $try_fluxons = int($try_fluxons);
            $floc        = fluxon_placement_hilbert( $smag, $try_fluxons );
            $N_actual    = $floc->nelem / 3;
            $count++;

            print("\t\t    Placing footpoints:: iter: $count/$maxreps. ".
                  "Wanted: $n_fluxons_wanted, Tried: $try_fluxons, Placed: $N_actual...");

            my $distance = $N_actual - $n_fluxons_wanted;
            if ( abs($distance) > $tolerance ) {
                # Adjust step size based on the distance from the desired number of fluxons
                my $factor = 1 + abs($distance) / ( $n_fluxons_wanted * 2 );
                $step *= $factor;
                $step *= 0.9;
                $try_fluxons += $distance > 0 ? -$step : $step;

                $iterate = 1;
                print "Retrying!\n";
            } else {
                print "Success!\n";
                $iterate = 0;
            }
        }

        print "\n\t\tWriting Result...";
        wcols $floc->transpose, $flocpath;
        print "Success! File saved to:";
        print "\n\t\t\t " . shorten_path($flocpath);
        print "\n";
    } else {
        $floc     = rcols $flocpath;
        $floc     = $floc->transpose;
        $N_actual = int( ( $floc->nelem ) / 3 );
        my $flocpath_short = shorten_path($flocpath);

        print "\n\t\tFound a $N_actual footpoint file on Disk:";
        print "\n\t\t\t$flocpath_short\n";
    }

    print "\n\t\t\t```````````````````````````````\n\n\n";
    return $floc, $N_actual;
}

1;  # End of module
