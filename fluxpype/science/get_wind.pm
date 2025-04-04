
=head1 NAME

get_wind - Calculate Solar Wind Plasma Parameters for a Given Carrington Rotation (CR)

=head1 SYNOPSIS

    use pipe_helper;
    my ($out_b, $out_fr, $out_wind) = get_wind($this_world, $datdir, $batch_name, $CR, $N_actual, $recompute, $n_want, $pythondir);

=cut

package get_wind;
# use strict;
use warnings;
use Exporter qw(import);
our @EXPORT_OK = qw(get_wind);
# use local::lib;
# use lib ".";
# use lib "..";
# # use lib "fluxpype/fluxpype/helpers/";
# # use lib "fluxpype/fluxpype/science/";

use FindBin;
use lib $FindBin::Bin;       # Adds the script's directory
use lib "$FindBin::Bin/..";  # Adds the parent directory
use lib "$FindBin::Bin/../plotting";  # Adds the neighbor's directory
use lib "$FindBin::Bin/../..";  # Adds the grandparent directory


use Term::ANSIColor;

my @required_modules = qw(
    Flux
    pipe_helper
    get_hilbert_footpoints
    make_world_from_pfss
    plot_world
    relax_pfss_world
    get_wind
);

foreach my $module (@required_modules) {
    eval {
        require "$module.pm";  # Dynamically find and load the module
        $module->import();     # Explicitly call the import method
    };
    if ($@) {
        die "Required module $module not found or failed to load. Please ensure it is installed and accessible.\nError: $@\n";
    }
}


use pipe_helper                     qw(shorten_path configurations find_highest_numbered_file_with_string);
use File::Path                      qw(mkpath);
use map_fluxon_b                    qw(map_fluxon_b);
use map_fluxon_b_all                qw(map_fluxon_b_all);
use map_fluxon_fr                   qw(map_fluxon_fr);
use map_fluxon_flow_parallel_master qw(map_fluxon_flow_parallel_master);
use Flux::World    qw(read_world);
use PDL::IO::FITS;
use Text::CSV;
use PDL;
# use PDL::Graphics::Simple;
use PDL::IO::Misc;


=head1 DESCRIPTION

This subroutine calculates various solar wind plasma parameters such as the radial magnetic field, radial expansion factor, and radial wind speed. It performs these calculations based on the provided Carrington Rotation (CR) and other parameters.

=head1 PARAMETERS

=over 4

=item * C<$this_world> - The relaxed world object containing fluxons.

=item * C<$datdir> - Data directory path.

=item * C<$batch_name> - Name of the batch.

=item * C<$CR> - Carrington Rotation number.

=item * C<$N_actual> - Actual number of footpoints.

=item * C<$recompute> - Flag to indicate whether to recompute the parameters or not.

=item * C<$n_want> - Number of footpoints wanted.

=item * C<$pythondir> - Python directory path (currently not used).

=back

=head1 WORKFLOW

1. Checks for the elong_xistence of output directories and files.
2. If necessary, creates output directories.
3. If the output files do not elong_xist or if recompute is true, performs the following calculations:
    - Updates neighbors in the relaxed world.
    - Calculates the radial magnetic field.
    - Calculates the radial expansion factor.
    - Calculates the radial wind speed.

=head1 OUTPUT

Returns the paths to the output files for radial magnetic field (C<$out_b>), radial expansion factor (C<$out_fr>), and radial wind speed (C<$out_wind>).

=head1 EXCEPTIONS

Dies if it fails to create the output directory.

=head1 AUTHOR

Gilly <gilly@swri.org> (and others!)

=head1 SEE ALSO

L<pipe_helper>, L<map_fluxon_b>, L<map_fluxon_fr>, L<map_fluxon_flow_parallel_master>

=cut

sub file_has_content {
    my ($filename) = @_;
    if (-f $filename) {
        open my $fh, '<', $filename or return 0; # Return false if file can't be opened
        my $line_count = 0;
        while (<$fh>) {
            $line_count++;
            last if $line_count >= 3; # Stop reading if we have 3 or more lines
        }
        close $fh;
        return $line_count >= 3;
    }
    return 0; # Return false if file doesn't elong_xist
}

sub get_wind {

    my ( $this_world, $CR, $n_want, $recompute, $flow_method_in, $do_stiff) = @_;


    # Read configurations from disk
    # print "Reading configurations...\n";
    my %configs = configurations();

    $recompute = $recompute // 0;

    $CR = $CR // ( $configs{rotations}->at(0) );
    $configs{CR} = $CR;

    my $n_fluxons_wanted = $n_want // ( $configs{fluxon_count}->at(0) );
    $configs{n_fluxons_wanted} = $n_fluxons_wanted;


    my $pythondir = $configs{pythondir};
    my $datdir = $configs{datdir};
    my $batch_name = $configs{batch_name};
    my $flow_method = $flow_method_in;
    my $flow_method_str = $flow_method;
    $flow_method_str = "_$flow_method_str";

    if ($do_stiff) {
        $stiffness = "\_stiff";
    }
    else {
        $stiffness = "";
    }



    my $do_wind_calc = 1;

    my $wind_out_dir   = $datdir . "/batches/$batch_name/data/cr" . $CR . '/wind';
    my $prefix         = "$wind_out_dir/cr$CR\_f$n_fluxons_wanted";
    my $out_b          = "$prefix\_radial_bmag.dat";
    my $out_b_all      = "$prefix$stiffness\_radial_bmag_all.dat";
    my $out_fr         = "$prefix\_radial_fr.dat";
    my $out_wind       = "$prefix\_radial_wind$flow_method_str.dat";
    my $ch_map_url     = "https://sun.njit.edu/coronal_holes/data/chs_synmap_cr$CR.fits";
    my $ch_map_path    = $datdir . "/CHmaps/chs_synmap_cr$CR.fits";
    my $short_out_wind = shorten_path($out_wind);
    my $skipstring =
      "\n\tWind Calculation Skipped! \n\t\tFound $short_out_wind\n\n";



    # Make the directory if necessary
    if ( !-d $wind_out_dir ) {
        $do_wind_calc = 1;
        mkpath($wind_out_dir)
          or die "Failed to create directory: $wind_out_dir $!\n";
    }

    # Check if the files exist
    if ( !-f $out_b
        or !-f $out_fr
        or !-f $out_wind
        or !-f $out_b_all
        or $recompute ) {
        $do_wind_calc = 1;
    }

    # Check if the files have at least 3 lines
    if (!file_has_content($out_b)
        or !file_has_content($out_fr)
        or !file_has_content($out_wind)
        or !file_has_content($out_b_all)
        or $recompute) {
        $do_wind_calc = 1;
    }
    print color("bright_cyan");
    print "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    print "(pdl) Calculating Coronal Plasma Parameters for CR$CR\n";
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    print color("reset");

    # Perform the calculation if necessary
    if ($do_wind_calc ) {

        #Initialize the world
        print "\n\tUpdating neighbors...";
        $this_world->update_force(0);
        my @fluxons = $this_world->fluxons;

        print "Done!\n";
        print "\t\tNumber of fluxons: " . scalar @fluxons . "\n";

        if (scalar @fluxons == 0) {
            die "\t\t\tNo fluxons found. Cannot proceed without data.";
        }

        # DB::single;
        # Calculate the radial magnetic field
        print "\n\tRadial Magnetic Field (B) Calculation...";
        map_fluxon_b( $out_b, \@fluxons );
        map_fluxon_b_all( $out_b_all, \@fluxons );


        # Calculate the radial expansion factor
        print "\n\tRadial Expansion Factor (Fr) Calculation...";
        map_fluxon_fr( $out_fr, \@fluxons );
        print "Done! Saved to \n\t$out_fr\n";

        # Calculate the radial wind speed

        print "\n\n\tRadial Wind Speed Calculation...\n";
        my $do_wind_map = 0 || $recompute || $configs{recompute_wind};
        # $do_wind_map=1; #OVERRIDE WIND MAP

        if ( !-e $out_wind || !file_has_content($out_wind)) { $do_wind_map = 1; }

        # Pass the image into the main function
        if ($do_wind_map) { map_fluxon_flow_parallel_master( $out_wind, \@fluxons, $flow_method, $CR, $n_want); }
        else { print $skipstring;}

        # #run the python script to plot the angles
        # system("python3 fluxon-mhd/fluxpype/fluxpype/plotting/plot_angles.py");

    }
    else {
        print $skipstring;
    }

    print "\t\t\t```````````````````````````````\n\n\n\n";

    return $out_b, $out_fr, $out_wind, $out_b_all;
}

if ($0 eq __FILE__) {
    # This code block will run only if the script is executed directly
    # and not when it's included as a module in another script.

    print("Running get_wind.pm directly\n");
    my %configs = configurations();

    my $n_want = $configs{fluxon_count}->at(0);
    my $CR = $configs{rotations}->at(0);
    my $world_dir = $configs{data_dir} . "/batches/" . $configs{batch_name} . "/data/cr" . $CR . "/world";

    my $search_string = $n_want . "_hmi_relaxed_s";  # Modify this to your desired search string
    my ($highest_file, $highest_number) = find_highest_numbered_file_with_string($world_dir, $search_string);

    my $this_world = read_world($highest_file);
    get_wind($this_world)

}




1;
