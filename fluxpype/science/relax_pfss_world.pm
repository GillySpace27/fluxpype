
=head1 NAME

relax_pfss_world - Relaxes the initial world state generated from PFSS (Potential Field Source Surface) models.

=cut

package relax_pfss_world;
use strict;
use warnings;
use Exporter qw(import);
our @EXPORT_OK = qw(relax_pfss_world);
use File::Basename qw(dirname);
use File::Path     qw(mkpath);
use pipe_helper    qw(find_highest_numbered_file check_second_file_presence);
use Flux::World    qw(read_world);
use Time::HiRes    qw(clock_gettime);
use simple_relaxer qw(simple_relaxer);
use pipe_helper    qw(shorten_path);

=head1 SYNOPSIS

    use relax_pfss_world;
    relax_pfss_world($world_out_dir, $full_world_path, $do_relax, $do_steps, $relax_threshold,
    $max_cycles, $timefile, $n_fluxons_wanted, $N_actual, $datdir, $batch_name, $CR);

=head1 DESCRIPTION

This Perl script relaxes the initial world state generated from PFSS (Potential Field Source Surface) models.
It reads the world state from a file, applies relaxation algorithms, and then saves the relaxed world state back to a file.

=head1 FUNCTIONS

=head2 relax_pfss_world

    relax_pfss_world($world_out_dir, $full_world_path, $do_relax, $do_steps, $relax_threshold,
    $max_cycles, $timefile, $n_fluxons_wanted, $N_actual, $datdir, $batch_name, $CR);

This function does the following:

=over

=item * Reads the world state from a file.

=item * Applies relaxation algorithms to the world state.

=item * Saves the relaxed world state back to a file.

=back

=head3 PARAMETERS

=over

=item * C<$world_out_dir>: Directory where the output world file will be saved.

=item * C<$full_world_path>: Full path to the world file.

=item * C<$do_relax>: Flag to indicate whether to perform relaxation.

=item * C<$do_steps>: Number of steps for the relaxation algorithm.

=item * C<$relax_threshold>: Threshold for the relaxation algorithm.

=item * C<$max_cycles>: Maximum number of cycles for the relaxation algorithm.

=item * C<$timefile>: File to log the time taken for relaxation.

=item * C<$n_fluxons_wanted>: Number of fluxons wanted.

=item * C<$N_actual>: The actual number of fluxons.

=item * C<$datdir>: The data directory.

=item * C<$batch_name>: Name of the batch.

=item * C<$CR>: Carrington Rotation.

=back

=head3 OUTPUT

This function returns the original and relaxed world states, along with the number of steps taken for relaxation.

=head1 AUTHOR

Gilly <gilly@swri.org> (and others!)

=head1 SEE ALSO

L<pipe_helper>, L<Storable>, L<File::Path>, L<Time::HiRes>

=cut

use Term::ANSIColor;


sub relax_pfss_world {
    my (
        $world_out_dir, $full_world_path,  $do_relax,
        $do_steps,      $relax_threshold,  $max_cycles,
        $timefile,      $n_fluxons_wanted, $N_actual,
        $datdir,        $batch_name,       $CR
    ) = @_;
    print color("bright_cyan");
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    print "(pdl) Relax the Initial World State from PFSS\n";
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    print "\n";
    print color("reset");

    my ( $found_file_path, $stepnum ) =
      find_highest_numbered_file($world_out_dir);
    my ( $second_file_present, $file_path_relaxed ) =
      check_second_file_presence($full_world_path);
    if ($second_file_present) {
        print "\tFound a relaxed file: $file_path_relaxed";
    }
    else {
        print
"\tNo relaxed file found, so we will relax the world (fairly slow).\n\n";
    }
    my $directory         = dirname($full_world_path);
    my $file_name_relaxed = $directory . "/" . $file_path_relaxed;
    $stepnum = 0;

    # exit();
    my $do_the_relax = ( $do_relax || not $second_file_present );

    # print "\n" . $found_file_path . "\n";
    # print $directory;

    if ($do_the_relax) {

        # print "Press Enter to Confirm...";
        print "\tLoading World...\n\n";
        my $this_world_orig = read_world($full_world_path);

        use Storable qw(dclone);
        our $this_world_relaxed = dclone($this_world_orig);

        # our $this_world_relaxed = $this_world_orig;
        $this_world_relaxed->forces( 'f_pressure_equi2b', 'f_curvature', 'f_vertex4', 'b_eqa' );    # OLD

        # $this_world_relaxed->forces('b_eqa', 'f_p_eqa_perp', 'f_curv_hm', 'f_vert4'); # NEW
        # $this_world_relaxed->{scale_b_power} = -1.0; ## Default 0

        $this_world_relaxed->{concurrency} = 12;

        # print();
        my $cycle       = 0;
        my $stiff       = 100;
        my $round_stiff = 100;
        my $broken      = 0;
        my $stepnum     = 0;
        my $flen        = 0;

        # # # Begin the Computation!! # # #
        my $starttime = clock_gettime();

        while ( $stiff > $relax_threshold
            and $cycle < $max_cycles
            and $broken < 3 )
        {
            $cycle = $cycle + 1;
            print
"\n\tRelaxing PFSS model for $do_steps steps to $relax_threshold stiffness...\n\n";

            simple_relaxer( $this_world_relaxed, 0, $do_steps,
                { disp_n => 0, movie_n => 50, print_n => 200 } );
            print("\n\n\n");
            $relax_threshold = $relax_threshold * 1.1;

            # #Trying this new thing
            # $this_world_relaxed->fix_proximity(0.5);
            # $this_world_relaxed->fix_curvature(0.25);

            my $h = $this_world_relaxed->fw_stats;
            $stiff = $h->{f_av} / $h->{fs_av} * 100;

            $round_stiff = sprintf( "%.2f", $stiff );
            $stepnum     = $cycle * $do_steps;
            my $round_time =
              sprintf( "%.2f", ( clock_gettime() - $starttime ) );

            print(
"\tCumulative relaxation time: $round_time seconds, $stepnum steps\n"
            );

            if ( $stiff > 99 ) {
                $broken = $broken + 1;
            }
            else {
                $broken = 0;
            }

            # print "\n \nSTIFFNESS IS $stiff, cycle = $cycle\n";

            if ( $broken >= 2 ) {
                open my $fhh, ">>", $timefile or die "Cannot open file: $!";

# printf $fhh, "Success: 0, fluxons: %d, steps: %d, stiff: %f, n_out: %d", $N_actual, $stepnum, $stiff, $flen;
                print $fhh
"n_want: $n_fluxons_wanted, n_actual: $N_actual, n_out: $flen, Success: 0, steps: $stepnum, stiff: $round_stiff";
                close $fhh;
                die "\n \n TOO MANY FLUXONS! Reduce N_flux.\n \n \n";

            }
        }
        print "\tRelaxation Complete in $stepnum Steps \n";

# print("Cumulative relaxation time: ".(clock_gettime()-$starttime)." seconds\n");

        open my $fhh, ">>", $timefile or die "Cannot open file: $!";

# printf $fhh, "Success: 1, fluxons: %d, steps: %d, stiff %.2f, n_out: %d", $N_actual, $stepnum, $stiff, $flen;
        print $fhh
"n_want: $n_fluxons_wanted, n_actual: $N_actual, n_out: $flen, Success: 1, steps: $stepnum, stiff: $round_stiff, ";
        close $fhh;

        # Save the World
        # $world_out_dir = $datdir."/batches/$batch_name/cr".$CR.'/rlx/';

        ## print $out_world;
        my $out_world = $full_world_path;
        substr( $out_world, -5 ) = "_relaxed_s$stepnum.flux";
        my $short_world_path = shorten_path($out_world);
        print "\n\n\tSaving the World to $short_world_path\n";
        my $world_out_dir = File::Basename::dirname($out_world);
        if ( !-d $world_out_dir ) {
            mkpath($world_out_dir)
              or die "Failed to create directory: $world_out_dir $!\n";
        }
        $this_world_relaxed->write_world($out_world);

        return $this_world_orig, $this_world_relaxed, $stepnum;

    }
    else {
        print
          "\n\tSkipped relaxation! Already have relaxed file. Loading...\n \n";

        # # print $found_file_path;

        print "\t\tLoading relaxed world...\n";
        my $this_world_relaxed = read_world($file_name_relaxed);

        if ( $file_name_relaxed =~ /_s(\d+)\.flux$/ ) {
            $stepnum = $1;

            # print "A STEPS ARE $stepnum";

        }
        else {
            print "Couldn't load relaxed world file!";
            die;
        }

        # print "\n";
        print "\n\t\tLoading pfss world...\n";
        my $this_world_orig = read_world($full_world_path);

        print "\n\n\tWorlds Successfully loaded!";
        print "\n \n";
        print "\t\t\t```````````````````````````````\n \n\n";

        # print($found_file_path)
        # $this_world_orig = read_world($found_file_path);
        return $this_world_orig, $this_world_relaxed, $stepnum;
    }

}
1;
