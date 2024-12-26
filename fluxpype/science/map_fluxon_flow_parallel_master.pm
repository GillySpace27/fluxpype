package map_fluxon_flow_parallel_master;

use strict;
use warnings;
use Exporter qw(import);
our @EXPORT_OK = qw(map_fluxon_flow_parallel_master);

use File::Basename qw(fileparse);
use File::Spec;
use File::Path qw(make_path rmtree);
use Time::HiRes qw(clock_gettime);
use Parallel::ForkManager;
use Data::Dumper;
use PDL;
use PDL::NiceSlice;
use PDL::Options;
use PDL::IO::Storable;
use PDL::Graphics::Gnuplot;
use Chart::Gnuplot;

use lib "fluxpype/science";
use lib "fluxpype/helpers";
use lib ".";

use local::lib;

use gen_fluxon_tflow       qw(gen_fluxon_tflow);
use gen_fluxon_schonflow   qw(gen_fluxon_schonflow);
use gen_fluxon_wsaflow     qw(gen_fluxon_wsaflow do_image_plot);
use gen_fluxon_cranmerflow qw(gen_fluxon_cranmerflow);
use gen_fluxon_ghostsflow  qw(gen_fluxon_ghostsflow);
use gen_fluxon_tempestflow qw(gen_fluxon_tempestflow);
use make_tempest_file      qw(make_tempest_file);
use pipe_helper            qw(configurations);

=head1 NAME

Fluxon Flow Mapper - Parallelized Solar Wind Flow Mapping Along Fluxon Structures

=head1 DESCRIPTION

This Perl module provides functionality to map the solar wind flow along fluxon structures in a parallelized manner.
It uses L<Parallel::ForkManager> for concurrency and L<PDL> for numerical computations.

=head1 AUTHOR

Gilly E<lt>gilly@swri.orgE<gt> and others!

=head1 LICENSE

Free software; you can redistribute it and/or modify under the same terms as Perl.

=cut

#------------------------------------------------------------------------
# Global/Module-level variables (or config)
#------------------------------------------------------------------------

my %configs    = configurations();
my $concurrency = $configs{concurrency} // 4;
my $temp_dir   = "temp_dir";

# Control Flags
$PDL::verbose = 0;

#------------------------------------------------------------------------
# Utility subroutines
#------------------------------------------------------------------------

=head2 listy($pdl_obj)

Convert a PDL object into a plain Perl array.

=cut

sub listy {
    my $pdl_obj = shift;
    return $pdl_obj->list;  # Convert PDL object to Perl array
}

#------------------------------------------------------------------------
# save_full_velocity_profiles(\@results, $file_path)
#
# Safely writes out a CSV file with columns: fluxon_position,radius,velocity
#------------------------------------------------------------------------
sub save_full_velocity_profiles {
    my ($results, $file_path) = @_;

    # Use Data::Dumper to confirm structure
    # print "33: Results in save_full_velocity_profiles:\n", Dumper($results);

    # Ensure $results is an array ref
    die "ERROR: save_full_velocity_profiles expects an ARRAY ref, got: " . ref($results)
        unless ref($results) eq 'ARRAY';

    # Sort by fluxon_position (guarding against missing fluxon_position)
    my @sorted_results = sort {
        (defined $a->{fluxon_position} ? $a->{fluxon_position}->sclr : 0)
            <=>
        (defined $b->{fluxon_position} ? $b->{fluxon_position}->sclr : 0)
    } @$results;

    open(my $fh, '>', $file_path)
        or die "Could not open file '$file_path': $!";

    print $fh "fluxon_position,radius,velocity\n";

    foreach my $result (@sorted_results) {

        # Skip if fluxon_position is missing or not a PDL
        unless (   defined $result->{fluxon_position}
                && ref($result->{fluxon_position}) eq 'PDL' )
        {
            warn "fluxon_position is missing/undefined for result:\n", Dumper($result);
            next;
        }
        my $fluxon_id = $result->{fluxon_position}->sclr;

        # Check r and vr
        unless (defined $result->{r} && ref($result->{r}) eq 'PDL') {
            warn "'r' is missing or not a PDL for fluxon $fluxon_id:\n", Dumper($result);
            next;
        }
        unless (defined $result->{vr} && ref($result->{vr}) eq 'PDL') {
            warn "'vr' is missing or not a PDL for fluxon $fluxon_id:\n", Dumper($result);
            next;
        }

        my @r  = listy($result->{r});
        my @vr = listy($result->{vr});

        if (scalar(@r) != scalar(@vr)) {
            warn "Mismatched array lengths for fluxon ID $fluxon_id: ",
                 "r length=", scalar(@r), ", vr length=", scalar(@vr);
            next;
        }

        for my $i (0 .. $#r) {
            print $fh join(",", $fluxon_id, $r[$i], $vr[$i]), "\n";
        }
    }

    close $fh;
    return;
}

#------------------------------------------------------------------------
# map_fluxon_flow_parallel_master($output_file_name, \@fluxons, $flow_method, $CR, $n_want, [$max_processes])
#
# Primary subroutine to coordinate parallel flow calculations on fluxons.
#------------------------------------------------------------------------
sub map_fluxon_flow_parallel_master {
    my ($output_file_name, $fluxon_list, $flow_method, $CR, $n_want, $max_processes_arg) = @_;

    # Safely extract fluxons array
    die "ERROR: fluxon_list must be an array ref"
        unless ref($fluxon_list) eq 'ARRAY';
    my @fluxons = @$fluxon_list;

    my $max_processes = $max_processes_arg // $concurrency;  # concurrency from config if not given
    make_path($temp_dir) unless -d $temp_dir;

    # For timing
    my $before_time = clock_gettime();

    # Some local flags/variables
    my $do_plot_charts     = 0;
    my $choke              = 0;   # Possibly limit fluxons for debugging
    my $n_choke            = 12;
    my $highest_fluxon_done = 0;

    # Print some info
    print "\nINFO: Running with $max_processes Cores.\n";

    # Count open vs closed fluxons
    my $num_closed_fluxons   = 0;
    my $num_open_fluxons     = 0;
    my @open_fluxon_indices  = ();

    for my $fluxon_index (0 .. $#fluxons) {
        my $fluxon = $fluxons[$fluxon_index];
        # Check for open fieldline
        my $start_open = ($fluxon->{fc_start}->{label} // 0) == -1;
        my $end_open   = ($fluxon->{fc_end}->{label}   // 0) == -2;

        # If not purely open at one end, count as closed.
        if ($fluxon->{plasmoid} || ($start_open + $end_open != 1)) {
            $num_closed_fluxons++;
        }
        else {
            $num_open_fluxons++;
            push(@open_fluxon_indices, $fluxon_index);
        }
    }

    my $num_total_fluxons  = scalar(@fluxons);
    my $num_active_fluxons = 2 * ($num_total_fluxons - $num_open_fluxons) + $num_open_fluxons;

    print "INFO: $num_open_fluxons open fluxons / $num_total_fluxons total\n";
    print "     -> $num_active_fluxons footpoints total.\n\n";
    print "     Beginning Calculation...\n\n";

    # If requested, run a precomputation step
    if ($flow_method eq "tempest") {
        # Make the tempest file
        my $fmap_command = "$configs{python_dir} fluxpype/fluxpype/science/tempest.py --cr $CR --nwant $n_want";
        system($fmap_command) == 0 or die "Python script returned error $?";
    }
    elsif ($flow_method eq "cranmer") {
        my $fmap_command = "$configs{python_dir} fluxpype/fluxpype/science/cranmer_wind.py --cr $CR --nwant $n_want";
        system($fmap_command) == 0 or die "Python script returned error $?";
    }
    elsif ($flow_method eq "wsa") {
        # run the python file footpoint_distances.py
        system("$configs{python_dir} fluxpype/fluxpype/science/footpoint_distances_2.py --cr $CR") == 0
            or die "footpoint_distances_2.py returned error $?";

        my $distance_file = $configs{data_dir} . "/batches/" . $configs{batch_name} . "/data/cr" . $CR . "/floc/distances.csv";
        open my $fh, '<', $distance_file or die "Could not open '$distance_file': $!";

        my @rows;
        while (my $line = <$fh>) {
            chomp $line;
            my @values = split /, /, $line;  # split on comma+space
            push @rows, pdl(@values);
        }
        close $fh;

        # Combine all rows into a 2D PDL array
        our $distance_array_degrees = cat(@rows);
    }
    elsif ($flow_method =~ /^(parker|schonfeld|psw|ghosts)$/ ) {
        # No special precomputation needed (assuming these exist).
    }
    else {
        die "Invalid flow method: $flow_method";
    }

    print "\t\tThe flow method is '$flow_method'.\n\n";

    # Possibly limit fluxons for debugging
    my $max_fluxon_id = scalar(@open_fluxon_indices) - 1;
    $max_fluxon_id = $n_choke if $choke && ($n_choke < $max_fluxon_id);

    my $iteration_count = -1;
    my @results;

    # Set up parallel manager
    my $fork_manager = Parallel::ForkManager->new($max_processes, $temp_dir);

    # This runs in the parent process after each child finishes
    $fork_manager->run_on_finish(sub {
        $iteration_count++;
        my ($pid, $exit_code, $ident, $exit_signal, $core_dump, $result) = @_;
        return if $exit_code != 0;   # The child died or was killed?

        push(@results, $result);

        # Update progress
        if (defined $result->{fluxon_position}) {
            my $this_fluxon_id = $result->{fluxon_position}->sclr + 1;
            if ($this_fluxon_id > $highest_fluxon_done) {
                $highest_fluxon_done = $this_fluxon_id;
                my $after_time = clock_gettime();
                my $elapsed_time = $after_time - $before_time;
                my $rounded_elapsed = sprintf("%.1f", $elapsed_time);

                my $remaining_fluxons  = $num_open_fluxons - $highest_fluxon_done;
                my $time_each_fluxon   = $elapsed_time / $highest_fluxon_done;
                my $time_remaining     = $remaining_fluxons * $time_each_fluxon;
                my $rounded_remaining  = sprintf("%.1f", $time_remaining);
                my $rounded_remaining_mins = sprintf("%.1f", $time_remaining / 60);

                print "\rCalculated $flow_method: $highest_fluxon_done of $num_open_fluxons, ",
                      "$rounded_elapsed(s) elapsed, ",
                      "$rounded_remaining(s) [$rounded_remaining_mins mins] remaining...  ";
            }
        }
    });

    # Launch parallel tasks
    for my $fluxon_id (0 .. $max_fluxon_id - 1) {

        $fork_manager->start and next;  # child will skip

        my $fluxon = $fluxons[$fluxon_id];

        # Some local variables we expect from the flow subroutines
        my ($r_vr_scaled, $r_fr_scaled, $thetas, $phis);

        # Dispatch to correct flow subroutine
        if ($flow_method eq 'wsa') {
            our $distance_array_degrees;  # Must be "our" for lexical scoping if truly global
            ($r_vr_scaled, $r_fr_scaled, $thetas, $phis)
                = gen_fluxon_wsaflow($fluxon, $distance_array_degrees, $fluxon_id);

        }
        elsif ($flow_method eq 'parker') {
            ($r_vr_scaled, $r_fr_scaled, $thetas, $phis) = gen_fluxon_tflow($fluxon);

        }
        elsif ($flow_method eq 'schonfeld') {
            ($r_vr_scaled, $r_fr_scaled, $thetas, $phis) = gen_fluxon_schonflow($fluxon);

        }
        elsif ($flow_method eq 'psw') {
            # Presumably gen_fluxon_pswflow exists
            ($r_vr_scaled, $r_fr_scaled, $thetas, $phis) = gen_fluxon_pswflow($fluxon);

        }
        elsif ($flow_method eq 'tempest') {
            ($r_vr_scaled, $r_fr_scaled, $thetas, $phis)
                = gen_fluxon_tempestflow($fluxon, $fluxon_id, $output_file_name);

        }
        elsif ($flow_method eq 'cranmer') {
            ($r_vr_scaled, $r_fr_scaled, $thetas, $phis)
                = gen_fluxon_cranmerflow($fluxon, $fluxon_id);

        }
        elsif ($flow_method eq 'ghosts') {
            ($r_vr_scaled, $r_fr_scaled, $thetas, $phis)
                = gen_fluxon_ghostsflow($fluxon, $fluxon_id);

        }
        else {
            die "Invalid flow method: $flow_method";
        }

        # Extract columns from r_vr_scaled / r_fr_scaled
        our $vr;
        our $fr;
        our $rn;

        if ($flow_method eq 'parker') {
            # Parker: dimension layout from gen_fluxon_tflow
            $vr = $r_vr_scaled(1, :)->transpose;   # row 1 => velocity
            $rn = $r_vr_scaled(0, :)->transpose;   # row 0 => radius
            $fr = $r_fr_scaled(:, 1);              # flux expansion in column 1
        }
        else {
            # Many flows have $r_vr_scaled(:,0)=r, $r_vr_scaled(:,1)=vr, etc.
            $vr = $r_vr_scaled(:, 1);
            $rn = $r_fr_scaled(:, 0);
            $fr = $r_fr_scaled(:, 1);
        }

        # The code below sets up a “result” structure to pass back
        my $zn = $rn - 1;      # Just r/R - 1
        my $bot_ind = 1;
        my $top_ind = -2;

        # Create a result hash
        my $result = {
            fluxon_position => pdl($fluxon_id),
            r               => $rn,
            vr              => $vr,
            phi_base        => squeeze(pdl($phis(0))),
            theta_base      => squeeze(pdl($thetas(0))),
            phi_end         => squeeze(pdl($phis($top_ind))),
            theta_end       => squeeze(pdl($thetas($top_ind))),
            radial_velocity_base => squeeze(pdl($vr($bot_ind))),
            radial_velocity_end  => squeeze(pdl($vr($top_ind))),
            flux_expansion_base  => squeeze(pdl($fr($bot_ind))),
            flux_expansion_end   => squeeze(pdl($fr($top_ind))),
        };

        # Return the result to the parent
        $fork_manager->finish(0, $result);
    }

    # Wait for all children
    $fork_manager->wait_all_children;

    # Create results directory
    my ($output_filename, $output_dir) = fileparse($output_file_name);
    my $new_results_dir = File::Spec->catdir($output_dir, "full_velocity_profiles");
    make_path($new_results_dir) unless -d $new_results_dir;

    # Build final CSV filename
    my $results_file = File::Spec->catfile($new_results_dir, "results_${flow_method}_full_velocity.dat");
    print "\nINFO: Writing full velocity profiles to: $results_file\n";

    # Save to disk
    save_full_velocity_profiles(\@results, $results_file);

    my $after_time       = clock_gettime();
    my $elapsed_time     = $after_time - $before_time;
    my $rounded_elapsed  = sprintf("%.1f", $elapsed_time);

    print "\n\n\tWind Calculation Complete in $rounded_elapsed seconds.\n\n";

    # If WSA, optionally do an image plot
    # if ($flow_method eq "wsa"){
    #     do_image_plot();
    # }

    # Sort results by fluxon_position for final console output
    @results = sort { $a->{fluxon_position} <=> $b->{fluxon_position} } @results;

    # Example printing with wcols-like output
    # Use `PDL::Simple` or your own macros if you want to replicate wcols exactly
    wcols(
        pdl(map { $_->{fluxon_position} }   @results),
        pdl(map { $_->{phi_base} }          @results),
        pdl(map { $_->{theta_base} }        @results),
        pdl(map { $_->{phi_end} }           @results),
        pdl(map { $_->{theta_end} }         @results),
        squeeze(pdl(map { $_->{radial_velocity_base} } @results)),
        squeeze(pdl(map { $_->{radial_velocity_end} }  @results)),
        squeeze(pdl(map { $_->{flux_expansion_base} }  @results)),
        squeeze(pdl(map { $_->{flux_expansion_end} }   @results)),
        $output_file_name
    );

    # Delete temp directory
    rmtree($temp_dir);

    return;
}

# If this file is run directly, do a small example/demo (optional):
if ($0 eq __FILE__) {
    use PDL::AutoLoader;
    use PDL;
    use PDL::Transform;
    use PDL::NiceSlice;
    use PDL::Options;
    use Flux;
    use PDL::IO::Misc;
    use File::Path;
    use Time::HiRes qw(clock_gettime);
    use File::Basename qw(fileparse);
    use pipe_helper qw(configurations);

    my %configs  = configurations();
    my $datdir   = $configs{datdir};
    my $cr       = 2160;
    my $batch_name = "fluxon_paperfigs";

    my $world_out_dir     = $datdir . "$batch_name/cr$cr/rlx/";
    my $full_world_path   = $world_out_dir . "cr2160_relaxed_s4000.flux";
    my $wind_out_dir      = $datdir . "$batch_name/cr$cr/wind";
    my $wind_out_file     = "$wind_out_dir/radial_wind.dat";

    # Load world
    my $this_world_relaxed = read_world($full_world_path);
    $this_world_relaxed->update_force(0);

    my @fluxons = $this_world_relaxed->fluxons;

    # Example call
    map_fluxon_flow_parallel_master($wind_out_file, \@fluxons, 'parker', $cr, 100);
}

1;

__END__