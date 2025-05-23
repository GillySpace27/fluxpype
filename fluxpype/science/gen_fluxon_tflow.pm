=head1 NAME

gen_fluxon_tflow - Generate Transonic Solar Wind Solution for a Given Fluxon

=cut

# package gen_fluxon_flow;


package gen_fluxon_tflow;
use strict;
use warnings;
use Exporter qw(import);
use lib "fluxpype/science";
use local::lib;
our @EXPORT_OK = qw(gen_fluxon_tflow);
use gen_fluxon_flow qw(gen_fluxon_flow);

use PDL::NiceSlice;
use PDL::Options;
$PDL::verbose = 0;


=head1 SYNOPSIS

This function iteratively generates a transonic solar wind solution for a given fluxon using a clever method to find the ideal solution.

=head1 DESCRIPTION

The function takes a fluxon and optional user-defined settings as input. It returns a set of PDL arrays representing the final fluxon array, fluxon radius, and magnetic field components (theta and phi).

If the fluxon is either doubly open, doubly closed, or labeled as a plasmoid, the function returns undef.

=head1 USAGE

    use gen_fluxon_tflow;
    my ($r_vr_scaled, $r_fr_scaled, $b_theta, $b_phi) = gen_fluxon_tflow($fluxon, \%options);

=head1 OPTIONS

=over 4

=item * initial_temperature: Initial temperature in Kelvin. Default is 1e6.

=item * sound_speed: Speed of sound based on the initial temperature. Calculated internally.

=item * velocity_increment: Incremental step for velocity search. Default is 25 km/s.

=back

=head1 DEPENDENCIES

This module depends on the following Perl modules:

=over 4

=item * PDL::NiceSlice
=item * PDL::Options

=back

=head1 DIAGNOSTICS

Set the C<$verbose> flag to 1 for diagnostic output.

=head1 SEE ALSO

Other relevant modules and documentation.

=cut

=head2 gen_fluxon_tflow

    gen_fluxon_tflow($fluxon, \%options)

Generates a transonic solar wind solution for a given fluxon.

=head3 Parameters

=over 4

=item * C<$fluxon>: A hash reference representing the fluxon.

=item * C<\%options> (optional): A hash reference for user-defined settings.

=back

=head3 Returns

Returns a list containing the final fluxon array, fluxon radius, and magnetic field components (theta and phi). Returns C<undef> if the fluxon is either doubly open, doubly closed, or labeled as a plasmoid.

=head3 Example

    my ($r_vr_scaled, $r_fr_scaled, $b_theta, $b_phi) = gen_fluxon_tflow($fluxon);

=cut


my $verbose = 0;
sub gen_fluxon_tflow {
    my $fluxon = shift;
    my $user_options = shift // {};

    # Check if the start and end points of the fluxon are open
    my $is_start_open = ($fluxon->{fc_start}->{label} == -1);
    my $is_end_open = ($fluxon->{fc_end}->{label} == -2);
    my $transonic_velocity = 0;

    # If the fluxon is labeled as a plasmoid, or if both ends are open, return undefined
    return undef if($fluxon->{plasmoid} || ($is_start_open + $is_end_open != 1));

    # Flag for plotting while diagnosing things
    my $should_plot = 0;

    # Define any other required quantities
    my $initial_temperature = 1e6;
    my $sound_speed = sqrt(2 * 1.381e-23 * $initial_temperature / 0.5 / 1.673e-27) / 1e3;
    my $velocity_increment = 20;

    # Define an initial lower bound with a breeze solution
    my $lower_velocity_bound = 50;
    my $upper_velocity_bound = 800;

    (our $r_vr_scaled, my $r_fr_scaled, my $b_theta, my $b_phi) = gen_fluxon_flow($fluxon, {'v0'=>$lower_velocity_bound, 'cs'=>$sound_speed});

    if ($verbose) {print "\n\tInitial Lower Bound: $lower_velocity_bound\n";}

    # Jump up in large steps to find a transonic solution, updating the lower breeze bound along the way
    my $search_for_transonic_velocity = 1;
    my $transonic_velocity_threshold = $lower_velocity_bound;
    my $iterations = 0;
    while ($search_for_transonic_velocity && $iterations < 20) {
        $lower_velocity_bound = $transonic_velocity_threshold;
        $transonic_velocity_threshold = $transonic_velocity_threshold + $velocity_increment;
        ($r_vr_scaled, $r_fr_scaled, $b_theta, $b_phi) = gen_fluxon_flow($fluxon, {'v0'=>$transonic_velocity_threshold, 'cs'=>$sound_speed});
        # print "Iteration $iterations: Transonic velocity threshold: $transonic_velocity_threshold\n";
        $iterations += 1;

        # Check for pseudo-discontinuities in the r_vr_scaled
        my $has_pseudo_discontinuities = 0;
        my $n_array = $r_vr_scaled->dim(1);  # Assuming $r_vr_scaled is a 2D piddle
        for (my $i = 0; $i < $n_array - 1; $i++) {
            my $current_value = $r_vr_scaled->slice("1,($i)")->sclr;
            my $next_value = $r_vr_scaled->slice("1,($i+1)")->sclr;
            if ($next_value > 1.25 * $current_value) {
                $has_pseudo_discontinuities = 1;
                last;
            }
        }

        if (($has_pseudo_discontinuities == 0) && ($r_vr_scaled(1,-1) == ($r_vr_scaled(1,:)->max()))) {
            $search_for_transonic_velocity = 0;
            $upper_velocity_bound = $transonic_velocity_threshold;
            if ($verbose) {print "\t\tTransonic velocity found! Range: $lower_velocity_bound to $upper_velocity_bound\n";}
        }
   }

    # Hone in on the ideal transonic solution
    my $iteration_count = 0;
    while ($iteration_count < 20) {
        # Test the mean velocity between the upper (transonic / misbehaved) and lower (breeze) bounds.
        # Define it as the new upper or lower bound accordingly.
        my $transonic_velocity_test = ($lower_velocity_bound + $upper_velocity_bound) / 2;
        # Cutoff wind velocities of unusual size
        if ($transonic_velocity_test > 800) {$upper_velocity_bound = 800;}
        if ($transonic_velocity_test < 400) {$lower_velocity_bound = 400;}
        $transonic_velocity_test = ($lower_velocity_bound + $upper_velocity_bound) / 2;

        # Generate a wind solution with the test velocity
        ($r_vr_scaled, $r_fr_scaled, $b_theta, $b_phi) = gen_fluxon_flow($fluxon, {'v0'=>$transonic_velocity_test, 'cs'=>$sound_speed});

        # Check for pseudo-discontinuities in the r_vr_scaled
        my $has_pseudo_discontinuities = 0;
        my $n_array = $r_vr_scaled->dim(1);
        for (my $i = 0; $i < $n_array - 1; $i++) {
            my $current_value = $r_vr_scaled->slice("1,($i)")->sclr;
            my $next_value = $r_vr_scaled->slice("1,($i+1)")->sclr;
            if ($next_value > 1.25 * $current_value) {
                $has_pseudo_discontinuities = 1;
                last;
            }
        }

        if (($has_pseudo_discontinuities == 0) && ($r_vr_scaled(1,-1) == ($r_vr_scaled(1,:)->max()))) {
            # This was a successful try! Set the new upper bound to the test velocity.
            $upper_velocity_bound = $transonic_velocity_test;

            if ($upper_velocity_bound - $lower_velocity_bound < 1) {
                # If the upper and lower bounds are within 1 km/s of each other, we're done!
                $transonic_velocity = $transonic_velocity_test;
                if ($verbose) {print "\t\t\tExact transonic velocity found! $transonic_velocity\n";}
                last;
            }
        } else {
            # This was an unsuccessful try. Set the new lower bound to the test velocity.
            $lower_velocity_bound = $transonic_velocity_test;
        }



        $iteration_count += 1;
        # print "\t\tIteration $iteration_count: $lower_velocity_bound, $transonic_velocity_test, $upper_velocity_bound\n";
    }

    # Generate a wind solution with the final upper limit
    ($r_vr_scaled, $r_fr_scaled, $b_theta, $b_phi) = gen_fluxon_flow($fluxon, {'v0'=>$transonic_velocity, 'cs'=>$sound_speed});

    # Return the final fluxon array, fluxon radius, and theta and phi
    return ($r_vr_scaled, $r_fr_scaled, $b_theta, $b_phi);
}
1;
__END__
