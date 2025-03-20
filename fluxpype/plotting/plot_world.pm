=head1 NAME

plot_worlds - Perl PDL Script for Plotting Initial and Relaxed Worlds

=cut

package plot_world;
use strict;
use warnings;
use Exporter qw(import);
our @EXPORT_OK = qw(plot_world plot_worlds);
use PDL;
use File::Basename         qw(dirname fileparse);
use File::Path             qw(mkpath);
use lib "fluxpype";
use pipe_helper            qw(shorten_path);
use PDL::Graphics::Gnuplot qw(gpwin);
use PDL::AutoLoader;
use Term::ANSIColor;

=head1 SYNOPSIS

    use warnings;
    use PDL::AutoLoader;
    use PDL;
    use pipe_helper;
    plot_worlds($this_world_orig, $this_world_relaxed, $do_interactive, $do_png,
                $full_world_path, $datdir, $batch_name, $CR, $N_actual, $nwant,
                $lim, $lim2, $stepnum);

=head1 DESCRIPTION

This script plots the initial and relaxed states of worlds using Perl's PDL.

=head1 FUNCTIONS

=head2 plot_world($world, $datdir, $batch_name, $CR, $reduction, $n_fluxons_wanted, $adapt, $force_make_world, $lim, $lim2, $configs, $which)

Plots a single world (narrow, middle, wide) into subdirectories 0_narrow, 1_middle, 2_wide.

=cut

sub plot_world {
    my ($world, $datdir, $batch_name, $CR,
        $reduction, $n_fluxons_wanted,
        $adapt, $force_make_world,
        $lim, $lim2, $configs,
        $which
    ) = @_;

    ## Display
    print "\n \n\tPlotting the $which World...";

    # Set Ranges of Plots
    my $narr_factor = 0.5;
    my $wide_factor = 2.0;
    my $narr_lim    = $lim  * $narr_factor;
    my $wide_lim    = $lim  * $wide_factor;

    my $range_inner  = [ -$narr_lim, $narr_lim, -$narr_lim, $narr_lim, -$narr_lim, $narr_lim ];
    my $range_middle = [ -$lim, $lim, -$lim, $lim, -$lim, $lim ];
    my $range_wide   = [ -$wide_lim, $wide_lim, -$wide_lim, $wide_lim, -$wide_lim, $wide_lim ];

    my $narrow_dir = $datdir . "/batches/$batch_name/imgs/world/0_narrow/";
    my $mid_dir    = $datdir . "/batches/$batch_name/imgs/world/1_middle/";
    my $wide_dir   = $datdir . "/batches/$batch_name/imgs/world/2_wide/";

    mkpath($narrow_dir) or die "Failed to create $narrow_dir: $!\n" unless -d $narrow_dir;
    mkpath($mid_dir)    or die "Failed to create $mid_dir: $!\n"    unless -d $mid_dir;
    mkpath($wide_dir)   or die "Failed to create $wide_dir: $!\n"   unless -d $wide_dir;

    my $ext = 'png';
    my $renderer = "pngcairo";  # Force pngcairo

    my $world_png_path_narrow = $narrow_dir."cr$CR\_f". $n_fluxons_wanted. "_$which\_narrow.$ext";
    my $world_png_path_middle = $mid_dir   ."cr$CR\_f". $n_fluxons_wanted. "_$which\_middle.$ext";
    my $world_png_path_wide   = $wide_dir  ."cr$CR\_f". $n_fluxons_wanted. "_$which\_wide.$ext";

    # Simple skip if we've done it already
    if (-e $world_png_path_narrow) {
        print " Skipping existing plots!\n";
        return;
    }

    # Plot narrow
    my $win_narrow = gpwin($renderer, size=>[9,9], dashed=>0, output=>$world_png_path_narrow);
    $world->render({ window => $win_narrow, range => $range_inner, view => [0,0] });

    # Plot middle
    my $win_middle = gpwin($renderer, size=>[9,9], dashed=>0, output=>$world_png_path_middle);
    $world->render({ window => $win_middle, range => $range_middle, view => [0,0] });

    # Plot wide
    my $win_wide = gpwin($renderer, size=>[9,9], dashed=>0, output=>$world_png_path_wide);
    $world->render({ window => $win_wide, range => $range_wide, view => [0,0] });

    print "\t\tDone!\n";
}


=head2 plot_worlds(...)

Plots both the initial and relaxed states (using plot_world for triple zoom),
optionally to an interactive window as well.

=cut

sub plot_worlds {
    our (
        $this_world_orig,   $this_world_relaxed,
        $do_interactive,    $do_png,
        $full_world_path,   $datdir,
        $batch_name,        $CR,
        $N_actual,          $nwant,
        $lim,               $lim2,
        $stepnum
    ) = @_;

    my $range_i  = [ -$lim,  $lim,  -$lim,  $lim,  -$lim,  $lim ];
    my $range_f  = [ -$lim,  $lim,  -$lim,  $lim,  -$lim,  $lim ];

    # If interactive, show them side-by-side on the screen
    if ($do_interactive) {
        my $window1 = gpwin('qt', size => [9,9], dashed => 0, title => 'Initial');
        my $window2 = gpwin('qt', size => [9,9], dashed => 0, title => 'Relaxed');

        $this_world_orig->render(   { 'window' => $window1, range => $range_i } );
        $this_world_relaxed->render({ 'window' => $window2, range => $range_f } );
    }

    print color("bright_cyan");
    print "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    print "(pdl) Plotting the Initial and Relaxed Worlds\n";
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n \n";
    print color("reset");

    $stepnum ||= 0;

    # Instead of manual gpwin calls, reuse plot_world for triple-zoom
    if (1) {
        print "\tRendering triple-zoom images for initial & relaxed...\n";

        # For "initial"
        plot_world(
            $this_world_orig,   # the world data
            $datdir,
            $batch_name,
            $CR,
            $N_actual,          # 'reduction' param
            $nwant,             # fluxons
            0,                  # adapt
            0,                  # force_make_world
            $lim,
            $lim2,
            undef,              # configs
            'initial'
        );

        # For "relaxed"
        plot_world(
            $this_world_relaxed,
            $datdir,
            $batch_name,
            $CR,
            $N_actual,
            $nwant,
            0,
            0,
            $lim,
            $lim2,
            undef,
            'relaxed'
        );
    }
    else {
        print "\tSkipping PNG output (do_png not set)\n";
    }

    print "\n\tDone with Plotting!\n\n";
    return;
}

1;