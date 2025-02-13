
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
    asdf
    use warnings;
    use PDL::AutoLoader;
    use PDL;
    use pipe_helper;
    plot_worlds($this_world_orig, $this_world_relaxed, $do_interactive, $do_png, $full_world_path, $datdir, $batch_name, $CR, $N_actual, $nwant, $lim, $lim2, $stepnum);

=head1 DESCRIPTION

This script is designed to plot the initial and relaxed states of worlds using Perl's PDL (Perl Data Language). It supports both interactive plotting and PNG output.

=head1 FUNCTIONS

=head2 plot_worlds

    plot_worlds($this_world_orig, $this_world_relaxed, $do_interactive, $do_png, $full_world_path, $datdir, $batch_name, $CR, $N_actual, $nwant, $lim, $lim2, $stepnum);

Plots the initial and relaxed states of worlds.

=over

=item * C<$this_world_orig> - The initial world state.

=item * C<$this_world_relaxed> - The relaxed world state.

=item * C<$do_interactive> - Flag to enable interactive plotting.

=item * C<$do_png> - Flag to enable PNG output.

=item * C<$full_world_path> - Full path to the world data.

=item * C<$datdir> - Data directory.

=item * C<$batch_name> - Batch name.

=item * C<$CR> - Carrington Rotation.

=item * C<$N_actual> - Actual number of steps.

=item * C<$nwant> - Desired number of steps.

=item * C<$lim> - Limit for initial state plot.

=item * C<$lim2> - Limit for relaxed state plot.

=item * C<$stepnum> - Step number for relaxed state.

=back

=head1 EXAMPLES

    # To plot the worlds interactively
    plot_worlds($world1, $world2, 1, 0, $path, $data_dir, "batch1", 2163, 100, 50, 1, 1, 0);

=head1 AUTHOR

Gilly <gilly@swri.org> (and others!)

=head1 SEE ALSO

L<PDL>, L<PDL::AutoLoader>, L<pipe_helper>

=cut


sub plot_world {
        my ($world, $datdir, $batch_name, $CR, $reduction, $n_fluxons_wanted, $adapt, $force_make_world, $lim, $lim2, $configs, $which) = @_;

        ## Display ####################################################
        print "\n \n\tPlotting the $which World...";
        # Set Ranges of Plots
        my $narr_factor = 0.5;
        my $wide_factor = 2.0;
        my $narr_lim = $lim * $narr_factor;
        my $wide_lim = $lim * $wide_factor;
        my $range_inner =   [-$narr_lim, $narr_lim, -$narr_lim, $narr_lim, -$narr_lim, $narr_lim];
        my $range_middle =  [-$lim, $lim, -$lim, $lim, -$lim, $lim];
        my $range_wide =    [-$wide_lim, $wide_lim, -$wide_lim, $wide_lim, -$wide_lim, $wide_lim];

        # number of open fluxons, number of fluxons, number of fluxons requested
        my $narrow_dir  = $datdir.   "/batches/$batch_name/imgs/world/0_narrow/";
        my $mid_dir     = $datdir.   "/batches/$batch_name/imgs/world/1_middle/";
        my $wide_dir    = $datdir.   "/batches/$batch_name/imgs/world/2_wide/";

        if (! -d $narrow_dir ) {mkpath($narrow_dir) or die "Failed to create directory: $narrow_dir $!\n";}
        if (! -d $mid_dir ) {mkpath($mid_dir) or die "Failed to create directory: $mid_dir $!\n";}
        if (! -d $wide_dir ) {mkpath($wide_dir) or die "Failed to create directory: $wide_dir $!\n";}

        my $ext = 'png';
        my $renderer = $ext.'cairo';
        my $world_png_path_narrow = $narrow_dir."cr$CR\_f". $n_fluxons_wanted. "_$which\_narrow.$ext";
        my $world_png_path_top    = $mid_dir   ."cr$CR\_f". $n_fluxons_wanted. "_$which\_middle.$ext";
        my $world_png_path_wide   = $wide_dir.  "cr$CR\_f". $n_fluxons_wanted. "_$which\_wide.$ext";

        my $window00 = gpwin($renderer,size=>[9,9], dashed=>0, output=> $world_png_path_narrow);
        $world->render( {'window'=>$window00, range=>$range_inner, view=>[0,0]});

        my $window000 = gpwin($renderer,size=>[9,9], dashed=>0, output=> $world_png_path_top);
        $world->render( {'window'=>$window000, range=>$range_middle, view=>[0,0]});

        my $window01 = gpwin($renderer,size=>[9,9], dashed=>0, output=> $world_png_path_wide);
        $world->render( {'window'=>$window01, range=>$range_wide, view=>[0,0]});
        print "\t\tDone!\n";
        }


sub plot_worlds {
    our (
        $this_world_orig, $this_world_relaxed, $do_interactive,
        $do_png,          $full_world_path,    $datdir,
        $batch_name,      $CR,                 $N_actual,
        $nwant,           $lim,                $lim2,
        $stepnum
    ) = @_;

    my $range_i  = [ -$lim,  $lim,  -$lim,  $lim,  -$lim,  $lim ];
    my $range_f  = [ -$lim,  $lim,  -$lim,  $lim,  -$lim,  $lim ];
    my $range_f2 = [ -$lim2, $lim2, -$lim2, $lim2, -$lim2, $lim2 ];

    if ($do_interactive) {

        # Plot to interactive window

        my $window1 = gpwin(
            'qt',
            size   => [ 9, 9 ],
            dashed => 0,
            title  => 'Initial Conditions'
        );
        my $window2 = gpwin(
            'qt',
            size   => [ 9, 9 ],
            dashed => 0,
            title  => 'After Relaxation'
        );
        ## Create a sphere with radius 1
        ## $sphere = sphere(50) * 1;
        ## $window2->splot($sphere);
        ## $window2->gnuplot("set style data lines");
        ## $window2->gnuplot("set hidden3d");

        # $this_world_orig->render( { 'window' => $window1, range => $range_i } );
        $this_world_relaxed->render(
            { 'window' => $window2, range => $range_f } );    #, hull=>'1'});
        ## use PDL;

        ## my $plot = gpwin();
        ## $plot->gnuplot("set view equal xyz");
        ## $plot->gnuplot("set xrange [-1:1]");
        ## $plot->gnuplot("set yrange [-1:1]");
        ## $plot->gnuplot("set zrange [-1:1]");
    }

    # print "\n \n**Plotting the Worlds...";
    print color("bright_cyan");
    print "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
    print "(pdl) Plotting the Initial and Relaxed Worlds\n";
    print
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n \n";
    print color("reset");

    ## Plot to Png
    $stepnum = $stepnum || 0;

    my $path = $full_world_path;
    my ( $filename, $directories, $suffix ) = fileparse( $path, qr/\.[^.]*/ );
    my $new_filename_initial = $directories . $filename . "_initial.png";
    my $new_filename_relaxed =
      $directories . $filename . "_relaxed\_s$stepnum.png";

    $do_png = $do_png || ( !-e $new_filename_relaxed );

    if ($do_png) {

        print "\tRendering Images...\n";

   # my $new_filename_initial2 = $directories . $filename . "_initial-wide.png";
        my $new_filename_relaxed =
          $directories . $filename . "_relaxed\_s$stepnum.png";
        my $new_filename_relaxed2 =
          $directories . $filename . "_relaxed\_s$stepnum-wide.png";

        my $top_path = $datdir . "/batches/$batch_name/imgs/world/";
        if ( !-d $top_path ) {
            mkpath($top_path)
              or die "Failed to create directory: $top_path $!\n";
        }
        my $top_filename_relaxed =
          $top_path . "cr" . $CR . "_f$nwant\_relaxed\_s$stepnum.png";



# my $short_new_filename_initial = shorten_path($new_filename_initial);
# print "\t\tRendering $short_new_filename_initial\n";
# my $window25=   gpwin('pngcairo',size=>[9,9],dashed=>0, output=> $new_filename_initial);
# $this_world_orig->render( {'window'=>$window25, range=>$range_i});
# $window25 = null;

# my $short_new_filename_initial2 = shorten_path($new_filename_initial2);
# print "\t\tRendering $short_new_filename_initial2\n";
# my $window3 =   gpwin('pngcairo',size=>[9,9],dashed=>0, output=> $new_filename_initial2);
# $this_world_orig->render( {'window'=>$window3, range=>$range_f2});
# $window3 = null;
        # use File::Basename;

        my $short_new_filename_relaxed = shorten_path($new_filename_relaxed);
        my $dir_name = dirname($new_filename_relaxed);

        print dirname($dir_name) . "\n\n\n";

        if ( !-d $dir_name ) {
            print "Making things";
            mkpath(dirname($dir_name)) or die "Failed to create parent directory: " . dirname($dir_name) . " $!\n";
            mkpath($dir_name) or die "Failed to create directory: $dir_name $!\n";
        }


        print "\t\tRendering $short_new_filename_relaxed\n";
        my $window4 = gpwin(
            'pngcairo',
            size   => [ 9, 9 ],
            dashed => 0,
            output => $new_filename_relaxed
        );
        $this_world_relaxed->render(
            { 'window' => $window4, range => $range_f } );
        $window4 = null;

        my $short_new_filename_relaxed2 = shorten_path($new_filename_relaxed2);

        print "\t\tRendering $short_new_filename_relaxed2\n";
        my $window5 = gpwin(
            'pngcairo',
            size   => [ 9, 9 ],
            dashed => 0,
            output => $new_filename_relaxed2
        );
        $this_world_relaxed->render(
            { 'window' => $window5, range => $range_f2 } );
        $window5 = null;

        my $short_top_filename_relaxed = shorten_path($top_filename_relaxed);

        print "\t\tRendering $short_top_filename_relaxed\n";
        my $windowTop = gpwin(
            'pngcairo',
            size   => [ 9, 9 ],
            dashed => 0,
            output => $top_filename_relaxed
        );
        $this_world_relaxed->render(
            { 'window' => $windowTop, range => $range_f } );
        $windowTop = null;

        # $this_world_orig    = null;
        # $this_world_relaxed = null;

        ## Create a sphere with radius 1
        # $sphere = sphere(50) * 1;

        # $window25->splot($sphere);
        # $window3->splot($sphere);
        # $window4->splot($sphere);
        # $window5->splot($sphere);
    }

    my $short_name   = shorten_path($new_filename_initial);
    my $short_name_r = shorten_path($new_filename_relaxed);

    if ( not $do_png and not $do_interactive ) {
        print
"\tSkipped Plotting! Found: \n\t\t$short_name and \n\t\t$short_name_r\n";
    }
    else {
        print "\tDone with Plotting!\n";
    }

    print "\n\t\t\t```````````````````````````````\n \n\n\n";
    return;
}

1;
