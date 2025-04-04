
=head2 simple_relaxer - sample relaxation code using FLUX

=for usage

 use Flux;
 $world = read_world('menagerie/twisted.flux')
 $dt = 0.3;
 $step = 0;
 simple_relaxer($world,0,300,$opt);

=for ref

simple_relaxer handles the top level relaxation loop -- it is written in perl for
ease of manipulation and to illustrate relaxation code in action.

It takes up to five arguments:

=over 3

=item $world

The first argument is a Flux::World object to relax

=item $global

The second argument sets the global flag for update_neighbors() and
update_force() calls.  It's here for debugging -- if you actuallly
want your simulation to ever complete, you want to set this to 0.
(Global neighbor recognition runs in O(n^2) time and is MUCH slower
than local neighbor expansion, which runs in O(n) time).

=item $final

The third argument is the step number at which to terminate.  See
below for how steps are kept track of.

=item $opt

The fourth argument, if present, is a hash ref containing options about
how the relaxation is to proceed.

=back

In addition, three global variables are used: $dt, $auto, and $step.  $dt keeps
track of delta-tau between steps.  $auto adjust delta-tau with a slightly more
sophisticated algorithm than the C library -- it shouldn't be necessary in most
cases.  $step keeps track of the current step number.  Setting it to zero causes a
global neighbor find to be done before the first step takes place.

The following options are useful in the options hash:

=over 3

=item movie_n

Specifies that a relaxation movie is to be made by capturing 3-D
renders every <n> time steps.

=item movie_f

This is the format string to be used to name relaxation movie frames.
The format string acts on a single integer -- the step number -- for
formatting.  The default value is "flux-movie-%5.5d.png".  The
formatted string is fed to L<wim|wim>, so the suffix determines the
file type of the images.

=item disp_n

Sets how often the on-screen rendering should be updated.  Set this to 0
to disable on-screen rendering.  (Default value is 10).

=item save_f

Format string for naming relaxation save-sequence frames (see below)

=item save_n

If this is specified, the state of the simulation is saved every n frames.

=item print_n

If this is specified, the state of the simulation is printed to the console every n frames.

=item render_opt

Contains a hash ref to be fed to Flux::World::render() when the field is
rendered.

=item range

A 3x2 PDL containing the (minimum, maximum) corners of the 3-cube to
render.  See L<Flux::World::render>.

=item log

Name of a log file. Number of vertices, stiffness, and information
about the forces is written here.

=item loghdr

Some text to go at the beginning of the log file. Has no effect if
C<log> is not set.

=back

Note that simple_relaxer is really a demonstation kludge -- it started
as a simple demo code but has grown and morphed into the main relaxation
engine control loop.

VERSION

This file is part of FLUX 2.0, released 31-Oct-2007.

=cut

package simple_relaxer;
use strict;
use warnings;
use Exporter qw(import);
our @EXPORT_OK = qw(simple_relaxer);
use PDL::Graphics::Gnuplot 2.006_001;
use Time::HiRes qw( clock_gettime);

sub simple_relaxer {
    if (   defined( $_[1] )
        && defined( $_[2] )
        && defined( $_[3] )
        && !ref( $_[1] )
        && !ref( $_[2] )
        && !ref( $_[3] ) )
    {
        die
"Looks like you're using the old calling convention for simple_relaxer.  Please delete the 'interactive' flag from your call.";
    }

    my $w       = shift;
    my $global  = shift || 0;
    my $n       = shift || 0;
    my $opt     = shift || {};
    my $auto    = $opt->{auto};
    my $save_n  = $opt->{save_n} || 0;
    my $save_f  = $opt->{save_f} || "step-%5.5d.flux.gz";
    my $movie_n = $opt->{movie_n};
    my $movie_f = $opt->{movie_f} || "flux-movie-%5.5d.png";
    my $disp_n =
      defined( $opt->{disp_n} ) ? $opt->{disp_n} : ( $movie_n || 10 );
    my $conv       = 180.0 / ( atan2( 1, 1 ) * 4. );
    my $print_n    = $opt->{print_n} || 1;
    my $do_print   = 1;
    my $step       = $w->{rel_step} || 0;
    my $final_step = $n;
    $final_step = $step + $n if $final_step <= $step;

    #    $w = read_world($w)
    #      unless(ref $w) ;

    my ( $display_term, $png_term ) = gnuplot_terms();

    PDL::Graphics::Gnuplot::options( terminal => $display_term );
    $w->render_lines() unless ( $disp_n == 0 );

    my $dt = $w->{dtau} || 0.1;

    if ( $opt->{log} ) {
        open LOG, ">$opt->{log}";
        print LOG $opt->{loghdr};
    }
    my $before = clock_gettime();

    while ( !$final_step || $step < $final_step ) {

        if ( $step % $print_n == 0 ) {
            $do_print = 1;
        }
        else { $do_print = 0; }

        if ( $w->{auto_open} ) {
            $w->f_length_check($global);
        }

        # $step % $disp_n
        if ( $w->{concurrency} ) {
            if ( $w->{rel_step} == 0 ) {
                printf("\tComputing initial fast neighbor search...\n");
                $w->update_neighbors(-1);
            }

            my $fmax = $w->update_force_parallel($global);

            $w->relax_step_parallel($dt);
        }
        else {
            ##problem here!
            if ( $w->{rel_step} == 0 ) {
                printf("computing inital fast neighbor search...\n");
                $w->update_neighbors(-1);
            }
            my $fmax = $w->update_force($global);
            $w->relax_step($dt);
        }
        my $t += $dt;
        $w->{rel_step}++;

        my $h = $w->fw_stats;

        if ($do_print) {
            eval {
                printf(
"\nstep $step: time is now %10.4g \n(%d vertices; stiffness=%7.3g%%, fmax=%7.3g, f_av=%7.3g, fs_av=%7.3g, n_av=%7.3g max_angle: %6.2f, mean_angle: %6.2f)\n",
                    $t,                             $h->{n},
                    $h->{f_av} / $h->{fs_av} * 100, $h->{f_ma},
                    $h->{f_av},                     $h->{fs_av},
                    $h->{n_av},                     $w->{max_angle},
                    $w->{mean_angle}
                );
            };
        }
        eval {
            printf LOG
" time is now %10.4g (%d vertices; stiffness=%7.3g%%, fmax=%7.3g, f_av=%7.3g, fs_av=%7.3g, n_av=%7.3g)\n",
              $t, $h->{n}, $h->{f_av} / $h->{fs_av} * 100, $h->{f_ma},
              $h->{f_av}, $h->{fs_av}, $h->{n_av}
              if ( $opt->{log} );
        };

        if ($@) { print "printf err:", $@; $@ = ''; }

        push( my @f_av,  $h->{f_av} );
        push( my @fs_av, $h->{fs_av} );

        unless ( $save_n == 0 || $step % $save_n ) {
            my $s = sprintf( $save_f, $step );
            print "saving %s...\n";
            $w->write_world($s);
        }

        unless ( $disp_n == 0 || $step % $disp_n ) {
            $w->render_lines( $opt->{render_opt} || { label => 0 } )
              ;    #{hull=>1,neighbors=>1,nscale=>1});
            if ($movie_n)
            { #DAL: this doesn't agree with the description of movie_n in the POD.
                my $fname = sprintf( $movie_f, $step );
                $w->render_lines(
                    $opt->{movie_opt} // $opt->{render_opt},
                    my $gpw = gpwin( $png_term, output => $fname, wait => 20)
                );
                $gpw->close;
                print "wrote $fname...\n";
            }
        }

        if ($auto) {

            # if (not defined my $dt0){
            #    my $dt0 = $dt;}
            #    my $dt0 = $dt unless(defined $dt0);
            my $ratio = $h->{f_av} / $h->{fs_av};
            my $dt    = $dt * $ratio;
            print "\t\tratio=$ratio; new dt=$dt\n";
        }

        $step++;

    }

    # ARD Update world
    $w->{dtau}     = $dt;
    $w->{rel_step} = $step;

    my $after   = clock_gettime();
    my $elapsed = $after - $before;
    my $concs   = $w->{concurrency} || 1;
    print
"\nTime Elapsed: $elapsed with print_n = $print_n and Concurrency = $concs";
}

=pod

=head2 gnuplot_terms

=for ref

Get the preferred display and png gnuplot terminals

=for example

my ($display,$png) = gnuplot_terms();
PDL::Graphics::Gnuplot::options(terminal=>$display);
PDL::Graphics::Gnuplot::options(terminal=>$png,output=>'file.png');

=for method

This is not super-portable, but should at least not break terribly on MacOS, Linux, and Windows. We prefer the wxt driver for interactivity since it is the most cross-platform. (The 'windows' driver has not been tested at all, since the authors do not have a Windows machine with Gnuplot installed.)

This is also really dumb.  Could be done using $PDL::Graphics::Gnuplot::globalPlot->{validterms}. Or through $Alien::Gnuplot::terms (preferred).

=cut

sub gnuplot_terms {

    my ( @display_terms, @png_terms, $display_term, $png_term );
    my @terminfo_lines = split "\n",
      PDL::Graphics::Gnuplot::terminfo( undef, undef, 1 )
      ;    #the '1' sets the "don't print" flag
    shift @terminfo_lines until $terminfo_lines[0] =~ /DISPLAY TERMINALS/;
    shift @terminfo_lines;
    while ( $terminfo_lines[0] =~ m/^\s*(\w*):/ ) {
        push @display_terms, $1;
        shift @terminfo_lines;
    }
    shift @terminfo_lines until $terminfo_lines[0] =~ /FILE TERMINALS/;
    shift @terminfo_lines;
    foreach (@terminfo_lines) {
        if ( $_ =~ /^\s*(png\w*):/ ) { push @png_terms, $1; }
    }
    foreach my $iterm ( 'qt', 'x11', 'aqua', 'windows' )
    {    #in order of preference
        if ( grep { $_ eq $iterm } @display_terms ) {
            $display_term = $iterm;
            last;
        }
    }
    foreach my $pterm ( 'pngcairo', 'png' ) {    #in order of preference
        if ( grep { $_ eq $pterm } @png_terms ) {
            $png_term = $pterm;
            last;
        }
    }
    return ( $display_term, $png_term );
}

1;
