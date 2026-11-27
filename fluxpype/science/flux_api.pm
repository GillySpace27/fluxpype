package flux_api;

=head1 NAME

flux_api - the one fluxpype-side Perl boundary to the FLUX core (audit A1-01, A1-09)

=head1 SYNOPSIS

    use lib 'fluxpype/science';
    use flux_api qw(load relax_cycle fix stats save done finish);

    my $w = load($in);                      # read_world; the object is kept alive until exit
    relax_cycle($w, 100);                   # update_neighbors(-1) on step 0, then update_force + relax_step per step
    my $h = stats($w);                      # fw_stats: f_av, fs_av, n, ...
    fix($w, 0.3, 0.1, 2.5);                 # fix_curvature(hi, lo) then fix_proximity(prox)
    save($w, $out);                         # write_world
    finish();                               # print DONE, flush, POSIX::_exit(0): never returns

=head1 DESCRIPTION

FLUX core (fluxon-mhd) is frozen, so the rules fluxpype needs to follow when it calls it live here, in one
file, instead of being copied into each driver. See docs/RULES-FOR-CALLING-FLUX.md for the rules and the core
defect behind each one. Nothing here changes what FLUX computes: every function makes the same core calls, in the
same order, that the drivers made before.

Loading this module does not load Flux (the core is required on first use of C<load> or C<read_str>), so a module
that only wants the fork-child exit rule can use it without changing what is loaded when.

=cut

use strict;
use warnings;
use Exporter qw(import);
use POSIX ();

our @EXPORT_OK = qw(load read_str keep neighbors_init relax_cycle fix stats save done finish fork_child);

# Rule 1 (A1-03, K04): freeing a world segfaults (exit 139) when it has a one-sided neighbor link. Every Flux object
# the facade hands out is also pushed here, so nothing is destroyed before finish() leaves with POSIX::_exit.
our @KEEP;

# Rule 2 (A1-09 #2): Flux's DESTROY in a forked child frees the parent's world. A child marked with fork_child()
# leaves through POSIX::_exit at END, without global destruction. Its result must already be handed to the parent.
our $FORK_CHILD = 0;
END { POSIX::_exit($?) if $FORK_CHILD }

=head2 keep(@objects)

Keep Flux objects (or anything holding them, such as a list from C<< $w->vertices >>) alive until exit.
Returns its arguments.

=cut

sub keep { push @KEEP, @_; return wantarray ? @_ : $_[0] }

=head2 load($path)

C<read_world($path)>, kept alive. Dies like read_world does.

=cut

sub load {
    my ($path) = @_;
    require Flux;
    require Flux::World;
    my $w = Flux::World::read_world($path);
    keep($w);
    return $w;
}

=head2 read_str($text)

C<Flux::World::str2world($text)>, kept alive. Note the core's str2world ignores a failed temp-file open
(A1-09), so check the result.

=cut

sub read_str {
    my ($text) = @_;
    require Flux;
    require Flux::World;
    my $w = Flux::World::str2world($text);
    keep($w);
    return $w;
}

=head2 neighbors_init($world)

Rule 5 (A1-09 #5): C<update_force> before any C<update_neighbors(-1)> segfaults instead of failing. Runs
C<update_neighbors(-1)> when C<rel_step> is 0 (or unset); does nothing after that. Returns true if it ran.

=cut

sub neighbors_init {
    my ($w) = @_;
    return 0 if ( $w->{rel_step} || 0 ) != 0;
    $w->update_neighbors(-1);
    return 1;
}

=head2 relax_cycle($world, $steps, $dt)

C<$steps> serial relaxation steps: C<neighbors_init>, C<update_force(0)>, C<relax_step($dt)>, C<rel_step++>.
C<$dt> defaults to C<< $world->{dtau} || 0.1 >> read at call time; the core changes C<dtau> while it relaxes, so a
caller that must keep one step size across cycles reads it once and passes it. Serial only; the parallel path (relax_concurrency) stays in
simple_relaxer.pm for now. Returns the number of steps taken.

=cut

sub relax_cycle {
    my ( $w, $steps, $dt ) = @_;
    $dt = $w->{dtau} || 0.1 unless defined $dt;
    for ( 1 .. $steps ) {
        neighbors_init($w);
        $w->update_force(0);
        $w->relax_step($dt);
        $w->{rel_step}++;
    }
    return $steps;
}

=head2 fix($world, $curv_hi, $curv_lo, $proximity)

C<fix_curvature($hi, $lo)> then C<fix_proximity($proximity)>, in that order (the order changes the result).

=cut

sub fix {
    my ( $w, $hi, $lo, $prox ) = @_;
    $w->fix_curvature( $hi, $lo );
    $w->fix_proximity($prox);
    return;
}

=head2 stats($world)

C<fw_stats>: a hash ref with f_av, fs_av, n and the other world statistics.

=cut

sub stats { return $_[0]->fw_stats }

=head2 save($world, $path)

C<write_world($path)>. Call it last: after the last write the process should reach C<finish>.

=cut

sub save {
    my ( $w, $path ) = @_;
    $w->write_world($path);
    return $path;
}

=head2 done()

Prints C<DONE> and a newline to STDOUT and flushes. The protocol line fluxpype's Python side waits for. C
printf output from the core can be glued in front of it (A1-09 #4), so readers match it with C<endswith>.

=cut

sub done {
    local $| = 1;
    print STDOUT "DONE\n";
    return;
}

=head2 finish()

Rule 1: flush STDOUT and STDERR, then C<POSIX::_exit(0)>. Never returns; no destructor runs, so the K04 teardown
segfault cannot happen. Call it after the last write, once C<done> has printed. Anything not yet flushed is lost.

=cut

sub finish {
    my ($code) = @_;
    STDOUT->flush;
    STDERR->flush;
    POSIX::_exit( $code // 0 );
}

=head2 fork_child()

Rule 2: call it first thing in a forked child (after C<< $fork_manager->start >> returned 0 in the child).
From then on the process leaves through C<POSIX::_exit($?)> at END.

=cut

sub fork_child { $FORK_CHILD = 1; return }

1;
