=head1 NAME

fluxpype::events - structured progress events for the FLUXpype pipeline

=head1 SYNOPSIS

    use fluxpype::events;
    emit_event('metric', { name => 'n_actual', value => 812, unit => 'fluxons' });
    stage_start('relax');
    stage_end('relax', 'ok', 'detail text');

=head1 DESCRIPTION

Appends one JSON line per event to the file named by C<$ENV{FLUXPYPE_EVENTS}>.
Does nothing when that variable is unset, and never dies: the whole body runs
inside C<eval>. The schema matches F<fluxpype/events.py> (see docs/gui/EVENTS.md).

C<stage_start>, C<stage_end>, C<stage_skipped> and C<stage_detail> are thin wrappers that
time a stage and fill in C<elapsed_s>. If the process ends (die or exit) with a stage still
open, an END block closes it with C<status=failed>.

The process id is recorded when the module loads, and events are only written
while C<$$> still equals it, so forked Flux daughters never write.

=cut

package fluxpype::events;

use strict;
use warnings;
use Exporter qw(import);
use Fcntl qw(O_WRONLY O_APPEND O_CREAT);
use JSON::PP ();
use Time::HiRes ();

our @EXPORT = qw(emit_event stage_start stage_end stage_skipped stage_detail);
our $VERSION = '1';

my $OWNER_PID = $$;
my $JSON = JSON::PP->new->canonical->allow_nonref;
my (%STAGE_T, %STAGE_DETAIL, @OPEN);

# Flatten blessed values (PDL scalars) to plain numbers or strings so JSON::PP accepts them.
sub _plain {
    my ($v) = @_;
    return undef unless defined $v;
    if (ref $v eq 'HASH')  { return { map { $_ => _plain($v->{$_}) } keys %$v } }
    if (ref $v eq 'ARRAY') { return [ map { _plain($_) } @$v ] }
    if (ref $v) { $v = "$v" }
    return 0 + $v if $v =~ /^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*$/;
    return $v;
}

sub emit_event {
    my ($type, $fields) = @_;
    eval {
        my $path = $ENV{FLUXPYPE_EVENTS};
        return unless defined $path && length $path;
        return unless $$ == $OWNER_PID;
        $fields = {} unless ref $fields eq 'HASH';
        my $task = $ENV{FLUXPYPE_TASK_ID};
        my %record = (
            v    => 1,
            t    => 0 + sprintf('%.6f', Time::HiRes::time()),
            type => "$type",
            pid  => 0 + $$,
            task => (defined $task && length $task) ? "$task" : undef,
            %{ _plain($fields) },
        );
        my $line = $JSON->encode(\%record) . "\n";
        sysopen(my $fh, $path, O_WRONLY | O_APPEND | O_CREAT, 0644) or return;
        syswrite($fh, $line);
        close($fh);
    };
    return;
}

sub stage_start {
    my ($stage) = @_;
    eval {
        $STAGE_T{$stage} = Time::HiRes::time();
        push @OPEN, $stage;
        emit_event('stage.start', { stage => $stage });
    };
    return;
}

sub stage_detail {
    my ($stage, $detail) = @_;
    $STAGE_DETAIL{$stage} = $detail;
    return;
}

sub stage_end {
    my ($stage, $status, $detail) = @_;
    eval {
        $status = 'ok' unless defined $status && length $status;
        $detail = $STAGE_DETAIL{$stage} unless defined $detail;
        my $t0 = delete $STAGE_T{$stage};
        @OPEN = grep { $_ ne $stage } @OPEN;
        emit_event('stage.end', {
            stage     => $stage,
            status    => $status,
            elapsed_s => defined $t0 ? 0 + sprintf('%.2f', Time::HiRes::time() - $t0) : ($status eq 'skipped' ? 0 : undef),
            detail    => $detail,
        });
    };
    return;
}

sub stage_skipped {
    my ($detail, @stages) = @_;
    stage_end($_, 'skipped', $detail) for @stages;
    return;
}

END {
    my $code = $?;
    eval {
        if ($$ == $OWNER_PID && @OPEN) {
            my $msg = $code ? "process exited with status $code" : 'stage left open at exit';
            stage_end($_, 'failed', $msg) for reverse @OPEN;
        }
    };
    $? = $code;
}

1;
