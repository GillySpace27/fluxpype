=head1 NAME

fluxpype::events - structured progress events for the FLUXpype pipeline

=head1 SYNOPSIS

    use fluxpype::events;
    emit_event('stage.start', { stage => 'relax' });

=head1 DESCRIPTION

Appends one JSON line per event to the file named by C<$ENV{FLUXPYPE_EVENTS}>.
Does nothing when that variable is unset, and never dies: the whole body runs
inside C<eval>. The schema matches F<fluxpype/events.py> (see docs/gui/EVENTS.md).

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

our @EXPORT = qw(emit_event);
our $VERSION = '1';

my $OWNER_PID = $$;
my $JSON = JSON::PP->new->canonical->allow_nonref;

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
            %$fields,
        );
        my $line = $JSON->encode(\%record) . "\n";
        sysopen(my $fh, $path, O_WRONLY | O_APPEND | O_CREAT, 0644) or return;
        syswrite($fh, $line);
        close($fh);
    };
    return;
}

1;
