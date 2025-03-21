package map_fluxon_fr;

use strict;
use warnings;

use Exporter qw(import);
our @EXPORT_OK = qw(map_fluxon_fr);

use PDL qw(squeeze);
use PDL::NiceSlice;
=head2 map_fluxon_fr

=for ref

Given a world and list of fluxons, generates a mapping of the representative expansion factor from these fluxons.

=cut
use PDL::Options;

sub map_fluxon_fr {
    my $file_handle = shift;
    my $fluxons = shift;
    my @fluxons = @{$fluxons};

    die "I need something to work on!" if scalar(@fluxons)<0;

    ## Open an output file
    open(my $fh, '>', $file_handle) or die "Could not open file '$file_handle': $!";

    ## Loop through open fluxons and generate expansion profiles
    for my $fid(0..scalar(@fluxons)-1){

        my $me = $fluxons[$fid];

        # Check for open fieldlines
        my $st_open = ($me->{fc_start}->{label}==-1);
        my $en_open = ($me->{fc_end}->{label}==-2);
        if ($me->{plasmoid} || ($st_open + $en_open != 1)){
        next;
        }

        my $r0 = 696e6;
        my $x, my $y, my $z;
        my $r, my $r1;
        my $th;
        my $ph;
        my $A;
        my $fr;

        ## Grab coordinates
        if(1) { #$en_open) {
            $x = squeeze($me->dump_vecs->(0));
            $y = squeeze($me->dump_vecs->(1));
            $z = squeeze($me->dump_vecs->(2));
            $r1 = ($x**2 + $y**2 + $z**2)->sqrt * $r0;
            $r = 0.5 * $r1->range([[0],[1]],[$r1->dim(0)],'e')->sumover;
            $th = acos($z/$r1*$r0);
            $ph = atan2($y, $x);
            $A = pdl(map {$_->{A}} ($me->vertices)) * $r0 * $r0;
            $A(-1) .= $A(-2);
        } else {
            $x = squeeze($me->dump_vecs->(0,-1:0:-1));
            $y = squeeze($me->dump_vecs->(1,-1:0:-1));
            $z = squeeze($me->dump_vecs->(2,-1:0:-1));
            $r1 = ($x**2 + $y**2 + $z**2)->sqrt * $r0;
            $r = 0.5 * $r1->range([[0],[1]],[$r1->dim(0)],'e')->sumover;
            $th = acos($z/$r1*$r0);
            $ph = atan2($y, $x);
            $A = pdl(map { $_->{A}} ($me->vertices))->(-1:0:-1) * $r0 * $r0;
            $A(0) .= $A(1);
        }

        ## Calculate the expansion factor
        $fr = $A / $A->((1)) * ($r->((1))) * ($r->((1))) / $r / $r;


        ## Write to file
        for my $i(0..$r->getdim(0)-1){
            printf $fh "%05d %.8f %.8f %.8f %.8e %.8f %.8f %.8e %.8e\n",
            $fid, $x($i)->sclr, $y($i)->sclr, $z($i)->sclr, $r($i)->sclr, $th($i)->sclr, $ph($i)->sclr,
            $A($i)->sclr, $fr($i)->sclr;
        }
    }

    ## Close output file
    close $fh;

}
1;