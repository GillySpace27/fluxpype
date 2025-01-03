=pod

=head2 fluxon_placement_hilbert

=for usage

$fluxon_locations = fluxon_placement_hilbert($im, $count);
$fluxon_locations = fluxon_placement_hilbert($im, -$flux);

=for ref

Generates a (3xN) set of fluxon placements, in magnetogram pixels, adjusting
the flux per fluxon either to place approximately $count fluxons or to the
value set (if the second argument is negative). It uses a variant of
Riemersma dither, which is a 1-D error-diffusion technique. Quantization
error is diffused along an area-filling 1-D fractal (the Hilbert curve).

fluxon_placement_hilbert differs from Riemersma dither in that there is
no decay term applied to the diffused quantization error.

Because the Hilbert curves are only defined on images with dimension 2**n x 2**n,
the error diffusion happens on a grid that is interpolated onto the original
image. (n is selected so that 2**n is larger than the greater of the two
dimensions of the input image).

Because the quantization error follows a prescribed path rather than
diffusing through the image as in Floyd-Steinberg dither, absolute flux is
much better conserved than with fluxon_placement_fs.

fluxon_placement_hilbert uses the auxiliary routine hilbert(), in hilbert.pdl,
and also generates an Inline PP subroutine to do the heavy lifting of the
error diffusion.

=cut

package fluxon_placement_hilbert;
use strict;
use warnings;
use Exporter qw(import);
our @EXPORT_OK = qw(fluxon_placement_hilbert);
use lib ".";
use lib "fluxpype/fluxpype/science";
use hilbert qw(hilbert);
# use PDL::Hilbert;
use PDL;
use PDL::NiceSlice;
use PDL::ImageND;
use POSIX qw(isfinite);

use PDL;
use PDL::Core qw(pdl);
use PDL::NiceSlice;

sub fl_hi_helper {
    my ($a) = @_;

    my $err = 0;
    my $n = $a->nelem;  # Get the number of elements in $a
    my @indices = ();

    for my $i (0 .. $n - 1) {
        $err += $a->at($i);

        if ($err > 0.667) {
            push @indices, $i;
            $err -= 1;
        } elsif ($err < -0.667) {
            push @indices, -$i;
            $err += 1;
        }

        # Set the value in $a
        $a->set($i, $err);
    }

    return \@indices;
}


sub fluxon_placement_hilbert {
    my $bgram        = shift;
    my $fluxon_count = shift;
    my $verb         = shift || 0;

    $bgram = $bgram->copy;

    # Smoothing and calculation
    my $smooth = $bgram->convolveND(ones(3, 3) / 9, { b => 'm' });
    my $sm_max = $smooth->abs->max;
    my $sm_us_sum = $smooth->abs->sum;

    # Default fluxon count
    unless ($fluxon_count) {
        print "Warning - no fluxon count specified; using 250\n";
        $fluxon_count = 250;
    }

    # Calculate flux per fluxon
    my $flux = $fluxon_count < 0 ? -$fluxon_count : $sm_us_sum / $fluxon_count;
    $fluxon_count = $smooth->abs->sum / $flux;

    # Calculate the size for Hilbert curve
    my $siz = pdl($bgram->dims)->(0:1)->maximum;
    my $density_mult = 2 * ( $sm_max / ( $sm_us_sum / $smooth->nelem ) ) * ( $fluxon_count / $smooth->nelem );
    $density_mult = 1 unless ( $density_mult > 1 );
    $density_mult = sqrt($density_mult);
    $siz = 2**( ( log( $siz * $density_mult ) / log(2) )->ceil->at(0) );

    # Path calculation along Hilbert curve
    my $path = double( hilbert( $siz, $siz ) );
    $path->( (0) ) *= $bgram->dim(0) / $siz;
    $path->( (1) ) *= $bgram->dim(1) / $siz;

    # Interpolation of background series
    my $bg2 = $bgram * $fluxon_count / $sm_us_sum * $bgram->dim(0) * $bgram->dim(1) / $siz / $siz;
    my $bgseries = $bg2->interpND($path)->sever;

    if ($verb) {
        print "density_mult is $density_mult\n";
        print "siz is $siz\n";
        print "sm_us_sum is " . $sm_us_sum . "\n";
        print "Calling helper...\n";
    }

    my $plist = [];
    PDL::fl_hi_helper($bgseries, $plist);
    my $points = pdl($plist);
    my $pl     = $plist;
    $points = ( $path->( :, $points->abs ) )->glue( 0, ( 1 - 2 * ( $points->(*1) < 0 ) ) );
    return $points;
}
1;


no PDL::NiceSlice;
use Inline Config => CLEAN_AFTER_BUILD => 0;
use Inline Pdlpp=><<'EOF'
pp_def('fl_hi_helper',
    Pars=>'a(n);',
    OtherPars=>'SV *plsvr;',
    Code=>q{
        int i;
        double err = 0;
        AV *pl= (AV *)SvRV($COMP(plsvr));

        av_clear(pl);

        for(i=0;i<$SIZE(n);i++) {
            err += $a(n=>i);
            if(err > 0.667) {
                av_push(pl, newSViv(i));
                err -= 1;
            } else if(err < -0.667) {
                av_push(pl, newSViv(-i));
                err += 1;
            }
            $a(n=>i) = err;
        }
    }
);
EOF



