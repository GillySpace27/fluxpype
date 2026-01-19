=head1 NAME

gen_fluxon_wsaflow - Comprehensive Module for Solar Wind Solution and Fluxon Flow Analysis

=cut

package gen_fluxon_wsaflow;
use strict;
use warnings;
use Exporter qw(import);
our @EXPORT_OK = qw(gen_fluxon_wsaflow do_image_plot load_predictor_maps);
use JSON::PP qw(decode_json);
use File::Spec;
use File::Basename qw(dirname);

use PDL::Graphics::Gnuplot;
use PDL;
use PDL::NiceSlice;
use PDL::Options;
use PDL::ImageND;
use PDL ('which_min'); # This imports which_min directly into your current package
use POSIX;
use Math::RungeKutta;
use PDL::GSL::INTEG;
use PDL qw(squeeze);
$PDL::verbose = 0;

=head1 SYNOPSIS

The module provides two primary functionalities:
1. gen_fluxon_wsaflow: Iteratively generates a transonic solar wind solution for a given fluxon.
2. gen_fluxon_wsaflow_phys: Calculates the flow and density in a fluxon, depending on its open or closed status.

=head1 DESCRIPTION

gen_fluxon_wsaflow - Takes a fluxon and optional user-defined settings as input, returning PDL arrays representing the final fluxon array, fluxon radius, and magnetic field components. Handles special cases like doubly open, doubly closed, or plasmoids.

gen_fluxon_wsaflow_phys - Provides detailed flow and density calculations in a fluxon, returning a 2xN PDL containing (r, vr), and updates these fields in the fluxon. Also returns theta and phi values for the fluxon, handling special cases similarly.

=head1 USAGE

For gen_fluxon_wsaflow:
    use gen_fluxon_wsaflow;
    my ($r_vr_scaled, $r_fr_scaled, $theta, $phi) = gen_fluxon_wsaflow($fluxon, \%options);

For gen_fluxon_wsaflow_phys:
    use gen_fluxon_wsaflow_phys;
    my $result = gen_fluxon_wsaflow_phys($fluxon, \%options);

=head1 OPTIONS

gen_fluxon_wsaflow options:
=over 4
=item * initial_temperature: Initial temperature in Kelvin. Default is 1e6.
=item * sound_speed: Speed of sound based on the initial temperature. Calculated internally.
=item * velocity_increment: Incremental step for velocity search. Default is 25 km/s.
=back

gen_fluxon_wsaflow_phys options:
=over 4
=item * steps: Number of steps for calculation. Default is 500.
=item * g0: Gravitational acceleration in m/s^2. Default is 280.
=item * r0: Radius in meters. Default is 696e6.
=item * v0: Velocity in km/sec. Default is a random number between 10 and 20.
=item * cs: Sound speed in km/sec at 2e6K and fully ionized protons. Default is 180.
=back

=head1 DEPENDENCIES

This module depends on the following Perl modules:
=over 4
=item * PDL::NiceSlice
=item * PDL::Options
=item * PDL::ImageND
=item * Math::RungeKutta
=item * PDL::GSL::INTEG
=back

=head1 DIAGNOSTICS

Set the C<$verbose> flag to 1 for diagnostic output.

=head1 AUTHOR

Gilly <gilly@swri.org> (and others!)

=head1 SEE ALSO

Other relevant modules and documentation.

=cut



my $verbose = 0;

# --- Helper: read a numeric 2D CSV into a PDL of shape (rows, cols)
sub _read_csv_2d {
    my ($path) = @_;
    open my $fh, "<", $path or die "[load] Cannot open $path: $!";
    my @rows;
    while (my $line = <$fh>) {
        chomp $line;
        next if $line =~ /^\s*$/;
        my @vals = map {
            my $v = $_;
            $v =~ s/^\s+|\s+$//g;
            $v = '0' if (!defined $v or $v eq '');
            0.0 + $v;
        } split(/\s*,\s*/, $line, -1);
        push @rows, \@vals;
    }
    close $fh;
    my $mat = pdl(\@rows)->double;   # dims: (rows, cols)
    return $mat;
}


sub _slurp {
    my ($path) = @_;
    local $/ = undef;
    open my $fh, "<", $path or return undef;
    my $txt = <$fh>;
    close $fh;
    return $txt;
}

# Helper: optionally flip latitude / longitude axes (rows/cols)
sub _flip_lat {
    my ($p) = @_;
    return undef unless defined $p;
    return $p(-1:0:-1, :);
}
sub _flip_lon {
    my ($p) = @_;
    return undef unless defined $p;
    return $p(:, -1:0:-1);
}

# Public: Load distance & expansion maps written by footpoint_distances_wsa.py
# Returns (distance_array_degrees, expansion_array_or_undef)
# Prefers *_latlon.csv (rows=lat, cols=lon) to match interpolate_2d_lonlat expectations.
sub load_predictor_maps {
    my ($base) = @_;
    die "[load] load_predictor_maps: base path is undefined" unless defined $base;
    my $dir = -d $base ? $base : dirname($base);
    my $meta_path = File::Spec->catfile($dir, "pfss_maps_meta.json");

    my ($dist_rel, $exp_rel, $lat_len, $lon_len);
    my ($flip_lat_meta, $flip_lon_meta);
    if (-e $meta_path) {
        my $jtxt = _slurp($meta_path);
        my $meta = eval { decode_json($jtxt) };
        if ($@ or !defined $meta) {
            print "[load] WARN: Failed to parse $meta_path, falling back to defaults.\n";
        } else {
            $dist_rel = $meta->{distance_csv_latlon} // "pfss_distances_latlon.csv";
            $exp_rel  = $meta->{expansion_csv_latlon};
            $lat_len  = $meta->{lat_len};
            $lon_len  = $meta->{lon_len};
            $flip_lat_meta = $meta->{flip_lat};
            $flip_lon_meta = $meta->{flip_lon};
        }
    }
    # Fallbacks if meta missing
    $dist_rel //= (-e File::Spec->catfile($dir, "pfss_distances_latlon.csv"))
              ? "pfss_distances_latlon.csv"
              : "pfss_distances.csv";
    if (!defined $exp_rel) {
        if (-e File::Spec->catfile($dir, "pfss_expansions_latlon.csv")) {
            $exp_rel = "pfss_expansions_latlon.csv";
        } elsif (-e File::Spec->catfile($dir, "pfss_expansions.csv")) {
            $exp_rel = "pfss_expansions.csv";
        }
    }

    my $dist_path = File::Spec->catfile($dir, $dist_rel);
    die "[load] Distance CSV not found: $dist_path" unless -e $dist_path;
    print "[load] distances: $dist_path\n";
    my $distance = _read_csv_2d($dist_path);

    my $expansion;
    if (defined $exp_rel) {
        my $exp_path = File::Spec->catfile($dir, $exp_rel);
        if (-e $exp_path) {
            print "[load] expansion: $exp_path\n";
            $expansion = _read_csv_2d($exp_path);
        } else {
            print "[load] WARN: expansion CSV not found: $exp_path (continuing with undef)\n";
        }
    }

    # Decide on optional axis flips from meta or environment overrides
    my $flip_lat = 0 + ($ENV{WSA_FLIP_LAT} // ($flip_lat_meta // 0));
    my $flip_lon = 0 + ($ENV{WSA_FLIP_LON} // ($flip_lon_meta // 0));

    if ($flip_lat) {
        print "[load] flipping LAT axis (rows reversed)\n";
        $distance = _flip_lat($distance);
        $expansion = _flip_lat($expansion) if defined $expansion;
    }
    if ($flip_lon) {
        print "[load] flipping LON axis (cols reversed)\n";
        $distance = _flip_lon($distance);
        $expansion = _flip_lon($expansion) if defined $expansion;
    }

    # Verify orientation against lat_len/lon_len if provided
    if (defined $lat_len and defined $lon_len) {
        my ($r, $c) = $distance->dims;
        if ($r != $lat_len or $c != $lon_len) {
            if ($c == $lat_len and $r == $lon_len) {
                print "[load] transposing distances to (rows=lat, cols=lon)\n";
                $distance = $distance->transpose;
            } else {
                print "[load] WARN: distance dims=($r,$c) don't match meta lat_len=$lat_len lon_len=$lon_len\n";
            }
        }
        if (defined $expansion) {
            my ($re, $ce) = $expansion->dims;
            if ($re != $lat_len or $ce != $lon_len) {
                if ($ce == $lat_len and $re == $lon_len) {
                    print "[load] transposing expansion to (rows=lat, cols=lon)\n";
                    $expansion = $expansion->transpose;
                } else {
                    print "[load] WARN: expansion dims=($re,$ce) don't match meta lat_len=$lat_len lon_len=$lon_len\n";
                }
            }
        }
    }

    {
        my ($nz_top_d, $nz_bot_d, $nz_top_e, $nz_bot_e);

        $nz_top_d = ($distance(0:9, :) != 0)->sum / (10 * $distance->dim(1));
        $nz_bot_d = ($distance(-10:-1, :) != 0)->sum / (10 * $distance->dim(1));
        printf "[load] distance nonzero frac top/bot 10 rows: %.3f / %.3f\n", $nz_top_d, $nz_bot_d;

        if (defined $expansion) {
            $nz_top_e = ($expansion(0:9, :) != 0)->sum / (10 * $expansion->dim(1));
            $nz_bot_e = ($expansion(-10:-1, :) != 0)->sum / (10 * $expansion->dim(1));
            printf "[load] expansion nonzero frac top/bot 10 rows: %.3f / %.3f\n", $nz_top_e, $nz_bot_e;
        }
    }

    # Final debug
    my ($dr,$dc) = $distance->dims;
    print "[load] distance dims=($dr,$dc), stats: min=", $distance->min, " max=", $distance->max, " mean=", $distance->avg, "\n";
    if (defined $expansion) {
        my ($er,$ec) = $expansion->dims;
        print "[load] expansion dims=($er,$ec), stats: min=", $expansion->min, " max=", $expansion->max, " mean=", $expansion->avg, "\n";
    }
    return ($distance, $expansion);
}



sub gen_fluxon_wsaflow {
    my $fluxon = shift;
    my $distance_array_degrees = shift;
    my $expansion_array = shift;
    my $fluxon_id = shift;

    # --- Autoload predictors if the arguments are not PDLs
    # Accept either:
    #  (a) $distance_array_degrees is a PDL (keep as-is), or
    #  (b) $distance_array_degrees is a directory or CSV path → load, and optionally load expansion
    #  (c) if both are PDLs, do nothing
    if (!defined $distance_array_degrees || ref($distance_array_degrees) ne 'PDL') {
        my $hint = $distance_array_degrees // '.';
        if (defined $hint && -d $hint) {
            my ($d_loaded, $e_loaded) = load_predictor_maps($hint);
            $distance_array_degrees = $d_loaded;
            # Only override expansion if caller didn't pass a PDL
            $expansion_array = $e_loaded if (!defined $expansion_array || ref($expansion_array) ne 'PDL');
        } elsif (defined $hint && -f $hint) {
            print "[autoload] reading distance CSV: $hint\n";
            $distance_array_degrees = _read_csv_2d($hint)->double;
        } else {
            # As a last resort, try current directory
            eval {
                my ($d_loaded, $e_loaded) = load_predictor_maps('.');
                $distance_array_degrees = $d_loaded;
                $expansion_array = $e_loaded if (!defined $expansion_array || ref($expansion_array) ne 'PDL');
            };
            die "[autoload] Failed to autoload predictor maps: $@" if $@;
        }
    }
    if (defined $expansion_array && ref($expansion_array) ne 'PDL') {
        if (-f $expansion_array) {
            print "[autoload] reading expansion CSV: $expansion_array\n";
            $expansion_array = _read_csv_2d($expansion_array)->double;
        } elsif (-d $expansion_array) {
            my (undef, $e_loaded) = load_predictor_maps($expansion_array);
            $expansion_array = $e_loaded;
        } else {
            print "[autoload] WARN: expansion arg is not a PDL and not a path; leaving undefined.\n";
            undef $expansion_array;
        }
    }

    # Check if the start and end    points of the fluxon are open
    my $is_start_open = ($fluxon->{fc_start}->{label} == -1);
    my $is_end_open = ($fluxon->{fc_end}->{label} == -2);
    my $transonic_velocity = 0;

    # If the fluxon is labeled as a plasmoid, or if both ends are open, return undefined
    return undef if($fluxon->{plasmoid} || ($is_start_open + $is_end_open != 1));

    (my $r_vr_scaled, my $r_fr_scaled, my $theta, my $phi) =
            gen_fluxon_wsaflow_phys($fluxon, $distance_array_degrees, $expansion_array, $fluxon_id);

    # Return the final fluxon array, fluxon radius, and magnetic field components
    return ($r_vr_scaled, $r_fr_scaled, $theta, $phi);
}

sub interpolate_2d_lonlat {
    # Bilinear interpolation on a (lat, lon) grid with optional "prefer nonzero" fallback
    # Call as: interpolate_2d_lonlat($image, $long_i, $latt_i, { prefer_nonzero => 1, search_radius => 3 })
    our ($image, $long_i, $latt_i, $opts) = @_;
    $opts ||= {};
    my $prefer_nonzero = $opts->{prefer_nonzero} // 0;
    my $search_radius  = $opts->{search_radius};
    $search_radius = $ENV{WSA_EXP_SEARCH_RADIUS} if defined $ENV{WSA_EXP_SEARCH_RADIUS};
    $search_radius = 3 unless defined $search_radius;

    $image = pdl($image);
    print "Type of image PDL: ", $image->type, "\n";

    # Dimensions: image is (lat, lon)
    my ($img_height, $img_width) = $image->dims;

    # Ensure phi wrapping to [0, 2*pi)
    my $two_pi = 2 * 3.14159265;
    $long_i = fmod($long_i, $two_pi);
    $long_i += $two_pi if $long_i < 0;

    # Map input (long, latt) to fractional pixel coordinates (x, y)
    # x in [0..img_width-1], y in [0..img_height-1] where latt_i in [-1,1]
    my $x = $long_i / $two_pi * ($img_width - 1);
    my $y = ($latt_i + 1) / 2.0 * ($img_height - 1);
    $y = 0 if $y < 0;
    $y = $img_height - 1 if $y > $img_height - 1;

    # Integer neighbors (with longitude wrapping)
    my $x0 = int($x);
    my $y0 = int($y);
    my $x1 = $x0 + 1;
    $x1 = 0 if $x1 >= $img_width;  # wrap
    my $y1 = $y0 + 1;
    $y1 = $img_height - 1 if $y1 >= $img_height;

    my $dx = $x - $x0;
    my $dy = $y - $y0;

    # Sample neighbors
    my $v00 = $image->at($y0, $x0);
    my $v10 = $image->at($y1, $x0);
    my $v01 = $image->at($y0, $x1);
    my $v11 = $image->at($y1, $x1);

    my ($w00,$w10,$w01,$w11) = ((1-$dx)*(1-$dy), (1-$dx)*$dy, $dx*(1-$dy), $dx*$dy);

    my $imval;
    if ($prefer_nonzero) {
        # Ignore zero-valued neighbors (treat as missing) and renormalize weights
        my $sumw = 0.0;
        my $acc  = 0.0;
        if ($v00 != 0) { $acc += $w00 * $v00; $sumw += $w00; }
        if ($v10 != 0) { $acc += $w10 * $v10; $sumw += $w10; }
        if ($v01 != 0) { $acc += $w01 * $v01; $sumw += $w01; }
        if ($v11 != 0) { $acc += $w11 * $v11; $sumw += $w11; }

        if ($sumw > 0) {
            $imval = $acc / $sumw;
        } else {
            # Local nearest non-zero fallback search within a small radius
            my $best_v;
            my $found = 0;
            OUTER: for (my $r = 1; $r <= $search_radius; $r++) {
                my $ymin = $y0 - $r; $ymin = 0 if $ymin < 0;
                my $ymax = $y0 + $r; $ymax = $img_height - 1 if $ymax > $img_height - 1;
                my $xmin = $x0 - $r;
                my $xmax = $x0 + $r;
                for (my $yy = $ymin; $yy <= $ymax; $yy++) {
                    for (my $xx = $xmin; $xx <= $xmax; $xx++) {
                        my $xxw = $xx;
                        # wrap longitude
                        $xxw %= $img_width; $xxw += $img_width if $xxw < 0;
                        my $v = $image->at($yy, $xxw);
                        next if $v == 0;
                        $best_v = $v;
                        $found = 1;
                        last OUTER;
                    }
                }
            }

            if ($found) {
                print "DEBUG: prefer_nonzero fallback used near ($y0,$x0) → $best_v\n";
                $imval = $best_v;
            } else {
                print "DEBUG: prefer_nonzero fallback not found within radius $search_radius at ($y0,$x0); returning 0\n";
                $imval = 0;
            }
        }
    } else {
        # Standard bilinear interpolation
        $imval = $w00*$v00 + $w10*$v10 + $w01*$v01 + $w11*$v11;
    }

    # Debug prints (kept terse)
    print "DEBUG: dims=($img_height,$img_width), ind_latt=$y0, ind_long=$x0\n";
    print "DEBUG: interpolate_2d_lonlat: long_i=$long_i, latt_i=$latt_i, ind_long=$x0, ind_latt=$y0\n";
    print "DEBUG: image stats => min=", $image->min, ", max=", $image->max, ", mean=", $image->avg, "\n";
    print "DEBUG: value at ($y0,$x0)=", $image->at($y0, $x0), "  bilinear=", $imval, "\n";

    return (ref($imval) && $imval->can('sclr')) ? $imval->sclr : $imval;
}

sub gen_fluxon_wsaflow_phys {
    my $me = shift;
    my $distance_array_degrees = shift;
    my $expansion_array = shift;
    my $fluxon_id = shift;
    my $u_opt = shift // {};

    # DIAGNOSTIC PRINTS for $expansion_array
    print "DEBUG: expansion_array type=", ref($expansion_array), ", dims=", (ref($expansion_array) eq 'PDL' ? join('x', $expansion_array->dims) : 'undef'), "\n";
    if (ref($expansion_array) eq 'PDL') {
        my $min = $expansion_array->min;
        my $max = $expansion_array->max;
        my $mean = $expansion_array->avg;
        print "DEBUG: expansion_array stats => min=$min, max=$max, mean=$mean\n";
    } else {
        print "DEBUG: expansion_array is not a PDL object.\n";
    }

    # DEBUG PLOT OF DISTANCE AND EXPANSION ARRAYS
    if ($fluxon_id == 0) {  # only first fluxon -> interactive (now: PNG output)
        eval {
            use PDL::Graphics::Gnuplot;
            my $gp = PDL::Graphics::Gnuplot->new(
                terminal => 'pngcairo size 1200,900',
                output => sprintf("debug_fluxon_%04d.png", $fluxon_id)
            );
            $gp->multiplot({layout => [2,1], title => 'WSA Predictors Debug'});
            $gp->plot({title=>'Distances', with=>'image'}, $distance_array_degrees);
            $gp->replot({title=>'Expansion Factors', with=>'image'}, $expansion_array);
            $gp->end_multi();
            $gp->close;
            # print "Press ENTER to continue after viewing the plot...\n";
            # <STDIN>;
        };
        print "DEBUG plot failed: $@\n" if $@;
    } else {
        eval {
            use PDL::Graphics::Gnuplot;
            my $gp = PDL::Graphics::Gnuplot->new(
                terminal => 'pngcairo size 1000,800',
                output => sprintf("wsa_debug_%04d.png", $fluxon_id)
            );
            $gp->multiplot({layout => [2,1], title => "Fluxon $fluxon_id"});
            $gp->plot({title=>'Distances', with=>'image'}, $distance_array_degrees);
            $gp->replot({title=>'Expansion Factors', with=>'image'}, $expansion_array);
            $gp->end_multi();
            $gp->close;
        };
        print "DEBUG plot failed for $fluxon_id: $@\n" if $@;
    }





    # Define and append optional option input
    my %opt = parse( {
        steps => 500,
        g0 => 280,      # m/s^2
        r0 => 696e6,    # m
        #v0 => 10,       # km/sec
        v0 => rand(10)+10,       # km/sec
        cs => 180,      # km/sec at 2e6K and fully ionized protons
                        },
                        $u_opt
        );

    # Pass options into local variables
    my $g0 = $opt{g0};
    my $r0 = $opt{r0};
    my $cs = $opt{cs} * 1e3; ## convert to m/s from km/s

    # Check for start and ending points being open
    our $en_open = ($me->{fc_end}->{label}==-2);

    # Calculate array of spherical coordinate positions and areas along the fluxon
    # Work along the correct direction depending on which end is open
    if($en_open) {
        my $x = squeeze($me->dump_vecs->(0));
        my $y = squeeze($me->dump_vecs->(1));
        my $z = squeeze($me->dump_vecs->(2));
        our $r1 = ($x**2 + $y**2 + $z**2)->sqrt * $opt{r0};
        our $r = 0.5 * $r1->range([[0],[1]],[$r1->dim(0)],'e')->sumover;
        our $th = acos($z/$r1*$opt{r0});
        our $ph = atan2($y, $x);
        our $A = pdl(map {$_->{A}} ($me->vertices));
        our $bfield = pdl(map {$_->{b_vec}} ($me->vertices));
        our $bmag = pdl(map {$_->{b_mag}} ($me->vertices));
        our $T = pdl(map {$_->{T}} ($me->vertices));
        our $rho = pdl(map {$_->{rho}} ($me->vertices));

    } else {
        my $x = squeeze($me->dump_vecs->(0,-1:0:-1));
        my $y = squeeze($me->dump_vecs->(1,-1:0:-1));
        my $z = squeeze($me->dump_vecs->(2,-1:0:-1));
        our $r1 = ($x**2 + $y**2 + $z**2)->sqrt * $opt{r0};
        our $r = 0.5 * $r1->range([[0],[1]],[$r1->dim(0)],'e')->sumover;
        our $th = acos($z/$r1*$opt{r0});
        our $ph = atan2($y, $x);
        our $A = pdl(map { $_->{A}} ($me->vertices))->(-1:0:-1);
        our $bfield = pdl(map {$_->{b_vec}} ($me->vertices))->(-1:0:-1);
        our $bmag = pdl(map {$_->{b_mag}} ($me->vertices))->(-1:0:-1);
        our $T = pdl(map {$_->{T}} ($me->vertices))->(-1:0:-1);
        our $rho = pdl(map {$_->{rho}} ($me->vertices))->(-1:0:-1);

    }

    our $A;
    our $bfield;
    our $bmag;
    our $r1;
    our $r;


    our $th;
    our $ph;
    our $T;
    our $rho;

    my $rn = $r1/$opt{r0};
    my $zn = $rn - 1;
    my $len = $rn->nelem - 1;

    # Get rid of end anomalies
    # $A(0:-2) .= $A(1:-1);
    # $A->((0)) .= $A->((1));
    # $A->((-1)) .= $A->((-2));

    # Fix the bottom magnetic field vertex
    $bmag->((1)) .= $bmag->((2)) * $A->((2)) / $A->((1));

    ## Declare Variables
    my $vnow;
    my $dr;
    my @rv;
    my $rnow;
    my @num;
    my @denom;
    my $step;
    my $vint;
    my $abserr;
    my $ierr;

    sub wind_speed {
        our ($fss, $theta_b) = @_;

        if (0){
            # Define constants (From Schonfeld 2022)
            my $c1 = 2/9;
            my $c2 = 0.8;
            my $c3 = 4.0; # 2
            my $c4 = 2.0;
            my $c5 = 3.0;

            my $v0 = 285+50; #km/s
            my $vm = 700 - $v0; #km/s
            our $speed = $v0 + ($vm / (1 + $fss)**$c1) * (1 - $c2 * exp(-($theta_b / $c3)**$c4))**$c5;
            # $speed->where($fss == 0) .= 'nan';

        } elsif (0){

            # Define constants (From Wiengarten 2014)
            my $c1 = 2/9;
            my $c2 = 0.8;
            my $c3 = 3.0; # degrees
            my $c4 = 2.0;
            my $c5 = 3.0;

            my $v0 = 200; #km/s
            my $vm = 675; #km/s
            our $speed = $v0 + ($vm / (1 + $fss)**$c1) * (1 - $c2 * exp(-($theta_b / $c3)**$c4))**$c5;
            # $speed->where($fss == 0) .= 'nan';
        } else {
            # Define constants (From McGregor 2011)
            my $c1 = 2/9;
            my $c2 = 0.8;
            my $c3 = 3.8; # phi
            my $c4 = 3.6; # beta
            my $c5 = 3.0;

            my $v0 = 200; #km/s
            my $vm = 750; #km/s
            our $speed = $v0 + ($vm / (1 + $fss)**$c1) * (1 - $c2 * exp(-($theta_b / $c3)**$c4))**$c5;

        }
        our $speed;
        our $fss;

        # print "$fss, $theta_b, $speed\n";

        if ($fss <= 0) {
            $speed = 'nan';
        }
        # $speed->where($fss == 0) .= 'nan';
        return pdl($speed);

    }


    # Interpolate an array given X and Y values along with a test point
    sub interpolate_1d {
        my ($x, $y, $x0) = @_;
        my $y0;
        my $len = $x->nelem;
        my $ind = abs($x - $x0)->minimum_ind;
        if ($ind == 0) {
            $y0 = $y->(($ind));
        } elsif ($ind == $len-1) {
            $y0 = $y->(($ind));
        } else {
            my $x1 = $x->(($ind));
            my $x2 = $x->(($ind+1));
            my $y1 = $y->(($ind));
            my $y2 = $y->(($ind+1));
            $y0 = $y1 + ($y2 - $y1) * ($x0 - $x1) / ($x2 - $x1);
        }
        return $y0;
    }

    # FIND THE SUN SURFACE
    my $first_ind = 1;
    my $r_sun = $rn->(($first_ind));
    my $B_sun = $bmag->(($first_ind));
    my $A_sun = $A->(($first_ind));

    my $phi0 = $ph->(($first_ind));
    my $sin_theta0 = sin($th)->(($first_ind));
    my $r00 = $rn->(($first_ind));

    # FIND THE SOURCE SURFACE
    my $r_ss = 2.5; # Solar radii
    my $B_ss = interpolate_1d($rn, $bmag, $r_ss);
    my $A_ss = interpolate_1d($rn, $A, $r_ss);
    my $f_ss = abs($A_ss / $A_sun) * ($r_sun * $r_sun) / ($r_ss * $r_ss);

    my $f_all= abs($A / $A_sun) * ($r_sun * $r_sun) /   ($rn * $rn);

    # FIND THE TOP OF THE DOMAIN
    my $last_ind = $ph->nelem - 1;
    my $phi1        = $ph->(($last_ind));
    my $r11         = $rn->(($last_ind));
    my $sin_theta1  = sin($th)->(($last_ind));

    if ($phi0 < 0) {$phi0 += 2 * 3.1415926;}
    if ($phi1 < 0) {$phi1 += 2 * 3.1415926;}


    # Find the distance from the coronal hole boundary at that index
    my $distance_degrees = interpolate_2d_lonlat($distance_array_degrees, $phi0, $sin_theta0, { prefer_nonzero => 0 });

    my $expansion_factor = interpolate_2d_lonlat($expansion_array, $phi0, $sin_theta0, { prefer_nonzero => 1 });

    print $expansion_factor, " ", $distance_degrees, "\n";

    # Calculate the wind speed at the given location
    # my $flow_field = wind_speed($An, $distance_degrees);
    # my $speed = interpolate_1d($rn, $flow_field, 5.0);

    my $speed = wind_speed($expansion_factor, $distance_degrees);


    use strict;
    use warnings;
    use PDL;
    use PDL::Graphics::Gnuplot;

    # Sample data
    my $ones = ones($r->dims);

    # # Plot the data with labels
    # my $plot = PDL::Graphics::Gnuplot->new(persist => 1);
    # $plot->plot({title=>$fluxon_id}, {legend=>"phi"},$rn, $ph);
    # $plot->replot({legend=>"phi", with=>"points"},$rn, $ph);
    # $plot->replot(legend=>"ph0", with=>"points", pointsize=>3, $r00, $phi0);
    # $plot->replot(legend=>"ph0", with=>"points", pointsize=>3, $r11, $phi1);

    # $plot->replot({legend=>"stheta"},$rn, sin($th));
    # $plot->replot({legend=>"stheta", with=>"points"},$rn, sin($th));
    # $plot->replot(legend=>"sth0", with=>"points", pointsize=>3 ,$r00, $sin_theta0);
    # $plot->replot(legend=>"sth0", with=>"points", pointsize=>3 ,$r11, $sin_theta1);
    # $plot->replot({legend=>"distance"},$rn, $ones * $distance_degrees);

    # my $plot = PDL::Graphics::Gnuplot->new(persist => 1);
    # my $top_speed = $flow_field->((-1));
    # $plot->plot({title=>"ID: $fluxon_id, Top Speed: $top_speed", logscale => 'xy'}, {legend=>"rn", with=>"points"},$rn, $rn);
    # $plot->replot({legend=>"bmag", with =>'points'},$rn, $bmag);
    # $plot->replot({legend=>"fss"},$rn, $ones * $fss);
    # $plot->replot({legend=>"fr_all"},$rn, $fr_all);
    # $plot->replot({legend=>"An"},$rn, $An);
    # $plot->replot({legend=>"A"},$rn, $A);
    # $plot->replot({legend=>"flow"},$rn, $flow_field);
    # $plot->replot({legend=>"flow", with=>"points"},$rn, $flow_field);
    # $plot->replot({legend=>"flow", with=>"points"},$rn->((-1)), $flow_field->((-1)));


    # Block execution
    # print "Press ENTER to continue...\n";
    # <STDIN>;


    ## Calculate Return Values ##
    my $speed_tall = $ones * $speed;
    # my $speed_tall = $speed;

    # This array is (r, v) in units of (r_sun, km/s)
                                            #To consider plotting:
                                                # $phi0, $sin_theta0, $distance_degrees, $flow_field
    # our $r_v_scaled = pdl($rn, $ones * ($flow_field->at($len)));
    our $r_v_scaled = pdl($rn, $speed_tall);
    # if (1-$en_open) {
    #     # print $r_v_scaled;
    # } else {
    #     our $r_v_scaled = pdl($rn, $ones * $flow_field->at(0));
    #     # print $r_v_scaled;
    # }

    # our $r_v_scaled;
    # This array is (r, fr) in units of (r_sun, unitless)
    my $r_fr_scaled = pdl($rn, $f_all);
    # print $fr_all;

    # Return the constructed arrays
    return ($r_v_scaled, $r_fr_scaled, $th, $ph);
}


use PDL;
use PDL::NiceSlice;

sub gradient_descent {
    my ($image, $latitude, $longitude) = @_;
    my $threshold = 0.1;

    # Convert geographic coordinates to image coordinates
    my $row = ($latitude);   # Assuming latitude and longitude are already in pixel coordinates
    my $col = ($longitude);

    # Check initial value
    my $initial_value = $image->at($row, $col);

    # Determine whether to perform descent or ascent
    my $is_ascent = $initial_value < $threshold;

    # Gradient descent or ascent
    my $max_steps = 500;  # Maximum number of steps to prevent infinite loops
    my $step_size = 1;    # Initial step size in pixels

    for (my $step = 0; $step < $max_steps; $step++) {
        my ($best_value, $best_row, $best_col) = ($initial_value, $row, $col);

        foreach my $r_offset (-$step_size, 0, $step_size) {
            foreach my $c_offset (-$step_size, 0, $step_size) {
                next if $r_offset == 0 && $c_offset == 0;  # Skip the current point

                my $new_row = $row + $r_offset;
                my $new_col = $col + $c_offset;

                # Ensure new coordinates are within bounds
                next if $new_row < 0 || $new_row >= $image->dim(0);
                next if $new_col < 0 || $new_col >= $image->dim(1);

                my $new_value = $image->at($new_row, $new_col);

                # Update best value found depending on descent or ascent
                if (($is_ascent && $new_value > $best_value) || (!$is_ascent && $new_value < $best_value)) {
                    ($best_value, $best_row, $best_col) = ($new_value, $new_row, $new_col);
                }
            }
        }

        # # Dynamically adjust the step size
        # if (abs($best_value - $initial_value) < 0.01) {
        #     $step_size += 1;
        # } else {
        #     $step_size = 1;  # Reset step size to initial value
        # }

        # Check if the best value found crosses the threshold
        if ((!$is_ascent && $best_value < $threshold) || ($is_ascent && $best_value >= $threshold)) {
            my $distance = sqrt(($best_row - $row)**2 + ($best_col - $col)**2);
            return ($best_row, $best_col, $distance);
        }

        # Update row and col for the next iteration
        $initial_value = $best_value;
        ($row, $col) = ($best_row, $best_col);
    }

    # Return NaN if threshold not crossed within max steps
    return ('NaN', 'NaN', 'NaN');
}


use PDL;
use PDL::Graphics::Gnuplot;

sub plot_image_with_points {
    my ($image_path, $original_point, $discovered_point, $fluxon_id) = @_;
    our $fh;
    # print $fluxon_id;
    # Save data and image path to a CSV file
    if ($fluxon_id == 0){
        open $fh, '>', 'data.csv' or die "Cannot open data.csv: $!";
        print $fh "x1,y1,x2,y2,image_path=$image_path\n";
    } else {
        open $fh, '>>', 'data.csv' or die "Cannot open data.csv: $!";
    }
    print $fh "$original_point->[1],$original_point->[0],$discovered_point->[1],$discovered_point->[0]\n";
    close $fh;


    # print "Data saved to data.csv\n";


}

sub do_image_plot{
    # Call the Python script for plotting
    my $python_script = 'plot_distance.py';
    my $exit_status = system("python $python_script");
    if ($exit_status == 0) {
        print "Python script executed successfully.\n";
    } else {
        print "Python script failed to execute.\n";
    }

    # <STDIN>;
}
# our $count = 0;

1;
__END__


    # # # Define a normalized expansion factor
    # my $Arat = $A / $A->((1));
    # my $Rrat = ($r->((1))) * ($r->((1))) / $r / $r;
    # my $Brat = $bmag / $bmag->((1));
    # our $fr1 = $Arat * $Rrat;

    # # Initialize Gnuplot object
    # use PDL::Graphics::Gnuplot;

    # gpwin('pngcairo')->plot(
    #     { title => 'Field vs. r', xlabel => 'r', ylabel => 'Field Value', logscale => 'xy' },
    #     with => 'lines', $r1/$opt{r0}, $Arat,
    #     with => 'lines', $r1/$opt{r0}, $fr1,
    #     # with => 'lines', $r, $r,
    #     with => 'lines', $r1/$opt{r0}, $bmag,
    # );

    # my $ch_map = rfits($ch_map_path);
    # my ($theta_bound, $phi_bound, $distance) = gradient_descent($ch_map, $th0_ind, $ph0_ind);

    # $distance = $distance * 3.14159265358 / 180;
    # plot_image_with_points($ch_map_path, [$th0_ind, $ph0_ind], [$theta_bound, $phi_bound], $fluxon_id);