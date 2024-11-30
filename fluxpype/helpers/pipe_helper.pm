=head1 NAME

pipe_helper - Utility Functions for File and Environment Management

=head1 SYNOPSIS

    use pipe_helper;
    shorten_path($string);
    find_highest_numbered_file($directory);
    set_env_variable($variable, $value);
    # ... and so on

=head1 DESCRIPTION

This Perl module provides utility functions for managing files and environment variables.
It includes functions for shortening file paths, finding the highest-numbered file in a directory,
setting and checking environment variables, and more.

=head1 FUNCTIONS

=head2 configurations

    configurations($adapt, $debug, $config_name, $config_filename);

Reads and processes a configuration file, returning a hash of the configuration settings.

=head2 shorten_path

    shorten_path($string);

Shortens the given file path by replacing the DATAPATH environment variable.

=head2 find_highest_numbered_file

    find_highest_numbered_file($directory);

Finds the highest-numbered file in the given directory.

=head2 set_env_variable

    set_env_variable($variable, $value);

Sets an environment variable to a given value.

=head2 get_env_variable

    get_env_variable($variable);

Gets the value of an environment variable.

=head2 check_env_variable

    check_env_variable($variable, $print);

Checks if an environment variable is set and optionally prints its value.

=head2 set_and_check_env_variable

    set_and_check_env_variable($variable, $value, $print);

Sets an environment variable and then checks if it is set.

=head2 calculate_directories

    calculate_directories($config_ref);

Calculates various directories based on the base directory and batch name.

=head2 configs_update_magdir

    configs_update_magdir($config_ref);

Updates the magnetogram directory in the configuration.

=head2 set_python_path

    set_python_path($pythonpath, $print);

Sets the PYTHONPATH environment variable.

=head2 print_banner

    print_banner($batch_name, $CR, $reduction, $n_fluxons_wanted, $recompute_string);

Prints a banner with various details.

=head2 search_files_in_directory

    search_files_in_directory($directory, $known_string, $extension);

Searches for files in a directory that match a known string and file extension.

=head2 check_second_file_presence

    check_second_file_presence($file_path);

Checks for the presence of a second file related to the given file path.

=head1 AUTHOR

Gilly <gilly@swri.org> (and others!)

=head1 SEE ALSO

L<PDL::AutoLoader>, L<PDL>, L<Time::Piece>

=cut

package pipe_helper;
use strict;
use warnings;
use local::lib;
use Exporter qw(import);
use Flux::World qw(read_world);
our @EXPORT_OK = qw(
    shorten_path
    find_highest_numbered_file
    find_highest_numbered_file_with_string
    set_env_variable
    get_env_variable
    check_env_variable
    configs_update_magdir
    set_and_check_env_variable
    calculate_directories
    set_python_path
    set_paths
    print_banner
    search_files_in_directory
    check_second_file_presence
    configurations
    load_highest_numbered_world
);
use File::Basename qw(dirname basename);
use PDL::AutoLoader;
use PDL;
use Time::Piece;
no warnings 'redefine';
use Scalar::Util qw(looks_like_number);
use List::Util qw(all);
use Config::IniFiles;
use Cwd qw(abs_path cwd);
use File::Spec::Functions qw(catfile catdir rel2abs);
use File::Temp qw(tempfile);
use PDL::Core qw(pdl);

# Shortens the given file path by replacing the DATAPATH environment variable.
sub shorten_path {
    my ($string) = @_;
    my $data_path = $ENV{'DATAPATH'} // '';
    $string =~ s/\Q$data_path\E/~/;
    return $string;
}

# Creates a PDL range
sub pdl_range {
    my ($start, $stop, $step) = @_;
    $step //= 1;  # Default step is 1 if not provided

    # Calculate the number of elements needed
    my $size = int(($stop - $start) / $step);
    $size++ if (($stop - $start) % $step) > 0;

    # Create the sequence
    return $start + $step * sequence($size);
}

# Reads and processes a configuration file
sub configurations {
    my ( $adapt, $debug, $config_name, $config_filename ) = @_;
    $adapt           //= 0;
    $config_name     //= "DEFAULT";
    $config_filename //= "config.ini";
    $debug           //= 0;

    # Locate and validate the configuration file
    my $config_path = find_config_file($config_filename);

    # Clean up the configuration file content
    my $clean_config = clean_config_file($config_path);

    # Read the cleaned configuration file
    my $cfg = Config::IniFiles->new( -file => $clean_config )
      or die "Failed to parse configuration file: $config_path";

    # Determine the configuration section to use
    $config_name = $cfg->val( 'DEFAULT', 'config_name' ) if $config_name eq 'DEFAULT';
    die "Configuration section '$config_name' not found in $config_filename"
      unless $cfg->SectionExists($config_name);

    # Load default and specified section values
    my %the_config = load_config_section( $cfg, 'DEFAULT' );
    %the_config = ( %the_config, load_config_section( $cfg, $config_name ) );

    # Perform additional processing on the configuration
    $the_config{'adapt'}       = $adapt;
    $the_config{'abs_rc_path'} = glob( $the_config{'rc_path'} );

    # Resolve placeholders like ${base_dir}
    my $base_dir = resolve_base_dir($config_path);
    $the_config{'base_dir'} = $base_dir;
    resolve_placeholders( \%the_config, { base_dir => $base_dir } );

    # Convert paths and compute derived values
    $the_config{"run_script"} =
      catfile( $the_config{"fl_mhdlib"}, $the_config{"run_script"} );

    # Parse and ensure correct handling of list or range values
    $the_config{"rotations"}    = parse_list_or_range( $the_config{"rotations"} );
    $the_config{"fluxon_count"} = parse_list_or_range( $the_config{"fluxon_count"} );
    $the_config{"adapts"}       = parse_list_or_range( $the_config{"adapts"} );
    $the_config{"flow_method"}  = parse_list_or_range( $the_config{"flow_method"} );

    # Handle flow_method specifically for non-numeric values
    if ( ref( $the_config{"flow_method"} ) eq 'ARRAY' && scalar @{ $the_config{"flow_method"} } == 1 ) {
        $the_config{"flow_method"} = $the_config{"flow_method"}->[0];
    }

    # Debugging
    if ( ref( $the_config{"flow_method"} ) eq 'ARRAY' ) {
        print "Flow Methods (Array): ", join( ', ', @{ $the_config{"flow_method"} } ), "\n";
    } else {
        print "Flow Method (Scalar): ", $the_config{"flow_method"}, "\n";
    }
    print "Rotations (PDL): ",    $the_config{"rotations"}->string, "\n";
    print "Fluxon Count (PDL): ", $the_config{"fluxon_count"}->string, "\n";
    print "Adapts (PDL): ",       $the_config{"adapts"}->string, "\n";

    # Validate that variables are PDL objects before computing job counts
    die "Rotations is not a PDL object" unless ref( $the_config{'rotations'} ) eq 'PDL';
    die "Fluxon Count is not a PDL object" unless ref( $the_config{'fluxon_count'} ) eq 'PDL';
    die "Adapts is not a PDL object" unless ref( $the_config{'adapts'} ) eq 'PDL';

    # Compute job counts
    $the_config{'n_jobs'} =
      $the_config{'rotations'}->nelem *
      $the_config{'fluxon_count'}->nelem *
      $the_config{'adapts'}->nelem;

    # Calculate directories and update magnetogram paths
    calculate_directories( \%the_config );
    configs_update_magdir( \%the_config );

    # Debugging output
    if ($debug) {
        print_debug_info( \%the_config );
    }

    return %the_config;
}

# Locate the configuration file
sub find_config_file {
    my ($config_filename) = @_;
    my $config_path = rel2abs(catfile( "fluxpype", "fluxpype", "config", $config_filename ));
    return $config_path if -e $config_path;

    die "Configuration file not found: $config_path";
}

# Clean up the configuration file to remove comments and trailing whitespace
sub clean_config_file {
    my ($config_path) = @_;
    my ( $fh, $temp_filename ) = tempfile();
    open( my $in, '<', $config_path )
      or die "Could not open '$config_path' for reading: $!";
    while (<$in>) {
        s/#.*$//;    # Remove inline comments
        s/\s+$//;    # Remove trailing whitespace
        print $fh "$_\n";
    }
    close $in;
    close $fh;
    return $temp_filename;
}

# Load configuration values from a specific section
sub load_config_section {
    my ( $cfg, $section ) = @_;
    my %config = ();
    for my $key ( $cfg->Parameters($section) ) {
        $config{$key} = $cfg->val( $section, $key );
    }
    return %config;
}

# Resolve the base directory dynamically
sub resolve_base_dir {
    my ($config_path) = @_;
    return abs_path( File::Spec->catdir( dirname($config_path), "..", ".." ) );
}

# Parse a list or range and return a PDL object or array reference
sub parse_list_or_range {
    my ($value) = @_;

    # Check for a list
    if ($value =~ /^\[/) {
        $value =~ s/[\[\]]//g;  # Remove brackets
        my @list = split(/\s*,\s*/, $value);

        # Check if all elements are numeric
        if (all { looks_like_number($_) } @list) {
            return PDL->new(@list);  # Return as PDL object for numeric values
        } else {
            return \@list;  # Return as array reference for non-numeric values
        }
    }
    # Check for a range
    elsif ($value =~ /^\(/) {
        $value =~ s/[\(\)]//g;  # Remove parentheses
        my ($start, $stop, $step) = split(/\s*,\s*/, $value);

        # Ensure numeric values
        if (looks_like_number($start) && looks_like_number($stop) && looks_like_number($step)) {
            return PDL->sequence(int(($stop - $start) / $step + 1))->multiply($step)->plus($start);
        } else {
            die "Invalid range values: start, stop, and step must be numeric.";
        }
    }
    # Single scalar value
    else {
        if (looks_like_number($value)) {
            return PDL->new($value);  # Return numeric as PDL
        } else {
            return $value;  # Return non-numeric as-is
        }
    }
}

# Resolve placeholders in configuration values
sub resolve_placeholders {
    my ($config, $placeholders) = @_;
    foreach my $key (keys %{$config}) {
        if (ref $config->{$key} eq '') {  # Only process scalar values
            my $value = $config->{$key};
            foreach my $placeholder (keys %{$placeholders}) {
                my $replacement = $placeholders->{$placeholder};
                $value =~ s/\$\{$placeholder\}/$replacement/g;
            }
            $config->{$key} = $value;
        }
    }
}

# Debugging function
sub print_debug_info {
    my ($config_ref) = @_;
    print "\nConfiguration Values:\n-----------------------------\n";
    for my $key ( sort keys %$config_ref ) {
        print "$key: $config_ref->{$key}\n";
    }
    print "-----------------------------\n\n";
}

=begin comment

Add the missing 'configs_update_magdir' function back into the module.

=end comment

=head2 configs_update_magdir

    configs_update_magdir($config_ref);

Updates the magnetogram directory in the configuration.

=cut

sub configs_update_magdir {
    my ($config_ref) = @_;

    # Update the 'mag_dir' in the configuration
    my $mag_dir = catdir( $config_ref->{'data_dir'}, 'magnetograms' );
    $config_ref->{'mag_dir'} = $mag_dir;
}

=head2 find_highest_numbered_file

Finds the highest-numbered file in the given directory.

=cut

sub find_highest_numbered_file {
    my ($directory) = @_;
    opendir( my $dir_handle, $directory ) or die "Cannot open directory: $!";
    my @files = grep { !/^\.{1,2}$/ } readdir($dir_handle);
    closedir $dir_handle;

    my $highest_numbered_file;
    my $highest_number = -1;
    for my $file_name (@files) {

        # Match file names containing "_relaxed_s", a number, and ".flux"
        if ( $file_name =~ /_relaxed_s(\d+)\.flux$/ ) {
            my $number = $1;
            if ( $number > $highest_number ) {
                $highest_number        = $number;
                $highest_numbered_file = $file_name;
            }
        }
    }
    return $highest_numbered_file ? "$directory/$highest_numbered_file" : undef,
      $highest_number;
}

sub find_highest_numbered_file_with_string {
    my ($directory, $search_string) = @_;
    opendir(my $dir_handle, $directory) or die "Cannot open directory $directory: $!";
    my @files = grep { !/^\.{1,2}$/ } readdir($dir_handle);
    closedir $dir_handle;

    my $highest_numbered_file;
    my $highest_number = -1;
    for my $file_name (@files) {

        # Match file names containing the search string, a number, and ".flux"
        if ( $file_name =~ /\Q$search_string\E(\d+)\.flux$/ ) {
            my $number = $1;
            if ( $number > $highest_number ) {
                $highest_number        = $number;
                $highest_numbered_file = $file_name;
            }
        }
    }
    return $highest_numbered_file ? "$directory/$highest_numbered_file" : undef,
      $highest_number;
}

=head2 set_env_variable

Sets an environment variable to a given value.

=cut

sub set_env_variable {
    my ( $variable, $value ) = @_;
    $ENV{$variable} = $value;
    return $ENV{$variable};
}

=head2 get_env_variable

Gets the value of an environment variable.

=cut

sub get_env_variable {
    my ($variable) = @_;
    return $ENV{$variable};
}

=head2 check_env_variable

Checks if an environment variable is set and optionally prints its value.

=cut

sub check_env_variable {
    my ( $variable, $print ) = @_;
    my $value = $ENV{$variable};
    if ( defined $value ) {
        if ( defined $print ) {
            if ($print) {
                print "\$$variable: \t$value\n";
            }
        }
        return $value;
    }
    else {
        print "\$$variable is not set\n";
        exit();
    }
}

=head2 set_and_check_env_variable

Sets an environment variable and then checks if it is set.

=cut

sub set_and_check_env_variable {
    my ( $variable, $value, $print ) = @_;
    set_env_variable( $variable, $value );
    return check_env_variable( $variable, $print );
}

=head2 calculate_directories

Calculates various directories based on the base directory and batch name.

=cut

sub calculate_directories {
    my ($config_ref) = @_;

    my $data_dir   = $config_ref->{'data_dir'};
    my $batch_name = $config_ref->{'batch_name'};

    $batch_name =~ s/^\s+|\s+$//g;
    if ( $config_ref->{'adapt'} && index( $batch_name, "adapt" ) == -1 ) {
        $batch_name = $batch_name . "_adapt";
    }

    my $pipedir = catdir( "fluxpype", "fluxpype" );
    my $pdldir  = catdir( "pdl", "PDL" );

    my $magdir   = catdir( $data_dir, "magnetograms" );
    my $batchdir = catdir( $data_dir, "batches", $batch_name );
    my $logfile  = catfile( $batchdir, "pipe_log.txt" );

    # Remove the tilde from the path
    my $home_dir = $ENV{'HOME'};
    $data_dir  =~ s{^~}{$home_dir};
    $pdldir    =~ s{^~}{$home_dir};
    $magdir    =~ s{^~}{$home_dir};
    $batchdir  =~ s{^~}{$home_dir};
    $logfile   =~ s{^~}{$home_dir};

    # Update the original config hash
    $config_ref->{'pipe_dir'}  = $pipedir;
    $config_ref->{'pdl_dir'}   = $pdldir;
    $config_ref->{'datdir'}    = $data_dir;
    $config_ref->{'data_dir'}  = $data_dir;
    $config_ref->{'mag_dir'}   = $magdir;
    $config_ref->{'batch_dir'} = $batchdir;
    $config_ref->{'logfile'}   = $logfile;

    set_and_check_env_variable( 'DATAPATH', $data_dir, 0 );
}

=head2 set_python_path

Sets the PYTHONPATH environment variable.

=cut

sub set_python_path {
    my ( $pythonpath, $print ) = @_;
    set_and_check_env_variable( 'PYTHONPATH', $pythonpath, $print );
    return $pythonpath;
}

=head2 print_banner

Prints a banner with various details.

=cut

sub print_banner {
    my ( $batch_name, $CR, $reduction, $n_fluxons_wanted, $recompute_string,
        $adapt, $flow_method )
      = @_;
    print "\n\n\n\n\n\n|\n|\n|\n|\n|\n|\n|\n|\n|\n|";
    print "\n\n";
    print
"--------------------------------------------------------------------------------------------------\n";
    print
"FLUXPype: Indicate a Carrington Rotation and this script will run the entire Flux Pipeline for it.\n";
    print
"--------------------------------------------------------------------------------------------------\n";
    print "\n\n";

    check_env_variable( 'DATAPATH', 1 );
    print
"\nBatch: $batch_name, CR: $CR, Reduction: $reduction, Fluxons: $n_fluxons_wanted, Adapt: $adapt, Wind: $flow_method\n";

    my $time  = localtime;
    my $ftime = $time->strftime('%m-%d-%Y %H:%M:%S');

    print "\n\n";
    print
"\t>>>>>>>>>>>>>>>>>>>>> Recompute = $recompute_string <<<<<<<<<<<<<<<<<<<<<<";
    print "\n\tStarting FLUXPype at $ftime ";
    print
      "\n\t>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
    print "\n\n";
    print
"--------------------------------------------------------------------------------------------------\n";
    return 1;
}

=head2 search_files_in_directory

Searches for files in a directory that match a known string and file extension.

=cut

sub search_files_in_directory {
    my ( $directory, $known_string, $extension ) = @_;

    # Escape the known string to avoid regex special characters
    my $escaped_string = quotemeta $known_string;

    # Generate the regular expression pattern
    my $pattern = $escaped_string . '.*' . $extension;

    opendir( my $dh, $directory ) or die "Failed to open directory: $!";
    while ( my $file = readdir($dh) ) {
        next if ( $file =~ /^\./ );    # Skip hidden files/directories
        next unless ( $file =~ /$pattern/ );
        print "$file\n";               # Process the matching file
        closedir($dh);
        return $file;
    }
    closedir($dh);
    return undef;
}

=head2 check_second_file_presence

Checks for the presence of a second file related to the given file path.

=cut

sub check_second_file_presence {
    my ($file_path) = @_;

    # Get the directory and base name of the first file
    my $directory = dirname($file_path);
    my $file_name = basename($file_path);

    # Generate the pattern for the second file
    my $second_file_pattern = $file_name;
    print "    File name: $file_name\n";
    $second_file_pattern =~ s/(\.[^.]+)$/_relaxed_.*${1}/;

    opendir( my $dh, $directory ) or die "Failed to open directory: $!";
    while ( my $file = readdir($dh) ) {
        next if ( $file =~ /^\./ );       # Skip hidden files/directories
        next if ( $file =~ /\.png$/ );    # Skip png files

        if ( $file =~ /^$second_file_pattern$/ ) {
            closedir($dh);
            return (1, $file);            # Second file found
        }
    }
    closedir($dh);

    return (0, undef);    # Second file not found
}

# Check if the environment variable is set
sub set_paths {
    my ($do_plot) = @_;
    if ( defined $ENV{'FL_PREFIX'} ) {
        my $envpath =
          "$ENV{'FL_PREFIX'}/fluxpype/fluxpype/helpers/perl_paths.pm";

        # Check if the file exists and is readable
        if ( -e $envpath && -r _ ) {
            require $envpath;
            print "\n\n";
        }
        else {
            warn "File does not exist or is not readable: $envpath\n\n";
        }
    }
    else {
        warn "Environment variable FL_PREFIX is not set.\n\n";
    }

    # Print the lists of directories
    if ($do_plot) {
        my @PDL_INC;
        my @PDLLIB;
        print "\n\nINC has:\n ";
        print map { " $_\n" } @INC;
        print "--------------------------\n";

        print "\nPDL_INC has:\n ";
        print map { " $_\n" } @PDL_INC;
        print "--------------------------\n";

        print "\nPDLLIB has:\n ";
        print map { " $_\n" } @PDLLIB;
        print "--------------------------\n\n";

        # Print each command-line argument
        foreach my $arg (@ARGV) {
            print "Argument: $arg\n";
        }
        print "\n\n";
    }
    return;
}

use File::Spec;

sub load_highest_numbered_world {
    my ( $datdir, $batch_name, $CR, $n_fluxons_wanted, $inst ) = @_;

    my $world_out_dir = File::Spec->catdir(
        $datdir, "batches", $batch_name, "data", "cr${CR}", "world"
    );
    my $file_pattern =
      qr/cr${CR}_f${n_fluxons_wanted}_${inst}_relaxed_s(\d+)\.flux$/;
    my $original_pattern = qr/cr${CR}_f${n_fluxons_wanted}_${inst}\.flux$/;

    my $max_d               = -1;
    my $selected_file_path  = undef;
    my $original_file_path  = undef;
    print "Searching for files in $world_out_dir\n";

    opendir( my $dh, $world_out_dir ) or die "Cannot open directory: $!";
    while ( my $file = readdir($dh) ) {
        if ( $file =~ /$file_pattern/ ) {
            my $d_value = $1;
            if ( $d_value > $max_d ) {
                $max_d              = $d_value;
                $selected_file_path = File::Spec->catfile( $world_out_dir, $file );
            }
        }
        if ( $file =~ /$original_pattern/ ) {
            $original_file_path = File::Spec->catfile( $world_out_dir, $file );
        }
    }
    closedir($dh);

    print "Selected file: $selected_file_path\n";
    print "Original file: $original_file_path\n";

    if ( defined $selected_file_path && defined $original_file_path ) {
        my $this_world_relaxed  = read_world($selected_file_path);
        my $this_world_original = read_world($original_file_path);
        my @fluxons             = $this_world_relaxed->fluxons;

        if ( scalar @fluxons == 0 ) {
            die "World loaded, but contains no fluxons.";
        }

        return ( $this_world_relaxed, $this_world_original );    # Successful load
    }
    else {
        die "No matching files found.";                          # No file found
    }
}

1;    # End of module