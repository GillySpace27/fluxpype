"""
Light configuration reader for FLUXpype (audit A5-02).

Holds ``configurations`` and its helpers with only standard-library imports (plus the small
``fluxpype.console``), so scripts that only need the config skip the heavy science imports of
``fluxpype.pipe_helper`` (sunpy.net, matplotlib.pyplot, pandas). ``pipe_helper`` re-exports every name
defined here, so ``from fluxpype.pipe_helper import configurations`` keeps working.
"""

import ast
import configparser
import os

from fluxpype.console import emit

# CONFIGURATION MANAGEMENT #######################################################


def configurations(config_name=None, config_filename="config.ini", args=None, debug=False):
    """
    Reads and sanitizes configuration settings from a specified config file.

    Args:
        config_name (str, optional): The specific configuration section to read.
                                     Defaults to the 'DEFAULT' section.
        config_filename (str, optional): The filename of the configuration file.
                                         Defaults to 'config.ini'.
        args (Namespace, optional): Command-line arguments to override config values.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        dict: Configuration settings as key-value pairs.
    """
    # Load configuration file
    config_path = find_config_file(config_filename)
    config_obj = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

    with open(config_path, "r") as f:
        clean_content = clean_config_content(f.readlines())
    config_obj.read_string(clean_content)

    # Determine the configuration section to use
    # FLUXPYPE_CONFIG (set by flux-config-run --config) wins over the file's config_name
    config_name = (
        config_name or os.environ.get("FLUXPYPE_CONFIG") or config_obj["DEFAULT"].get("config_name", "DEFAULT")
    )
    if config_name not in config_obj:
        available_sections = ", ".join(config_obj.sections())
        raise ValueError(
            f"Configuration section '{config_name}' not found in {config_filename}. "
            f"Available sections: {available_sections}"
        )

    # base_dir = auto (or empty): the folder that holds this fluxpype checkout,
    # so a fresh clone works wherever it lives. Set before any ${base_dir} is interpolated.
    if config_obj["DEFAULT"].get("base_dir", "").strip() in ("", "auto"):
        config_obj["DEFAULT"]["base_dir"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Create the configuration dictionary
    the_config = dict(config_obj[config_name])
    the_config["config_name"] = config_name  # the active profile, not the file default

    # Override with command-line arguments
    assimilate_args(the_config, args)

    # Resolve and compute configuration settings
    base_dir = resolve_base_dir(config_filename)
    the_config.update({"base_dir": base_dir})
    the_config = resolve_placeholders(the_config, {"base_dir": base_dir})

    # Process directories and derived values
    the_config.update(calculate_directories(the_config))
    if "batch_dir" not in the_config:
        raise KeyError("batch_dir is missing after calculate_directories.")
    update_magdir_paths(the_config)
    compute_configs(the_config)

    # Final type conversion
    the_config = {key: convert_value(value) for key, value in the_config.items()}

    # Debugging output
    if debug:
        print_debug_info(the_config)

    return the_config


def find_config_file(config_filename):
    """Find the configuration file: as given (absolute or relative to cwd), else the one
    shipped beside this module. The Perl side (find_config_file in pipe_helper.pm) also
    reads the shipped one, so both halves of a run see the same file from any cwd.

    FLUXPYPE_CONFIG_FILE, when set, wins over everything (the plumber uses it to freeze a
    per-job snapshot); if it is set but missing that is an error, not a silent fallback."""
    override = os.environ.get("FLUXPYPE_CONFIG_FILE")
    if override:
        override = os.path.expanduser(override)
        if not os.path.exists(override):
            raise FileNotFoundError(f"FLUXPYPE_CONFIG_FILE points to a missing file: {override}")
        return override
    config_path = os.path.expanduser(config_filename)
    if os.path.exists(config_path):
        return config_path

    # ponytail: was os.walk(os.getcwd()), which from ~ crawls the whole home dir
    # and can pick up an unrelated config.ini
    shipped = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(config_filename))
    if os.path.exists(shipped):
        return shipped

    raise FileNotFoundError(f"Configuration file '{config_filename}' not found.")


def clean_config_content(lines):
    """Remove comments and trailing whitespace from configuration file content."""
    return "\n".join(line.split("#", 1)[0].strip() for line in lines if line.strip())


def resolve_base_dir(config_path=None):
    """Determine the base directory dynamically."""
    return os.path.abspath(os.path.dirname(config_path)) if config_path else os.getcwd()


def assimilate_args(config, args=None):
    """Override configuration values with command-line arguments, if provided."""
    if args:
        for arg, value in vars(args).items():
            if value is not None:
                config[arg] = value


def resolve_placeholders(config, placeholders):
    """Resolve placeholders like ${base_dir} in the configuration dictionary."""
    for key, value in config.items():
        if isinstance(value, str):
            for placeholder, replacement in placeholders.items():
                value = value.replace(f"${{{placeholder}}}", replacement)
        config[key] = value
    return config


def calculate_directories(config):
    """Calculate and update directory paths in the configuration."""
    basedir = os.path.expanduser(config.get("base_dir", "").strip())
    batch_name = config.get("batch_name", "").strip()
    dat_dir = config.get("data_dir", os.path.join(basedir, "fluxpype", "data"))

    batch_dir = os.path.join(dat_dir, "batches", batch_name)

    return {
        "batch_dir": os.path.expanduser(batch_dir),
        "pipe_dir": os.path.join(basedir, "fluxpype", "fluxpype"),
        "pdl_dir": os.path.join(basedir, "pdl", "PDL"),
        "datdir": os.path.expanduser(dat_dir),
        "data_dir": os.path.expanduser(dat_dir),
        "mag_dir": os.path.join(dat_dir, "magnetograms"),
        "logfile": os.path.join(batch_dir, "pipe_log.txt"),
    }


def update_magdir_paths(config):
    """Update paths related to magnetograms and fluxons."""
    CR = config.get("cr", config.get("rotations", [2100])[0])
    n_fluxons_wanted = config.get("fluxon_count")
    if not CR or not n_fluxons_wanted:
        raise ValueError("Instance values 'cr' or 'fluxon_count' not found in configuration.")

    adapt_select = config.get("adapt_select", 0)
    reduction = config.get("mag_reduce", 1)

    if config.get("adapt", False):
        magfile = f"CR{CR}_rf{adapt_select}_adapt.fits"
        flocfile = f"floc_cr{CR}_rf{adapt_select}_f{n_fluxons_wanted}_adapt.dat"
    else:
        magfile = f"CR{CR}_r{reduction}_hmi.fits"
        flocfile = f"floc_cr{CR}_r{reduction}_f{n_fluxons_wanted}_hmi.dat"

    config.update(
        {
            "magfile": magfile,
            "flocfile": flocfile,
            "magpath": os.path.join(config["mag_dir"], magfile),
        }
    )
    return config


def compute_configs(config):
    """Process and compute additional configuration settings."""
    config["abs_rc_path"] = os.path.expanduser(config.get("rc_path", ""))
    config["abs_fl_mhdlib"] = os.path.expanduser(config.get("fl_mhdlib", ""))

    # Parse lists or ranges
    config["rotations"] = parse_list_or_range(config.get("rotations", "[]"))
    config["flow_method"] = parse_list(config.get("flow_method", "[]"))
    config["fluxon_count"] = parse_list(config.get("fluxon_count", "[]"))
    config["adapts"] = parse_list(config.get("adapts", "[]"))

    # Compute derived values
    # config["cr"] = config["rotations"][0]
    # config["nwant"] = config["fluxon_count"][0]
    config["n_jobs"] = str(
        len(config["rotations"]) * len(config["fluxon_count"]) * len(config["adapts"]) * len(config["flow_method"])
    )


def parse_list_or_range(value):
    """Parse a list or range-like string into a Python list."""
    if value.startswith("[") and value.endswith("]"):  # Parse list
        value = value.strip("[]")
        return [item.strip() for item in value.split(",")]
    elif value.startswith("(") and value.endswith(")"):  # Parse range
        start, stop, step = map(int, value.strip("()").split(","))
        return [str(x) for x in range(start, stop, step)]
    return [value.strip()]  # Single value as list


def parse_list(value):
    """Parse a string into a list."""
    the_list = value.strip("[]").split(",") if value else []
    return [x.strip(" ").casefold() for x in the_list]


def convert_value(value):
    """Convert configuration values to appropriate types."""
    # If the value is already a type other than string, return it as-is
    if not isinstance(value, str):
        return value

    # Literals only (numbers, strings, lists, tuples, dicts, True/False/None); anything else stays a string.
    # Was eval(), which ran arbitrary expressions from config values and pipeline output.
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def get_flux_perl_lib(config):
    """Return the directory that contains Flux.pm for PERL5LIB, or None if not found.

    Looks for fluxon-perl under base_dir, or under fl_prefix, and finds the
    architecture-specific lib dir (e.g. darwin-2level) that contains Flux.pm.
    Ignores config paths that are files (e.g. perl binary) or another user's home
    (avoids FLUX/relaxation failures from wrong-user paths).
    """
    my_home = os.path.expanduser("~")
    # Explicit config path (e.g. flux_perl_lib = ${base_dir}/../fluxon-perl/lib/perl5/darwin-2level)
    for key in ("flux_perl_lib", "fl_perl_lib"):
        path = config.get(key)
        if path:
            path = os.path.normpath(os.path.expanduser(str(path).strip()))
            # Skip if this is a file (e.g. mistaken perl binary path) or another user's path
            if os.path.isfile(path):
                continue
            if (path.startswith("/Users/") or path.startswith("/home/")) and not path.startswith(my_home):
                # Path is under another user's home (e.g. /Users/cgilbert/... when we're gilly)
                continue
            if os.path.isfile(os.path.join(path, "Flux.pm")):
                return path
            if os.path.isdir(path):
                for name in os.listdir(path):
                    sub = os.path.join(path, name)
                    if os.path.isfile(os.path.join(sub, "Flux.pm")):
                        return sub
    # Under base_dir or its parents: fluxon-perl/lib/perl5/<arch> (e.g. fluxons/fluxon-perl when cwd is fluxpype)
    basedir = os.path.expanduser(config.get("base_dir", "").strip())
    if basedir:
        base_candidates = [basedir, os.path.dirname(basedir), os.path.dirname(os.path.dirname(basedir))]
        for root in base_candidates:
            if not root or root == os.path.dirname(root):
                continue
            prefix = os.path.join(root, "fluxon-perl", "lib", "perl5")
            if os.path.isdir(prefix):
                if os.path.isfile(os.path.join(prefix, "Flux.pm")):
                    return prefix
                for name in os.listdir(prefix):
                    sub = os.path.join(prefix, name)
                    if os.path.isfile(os.path.join(sub, "Flux.pm")):
                        return sub
    # Under fl_prefix
    fl_prefix = os.path.expanduser(config.get("fl_prefix", "").strip())
    if fl_prefix:
        for prefix in (
            os.path.join(fl_prefix, "lib", "perl5"),
            os.path.join(fl_prefix, "lib", "perl5", "darwin-2level"),
        ):
            if os.path.isdir(prefix) and os.path.isfile(os.path.join(prefix, "Flux.pm")):
                return prefix
            if os.path.isdir(prefix):
                for name in os.listdir(prefix):
                    sub = os.path.join(prefix, name)
                    if os.path.isfile(os.path.join(sub, "Flux.pm")):
                        return sub
    return None


def print_debug_info(config):
    """Print debug information for the configuration."""
    emit("\nDebug Configuration Values:", style="dim")
    for key, value in sorted(config.items()):
        emit(f"  {key}: {value}", style="dim")
