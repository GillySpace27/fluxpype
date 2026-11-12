"""flux_perl picks the Perl every job stage runs; it is an executed path, so a per-job override may not set it."""

import os

from fluxpype.config_runner import _perl
from fluxpype.plumber.config_io import FORBIDDEN_KEYS


def test_perl_defaults_to_path_and_expands_a_configured_path():
    assert _perl({}) == "perl"
    assert _perl({"flux_perl": ""}) == "perl"
    assert _perl({"flux_perl": "~/x/perl"}) == os.path.expanduser("~/x/perl")


def test_flux_perl_cannot_be_overridden_per_job():
    assert "flux_perl" in FORBIDDEN_KEYS
