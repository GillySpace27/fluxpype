"""Entry point used by the runner tests: real config_runner, fake science stack."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_science  # noqa: E402

fake_science.install()

from fluxpype.config_runner import run  # noqa: E402

if __name__ == "__main__":
    run()
