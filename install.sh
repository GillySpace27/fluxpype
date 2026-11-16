#!/usr/bin/env bash
# Install fluxpype into ./.venv and verify the whole FLUX stack with flux-doctor.
# The FLUX core (C library + Perl/PDL) is built separately: see fluxon-mhd/doc.
#   ./install.sh                       # picks python3.12, 3.11 or 3.10
#   PYTHON=/path/to/python ./install.sh
set -euo pipefail
cd "$(dirname "$0")"

ok_py() { "$1" -c 'import sys; sys.exit(not (3,10) <= sys.version_info[:2] <= (3,12))' 2>/dev/null; }
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null && ok_py "$c"; then PY=$c; break; fi
  done
fi
if [ -z "$PY" ] || ! ok_py "$PY"; then
  echo "fluxpype needs Python 3.10-3.12 (numpy<2 has no wheels past 3.12)."
  echo "Install one (macOS: brew install python@3.12) or set PYTHON=/path/to/python."
  exit 1
fi

echo "Using $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[FLUXcore]"

# Verify in a fresh login shell: that is where stale PERL5LIB lines in rc files break Flux.
echo; echo "Checking the stack (flux-doctor, in a login shell):"
"${SHELL:-/bin/bash}" -lc "PATH=\"$PWD/.venv/bin:\$PATH\" flux-doctor"

echo
echo "Installed. Each session:  source $PWD/.venv/bin/activate"
echo "First run:                flux-config-run --config quickstart"
