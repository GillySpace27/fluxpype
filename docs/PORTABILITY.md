# Portable install: running record

Everything that had to change, or still has to change, in the instructions and tooling so that
another user can get FLUX and fluxpype running on a fresh machine. Append new items as they are
found; do not delete resolved ones, mark them instead.

Evidence tags: **verified** = observed in a command or log on Gilly's Mac; **proposal** = a
suggested change, not yet agreed or implemented; **open** = not investigated.

First entries come from the 2026-11-10/11 session that got the `quickstart` profile (CR 2225,
100 fluxons, parker) running under Plumber on macOS arm64 (conda `fluxenv` Perl 5.32.1, Python 3.12).

## Done

| # | Item | Status |
|---|------|--------|
| 1 | `flux-doctor` did not check `Inline::C` / `Inline::Pdlpp`. Without `Inline::C`, `hilbert.pm` fails with `Base class package "Inline::C" is empty`, and the log's "module get_hilbert_footpoints not found" points at the wrong module. | **Done**: `fluxpype/doctor.py` now checks both and prints the fix. Tested against the `fluxenv` Perl (ok) and `/usr/bin/perl` (FAIL, `Inline::Pdlpp` missing). Not covered: a compile smoke test. |

## To change in the instructions

| # | Item | Evidence | Change needed |
|---|------|----------|---------------|
| 2 | The Perl module list is incomplete. The FLUX core needs `Inline::C` and `Inline::Pdlpp` plus a C compiler (`Config{cc}` of the conda Perl is `arm64-apple-darwin20.0.0-clang`). | verified | Add them to the core build guide (fluxon-mhd/doc) and to the README step 1 completion test, e.g. `perl -e 'require Inline::C; require Inline::Pdlpp'`. |
| 3 | `cpanm` may belong to a different Perl than `perl` on PATH. Here `fluxenv/bin/cpanm` has the shebang `#!/Users/cgilbert/miniconda3a/envs/fluxenv/bin/perl`, so `cpanm Inline::C` reported "up to date" and later installed into a different directory from the one the jobs use. | verified | Document `perl $(which cpanm) Module`, or `perl -MCPAN`, so modules land in the active Perl. The doctor fix text already says this. |
| 4 | The README says `./install.sh` creates `./.venv`, but Plumber and the MCP server run jobs with `.venv-plumber`, and no `.venv` exists in this checkout. | verified | Say which venv Plumber uses and how it is created, or make Plumber use `./.venv`. |
| 5 | `.venv-plumber` contained `fluxpype` plus the Plumber and MCP packages only. Jobs run with that interpreter and failed with `No module named 'tqdm'`, then `rich`, then `matplotlib`. | verified | Create it with the full dependencies (`pip install -e ".[plumber,mcp]"`), not just the extras. |
| 6 | `requirements.txt` leaves `numpy` unpinned; `pyproject.toml` pins `numpy<2.0`. Installing `tqdm pandas` alone pulled `numpy` 2.5.4 and broke the pin. | verified | One source of truth: install from `pyproject.toml`, and delete or generate `requirements.txt`. |
| 7 | Shell state leaks into jobs. `PERL5LIB` points at `/Users/cgilbert/miniconda3a/...`, and `Flux.pm` is found there. `PERL5OPT=-Mlib=/opt/homebrew/opt/openssl@3/lib` sits first in `@INC`. `flux-doctor` already warns about stale `PERL5LIB`. | verified (values); effect of `PERL5OPT` unknown | Tell users to check `env \| grep PERL` and `perl -V:...` before building. |
| 8 | Two copies of the same env exist (`/Users/gilly/miniconda3a/envs/fluxenv` and `/Users/cgilbert/miniconda3a/envs/fluxenv`). | verified | Tell users to build the core in exactly one env and put its `bin` first on PATH. |

## Proposals

| # | Item | Change |
|---|------|--------|
| 9 | Plumber `--install` records the installing shell's `PATH` and `PERL5LIB` (README line 87), so the Perl a job uses is whatever that shell had. | Add an explicit `flux_perl` setting to `config.ini` or the Plumber home, and show it in `flux-doctor`. |
| 10 | `install.sh` installs only the Python half and says to build the core separately. | Optional `--core` flag that creates a conda env with Perl, PDL, gnuplot and a compiler, `cpanm`s the Inline modules, and runs `flux-doctor`. |
| 11 | `flux-doctor` reports a missing Python module as `config.ini: No module named 'pandas'`, which hides the real problem. | Add a check that imports the runtime dependencies, with its own label. |

## Open

| # | Item | Notes |
|---|------|-------|
| 12 | `pip check` reports `fluxpype 0.2.0 requires astropy<7.0, but you have astropy 7.2.3`. `sunpy` 8.0.1 was installed alongside it. The job ran fine. | Find where the `astropy<7.0` pin comes from and whether it can be loosened. |
| 13 | The wind stage logs read data under `/Users/gilly/vscode/fluxons/fluxpype/...`, not the `fluxons-plumber` checkout. | Find which setting (`base_dir`, `data_dir`?) is machine-specific. |
| 14 | The log had one `(in cleanup) PDL::Graphics::Gnuplot: subproc died` line per forked wind worker (51 per run), plus a `Read '...world.flux'` pair per worker. Cause: workers destroyed inherited world and Gnuplot objects at exit. Fixed in `map_fluxon_flow_parallel_master.pm` (workers `_exit`); log went from 461 to 305 lines. | Done; no plots were failing. One Gnuplot line may remain from the parent. |
| 15 | PyPI downloads timed out on this connection until `pip install --default-timeout 120 --retries 8` was used. | Environment-specific; mention in troubleshooting only. |
| 16 | The `fluxlight` stage (profile `light`) fails with `ModuleNotFoundError: No module named 'joblib'`. `thomson_scattering_flux.py` imports it, but `requirements.txt` and `pyproject.toml` do not declare it. The task still ends `ok` with `failure: process exited with status 1`, so the failure is easy to miss. | Fixed: `joblib` is now in `requirements.txt` and `pyproject.toml`. Still open: make a failed stage fail the task, or at least show it in the UI. |
| 17 | 2026-11-11: with `joblib` installed, the `fluxlight` stage failed on the next import, `ModuleNotFoundError: No module named 'sunkit_image'` (`fluxlight.py` line 13, job `j8fa25bd6`). Cause: `config_runner.py` puts the directory of its own interpreter first on `PATH`, so the Perl stage's bare `python3` is `.venv-plumber/bin/python3`, and that venv had only the declared dependencies. `sunkit-image` was imported but never declared. | Fixed: `sunkit-image` added to `requirements.txt` and `pyproject.toml` and installed in `.venv-plumber` (0.7.0). An AST scan of every import in `fluxpype/*.py`, `science/*.py` and `plotting/*.py` against `.venv-plumber` now finds nothing missing (47 top-level modules). Still open: `flux-doctor` does not run that scan (item 11). |
