# Fluxpype Plumber: implementation plan (v1)


Renamed 2026-11-10 from "FLUX Studio" (`flux-studio`) to "Fluxpype Plumber" (`flux-plumb`) to avoid clashing with HelioFITS Studio and with LLNL's Flux Framework scheduler.
Inputs: `SCOPE.md` (draft 1) and an independent review of it. The review found real errors in the scope; this plan supersedes SCOPE.md where they differ. Section "Review dispositions" at the end records each review item and what was done with it.

## 0. Ground rules for the implementer

- Branch `claude/project-thread-rx783j` on GillySpace27/fluxpype. Commit in logical steps. Never touch `lowderchris/*`.
- Terminal behavior must be unchanged when none of the new env vars are set. Every pipeline edit is additive and guarded.
- Never use the em dash character in code, comments, docs, or UI strings. Use colons, semicolons, parentheses or periods.
- No FLUX, PDL or Flux in the build container. System `perl` 5.38 (with core JSON::PP) and Node 22 are available. Python there is 3.13; fluxpype pins `<3.13`, so the plumber package and its tests must not import the science stack (see P0.1) and must run under 3.10 to 3.13.
- Edits to `.pdl`/`.pm` files that cannot be executed here are limited to single guarded lines that call a helper, and must pass `perl -c` where the file can be compiled standalone (`events.pm` can; the pipeline files cannot, so keep them trivially correct).

## 1. v1 deliverable (what ships in this PR)

1. P0 pipeline hooks (config snapshot, task selection, events, log file, timeout kill).
2. `fluxpype.plumber`: FastAPI server, SQLite job store, a local queue with key locks and CPU-slot budget, detached launcher with reattach, the monitor (events primary, log + file fallback with per-field source), a read-mostly profile service with per-job overrides, a cache index with trash.
3. A pluggable `Backend` protocol with `LocalBackend` and `FakeBackend` (tests and demo). SSH and Slurm are designed in SCOPE.md 4.5 and deferred to a follow-up PR, because the cluster scheduler is unknown and neither can be exercised here.
4. Frontend: React + TS + Vite + Tailwind, prebuilt into `fluxpype/plumber/web/dist` and shipped as package data. Screens: Overview, Profiles, Job detail, Cache. Dark-first, polished.
5. A demo mode (`flux-plumb --demo`) that runs fake jobs end to end so the UI can be seen without FLUX.
6. Tests that pass in the container; a README section; screenshots of the demo in `docs/gui/`.

Deferred (named so nothing is silently dropped): SSH/Slurm/PBS backends; job dependencies; priorities; prune rules, treemap, per-CR timeline; gallery compare; run-history comparison; ETA; peak RSS/machine load; adopting terminal logs; profile editing that writes config.ini (v1 is read-only on config.ini plus per-job overrides; see P1.3); 3D world viewer.

## 2. P0: pipeline hooks (small, additive)

**P0.1 Make `import fluxpype` cheap.**
- `fluxpype/__init__.py`: replace eager imports with a module `__getattr__` that lazily imports `config_runner`/`pipe_helper` and exposes `run`. Keep `__all__`.
- `config_runner.py`: move `os.nice(16)` from module level into `run()`.
- `fluxpype/plumber/**` and `fluxpype/events.py` import only stdlib, plus FastAPI/uvicorn/pydantic in the server modules. Add a `[plumber]` optional extra in `pyproject.toml`: `fastapi`, `uvicorn[standard]`. Entry point `flux-plumb = fluxpype.plumber.cli:main`. Add `fluxpype = ["config.ini", "plumber/web/dist/**/*", "events.pm"]` to package-data, and update `MANIFEST.in` to match.

**P0.2 `FLUXPYPE_CONFIG_FILE`.**
- `pipe_helper.py:find_config_file`: if the env var is set, return it (expanduser); if set but missing, raise `FileNotFoundError` naming it. Do not change how `base_dir` is derived (`resolve_base_dir(config_filename)` still receives the argument `"config.ini"`, and `base_dir = auto` still resolves from `__file__`).
- `pipe_helper.pm:find_config_file`: same, with `die` if set but missing.
- The plumber writes the snapshot as a verbatim copy of `config.ini`. Per-job overrides are applied by a line editor (P1.3) that rewrites only `key = value` lines inside the job's section, or appends a key to the end of that section. It never re-serializes with configparser.
- Also set `base_dir` explicitly in the snapshot's `[DEFAULT]` to the resolved checkout path. `auto` derives from the module file, so this is belt and braces.

**P0.3 Task selection: `--only CR,NFLUX,ADAPT,METHOD`.**
- `config_runner.run`: new option. When given, run exactly that one task (still validated against the profile's lists; error if not in the product). Also add `--capabilities`, which prints JSON `{"version": ..., "features": ["only","events","config_file","log_file","task_timeout"]}` and exits 0. The plumber probes this before submitting, so an older install can't silently run a whole profile.
- Launch command used by the plumber: `<python> -m fluxpype.config_runner --config P --only ...`. Make sure `config_runner.py` has a single `if __name__ == "__main__": run()` and that lazy `__init__` avoids double import side effects.

**P0.4 Log file and timeout.**
- `console.init_log`: if `FLUXPYPE_LOG_FILE` is set, open that path in append mode instead of the timestamped one.
- Timeout: read `FLUXPYPE_TASK_TIMEOUT` (seconds, default 2000). Replace the decorator with an explicit deadline: launch Perl with `start_new_session=True`, read lines with a reader thread or `selectors` and a deadline check, and on timeout `os.killpg(SIGTERM)`, wait 10 s, then `SIGKILL`. Always kill the process group in a `finally`. Exit code: 124 if any task timed out and none failed otherwise; 1 for other failures (unchanged). Keep `timeout_decorator` in dependencies (harmless) but stop using it here.

**P0.5 Events.**
- `fluxpype/events.py`: `emit(type, **fields)`. No-op unless `FLUXPYPE_EVENTS` is set. Each event gets `{"v":1,"t":<epoch float>,"type":...,"pid":...,"task":<from FLUXPYPE_TASK_ID env or null>}`. Writes with `os.open(path, O_WRONLY|O_APPEND|O_CREAT)` and a single `os.write` per line, then close. Swallows all exceptions.
- `fluxpype/events.pm` (package `fluxpype_events`, functions `emit_event($type, \%fields)`): same semantics with core `JSON::PP`, `sysopen` + `syswrite` + close, the whole body inside `eval {}`. It records `$$` in a package variable at load time and only emits when `$$` still equals it, so forked Flux daughters never write.
- Schema v1 (document it in `docs/gui/EVENTS.md`):
  - `run.start` {cr, nflux, adapt, method, profile, host, fluxpype_sha (best effort `git rev-parse` in REPO_ROOT, null on failure)}
  - `run.end` {status: ok|failed|timeout, exit_code, elapsed_s}
  - `stage.start` / `stage.end` {stage, status: ok|failed|skipped, elapsed_s, detail}
  - `relax.step` {step, stiffness, fmax, f_av, fs_av, n_av, vertices}
  - `relax.cycle` {cycle, threshold, stiffness, cum_time_s, steps, nc_curve, nc_proxi, broken}
  - `metric` {name, value, unit}
  - `artifact` {path, kind}
  - `log` {level: warn|error, message}
- Stage ids (correcting the scope table): `magnetogram`, `footpoints`, `pfss`, `world`, `relax`, `world_plots`, `fieldmap`, `wind`, `wind_plots`, `fluxlight`. Stages 1 to 7 (`magnetogram` through `fieldmap`) are all skipped when `wind_only` is set.
- Instrumentation, one guarded line per point:
  - `config_runner.run_pdl_script`: `run.start`, `run.end`. Set `FLUXPYPE_TASK_ID` in the child env when the plumber provided one.
  - `magnetogram2wind.pdl`: add `use fluxpype_events;` near the other `use` lines (the pipeline already puts `fluxpype/` on `@INC`; verify by reading how `pipe_helper` is found), then `stage.start`/`stage.end` around each block listed above. If `wind_only` skips the block, emit `stage.end status=skipped` for each skipped stage.
  - `relax_pfss_world.pm`: `relax.cycle` after the stats line; a `stage.end` detail that marks "reused relaxed world" when the skip branch is taken.
  - `simple_relaxer.pm`: `relax.step` inside the existing `if ($do_print)` block, reusing `$h`.
  - `fluxlight.py`: `artifact` after each `plt.savefig`.

**P0 tests (pytest, `tests/test_events.py`, `tests/test_runner_hooks.py`):**
- `events.py`: no-op without env; appends valid JSON lines with env; never raises on an unwritable path.
- `events.pm`: run `perl -I fluxpype -e 'use fluxpype_events; emit_event(...)'` and parse the output with Python (a cross-language schema test). Run `perl -c fluxpype/events.pm`.
- Runner: put a shim `perl` script first on `PATH` (a Python script named `perl`) that, given the `.pdl` path and args, replays a fixture: prints log lines, writes events to `$FLUXPYPE_EVENTS` the way the real pipeline would, sleeps, and exits with a code chosen by an env var. Test `--only` selection, `--capabilities`, env propagation (`FLUXPYPE_CONFIG_FILE`, `FLUXPYPE_EVENTS`, `FLUXPYPE_TASK_ID`), log file override, timeout kill (the shim sleeps forever; assert the process group is gone and the exit code is 124), and failure exit 1.
- The runner tests need `rich` and `tqdm` installed (pip in the container). The shim must also answer the runner's `perl -e "use PDL..."` and `perl -e "use Flux..."` probes by printing `ok`.

## 3. P1: plumber backend (`fluxpype/plumber/`)

Modules:

- `paths.py`: plumber home `~/.fluxpype/plumber` (override with `FLUX_PLUMBER_HOME`). Holds `plumber.db`, `runs/<job_id>/<task_id>/`, `token`.
- `config_io.py`: read-only parse of `config.ini` for display (sections, raw values, inline comments, DEFAULT inheritance, which keys a section overrides). Expand the task product using the same semantics as `pipe_helper.compute_configs` (`parse_list_or_range`, `parse_list`), implemented locally so the science stack is not imported; drop empty items and flag them as a warning (e.g. a trailing comma in `flow_method`). The line editor `apply_overrides(text, section, overrides) -> text` only touches uncommented `key = value` lines inside the section, and keeps inline comments. Override validation: keys must already exist in DEFAULT or the section; values must match `^[\w.\-+\[\], /~]*$`, no `#`, `$`, `;` or newline; `run_script`, `base_dir`, `data_dir`, `python_dir`, `rc_path`, `fl_prefix`, `pl_prefix`, `fl_mhdlib`, `working_dir`, `flux_perl_lib` are not overridable (`convert_value` calls `eval` on values, so this validation is a security boundary).
- `store.py`: SQLite (stdlib `sqlite3`, WAL). Tables: `jobs(id, profile, overrides_json, backend, status, created, started, ended, snapshot_path)` and `tasks(id, job_id, cr, nflux, adapt, method, status, pid, pid_start, exit_code, started, ended, run_dir, lock_key)`.
- `backends/base.py`: `Backend` Protocol + `TaskSpec`, `Handle`, `TaskStatus` dataclasses, as SCOPE.md 4.5 describes (with `capabilities`, `submit`, `poll`, `cancel`, `read_file(handle, name, offset)`).
- `backends/local.py`:
  - `submit` spawns `python -m fluxpype.plumber.launcher <run_dir>` detached (`start_new_session=True`, stdin DEVNULL, stdout/stderr to `run_dir/launcher.log`).
  - The launcher (stdlib only) writes `pid` and `pid_start` (start time from `ps -o lstart= -p PID` on macOS, `/proc/PID/stat` on Linux), runs the runner command with stdout/stderr to `run_dir/stdout.log` and the env vars set (`FLUXPYPE_CONFIG_FILE`, `FLUXPYPE_EVENTS`, `FLUXPYPE_LOG_FILE`, `FLUXPYPE_TASK_ID`, `FLUXPYPE_TASK_TIMEOUT`), then writes `exit_code` atomically (write tmp + rename).
  - `poll` reads those files and checks PID liveness with matching start time.
  - `cancel` does `killpg(SIGTERM)` then `SIGKILL` after 10 s, and writes `cancelled`.
- `backends/fake.py`: same as local, but the command is `python -m fluxpype.plumber.fake_flux` with a scenario (`ok`, `fail_in_pfss`, `slow_relax`, `no_events` to exercise the fallback). It emits realistic log lines and events with time compression. Used by `--demo` and tests.
- `scheduler.py`: an asyncio loop. A queue of pending tasks in creation order. Limits:
  - Global CPU budget (default `os.cpu_count()`).
  - Task slots = `max(relax_concurrency, concurrency)` from the effective config, clamped to the budget.
  - `max_parallel_tasks` (default 1, settable in the UI).
  - Lock keys: (`data_dir`, `batch_name`, `cr`) for outputs, plus (`data_dir`, `cr`, `inst`) for the magnetogram/PFSS caches. A task starts only if all its keys are free.

  Operations: pause/resume the queue, cancel a job or task, retry the failed tasks of a job. On server start, reattach: every task in a running state is polled and its status reconciled.
- `monitor.py`: builds a `TaskState` per task from:
  1. `events.jsonl`, tailed by byte offset, complete lines only;
  2. the fallback parser over the `FLUXPYPE_LOG_FILE` text (ANSI-stripped), with regexes for the existing banners: `(pdl) Relax the Initial World State`, `step N: time is now ... stiffness= X%, fmax= ...`, `Cycle N, Cumulative relaxation time: T seconds, S steps`, `Relaxation Complete in`, `Skipped relaxation!`, `get_magnetograms.py failed`, `get_pfss.py failed`, `plot_fieldmap.py failed`, `Program Complete for CR(\d+) in ([\d.]+) Seconds`, `TOO MANY FLUXONS`, and the Python-side banners in get_magnetograms/get_pfss/get_wind/plots (read those files to collect their exact banner strings);
  3. file probes: the expected artifacts per stage (paths per review B6: `batches/{b}/data/cr{cr}/floc/floc_cr{cr}_r{red}_f{n}_{inst}.dat`, `data_dir/pfss/pfss_cr{cr}_r{red}_{inst}.pkl`, `batches/{b}/data/cr{cr}/world/cr{cr}_f{n}_{inst}.flux` and `_relaxed_s*.flux`, magnetogram names from `pipe_helper.update_magdir_paths`), plus PID liveness and the log/event mtimes.

  Each stage and the relax metrics carry `source: "event"|"log"|"file"|"none"`. Events win when present. If the events file has no new line for 120 s, or longer than 3 times the observed relax print interval, while the log is still advancing, the task is marked `degraded` and log-derived fields take over. Verification: on `stage.end ok`, the stage's file probe must be satisfied (for stages that have probes), else `verified: false`. If the PID is dead with no `run.end`, the task is `crashed` and gets its exit code from the launcher. No heartbeat thread.
- `cache.py`: scans `data_dir` (from the active `config.ini` DEFAULT via `pipe_helper`-compatible resolution, done locally) into typed records: magnetogram, pfss_pickle, footpoints, world, relaxed_world, wind, plots, fluxlight, other. Each record has its params parsed from the filename, size and mtime. Results are cached in memory and rescanned on demand.
  - Actions: move to `data_dir/.flux_trash/<timestamp>/` (realpath-confined to `data_dir`), list trash, restore, empty trash.
  - No deletion outside `data_dir`, ever.
- `server.py` (FastAPI):
  - REST: `/api/profiles`, `/api/profiles/{name}` (raw + effective + task preview), `/api/jobs` (GET list, POST create {profile, overrides, backend, tasks subset?}), `/api/jobs/{id}`, `/api/jobs/{id}/cancel`, `/api/jobs/{id}/retry`, `/api/tasks/{id}` (state), `/api/tasks/{id}/log?offset=`, `/api/queue` (GET/PATCH pause, max_parallel, budget), `/api/cache` (GET summary + records), `/api/cache/trash` (POST move, GET list, POST restore, DELETE empty), `/api/artifacts?path=` (realpath-confined to `data_dir` and the run dirs; images only).
  - A single WebSocket `/api/ws` pushes `task_state`, `job_state`, `queue_state` and `log_append` messages.
  - Auth: a random token generated at launch and printed in the URL once (`/?token=...`). The server swaps it for an HttpOnly, SameSite=Strict cookie and redirects to strip the query. All API and WS requests require the cookie or an `X-Flux-Token` header. Check `Host` (localhost/127.0.0.1 or `--host` value) and `Origin` on the WS. Bind 127.0.0.1 by default.
  - Serves `web/dist` at `/`.
- `cli.py`: `flux-plumb [--host 127.0.0.1] [--port 8765] [--no-browser] [--demo] [--config PATH]`. Prints the tokenized URL and opens the browser unless `--no-browser`. `--demo` uses `FakeBackend` and a temp plumber home seeded with a sample `config.ini`.

P1 tests: `config_io` (parse the real `fluxpype/config.ini`, overrides keep comments, rejected keys/values, task expansion matches the Python semantics including ranges and trailing commas); `store`; `scheduler` with `FakeBackend` (lock serialization, slot budget, cancel, retry, reattach after a simulated server restart); `monitor` (events-only, log-only, mixed, stale events causing degradation, crash detection, verification failure); `cache` (classification on a synthetic tree, trash confinement rejects `..` and symlink escapes); `server` via FastAPI `TestClient` (auth required, token cookie flow, job lifecycle with fake backend, artifact path traversal blocked).

Fixture honesty: the log fallback regexes are written from the print statements in the code, not from a real captured log. Mark those tests as "synthesized from source" in their docstrings, and in the PR body ask Gilly for one real `quickstart` log to add as a fixture.

## 4. P2: frontend (`fluxpype/plumber/web/`)

- Vite + React 18 + TypeScript + Tailwind 3. Radix primitives where needed (dialog, tabs, tooltip, dropdown), lucide-react icons, TanStack Query, uPlot for live charts. Pin exact versions in `package.json` and commit `package-lock.json`. `npm run build` outputs to `web/dist` (committed).
- Design: dark-first (near-black slate background, subtle panels, one accent: a solar orange/amber; status colors: running = amber pulse, ok = emerald, failed = rose, skipped/cached = slate, degraded = violet outline). Inter or system font; tabular numerals for metrics. Light theme toggle. Responsive to laptop widths; usable at phone width.
- Layout: left nav (Overview, Profiles, Jobs, Cache, Settings); top bar with the queue status pill (running / queued / paused) and a connection indicator for the WebSocket.
- Overview:
  - Running tasks as cards, each with the 10-stage pipeline as chips (state-colored; an outline style when `source` is not `event`), the current stage name, elapsed time, a stiffness sparkline, and cancel.
  - The queue below as a list with drag-free up/down controls deferred (FIFO only in v1).
  - Recent finished tasks with status and duration, and a thumbnail of the newest image artifact.
- Profiles: list of sections with a search box.
  - Detail shows the effective values in a table (key, value, source: DEFAULT or section, inline comment as help text), the task product preview (CR x N x adapt x method) with warnings, and cache hits per task (whether magnetogram, pfss pickle, world, relaxed world exist).
  - A "Run" drawer edits per-job overrides (allowed keys only) and the task subset, then submits.
- Job detail:
  - Header with profile, overrides, and status.
  - Per-task tabs, each with:
    - a stage timeline (horizontal bars by duration, colored by status, source badge);
    - live relax charts (stiffness on a log y-axis vs step, with the cycle threshold as a step line; fmax and f_av/fs_av on a second chart; cycle boundaries as vertical markers);
    - a metrics strip (current stage, steps/s, cycle, stiffness, vertices, elapsed);
    - a log pane (virtualized, follows the tail, search, stage anchors, ANSI stripped);
    - an artifacts grid with a lightbox for images;
    - a "data sources" panel showing event/log/file status, last update times, and verification flags.
- Cache: summary tiles by kind (count, total size), a filterable table (kind, CR, batch, n, inst, size, mtime) with multi-select, "move to trash", and a trash view with restore/empty.
- Settings: queue limits (max parallel, CPU budget), theme, plumber paths (read-only), backend list (Local active; SSH and Slurm shown as "coming next").
- Empty, loading and error states for every view; a toast on actions.

## 5. P3: docs, demo, screenshots

- `README.md`: a "Fluxpype Plumber" section (install `pip install -e ".[plumber]"`, `flux-plumb`, `flux-plumb --demo`, the tunnel recipe `ssh -L 8765:localhost:8765 host` then `flux-plumb --host 127.0.0.1` on the remote).
- `docs/gui/EVENTS.md`: the event schema and how the fallback works.
- Screenshots: run `flux-plumb --demo --no-browser`, drive it with Playwright (preinstalled), and save `docs/gui/screenshots/{overview,job,profiles,cache}.png` at deviceScaleFactor 2.

## 6. Order of work and commits

1. P0.1 lazy import + package metadata.
2. P0.2 to P0.4 runner hooks + runner tests with the perl shim.
3. P0.5 events (py, pm, instrumentation) + tests + EVENTS.md.
4. Plumber backend modules + tests.
5. Frontend + build.
6. Demo, screenshots, README.
Run `pytest -m "not integration"` for plumber and events tests at the end; report anything that was skipped.

## Review dispositions

| Review item | Disposition |
|---|---|
| B1 import side effects | Adopted (P0.1). |
| B2 timeout orphan | Adopted (P0.4), exit 124. |
| B3 `--only` ignored by old installs | Adopted: `--capabilities` probe (P0.3). |
| B4 log clobber / Rich wrapping | Adopted: `FLUXPYPE_LOG_FILE`, parse that file. |
| B5 stage table and flags wrong | Adopted: corrected stage ids, `wind_only`; no per-stage force UI in v1. |
| B6 artifact paths and keys | Adopted: corrected probes; locks on (data_dir,batch,cr) + (data_dir,cr,inst); default max_parallel 1. |
| B7 Perl config resolution, base_dir trap | Adopted (P0.2); verbatim snapshot + line editor. |
| I1 security | Adopted: override whitelist/regex, cookie token, Host/Origin checks, realpath confinement. |
| I2 detach/reattach | Adopted: launcher process writes pid, pid_start, exit_code atomically. |
| I3 events and forks | Adopted: PID guard at load time, syswrite, eval. |
| I4 heartbeat flapping | Adopted: no heartbeat; staleness relative to the observed print interval. |
| I5 CPU slots | Adopted: max(concurrency, relax_concurrency). |
| I6 config.ini writes | Adopted by deferral: v1 never writes config.ini. |
| I7 cluster assumptions | Recorded for the SSH/Slurm follow-up. |
| Over-scoping | Mostly adopted; kept the cache table + trash and the profile cache-hit preview. |
| Testability | Adopted: perl shim, cross-language event test, honest fixtures, package-data. |

## Implementation notes

Deviations and caveats recorded after implementation.

- The real pipeline (PDL and Flux) was not available in the build environment. Everything that touches it is covered by the Perl shim, the fake science modules and demo mode, not by a real run.
- `flux-plumb` reports its version as `dev` when the package is not installed with metadata.
- The run drawer exposes only settings the profile marks editable; the first group shown is a fixed list of common knobs, the rest sit behind "Show more".
- SSH and Slurm backends appear in Settings as "coming next" and are not implemented.
- Cache table renders at most 200 rows at a time (with "show more"); the API returns up to 20000 records.
