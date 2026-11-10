# FLUX Studio: scope for a modern GUI over fluxpype

Status: scope draft 1 (2026-11-10). Base: `origin/main` at `0b635d2`.

## 1. What Gilly asked for

- A modern, impressive-looking GUI for FLUX, replacing "edit config.ini, run in a terminal".
- Granular readouts: exactly what FLUX is doing, at which stage, and how it is performing.
- Modern affordances: job queueing, cache management, and the like.
- Python, wrapping fluxpype (config.ini profiles), keeping the whole fluxpype codebase relevant.
- Jobs run on his Mac now, and later on the NWRA cluster and on clusters that other users configure. The NWRA scheduler type is unknown, so execution backends must be pluggable.
- Structured progress events added to fluxpype, with terminal runs behaving exactly as today, plus a fallback that reads logs and output files. The fallback verifies the happy path, provides redundancy, and degrades in a structured way when events are missing.
- Lands on GillySpace27/fluxpype as a PR. Never on lowderchris/fluxon-mhd.

Working name: **FLUX Studio** (`flux-studio` command). Rename freely.

## 2. How a run works today (verified by reading the code at `0b635d2`)

`flux-config-run [--config PROFILE]` (`fluxpype/config_runner.py`):

1. Sets `FLUXPYPE_CONFIG=PROFILE` in the environment. Every Python and Perl child re-reads `fluxpype/config.ini` and picks the profile from that variable.
2. Expands the profile into the Cartesian product `adapts x rotations x fluxon_count x flow_method`. Each element is one **task**: `perl magnetogram2wind.pdl CR NFLUX ADAPT METHOD`, run serially, with a 2000 s timeout per task (`timeout_decorator`).
3. Streams merged stdout/stderr to the terminal and to `logs/fluxpype_<timestamp>_<batch>.log` via `console.emit_raw`.

Inside `magnetogram2wind.pdl`, one task runs these stages:

| # | Stage | Code | Main artifacts (under `data_dir`) | Already skips if cached? |
|---|---|---|---|---|
| 1 | Magnetogram fetch/reduce | `get_magnetograms.py` (subprocess) | `magnetograms/CR{cr}_r{red}_hmi.fits` or `CR{cr}_rf{sel}_adapt.fits` | yes, unless `do_download` |
| 2 | Footpoints (Hilbert) | `get_hilbert_footpoints.pm` | `batches/{b}/cr{cr}/floc/floc_*.dat` | yes, unless `force_process_magnetogram` |
| 3 | PFSS trace | `get_pfss.py` (subprocess) | PFSS open/closed field-line files | partly |
| 4 | World from PFSS | `make_world_from_pfss.pm` | `batches/{b}/data/cr{cr}/world/cr{cr}_f{n}_{inst}.flux` | yes |
| 5 | Relaxation (FLUX core) | `relax_pfss_world.pm` -> `simple_relaxer.pm` | `..._relaxed_s{steps}.flux`, `flux-movie-*.png` | yes, if a relaxed file exists |
| 6 | World plots | `plot_worlds`, `plot_fieldmap.py` | PNGs | no |
| 7 | Wind | `get_wind.pm` (+ `flow_method` module) | wind / B / f(r) data files | `recompute` forced to 1 |
| 8 | Wind plots | `run_all_plots` | PNGs under the batch | no |
| 9 | FLUXlight (if `light=1`) | `fluxlight.py` | `cr{cr}_fluxlight_*.png` | no |

Relaxation is where the time goes and where the numbers live. `simple_relaxer` already computes these every step through `fw_stats`: vertex count, stiffness `f_av/fs_av*100`, `fmax`, `f_av`, `fs_av`, `n_av`, max/mean angle, and `dt`. `relax_pfss_world` adds the per-cycle numbers: cycle, cumulative time, curvature/proximity fixes, the "broken" counter, and the stopping threshold, which grows 1.1x per cycle. Right now all of this only goes to stdout as free text, printed every `do_steps/5` steps.

## 3. Findings that shape the design

These are the main constraints. Each one is verified in the code.

**F1. `config.ini` is shared, mutable, global state.** Every child process re-reads the one shipped `config.ini` while the run is going. `find_config_file` tries the given name relative to cwd, then falls back to the copy beside the module, and `pipe_helper.pm` does the same. So if the GUI edits a profile while a job is running, the change silently lands in that job partway through. **Requirement:** every job runs against a frozen snapshot of its config. That needs a small, backward-compatible hook: a `FLUXPYPE_CONFIG_FILE` environment variable that both `pipe_helper.py:find_config_file` and `pipe_helper.pm` honor ahead of the shipped file. Terminal behavior is unchanged when the variable is unset.

**F2. Output paths are keyed only by `batch_name`, `cr`, `n`, `adapt` (and `method` for wind).** Two queued jobs that share those keys race on the same files. **Requirement:** the scheduler holds a lock on the output key (`data_dir`, `batch_name`, `cr`, `nflux`, `adapt`). Jobs with conflicting keys serialize; jobs without conflicts run in parallel.

**F3. Magnetograms are a cache shared across all batches.** Concurrent first-time downloads of the same CR could collide. Same treatment: a lock per magnetogram key, held by the fetch stage.

**F4. CPU oversubscription.** `concurrency` and `relax_concurrency` (up to 33) are per task, and `os.nice(16)` is applied globally. A local queue that runs N tasks at 33 threads each would thrash the machine. **Requirement:** the local backend schedules by declared CPU slots, where slots = `relax_concurrency` (or `concurrency`), against a configurable machine budget that defaults to the core count.

**F5. Stage boundaries are not machine-readable.** There are only banner prints. Free-text parsing works today but breaks on any wording change. This is why Gilly picked option 4a: explicit events, with log parsing as the fallback.

**F6. Per-task timeout and exit codes.** PR #45 made the runner exit nonzero on failure, so exit codes are reliable now. The 2000 s timeout is hard-coded; a GUI job should be able to override it.

**F7. Cache semantics are implicit.** "Skip if the file exists" checks are scattered across stages, and invalidation happens through flags (`recompute`, `do_download`, `force_process_magnetogram`) or by deleting files by hand. A cache manager has to model these artifacts explicitly rather than invent a new cache.

## 4. Proposed architecture

```
 Browser (React + TS, prebuilt, served by FastAPI)
   |  REST (CRUD: profiles, jobs, cache, artifacts)   WebSocket (live events, log tail)
 FastAPI app: fluxpype/studio/server
   |-- ProfileService    read/validate/diff/save config.ini (comment-preserving), snapshots
   |-- JobService        queue, priorities, dependencies, cancel/retry, history (SQLite)
   |-- Backends          LocalBackend | SSHBackend | SlurmBackend   (one interface)
   |-- Monitor           event ingest (JSONL) + log/file fallback reconciler -> unified run state
   |-- CacheService      artifact index over data_dir, sizes, provenance, invalidate/prune
   `-- ArtifactService   images, .flux worlds, wind maps; thumbnails
 fluxpype (existing): config_runner -> magnetogram2wind.pdl -> stages
   `-- + events.py / events.pm   (no-op unless FLUXPYPE_EVENTS is set)
```

### 4.1 Process model

- `flux-studio` launches the server (uvicorn) on `127.0.0.1:8765` and opens the browser. `--host`/`--port` options exist for cluster head nodes, which are reached over an SSH tunnel. There is no auth beyond a random token printed at launch (Jupyter style) and passed as a cookie/query parameter. That is enough for localhost and tunnels.
- The server is a supervisor, not the executor. Jobs run as **detached** processes (`start_new_session=True`), each writing its own run directory. So restarting or closing the GUI never kills a running FLUX job, and the server re-attaches on startup by reading run directories and PIDs. This matters because jobs run for hours.
- State lives in SQLite (`~/.fluxpype/studio/studio.db`). Run directories live under `~/.fluxpype/studio/runs/<job_id>/` and hold `config.snapshot.ini`, `job.json`, `events.jsonl`, `stdout.log`, `exit_code`, and `pid`.

### 4.2 Job model

- **Job** = one profile snapshot + overrides + backend + priority. It expands into **tasks**, one per (cr, nflux, adapt, method) exactly as `config_runner.run` does, each with stage-level progress.
- To launch a task, the backend runs `python -m fluxpype.config_runner --config <profile>` with `FLUXPYPE_CONFIG_FILE=<snapshot>` and `FLUXPYPE_EVENTS=<run_dir>/events.jsonl` in the environment, plus a new `--only cr,nflux,adapt,method` option. Every task therefore goes through the existing runner code path, and the CLI and GUI cannot drift apart. *(Alternative: one runner process per job that loops over its tasks itself. Simpler, but it loses per-task scheduling and parallelism. Use it as the fallback if `--only` meets resistance.)*
- Queue features: FIFO with priorities, pause/resume the queue, cancel (SIGTERM to the process group, then SIGKILL), retry the failed tasks only, duplicate-and-edit, per-job timeout override, output-key locks (F2), CPU-slot accounting (F4), and job dependencies (for example, run the FLUXlight-only re-render after relax finishes).

### 4.3 Structured events (option 4a)

- One schema, JSON Lines, appended to `$FLUXPYPE_EVENTS`. When the variable is unset, emitting does nothing, so terminal runs are byte-identical to today.
- Emitters: `fluxpype/events.py` (Python) and `fluxpype/events.pm` (Perl, using `JSON::PP`, which is core Perl, so there is no new dependency). Writes are `O_APPEND`, one `write()` per line, and lines stay well under PIPE_BUF.
- Event types (v1):
  - `run.start/end` (task identity, host, pid, versions: git sha of fluxpype and fluxon-mhd, Perl/PDL versions)
  - `stage.start/end` (stage id from the table in section 2, status ok|skipped_cached|failed, duration, artifacts produced or reused)
  - `relax.step` (step, stiffness, fmax, f_av, fs_av, n_av, vertices, dt, wall time), throttled to the existing `print_n` cadence or 1 Hz
  - `relax.cycle` (cycle, threshold, fixes, broken count)
  - `metric` (free-form name/value/unit, for example fluxon N_actual vs N_wanted)
  - `artifact` (path, kind, size)
  - `warn` / `error` (message, stage, traceback tail)
  - `heartbeat` (every 10 s from the runner, so a hung process is detected)
- Instrumentation points: `config_runner.run_pdl_script` (run start/end), `magnetogram2wind.pdl` (each stage boundary, 9 pairs), `relax_pfss_world.pm` (cycle), `simple_relaxer.pm` (step stats it already computes), `get_magnetograms.py`, `get_pfss.py`, `fluxlight.py` (artifacts). The total diff is small and additive.

### 4.4 Fallback monitor (the "4a with fallback" part)

A reconciler runs for every task and builds the same `RunState` from three independent sources:

1. **Events** (primary).
2. **Log parsing**: regexes over `stdout.log` for the existing banners (`(pdl) Relax the Initial World State`, `step N: ... stiffness=`, `Cycle N, Cumulative relaxation time`, `Program Complete for CR... in X Seconds`, `get_pfss.py failed`, and so on).
3. **Filesystem probes**: whether the expected artifacts from section 2 exist, plus their mtimes and sizes, and whether the PID is alive.

Rules:

- **Happy path (verification):** when events say stage X ended ok, the reconciler checks that X's artifacts exist on disk. If they don't, it raises a `verification_failed` flag on that stage in the UI.
- **Redundancy:** if no event has arrived for a while (`stale_after`, about 30 s, using heartbeats), the state source for that task switches to logs+files. The UI shows a "degraded: inferred from logs" badge.
- **Structured degradation:** each field in `RunState` carries `source` = event | log | file | unknown and a `confidence`. The UI renders lower-confidence data muted rather than hiding it. If all three sources are silent and the PID is alive, the task shows as `stalled?`. If the PID is gone with no end event, it shows as `crashed` and the exit code comes from the backend.

This also covers old jobs, and runs launched from a terminal without `FLUXPYPE_EVENTS`: the GUI can "adopt" a plain `flux-config-run` log file and show inferred progress.

### 4.5 Execution backends (pluggable)

```python
class Backend(Protocol):
    name: str
    def capabilities(self) -> BackendCaps: ...         # slots, gpu?, shared_fs?
    def submit(self, task: TaskSpec) -> Handle: ...
    def poll(self, h: Handle) -> TaskStatus: ...
    def cancel(self, h: Handle) -> None: ...
    def stream(self, h: Handle, path: str, offset: int) -> bytes: ...   # events/log tail
    def fetch(self, h: Handle, remote: str, local: str) -> None: ...    # artifacts
```

- **LocalBackend**: detached subprocess, CPU-slot scheduler. This is the MVP.
- **SSHBackend**: runs the same command on a remote host that already has fluxpype installed (it checks with `flux-doctor` over SSH first). Tails `events.jsonl` through the SSH channel and pulls artifacts with rsync/sftp. Uses the system `ssh` binary and `~/.ssh/config`, so ProxyJump, keys and ControlMaster all work, and there is no paramiko dependency.
- **SlurmBackend**: an SSH (or local) transport plus `sbatch` with a generated script (`--cpus-per-task` = slots, time limit = timeout), `squeue`/`sacct` for status, and `scancel` for cancel. Events go to a shared-filesystem path and are tailed over SSH.
- **PBS/SGE**: same shape as Slurm. Left out of v1 but the interface allows it. Once Gilly learns what the NWRA cluster runs, adding it is a template plus a status parser.
- Clusters are defined in `~/.fluxpype/studio/clusters.toml` and edited in the GUI: name, kind, host, remote fluxpype path, Python executable, scratch/data dir, and default partition/account.

### 4.6 Cache manager

- Indexes `data_dir` into typed artifacts: magnetogram (HMI/ADAPT, CR, reduction), footpoints, PFSS, world (unrelaxed/relaxed, steps), wind outputs, plots, FLUXlight renders, and relax movie frames. Each record has its size, mtime, the producing job (when known, from events), and the parameters parsed from the filename.
- Views: totals by kind and by batch, a treemap of disk use, and a per-CR timeline.
- Actions: invalidate (delete with confirmation; files move to a `.trash` folder inside `data_dir` first, with an "empty trash" step), "force recompute this stage" (sets the matching existing flag in the job overrides rather than deleting), and prune rules (for example, keep only the latest relaxed world per key, or drop movie frames older than N days).
- Reuse awareness: when a job is queued, it shows which stages will be skipped because their artifacts already exist (cache hit) and which will run.

### 4.7 Frontend

- Stack: React 18 + TypeScript + Vite, Tailwind with shadcn/ui-style components (Radix primitives), TanStack Query for REST and a single WebSocket for live updates, uPlot for fast streaming time series, and lucide icons. The build output is committed under `fluxpype/studio/web/dist/` and shipped as package data, so users need no Node toolchain. Node is only needed to *develop* the frontend.
- Look: a dark-first scientific-dashboard aesthetic, with a light theme toggle.
- Screens:
  1. **Overview**: queue strip, running tasks with stage pipeline chips (9 stages, colored by state and source), machine load (CPU, memory, disk), recent results gallery.
  2. **Profiles**: list of `config.ini` sections; a form editor generated from a key schema (types, units, help text taken from the existing inline comments); the DEFAULT inheritance shown visually; diff before save; "run this profile" with a preview of the task expansion and cache hits.
  3. **Job detail**: per-task stage timeline (Gantt-style durations); live relaxation charts (stiffness vs step on a log axis, fmax, f_av/fs_av, dt, cycle markers, the threshold line); a log pane with search and stage anchors; an artifacts tab; source/confidence badges.
  4. **Cache**: the section 4.6 views.
  5. **Clusters**: backend config, connection test, remote doctor output.
  6. **Gallery**: browse outputs across batches (FLUXlight renders, wind maps, field maps), compare side by side.
- Performance readouts: per-stage wall time and its share of the total, steps per second, the stiffness convergence rate, an ETA (from cycle-threshold extrapolation, labeled as an estimate), peak RSS (from the backend's `ps` sampling), and a history comparison across past runs with the same (n, method).

## 5. Explicitly out of scope for v1

- Editing fluxon-mhd C/Perl internals, or exposing new physics knobs that are not already in `config.ini`.
- A 3D interactive fluxon world viewer (high-value v2: three.js over `open_world_python.read_flux_world`).
- Multi-user auth and shared servers.
- PBS/SGE backends (the interface supports them; implementations wait for the NWRA answer).
- Rewriting `config.ini` into another format. The GUI keeps `config.ini` as the source of truth, so terminal workflows keep working.

## 6. Phasing and effort (estimates, not measurements)

| Phase | Content | Rough size |
|---|---|---|
| P0 | `FLUXPYPE_CONFIG_FILE`, `--only`, events.py/.pm + instrumentation, unit tests that terminal output is unchanged | ~600 LOC |
| P1 | Server core: profiles, SQLite job store, LocalBackend + scheduler/locks, monitor + fallback reconciler, WebSocket | ~2,000 LOC Py |
| P2 | Frontend: overview, profiles, job detail with live charts, logs | ~3,500 LOC TS |
| P3 | Cache manager + gallery | ~1,200 LOC |
| P4 | SSH + Slurm backends + clusters UI | ~1,200 LOC |
| P5 | Polish: theming, empty/error states, docs, screenshots | small |

The sequencing decision for this PR: implement P0–P3 fully with LocalBackend, and SSH/Slurm (P4) behind the same interface with a fake-scheduler test harness, because neither a cluster nor FLUX itself can run inside the build environment.

## 7. Risks

- **Nothing in the build environment can run FLUX** (no PDL/Flux there). Mitigation: a `fake_flux` task (a Python script that replays a recorded log and event stream with realistic timing) drives tests, UI development, and the demo mode. The first real run happens on Gilly's Mac.
- **Comment-preserving `config.ini` writes.** `configparser` drops comments. Mitigation: a line-level editor that rewrites only the `key = value` part of changed lines and keeps the inline comments.
- **Perl event emission inside `simple_relaxer` per step** costs a little. Mitigation: throttle, and emit only when `FLUXPYPE_EVENTS` is set.
- **Detached jobs plus re-attach** is the subtle part (PID reuse, partial lines). Mitigation: store the PID together with the process start time, and only parse complete lines.
- **Conflict with the active `fluxlight-first-light` branch** (one commit touching FLUXlight and ADAPT). The instrumentation touches `magnetogram2wind.pdl` and `fluxlight.py`; keep the hunks minimal so the rebase stays trivial.
- **Frontend dist in git** adds about 300–600 KB of build output. That's acceptable for a research package; the alternative is a build step in CI.
