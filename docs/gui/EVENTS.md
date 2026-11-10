# FLUXpype progress events (schema v1)

FLUXpype can describe what it is doing as a stream of JSON Lines. Fluxpype Plumber
reads that stream to draw live progress. Nothing changes for terminal runs:
**when `FLUXPYPE_EVENTS` is unset, every emitter is a no-op.**

## Switching it on

| Variable | Meaning |
|---|---|
| `FLUXPYPE_EVENTS` | Path of the JSONL file to append to. Unset or empty means off. |
| `FLUXPYPE_TASK_ID` | Optional id copied into every event's `task` field. |
| `FLUXPYPE_LOG_FILE` | Console log is appended to this file instead of a timestamped one. |
| `FLUXPYPE_TASK_TIMEOUT` | Per-task time limit in seconds (default 2000). |
| `FLUXPYPE_CONFIG_FILE` | Use this file instead of the shipped `config.ini` (Python and Perl). |

`flux-config-run --capabilities` prints the supported features as JSON, and
`flux-config-run --config P --only CR,NFLUX,ADAPT,METHOD` runs exactly one task.
Exit codes: `0` ok, `1` a task failed, `124` the only failures were timeouts,
`2` `--only` named a task outside the profile.

## Writers

* Python: `fluxpype.events.emit(type, **fields)`.
* Perl: `use fluxpype::events;` then `emit_event($type, \%fields)`, plus the
  stage helpers `stage_start($id)`, `stage_end($id, $status, $detail)`,
  `stage_skipped($detail, @ids)` and `stage_detail($id, $text)`.

Both write each line with a single `O_APPEND` write, swallow every error, and
never raise. The Perl side remembers its pid at load time and only writes while
`$$` still equals it, so forked Flux daughters stay silent. If a stage is still
open when the Perl process dies, an `END` block closes it with `status=failed`.
Edits to the pipeline load the module through a soft-load `BEGIN` line, so a
checkout without `events.pm` still runs, just without events.

## Envelope

Every line has:

```json
{"v": 1, "t": 1789000000.123, "type": "stage.start", "pid": 4242, "task": "t-7", "...": "..."}
```

`t` is epoch seconds, `task` is the value of `FLUXPYPE_TASK_ID` or `null`.

## Event types

| type | fields |
|---|---|
| `run.start` | `cr`, `nflux`, `adapt`, `method`, `profile`, `host`, `fluxpype_sha` (best effort, `null` if git fails) |
| `run.end` | `status` (`ok`, `failed`, `timeout`), `exit_code`, `elapsed_s` |
| `stage.start` | `stage` |
| `stage.end` | `stage`, `status` (`ok`, `failed`, `skipped`), `elapsed_s`, `detail` |
| `relax.step` | `step`, `stiffness` (percent), `fmax`, `f_av`, `fs_av`, `n_av`, `vertices` |
| `relax.cycle` | `cycle`, `threshold` (the stiffness target the next check compares against), `stiffness`, `cum_time_s`, `steps`, `nc_curve`, `nc_proxi`, `broken` |
| `metric` | `name`, `value`, `unit` |
| `artifact` | `path`, `kind` (`image`) |
| `log` | `level` (`warn`, `error`), `message` |

## Stage ids

In pipeline order: `magnetogram`, `footpoints`, `pfss`, `world`, `relax`,
`world_plots`, `fieldmap`, `wind`, `wind_plots`, `fluxlight`.

* The first seven are all reported `skipped` (detail `wind_only`) when `wind_only` is set.
* `fluxlight` is reported `skipped` (detail `light=0`) unless `light` is set.
* When relaxation finds an existing relaxed world, `relax` ends `ok` with detail
  `reused relaxed world`. Other stages that find cached output are not
  distinguished from fresh work in v1; the plumber's file probes show what exists.

## Where the instrumentation lives

| Point | File |
|---|---|
| `run.start`, `run.end` | `config_runner.run_pdl_script` |
| `stage.*`, `metric n_actual` | `science/magnetogram2wind.pdl` |
| `relax.cycle`, reused-world detail | `science/relax_pfss_world.pm` |
| `relax.step` | `science/simple_relaxer.pm` (inside the existing print block) |
| `artifact` | `science/fluxlight.py` (after each `savefig`) |

## How the plumber degrades without events

The plumber never trusts a single source. For each task it combines:

1. **events** from `FLUXPYPE_EVENTS` (primary),
2. **log parsing** of `FLUXPYPE_LOG_FILE` (ANSI stripped) for the existing banners,
   such as `(pdl) Relax the Initial World State`, `step N: time is now ... stiffness=`,
   `Cycle N, Cumulative relaxation time`, `Skipped relaxation!`, `get_pfss.py failed`
   and `Program Complete for CR... in X Seconds`,
3. **file probes** for the artifacts each stage should leave under `data_dir`.

Each stage and the relaxation metrics carry a `source` of `event`, `log`, `file`
or `none`. Events win when present. If no event has arrived for 120 s, or for 3
times the observed relax print interval, while the log keeps growing, the task
is marked `degraded` and log-derived fields take over. When an event says a
stage finished `ok` and that stage has a file probe, the probe must be
satisfied or the stage is flagged `verified: false`. A dead process with no
`run.end` is `crashed`, with its exit code taken from the launcher.

The log regexes are written from the print statements in the code, not from a
captured real log; see the test docstrings.
