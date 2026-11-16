# Plumber: a clearer job page and a safe fix-and-rerun loop (scope, draft 2)

Status: proposal, nothing built. Draft 1 written 2026-11-11, revised the same day after an Opus critique. Evidence labels: **checked** means I read the code line myself; **critic** means the Opus review reported it and I have not re-read it.

## What Gilly asked for

1. The stage chips (the green radio boxes) and the tab bar (Progress, Log, Results, Files, Settings used, Evidence) disagree: some tabs follow the selected stage, others ignore it.
2. The job page makes sense to the expert but is not intuitive to a new user. Make it more intuitive and more useful to a scientist.
3. A better loop: see what went wrong, duplicate the job, change something, re-run.
4. The configuration form needs tooltips and clear labels on the most important parameters.

Assumption: "configuration page" means the New run form and the preset editor, not Settings (data folder, backends). Say if it is the other one.

Already done: reused stages are purple, not hatched; `.flux`, `.dat` and `.npy` files show thumbnails.

## Finding zero: changing a setting and re-running does not recompute anything

This changes the whole scope, so it comes first.

- **checked.** The relaxation stage runs only if `recompute` is 1 or no relaxed file exists yet (`science/relax_pfss_world.pm:139`, `$do_the_relax = ($do_relax || not $second_file_present)`). `$do_relax` is set only inside `if ($recompute)` (`science/magnetogram2wind.pdl:201-209`).
- **checked.** Even with `recompute=1` the world is not rebuilt: `$force_make_world = 0` is hard-coded (`magnetogram2wind.pdl:285`).
- **critic.** Reuse is decided by whether the output file exists, and the file name encodes batch, CR, fluxon count, instrument and reduce or ADAPT selection, not the other settings. So with the same `batch_name`, changing `relax_threshold`, `max_cycles`, `do_steps` or `fix_*` prints "Skipped relaxation!" and reuses the old world, while the job's "Settings used" still shows the new values. Wind loads the highest-step relaxed file in the batch and old files are never deleted, so a looser threshold that makes `_s300` can still feed wind from an older `_s900`.

What this means: today the "change a setting and re-run" loop can silently return the old result, labelled with settings that did not produce it. A scientist could publish on that. Everything below is built around making this visible first.

## What is there now

- The job page (`pages/JobDetail.tsx`, `TaskView`) has the chip row and tabs. The Progress tab has **two** stage selectors bound to the same `?stage=` (the chips and `StageTimeline`) **(critic)**. Settings used is per job, Files is per task **(critic)**. `StagePanel` already filters the plot catalog by stage **(critic)**.
- "Duplicate as new run" goes to `/jobs/new?from=<job>` and copies the job's overrides onto the **current** preset, not the job's own snapshot **(critic)**; if the preset changed or was deleted since, the copy differs or fails.
- The form is not a flat list (draft 1 was wrong): `COMMON` (rotations, fluxon_count, adapts, flow_method, max_cycles, relax_threshold, do_steps, light, do_wind_calc) is shown first and the rest sit behind "Show N more" (`RunForm.tsx:10`, **checked**).
- Retry exists on the server (`/api/jobs/{id}/retry`, same snapshot, failed tasks only) but only on the Jobs list, not the job page **(critic)**.
- The cache probe (`first_match_for_context`, `ProbeContext.patterns`) is a file-existence check and ignores overrides **(critic)**; `preset_cache_summary` gives aggregate counts for the saved preset, not per task **(critic)**.
- Nothing records that job B was made from job A.

## Proposal

### Phase A: tell the truth, then clean up

1. **Reuse warning and honest provenance.** When a planned or finished task has a stage whose output already exists, say so in plain words on the job page and in the form: `Relaxed world exists (cr2150_f1000_hmi_relaxed_s300.flux). Your relaxation settings will have no effect unless recompute is on or the batch name changes.` For a finished job, a reused stage says which file it reused and, where known, which job made it; Settings used marks the settings that could not have affected reused stages. This needs only the existing file probe, run against the draft's batch and overrides.
2. **Simpler selection model (replaces the "scope pill").** The chips in the header become a status display: clicking one opens the Progress tab at that stage. Remove the second selector inside Progress. The other tabs are explicitly whole-task and say so. This removes the inconsistency without building stage filters for Files, Settings or Evidence, which would rest on mtime guesses and a setting-to-stage map that does not exist.
3. **Verdict line** above the chips, composed only from fields the page already has: `Succeeded in 7 m 32 s; 5 of 10 stages reused, 5 ran.` / `Failed in Relaxation: <stage detail>. [Retry as is] [Change settings and re-run] [Read the log]` / `Running Relaxation, step 260.` No cause table yet: the "TOO MANY FLUXONS" check fires on stiffness above 99% for two cycles and "reduce the fluxon count" is the code author's guess **(critic)**, so a table would repeat a guess.
4. **Parameter registry and tooltips** for a short primary list (below). Group by stage, primary first, the rest under Advanced, plain label with the raw key in monospace, info icon with a keyboard-reachable tooltip.
5. **Rename for newcomers**, keeping the old name in the tooltip: "Evidence" to "Where this came from", "Settings used" to "Settings".

### Phase B: the loop

1. **Retry as is** on the job page.
2. **Change and re-run** as a drawer over the job page, prefilled from the job's own **snapshot** (so it is reproducible), the failed stage's parameters pinned at the top, changes marked against the source job with revert, and the Phase A reuse warning shown live as values change.
3. **Provenance**: `from_job` on `POST /api/jobs`, stored with the job; the header shows `Derived from <parent>` with the settings diff against the parent's snapshot (not only the overrides). The `?from` value has to survive in the draft; it is cleared today **(critic)**.
4. **A way to actually recompute.** The UI can only warn; to rerun with changed relaxation settings the user needs a new `batch_name` (cheap to offer as a one-click "new batch for this change") or `recompute=1` (which redoes everything, at hours of relaxation). A partial "re-run from stage X" does not exist in the pipeline today.
5. **Subset picker bug to check first.** Overriding `fluxon_count` or `rotations` while picking a subset of tasks may fail with "none of the requested tasks are in this profile" because the server rebuilds the task list after overrides (`scheduler.py:250-262`) **(critic, read from code, not run)**.

### Phase C: only after A and B

Compare view limited to the settings diff and overlaid relaxation curves; log lines scoped to a stage (stage line ranges are consecutive start markers, but the monitor must record offsets and there is no FLUXlight marker **(critic)**); a first-visit guide; show which relaxed file wind actually used.

Cut from draft 1: stage scoping for Settings and Evidence; the failure cause table; the full Compare tab.

## Parameter registry (Phase A item 4)

Entry: `key, label, group (stage), unit, default, typical range, one-sentence description, affects (stages), cost note, level, source`. Every description cites its source (config.ini comment, `flux_reference.md`, or a pipeline file and line) or is left out.

Starting list: the existing `COMMON` set, plus `recompute`, `batch_name`, `fix_proximity`, `fix_curvature`, `stop_fixing_after`, `mag_reduce`. Caveats from the critic, each to be confirmed before a tooltip is written:

- `do_wind_calc` appears dead: read in the pdl but the wind block is `if (1)`, and `get_wind.pm` uses its own variable. Drop it from `COMMON` unless confirmed live.
- `inner_fov` and `outer_fov`: config.ini says field of view in solar radii, but `make_world_from_pfss.pm:107` documents the matching variable as a plot range. Unresolved; Gilly knows which.
- `relax_threshold` is a percentage (`relax_pfss_world.pm:173`); say so in the unit.
- Some keys exist only in profiles (`flow_method`, `light`, `recompute_wind`, `observer_lon`), not in `[DEFAULT]`; the registry must not assume every key is in config.ini, and must survive keys being renamed.
- Not checked by anyone: whether `recompute` reaches `get_pfss.py` (its command line passes no force flag; `force_trace` appears at `get_pfss.py:145`).

## Open questions, all for Gilly

1. **Fix the pipeline first?** The stale-world reuse is a pipeline behaviour, not a UI bug. Options: put the relaxation settings (or a hash) in the relaxed file name, delete or rename old relaxed worlds when settings change, or leave it and rely on the warning plus new batch names. The UI work is much simpler if the pipeline stops reusing stale worlds.
2. Should Duplicate use the job's snapshot (reproducible) or the current preset (picks up preset edits)? Recommendation: the snapshot, with a visible "preset has changed since" note.
3. Which parameters are the important ones? Start from `COMMON` plus the six above and edit.
4. What do `inner_fov` and `outer_fov` actually control?
5. Who reviews the registry text? It is science content and should not ship unreviewed.
