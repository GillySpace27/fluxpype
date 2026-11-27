# Changelog

All notable changes to fluxpype. Versions follow [Semantic Versioning](https://semver.org/); before 1.0, a minor
bump (0.x.0) may change behavior, a patch bump (0.x.y) is fixes only. PR titles use conventional-commit prefixes, so
`git log --format=%s <last-tag>..` gives the raw list for the next entry.

## 0.5.0 (unreleased)

Since 0.4.0 (2026-11-13). Most of the second half comes from the 2026-11-26 FLUX stack audit; audit ids are in
parentheses.

### Changed (check before upgrading)

- Output file names for wind and FLUXlight now carry `_f{N}_{inst}`, so runs with different fluxon counts or
  instruments no longer overwrite each other (A7-01, #94). Old files are not renamed.
- FLUXlight writes one FITS file per channel and its cache is keyed on the render parameters; old `.light` caches
  re-render once (A7-03, #101).
- Plumber API errors are `{"detail", "code"}` with a stable code; an unknown job is now 404 (was 409) and an unknown
  task 404 (was 400). MCP errors show the code (A3-08, #113).
- MCP `get_task_image`: index -1 is the headline image and 0 and up follow the Results image list; artifact rows no
  longer carry `kind` (A3-07 #109, A4-04 #115).
- Dead science variants removed (`*_bkk`, `footpoint_distances*`, `map_fluxon_flow.pm` and others); they stay in git
  history (A2-11, #110). The wheel now ships the `.pm` and `.pdl` files.
- PyQt5, beautifulsoup4, lxml and zeep dropped from dependencies; `requirements.txt` is now `-e .` (A6-11, #107).
- Python 3.10 to 3.13 supported; numpy 2 supported (#73, A6-06 #100).
- `do_wind_calc = 0` now really skips the wind stage (#86).
- One install path: `install.sh` makes one `.venv` with the plumber and mcp extras (A6-05, #120).

### Added

- FLUXlight vs AIA side-by-side compare view, `GET /api/tasks/{id}/compare` and MCP `get_comparison` (A4-03, #114).
- FLUXlight EUV renderer, synthetic AIA DN/s, GHOSTS port, Cranmer and Gilly loop model, KD-tree neighbour seeds and
  full-precision worlds (#85).
- SSH and Slurm execution backends for Plumber (#72).
- Scene Designer: planar arcade scene spec and `.flux` compiler, typed edit operations with a version tree,
  potential-field seed and footpoint shear, analytic mirror-lid potential reference, validate/preview routes and a New
  scene form (#52, #63, #70, #71, A4-05 #104).
- Relaxation movie frames and GIF (`relax_movie_n`) (#74).
- Plum, the Ask agent: Co-design mode, hand-off work orders, skin-tone setting, static guide (#51, #55, #56, #57, #78).
- Plumber: inbox, live Runs feed, persisted job event log, `flux-plumb watch`, picture cards, external FLUX runs in the
  process monitor, a server-restart banner (A7-09 #95), per-stage timing report (A5-01 #105).
- `flux_api.pm` facade at the FLUX core boundary and `docs/RULES-FOR-CALLING-FLUX.md` (A1-01, A1-09, #117).
- `flux-doctor` reports the libflux optimization level (A1-04, #93); `flux_perl` setting picks the Perl for every stage.
- Real-FLUX golden test harness with relax provenance (A6-02, A7-07, #92).
- One stage manifest, `fluxpype/stages.py`, read by the monitor, results, fake backend and server (A2-09, #118).

### Fixed

- Magnetogram to FLUXlight path unblocked for the SBIR run: PFSS empty-line handling, AIA header, FLUXlight about 6x
  faster (#86).
- FLUXlight loads input-form worlds through one world parser (A2-06, #102).
- SSH backend no longer reports a finishing task as lost; pid-reuse guard (#103).
- Results, Overview and MCP agree on a task's headline image (A3-07, #109).
- Plumber: one owner for task state (A3-09, #99); `server.json` written only once listening (#59); cached footpoint
  row count (A5-07, #88); Plum's wind and speed guide corrected (A4-06, #89).
- `convert_value` uses `ast.literal_eval` instead of `eval`.
- ADAPT downloads from the NSO listing (#58).

### Performance

- Web: no polling while the WebSocket is up and no 2 Hz task pushes; measured 145 to 23 requests per 20 s on a job
  page (A4-07, #111). Lazy page routes; main bundle 710 KB to 442 KB (A4-10, #116).
- Gzip for API responses over 2 KiB (A3-05, #90); lazy heavy imports (A5-02, #91); pytest-xdist in CI (A6-08, #97).

### Known issues

- FLUX relax is not bit-repeatable run to run (A1-05): an uninitialized `r_cl` in fluxon-mhd's `new_vertex`. The fix
  goes into the GillySpace27 fork with a seed toggle; until it lands, a realflux golden can fail and pass on re-run.

## 0.4.0 (2026-11-13)

See the GitHub release notes for 0.4.0.
