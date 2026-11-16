# FLUX / fluxon-mhd reference

Reference for future Claude/Codex sessions. Built 2026-11-11 from the sources listed at the end.
Every statement is from a cited source unless marked **(inferred)**. Paths: `repo/` means the GitHub
repo `lowderchris/fluxon-mhd` (commit 2c1f563, 2026-10-21); `lib/*.c` in that repo is byte-identical to
`/Users/gilly/flux/lib/*.c`. The installed Perl modules under
`/Users/cgilbert/miniconda3a/lib/perl5/site_perl/lib/perl5/darwin-thread-multi-2level/Flux*` are identical
to `repo/pdl/Flux/*/*.pm`. The wiki (`fluxon-mhd.wiki.git`) was last edited 2018 and is stale in places
(see Disagreements).

## 1. What FLUX is

Sources: `repo/README.md`, `repo/doc/flux.md`, wiki `Home.md`, `User_Manual.md`, `Fluxel.md`,
`Image_fluxel.md`, `Voronoi_cell.md`, `Stiffness_coefficient.md`.

- FLUX = Field Line Universal relaXer. Original by Craig DeForest and Charles Kankelborg (SwRI/MSU);
  maintained by Chris Lowder. Reference paper named in the README: Lowder, Gilly & DeForest 2024
  (ADS bibcode 2024ApJ...965...13L); method paper DeForest & Kankelborg 2007 JASTP (bibcode 2007JATP...69..116D).
- Fluxon MHD: the field is a set of discretized field lines ("fluxons"), each carrying a **finite**
  quantum of flux, each a piecewise-linear chain of vertices. Topology is preserved exactly: no
  reconnection unless explicitly triggered. Coordinate-free (Cartesian storage, vector ops only).
- Mode of use: quasi-static nonlinear force-free relaxation with fixed topology, low beta. Relaxes
  "F - ma = 0" by moving each vertex along the net force. Relaxation steps take no physical time;
  a "time step" = boundary update + many relaxation steps.
- Forces: magnetic tension ("curvature"), magnetic pressure ("field gradient") and a vertex-distribution
  pseudoforce. Pressure/B are estimated from a 2-D Voronoi cell computed in the plane perpendicular to
  each fluxel (neighbors projected as points, non-Euclidean projection, hull algorithm).
- Force law queue = ordered list of function names in `lib/physics.c` `FLUX_FORCES[]`. Prefixes:
  `f_` accumulate force, `b_` estimate B, `e_` estimate energy. fluxpype uses
  `f_pressure_equi2b f_curvature f_vertex4 b_eqa`; `make_world_sphere` writes
  `f_pressure_equi2b f_curvature f_vertex4` (same as `new_world` default seen in output).
  Full list: b_simple, b_eqa, e_simple, e_simple2, e_open, e_eqa, f_curvature, f_curvature2, f_curvature3,
  f_pressure_equi, f_pressure_equi2, f_pressure_equi2a, f_pressure_equi2b, f_curv_hm, f_curv_m,
  f_p_eqa_perp, f_p_eqa_radial, f_vert, f_vert4, f_vertex, f_vertex2, f_vertex3, f_vertex4, f_vertex5.
- Step law (World.pm POD `step_scales`): `dx = F * dt * d^pd * B^pb * S^ps * DS^pds`; d = harmonic mean
  neighbor distance, B = field, S = stiffness, DS = difference stiffness. Recommended pb=0, pd=2, ps=0,
  pds=0 (wiki User_Manual). `dtau` default 0.1 (`lib/data.c`); POD says keep 0.3 to 0.5 early.
- Neighborhood acceleration "coefficients": final step = sum over neighbor generations of averaged
  naive steps times coefficient; `[1]` switches it off (wiki User_Manual, "experimental").
- Boundaries (wiki User_Manual, `lib/model.c`): global photosphere (plane, sphere, cylinder) enforced
  by **image fluxels** reflected through the (tangent) plane, plus a post-step push back inside; a
  second boundary `photosphere2` ("B2"). Image fluxels are not made for line-tied end fluxels.
- Fluxon end conditions (`bound` field of a concentration; names in `lib/model.c` `F_B_NAMES`):
  `fl_b_tied_inject` (default, `data.c:494`), `fl_b_tied_force`, `fl_b_open`, `fl_b_plasmoid`.
- Open fluxons: end on the special concentrations fc_ob (label -1, start) or fc_oe (label -2, end).
  Endpoint moves under forces, then is projected radially onto a sphere centred at the open FC's `x`
  with radius `locale_radius`. `auto_open=1` cuts and opens fluxons that cross that sphere (U-loops allowed).
- Plasmoids: fluxon from fc_pb (-3) to fc_pe (-4), `plasmoid` flag set; endpoints copied from the
  second/second-to-last vertices so the loop closes. Created explicitly or by self-reconnection.
- Reconnection: threshold criteria `rc_a_ad2`, `rc_a_ad2_h`, `rc_a_ad2_h_ad2hmax`, `rc_a_ad2_r_max`,
  `rc_a_ad2_loc` (`lib/physics.c` `FLUX_RECON[]`); run separately via `global_recon_check`.
- Vertex fixing (World.pm POD): `fix_proximity($thresh)` subdivides fluxels whose length/closest-approach
  ratio exceeds thresh; `fix_curvature($hi, $lo)` adds vertices where bend angle > hi (radians) and
  removes where < lo (only if proximity criteria also allow). Both need neighbors primed first.
- Neighbors: `update_neighbors(1)` global O(n^2) search; `(0)` local O(n). Stored VNEIGHBOR rows let a
  reloaded world skip the global search.
- Parallelism: `concurrency` = number of fork()ed daughters; only `*_parallel` entry points use it;
  results come back over pipes in binary dump format (wiki First-stage_parallelization).
- Gravity/mass code exists; `data.c:547` comment: "default: time in hours, distance in solar radii".

### Entry points

| Layer | What | Where |
|---|---|---|
| C library | `libflux.a` (data.c, geometry.c, io.c, model.c, physics.c); no standalone executables found | `repo/lib`, headers `repo/include` |
| Perl/PDL | `use PDL; use Flux;` objects Flux::World, ::Concentration, ::Fluxon, ::Vertex (tied hashes) | `repo/pdl/Flux` |
| PDL autoload scripts | `simple_relaxer`, `make_world_sphere`, `fluxon_placement_hilbert`, etc. | `repo/pdl/PDL/*.pdl` |
| Python (repo) | `rdworld.py`, `plt-world.py`, `plt-relax.py`, `vizworld.py` | `repo/py` |
| Build | top `Makefile` targets everything/install/condaflux | `repo/Makefile` |

Key Flux::World methods (World.pm POD): `read_world($f)` (`.gz` gunzipped via /tmp), `str2world($s)`,
`write_world($f)` (`.gz` gzipped), `string`, `fluxon_ids`, `fluxons(@ids)`, `vertex_ids`, `vertices`,
`concentrations`, `concentration($id)`, `fluxon($id)`, `vertex($id)`, `new_concentration`, `emerge`,
`forces(...)`, `reconnection(...)`, `update_neighbors($g)`, `update_force($g)`, `relax_step($dt)`,
`fw_stats` (hash: n, f_av, fs_av, f_ma, n_av ...), `fix_proximity`, `fix_curvature`, `render`,
`interpolate_value`, `bfield`, `energy`. Binary: `Flux::binary_dump($world,$fname)` and
`Flux::binary_read_dumpfile($fname)` in `Flux.pm`.

### How fluxpype drives FLUX

Sources: fluxpype `README.md`, `pyproject.toml`, `config_runner.py`, `science/magnetogram2wind.pdl`,
`science/make_world_from_pfss.pm`, `science/make_world_sphere.pm`, `science/relax_pfss_world.pm`,
`science/simple_relaxer.pm`, `science/pfss_funcs.py`, `science/fluxlight.py`, `science/open_world_python.py`.

1. CLI `flux-config-run` (also `flux-config-view`, `-edit`, `-gallery`, `flux-install`) runs
   `magnetogram2wind.pdl` per CR / fluxon count from `config.ini`.
2. `get_magnetograms.py` (HMI or ADAPT synoptic map, reduced) then `get_pfss.py` (PFSS, footpoints,
   traced field lines) write `floc/floc_open_cr<CR>_r<reduce>_f<N>_<inst>.dat` and `floc_closed_...dat`.
3. `make_world_from_pfss.pm` reads those with `rcols`, converts (lon,lat,r) deg via `t_spherical`, calls
   `make_world_sphere(@lines, {rmax=>199})` -> text -> `str2world` -> `write_world` to
   `world/cr<CR>_f<N>_<inst>.flux`.
4. `relax_pfss_world.pm`: sets forces, `concurrency`, loops `simple_relaxer` (update_force[_parallel] +
   relax_step[_parallel]) in cycles of `do_steps`, `fix_curvature`/`fix_proximity` between cycles until
   stiffness (f_av/fs_av*100, percent) < threshold; writes `..._relaxed_s<stepnum>.flux`.
5. Wind/flow maps (`get_wind`, `gen_fluxon_*flow.pm`), plots, then `fluxlight.py` (Thomson-scattering
   image from the relaxed world, Python reader `open_world_python.read_flux_world`).

## 2. The text `.flux` world format

Authority: `lib/io.c` (`next_line`, `footpoint_action`, `read_world`, `fprint_world` and helpers).
Wiki `User_Manual:_ASCII_data_format.md` agrees in outline but is stale (section 6).

General rules (io.c):
- One directive per line, whitespace-delimited, parsed independently in order; files **modify** an
  existing (initially empty) world. Leading whitespace ignored (so tab-indented VERTEX is fine).
- Blank lines and lines whose first non-space char is `#` are skipped. No trailing comments.
- Dispatch is on the **first character** (case-sensitive uppercase), sometimes the second; e.g. any
  word starting `F`+`R` is FRAME. Unknown directive -> message, `footpoint_action` returns 1 and
  `read_world` stops reading (loop ends on any nonzero return). Recognized-but-failed -> stderr, returns 2, also stops.
- `.gz`: handled only in Perl `read_world`/`write_world` (shell gunzip/gzip via /tmp); C reads plain text.
- Numbers parsed with `%lf`-style (NUMCHAR) format; writer uses `%.17f`/`%.17g` for coordinates.

### Directives

| Row | Grammar | Notes (io.c) |
|---|---|---|
| FRAME | `FRAME n` | Needs world state NEW (or READY with larger n); sets frame_number, state LOADING. Writer emits it after a `######` comment block. |
| FINISH | `FINISH` (any `FI...`) | Sets state LOADED and stops parsing. Optional. |
| GLOBAL | `GLOBAL <KEY> args` | see table below |
| NEW | `NEW id x y z flux` or `NEW id x y z flux boundary locale_radius` | Creates FC. Negative id: warning "automagic FC ... ignoring", row skipped. boundary is a name from F_B_NAMES or `(null)` (= use world default_bound). Writer always emits 7-col form, tab-separated, and only for id >= 0. |
| MOVE | `MOVE id x y z flux` | Moves existing FC and sets its flux. Must be exactly 5 values. |
| LINE | 12-param: `LINE fl_id v_start v_end fc0 fc1 flux sx sy sz ex ey ez`; 6-param: `LINE fl_id v_start v_end fc0 fc1 flux`; 4-param: `LINE fl_id fc0 fc1 flux` | Only processed if `fl_id <= -11 or fl_id > 0` (ids -10..0 silently skipped). FCs must exist and have opposite-sign nonzero flux. If fc0 flux < 0, fc0/fc1 swapped. Stored flux = abs(flux). Plasmoid iff fc0=-3 and fc1=-4 (one alone is an error). v_start/v_end 0 or absent -> labels `-(2*fl_id+100)` and `-(2*fl_id+100)+1`. 6/4-param: end vertices placed at the FC positions; 12-param: at (sx,sy,sz),(ex,ey,ez). Existing fluxon with same FCs: only flux updated. |
| VERTEX | `VERTEX fl_id v_id pos x y z` | Interior vertex at position `pos` along fluxon (0 = start, -1 = end per comment; first interior is 1). Writer emits tab-indented, pos 1..n, end vertices omitted (they live on the LINE row). |
| VNEIGHBOR | `VNEIGHBOR v_id n_id` (also `VN`, `V_NEIGHBOR`; any `VN...`/`V_...`) | Adds n to v's neighbor list and v to n's nearby list. Rows with n_id 1 or 2 are ignored. Missing vertex -> warning only. Writer emits all neighbor relations after all LINE blocks. |
| DELETE, SPLIT, JOIN, EMERGE, CANCEL | first letter D/S/J/E/C | Recognized but "not implemented" -> error return 2 (stops the read). |

### GLOBAL keys (reader dispatch -> writer form)

Reader matches on leading letters; writer (`fprint_world`) emits the spelling in the first column.

| Writer form | Reader accepts | Meaning |
|---|---|---|
| `GLOBAL FORCES f1 f2 ...` | any key starting `F` | Force-law queue by name (or `0x...` pointer). |
| `GLOBAL STATE n  (NAME)` | `ST...` | World state: 0 NEW, 1 LOADING, 2 LOADED, 3 WORKING, 4 RELAXED, 5 READY (`include/data.h`). Trailing `(NAME)` ignored. Bookkeeping only. |
| `GLOBAL BOUNDARY  TYPE x y z n1 n2 n3` | `BO...`; TYPE first char `P`/`1`, `S`/`2`, `C`/`3`, `N`/`0` | Photosphere. PLANE: origin x, normal n. SPHERE: centre x, radius n1 (reader reads 4 numbers, forces n2=n3=0). CYL: origin x, axis n, radius = abs(n). |
| `GLOBAL B2  TYPE ...` | `B2...` | Second boundary, same syntax. |
| `GLOBAL SCALING B|D|S|DS v` | `SC...` | Step-law powers pb, pd, ps, pds. |
| `GLOBAL RSTEP n` | any `R...` (wiki: REL_STEP) | Relaxation step counter (bookkeeping). |
| `GLOBAL DTAU v` | any `D...` | Relaxation timestep. |
| `GLOBAL COEFFICIENTS n c1..cn` | `COE...` (wiki: COEFFS) | Neighborhood acceleration coefficients. |
| `GLOBAL OPEN x y z r auto` | any `O...`; 4 values accepted (auto=0, legacy warning) | Open sphere: centre x, radius r, stored on fc_ob and fc_oe (`locale_radius`); auto = auto_open flag. |
| `GLOBAL SKEW_HANDLING s` | `SK...` (wiki: SKEWFLAG) | Experimental projection hack; should be 0. |
| `GLOBAL CONCURRENCY n` | `CON...` | Number of parallel daughter processes. |
| (none) | `PHOTOSPHERE x y z nx ny nz [type]` or `<NONE>` | Legacy photosphere. |
| (none) | other `B...` (e.g. B_FLAG / B_NORMALIZATION) | Deprecated; sets scaling (b=0 or 1, d=2, s=0, ds=0). |

Any other GLOBAL key: "Unrecognized GLOBAL command", returns 1 and **stops the read**.

### Writer output order (`fprint_world`)
header string, `######` comment, `FRAME`, GLOBAL FORCES, STATE, BOUNDARY, B2, SCALING B/D/S/DS, RSTEP,
DTAU, COEFFICIENTS, OPEN, SKEW_HANDLING, CONCURRENCY, blank lines; NEW rows (id >= 0, tree order);
LINE blocks grouped by start FC with `# FC <label>` comments (including the two image fluxons, see
below); VNEIGHBOR rows.

### Built-in objects created by `new_world` (`lib/data.c`, `include/data.h`)

| Object | Label | Notes |
|---|---|---|
| fc_ob / fc_oe | -1 / -2 | open start (flux +1) / open end (flux -1), bound `fl_b_open` |
| fc_pb / fc_pe | -3 / -4 | plasmoid start (+1) / end (-1), bound `fl_b_plasmoid` |
| fc_im10, fc_im11, fc_im20, fc_im21 | -8, -9, -10, -11 | image FCs, flux 0 |
| image fluxons fl_im, fl_im2 | -9, -10 | written as `LINE -9 -1 -2 -8 -9 0 ...` and `LINE -10 -3 -4 -10 -11 0 ...`; reader skips them |
| image vertices | -1, -2, -3, -4 | ends of the image fluxons |

So 8 built-in concentrations, 2 built-in fluxons, 4 built-in vertices. Users should not use negative
FC labels in NEW (wiki: -1..-99 reserved; reader actually ignores all negative NEW ids).

### Counts convention (verified by loading files with the installed Flux, fluxenv perl)

For a file **written by FLUX** (which includes the two image LINE rows):

| Library quantity | Equals |
|---|---|
| vertices (`world_links.n`, `vertex_ids`, "Read ... N vertices") | VERTEX rows + 2 x (all LINE rows) |
| fluxons in "Read ..." message and `fluxon_ids` | all LINE rows (= real + 2 image) |
| `$world->fluxons` (drops ids -11..-1) | LINE rows with id > 0 |
| concentrations | NEW rows + 8 |

Check: `cr2150_f1000_hmi_relaxed_s300.flux` has LINE 545 (543 with id > 0), VERTEX 8464, NEW 1034;
Flux reports 9554 vertices, 545 fluxons, 1042 concentrations; `fluxons()` = 543.
For a hand-written file without image LINE rows, add 4 vertices and 2 fluxons to the raw formula.

### Units, coordinates, signs

- FLUX itself is unitless/coordinate-free (doc/flux.md). Coordinates are Cartesian x,y,z.
- In fluxpype worlds: `GLOBAL BOUNDARY SPHERE 0 0 0 1` (photosphere radius 1, i.e. **solar radii**;
  consistent with the `data.c` gravity comment) **(inferred: unit is R_sun)**.
- `GLOBAL OPEN 0 0 0 R 1`: `make_world_sphere` writes `R = rmax + 1` with auto_open=1. Current fluxpype
  passes `rmax=>199` (R=200); the commented-out older call used `rmax=>21.5`, which gives the
  `22.5` seen in the sample cr2150 file. Lines whose start/end radius >= rmax are attached to FC -1/-2.
  Verified: first open LINE in the sample starts at radius approx 22.5.
- Frame: footpoints/field lines come from the PFSS output `coordinate_frame` of a synoptic magnetogram,
  (lon, lat, r) in degrees, converted with PDL `t_spherical` **(inferred: Carrington heliographic,
  z = rotation axis; not stated anywhere)**. Not HEEQ as far as found.
- Flux sign: NEW flux > 0 = source (fluxon start), < 0 = sink (end); "convention, code does not enforce"
  (wiki), but the LINE reader reorders so fc0 has positive flux and rejects same-sign pairs.
  fluxpype writes NEW flux +1/-1 (one fluxon per FC) and LINE flux 1: flux is in units of the
  per-fluxon flux quantum, the physical value is not in the file **(inferred)**.
- `make_world_sphere` formats NEW/VERTEX with `%9.3g` (3 significant figures) and rescales the first/last
  point onto the sphere; relaxed files written by FLUX carry full `%.17` precision.

## 3. Other files fluxpype touches

| File | Format | Source / notes |
|---|---|---|
| `floc/floc_open_cr*_r*_f*_*.dat`, `floc_closed_...dat` | `np.savetxt` text, 5 cols: fluxon number, polarity sign, lat (deg), lon (deg), r | `pfss_funcs.py`; read as `($flnum,$flx,$lat,$lon,$rad)` in `make_world_from_pfss.pm`. Open lines extended to r = 200. |
| `world/cr<CR>_f<N>_<inst>.flux` | initial text world | `make_world_from_pfss.pm` |
| `world/..._relaxed_s<k>.flux` | relaxed text world | `relax_pfss_world.pm` |
| `world/<name>.light` | **Python pickle** (protocol 4), not a FLUX format | `fluxlight.py` `do_fluxlight`: path = `.flux` replaced by `.light`; cache of `simulate_thomson_scattering` results. |
| `light/cr<CR>_fluxlight_light_<method>_<nz>[...].png` | renders | `fluxlight.py` |
| `cr<CR>_f<N>_fluxlight_world.png`, `_world_zoom.png`, `_world_zoom_{x,-x,y,-y,z,-z}.png` | matplotlib 3-D plots | `open_world_python.py` `plot_all`-style method; zoom = axes limited to +/-3 (R_sun inferred) |
| magnetogram FITS | HMI or ADAPT synoptic, reduced by `reduce` | `get_magnetograms.py` (filenames not checked) |

### `.light` layout
Undocumented in the FLUX repo (FLUX knows nothing about it). From `thomson_scattering_flux.py` plus
`fluxlight.py`, the pickled dict has keys: `B_total`, `B_polarized`, `Polarization_angle`,
`Polarization_fraction`, `X`, `Y`, `impact_parameter`, `position_angle`, `fov`, `Column_Density`,
`influence_length`, `nz`, `nxy`, `world`, `light_path`. Opcode scan of the 31 MB sample (pickletools,
no unpickling) confirms header `80 04` (protocol 4), first key `B_total`, astropy `Quantity` values
(units incl. `solRad`), and `world` as a pickled `fluxpype.science.open_world_python.FluxWorld`.
Loading needs numpy, astropy and an importable fluxpype; unpickling executes code, so only load trusted files.

### Zoom images
Six views from `ax.view_init(elev, azim)`: x=(0,180), -x=(0,0), y=(0,90), -y=(0,-90), z=(90,0),
-z=(-90,0). In matplotlib azim=0 puts the camera on +x, so the "x" image may be viewed **from** -x
**(inferred, not checked by rendering)**.

### Binary dump format (`lib/io.c`, `include/io.h`; wiki io.c page)
- Machine-dependent, not portable, mainly for parallel pipes; not used by fluxpype for storage.
- Stream of fields: 3 C `long`s `[BD_FENCE=0xAB5C155A, type, len]` then `len` bytes; ends with `BD_END`.
- Types: 1 END, 2 FLUXON_PIPE (raw pointers, same-process only), 3 HDR, 4 WORLD, 5 CONCENTRATION,
  6 FLUXON, 7 NEIGHBORS, 8 POSITION, 9 STEP.
- HDR payload: 4 longs: 0xAABBCCDD (endianness), version 1, sizeof(NUM), float 3.1416.
- WORLD payload: version long 1, sizeof(WORLD), raw WORLD struct padded to 16, `WORLD_VAR_FENCE`
  0x5AFEBEEF, photosphere planes (6 NUM each if present), 80-byte name strings (default bound,
  f_funcs, m_funcs, rc_funcs), `WORLD_END_FENCE` 0x5AFED00D. Other fences: NEIGHBOR 0xBED57EAD,
  VERTEX_END 0x5AFEB055.
- Perl `Flux::binary_dump` writes WORLD + all concentrations + fluxons + neighbors (no HDR); does not
  write a complete standalone-portable world (wiki says full world dump "not yet" done).

## 4. Glossary

| Term | Meaning (source) |
|---|---|
| fluxon | discretized field line with finite flux; FLUXON struct + linked VERTEX list (wiki Fluxon) |
| fluxel | segment between two consecutive vertices ("flux element") (wiki Fluxel) |
| vertex | point on a fluxon; first/last are end vertices tied to the FCs (io.c) |
| concentration (FC) | fluxon anchor with location, flux, `bound`, `locale_radius` (Flux.pm POD) |
| footpoint | FC position on the photosphere where a closed fluxon or an open fluxon's inner end is tied (fluxpype usage) |
| neighbor | vertex whose fluxel bounds this fluxel's Voronoi cell; stored as VNEIGHBOR (World.pm) |
| image fluxel | reflection through the boundary tangent plane that confines Voronoi cells (wiki) |
| open / closed | open: one end on fc_ob/fc_oe (open sphere); closed: both ends on ordinary FCs |
| plasmoid | fluxon from fc_pb to fc_pe, free-floating loop (wiki User_Manual) |
| relaxation step | one `update_force` + `relax_step(dt)`; no physical time (doc/flux.md) |
| stiffness | magnitude-of-sum / sum-of-magnitudes of vertex forces; 0 = perfect balance; fluxpype prints f_av/fs_av*100 in percent (wiki, simple_relaxer.pm) |
| cycles | fluxpype outer loop: `do_steps` relaxation steps, then curvature/proximity fixing (relax_pfss_world.pm) |
| locale_radius | FC size; for open FCs the open-sphere radius (wiki Data_Structures) |

## 5. Gotchas

1. Raw row counts differ from library counts: see the counts table (image fluxons/vertices, 8 built-in FCs).
2. The reader **stops at the first unrecognized or failing line** (including unknown GLOBAL keys and the
   unimplemented DELETE/SPLIT/JOIN/EMERGE/CANCEL) and the rest of the file is silently not loaded.
3. LINE ids -10..0 are skipped; ids <= -11 are accepted by C but fluxpype's Python reader keeps only id > 0.
4. LINE with 7 to 11 numeric fields: treated as the 12-param form with unset endpoint coordinates
   (uninitialized) **(inferred from code)**.
5. Fc swap on LINE (fc0 negative) does not swap the given end-vertex labels/positions **(inferred)**.
6. For open fluxons the end vertex sits on the open sphere, not at FC -1/-2 (which are at the sphere centre);
   use the LINE row coordinates, not the NEW positions.
7. SPHERE inside/outside flag (sign of normal[1], `model.c`) cannot be set from a text file: reader zeroes n2.
8. fluxpype's Python reader requires the 12-column LINE form and stops at the first VNEIGHBOR row.
9. `f<N>` in filenames is the requested fluxon count; the sample `f1000` file has 543 real fluxons.
10. `_relaxed_s<k>`: k = cycle index x do_steps at the last cycle, not exactly the steps run
    (do_steps also doubles after `stop_fixing_after`) **(inferred from relax_pfss_world.pm)**.
11. Perl `.gz` handling and `str2world` use fixed `/tmp/...-$$` paths.
12. `make_world_sphere` coordinates have only 3 significant figures in initial worlds.
13. `.light` is a pickle tied to fluxpype class layout; renaming `FluxWorld` breaks old caches.

## 6. Disagreements: wiki vs code

| Wiki says | Code does |
|---|---|
| `GLOBAL BOUNDARY x1 x2 x3 n1 n2 n3 type` (type last) | type token **first**: `GLOBAL BOUNDARY SPHERE 0 0 0 1` (wiki's own example also puts it first) |
| SPHERE: n2 < 0 means system inside | text reader forces n2 = 0 |
| `GLOBAL FORCE`, `REL_STEP`, `COEFFS`, `SKEWFLAG` | accepted by prefix, but writer emits FORCES, RSTEP, COEFFICIENTS, SKEW_HANDLING; wiki omits CONCURRENCY, B2 details |
| LINE has 4- and 6-number forms | also a 12-number form (writer always uses it) |
| `fl_b_start_open`, `fl_b_end_open`, `fl_b_start_plasmoid`... (model.c page) | single `fl_b_open`, `fl_b_plasmoid` in F_B_NAMES |
| neg. FC labels -1..-99 reserved | reader ignores every negative NEW id; built-ins are -1..-4, -8..-11 |
| example VNEIGHBOR to vertex `1` (old image label) | image vertices now -1..-4; reader drops neighbor ids 1 and 2 |
| wiki GLOBAL OPEN `auto` always present | 4-value form accepted with legacy warning |
| io.c page: macro `MD_MAX_TYPENO` | `BD_MAX_TYPENO` |

## Sources consulted

- https://github.com/lowderchris/fluxon-mhd (README.md, doc/README.md, doc/flux.md, lib/io.c, lib/data.c,
  lib/model.c, lib/physics.c, include/io.h, include/data.h, pdl/Flux/Flux.pm, Flux.xs, World/World.pm,
  World/World.xs, pdl/PDL/make_world_sphere.pdl, py/readme.md, py/rdworld.py)
- https://github.com/lowderchris/fluxon-mhd.wiki.git (Home, User_Manual, User_Manual:_ASCII_data_format,
  User_Manual:_Code:_io.c, User_Manual:_Code:_model.c, User_Manual:_force_laws, User_Manual:_Data_Structures,
  Fluxon, Fluxel, Image_fluxel, Voronoi_cell, Stiffness_coefficient, First-stage_parallelization)
- /Users/gilly/flux/lib/{io,data,model,physics,geometry}.c (identical to repo)
- /Users/cgilbert/miniconda3a/lib/perl5/site_perl/lib/perl5/darwin-thread-multi-2level/Flux.pm, Flux/*.pm
- /Users/gilly/vscode/fluxons/fluxpype: README.md, pyproject.toml, fluxpype/config_runner.py,
  science/{magnetogram2wind.pdl, make_world_from_pfss.pm, make_world_sphere.pm, relax_pfss_world.pm,
  simple_relaxer.pm, pfss_funcs.py, get_pfss.py, fluxlight.py, thomson_scattering_flux.py,
  open_world_python.py}
- Sample data: fluxpype/data/batches/fluxlight/data/cr2150/world/cr2150_f1000_hmi_relaxed_s300.{flux,light}
- Empirical: loaded a 1-fluxon test world and the cr2150 file with fluxenv perl + installed Flux.

## Not found / unresolved

- No document states the physical unit of length or the coordinate frame; R_sun and Carrington are inferred.
- Physical flux per fluxon is not stored in `.flux` files; where fluxpype records it was not checked.
- `.light` has no format spec; layout above is from fluxpype source plus an opcode scan.
- Field-line/footpoint file formats other than floc `.dat` (e.g. wind/flow outputs of `get_wind`) not documented here.
- Magnetogram file names/paths from `get_magnetograms.py` not verified.
- Zoom-image view direction labels not checked against a rendered image.
- Why the VNEIGHBOR reader drops neighbor ids 1 and 2 is not documented (presumed legacy image labels).
- No standalone FLUX command-line tools exist in the repo; "CLI" means fluxpype's `flux-config-*` scripts.
