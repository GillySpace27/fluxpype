# Plumber Scene Designer: scope

Status: scope draft 5 (2026-11-13). Draft 5 adds section 13: Gilly's decisions and the build order A, B, C. Nothing built. Long-term goal, not a sprint. Draft 2 adds the Designer panel (section 9: a shared editor that Gilly, Claude and scripts all edit live) and reorders the phases so the panel comes before the conversational designer. Draft 3 adds section 10: FLUXlight as fluxpype's general light layer. Draft 4 adds section 11: the 3-D graphical editor, structure gallery and library, drag-and-drop pipeline timeline, live script preview, test run, scene selector, help system, download manager and an affordance review; and section 12: an independent critique and what changed because of it.
Code read for this: `GillySpace27/fluxon-mhd` at `3f65ebd` and `GillySpace27/fluxpype` main at `4187b41`. I read the code; I did not run any planar or synthetic world. Each claim below is marked **checked** (I read the line), **inherited** (from the earlier "Run FLUX on a high-res photospheric patch" thread or `fluxpype/plumber/ask/flux_reference.md`, not re-read) or **uncertain**.

## 1. Bottom line

1. **The bottleneck is not the language model. It is that fluxpype has exactly one run type.** A task is the tuple `(adapt, rotation, fluxon_count, method)` (`config_runner.py:522-532`, `select_tasks`), the script receives exactly those four positional arguments (`config_runner.py:318-325`), Plumber's stage list is a fixed ten-stage global pipeline (`plumber/monitor.py:26-37`) recognized by log regexes, and `make_world_sphere.pm:38` always writes `GLOBAL BOUNDARY SPHERE`. Until "a run" can mean something other than "a Carrington rotation through PFSS", there is nothing for a scene designer to wire a new scene into.
2. **The propose-then-click machinery you want already exists.** The Ask bar (`plumber/ask.py`, `plumber/proposals.py`) runs Claude with read tools, and its write tools only store server-validated proposals that run when you click. The Scene Designer is that same gate with a richer thing being proposed: a scene spec instead of a preset override.
3. **FLUX core supports much more than fluxpype uses** (planar photosphere, a second boundary, an open sphere anywhere, synthetic tubes and loops, emergence, footpoint motion, reconnection, plasmoids). Several gaps are real, though: no periodic boundaries, at most two boundary surfaces plus the open sphere (so no true box), no Cartesian potential-field seed, no gravity or mass in the force list, and no renderer for a local scene. Per Gilly (2026-11-13), FLUXlight becomes the home for every light-producing code fluxpype uses or needs (section 10).
4. **Recommended shape:** a recipe registry in fluxpype (global PFSS becomes recipe #1), a declarative scene spec compiled by deterministic builders, and the model's job limited to (a) asking the questions that change the build, (b) writing a spec the compiler validates, and (c) for genuinely new physics or drivers, opening a pull request for a new recipe that you merge. The model never hot-loads code it wrote into the running server.
5. **Draft 2: one scene, many editors.** The scene spec becomes a live document edited only through typed operations. The Designer panel, Claude, and any script calling the REST API all send the same operations, and everyone sees them land through the existing `/api/ws` socket. Claude's edits arrive as suggestions you accept or reject (or, if you turn it on per scene, as direct edits you can undo), and its "looks good" verdicts must cite computed checks. Section 9 has the details.

## 2. What you asked for, restated as two request types

| Type | Your example | What the designer must do |
|---|---|---|
| **Specified scene** | "Cartesian scene, photosphere as a plane in a 1 Mm cube, PIL down the center, mean field 200 G" | Map words to spec fields, ask only about the gaps that change the build (resolution, side and top boundaries, initial connectivity, what "average" means), flag physics limits, show a cheap preview before relaxing. |
| **Research goal** | "Investigate sunspot umbra magnetic evolution" | Say what FLUX can and cannot answer about it, offer two or three concrete scene designs that FLUX can run, ask which observable matters, then proceed as a specified scene. |

Both end in the same artifact: a scene spec plus a recipe that runs it.

## 3. FLUX capability inventory

### 3.1 fluxon-mhd core (the engine)

| Feature | Status | Evidence |
|---|---|---|
| Planar photosphere | Supported | `GLOBAL BOUNDARY PLANE ox oy oz nx ny nz` parsed at `lib/io.c:564-578`; enforced with image fluxels, `image_find` at `lib/model.c:2016` (**checked**). Used by `pdl/menagerie/twisted.flux`, `potential.flux`, `several-open-lines.flux` and others (**checked**). |
| Spherical and cylindrical boundaries | Supported | `PHOT_SPHERE`, `PHOT_CYL` cases at `lib/io.c:582-614` (**checked**). |
| Second boundary (`GLOBAL B2`) | Supported in the parser | Same three types at `lib/io.c:657-707` (**checked**). Whether plane + cylinder B2 + open sphere has ever been relaxed together: **uncertain**, I found no example file that does it. |
| More than two walls (a box, a cube) | **Not supported** | Only `photosphere` and `photosphere2` exist; the Perl `boundary` POD lists types 0 to 3 only (`pdl/Flux/World/World.pm:828-857`, **checked**). |
| Periodic side boundaries | **Not supported** | No match for "periodic" anywhere in `lib/` or `include/` (**checked**). |
| Open field | Supported | `GLOBAL OPEN x y z r auto` (`lib/io.c:762-779`); the sphere can be centered anywhere, `auto_open` cuts lines that cross it (`lib/model.c:174` onward) (**checked**). |
| Force laws | 24 registered | `FLUX_FORCES[]` at `lib/physics.c:48-71`; fluxpype uses `f_pressure_equi2b f_curvature f_vertex4 b_eqa` (**checked**). Newer laws (`f_curv_hm`, `f_p_eqa_perp`, `f_p_eqa_radial`, `e_open`) are unused by fluxpype. |
| Gravity / mass loading | **Not usable** | `m_hydrostatic` is commented out of the force table (`lib/physics.c:72`); `gravity_type` defaults to 0 (`lib/data.c:543`) (**checked**). Treat FLUX as force-free and massless. |
| Reconnection | Supported, never used by fluxpype | Criteria `rc_a_ad2`, `rc_a_ad2_h`, `rc_a_ad2_h_ad2hmax`, `rc_a_ad2_r_max`, `rc_a_ad2_loc` (`lib/physics.c:79-83`); run explicitly through `$world->reconnect` (`World.xs:757`) (**checked**). No fluxpype file mentions it (**checked**, grep). |
| Plasmoids | Supported | `fluxon_plasmoid_cleanup` and plasmoid opening in `lib/model.c:174-376` (**checked**). |
| New flux concentrations, emergence | Supported in Perl | `new_concentration`, `emerge` (`World.pm:630-750`); `emerge` currently ignores the `$flux` argument per its own POD (**checked**). |
| Removing or reassigning flux | Supported in Perl | `Concentration::delete`, `abdicate`, `open`, `cancel`; `Fluxon::reattach` (**checked**). |
| Footpoint motion (driving) | Supported, scripted | Concentrations and vertices are writable tied hashes: `rotate_poles` sets `$c->{x}` and `$v->{x}` through `interp_duction` (`pdl/PDL/ductions/rotate_poles.pdl`) (**checked**). Also `rotation_spiro`, `rotation_breakout`, `rotate_starts`. A time loop is "move boundary, relax, write"; FLUX has no physical time inside a relaxation (**inherited**). |
| Synthetic initial fields | Several generators | `twisted_loop`, `vertical_fluxons`, `random_carpet` (randomly connected twisted carpet), `tubes/gold_hoyle_tube`, `tubes/lundquist_tube`, `tubes/twisted_straight_tube`, `gold_hoyle_line_cart`, `coronal_hole` (`whole_sun`) in `pdl/PDL/` (**checked** POD headers; none run). |
| World builders | `make_world` takes `{photosphere=>[type, origin, normal], photosphere2=>..., rmax=>...}` | `pdl/PDL/make_world.pdl:58-80` (**checked**). Note line 78 writes a unit sphere whenever no boundary is given, so the default is spherical. |
| Fluxon placement on a 2-D map | Supported | `fluxon_placement_hilbert`, `fluxon_placement_fs` (Floyd-Steinberg) in `pdl/PDL/`; works on any 2-D array, and a negative count fixes flux per fluxon (**inherited**). |
| Movies of a world sequence | Exists | `make_flux_movie.pdl` (**checked** POD header only). |
| Units | None | FLUX is unitless; fluxpype's global worlds use R_sun by convention (**inherited**). A local scene must choose its own unit (Mm is natural) and record it in the spec. |

### 3.2 fluxpype (the pipeline)

| Piece | Reusable for non-global scenes? | Evidence |
|---|---|---|
| `config_runner.run_pdl_script` | Yes for launching, no for arguments: it already reads a configurable `run_script` (`config_runner.py:286`) but passes CR, fluxon count, adapt and method positionally. | **checked** |
| `make_world_sphere.pm` | No: hardcodes `BOUNDARY SPHERE` and the force list (lines 37-39). | **checked** |
| `get_magnetograms.py`, `get_pfss.py` | No: global synoptic maps and global PFSS. | **inherited** |
| `relax_pfss_world.pm`, `simple_relaxer.pm` | Probably yes: the loop is `read_world`, force updates, `fix_curvature` and `fix_proximity`; the CR only enters the file names. | Core loop **checked**; full geometry independence **uncertain**. |
| Wind models (`gen_fluxon_*flow.pm`) | No: they assume radial solar-wind flow along open global fluxons. | **inherited** |
| FLUXlight (`science/fluxlight.py`, `science/thomson_scattering_flux.py`) | Not yet: one emission process (Thomson), a radial default density in R_sun (`thomson_scattering_flux.py:25-31`), `lower_bound=1.05` (`fluxlight.py:113`). Section 10 proposes making it fluxpype's general light layer. | **checked** |
| Plumber 3-D viewer (`plumber/fluxfile.py`) | Partly: it parses any `.flux` file, but the viewer always draws a sphere of radius R0 and fits the camera to it (`web/src/components/FluxViewer.tsx:308-309`). A plane view and editor is new viewer work, not reuse. | **checked** |
| Plumber task identity, stages, cache | No: keyed on CR, fluxon count, adapt; ten fixed stages; reuse is "does the output file exist" (see `docs/plumber-iteration-loop-scope.md`, finding zero). | **checked** |

## 4. Proposed architecture

```
 you (words)  ->  Scene Designer (Ask bar, Claude)  ->  scene spec (JSON, validated)  ->  recipe (code in fluxpype)  ->  scheduler  ->  results
                   asks questions, cites the               compiled by deterministic         global_pfss, planar_synthetic,
                   capability matrix, proposes             builders; unsupported combos       driven_sequence, ...
                                                           fail with a reason
                                                                                      new recipe needed?  ->  PR on a branch  ->  you merge
```

### 4.1 Layer 0: recipes (run types)

A recipe is a small Python module under `fluxpype/recipes/<name>/` that declares:

- a parameter schema (JSON Schema), which also drives the Plumber form;
- its stage list and progress events (replacing the global `STAGES` and log regexes for that recipe; the existing structured events in `events.py` / `events.pm` already carry stage names);
- a task key: the recipe name plus a hash of the normalized parameters. This also fixes the reuse problem in the iteration-loop scope, because a changed relaxation setting changes the hash;
- an entry point: a Perl or Python driver that receives one JSON task file instead of four positional arguments;
- its outputs and which viewers apply (3-D viewer, FLUXlight, energy plot).

The current pipeline becomes recipe `global_pfss` with no change in behavior: its entry point unpacks the JSON and calls `magnetogram2wind.pdl` with the old four arguments. Presets stay as they are for that recipe.

### 4.2 Layer 1: the scene spec

One declarative document per scene. Sketch of the top-level fields (names are proposals, not code):

| Field | Choices it would offer at first |
|---|---|
| `geometry` | `sphere` (global) or `plane`; unit (`R_sun`, `Mm`); extent |
| `boundaries` | lower (plane or sphere), optional `B2` (plane, sphere, cylinder side wall), optional open sphere (center, radius, `auto_open`) |
| `flux_source` | observed (HMI or ADAPT synoptic; later a SHARP CEA cutout) or synthetic (`bipole`, `pil_linear`, `sunspot` with umbra and penumbra radii, `random_carpet`, `tube` of type Gold-Hoyle, Lundquist or twisted straight) |
| `placement` | `hilbert` or `floyd_steinberg`; fluxon count or flux per fluxon |
| `initial_field` | `pfss` (global only), `point_source_potential` (plane), `twisted_loop`, `tube`, `vertical` |
| `physics` | force-law list (default fluxpype's), reconnection criteria (default off), `auto_open` |
| `driving` | none, footpoint rotation, translation, emergence schedule, flux removal, or an observed frame sequence; number of frames |
| `relaxation` | the existing `do_steps`, `relax_threshold`, `max_cycles`, `fix_*` |
| `outputs` | 3-D field lines, energy and stiffness series, connectivity map, movie, and any FLUXlight product (section 10): Thomson total and polarized brightness today; EUV and others as they are added, each for any geometry the plasma and camera layers support |
| `provenance` | the conversation that produced it, the recipe version, the fluxon-mhd commit |

The compiler is plain code. It turns a spec into a `.flux` world through `make_world` (which already takes plane boundaries) and the existing generators, and it rejects what FLUX cannot do with a sentence you can read ("FLUX has no periodic boundaries; choose a cylinder side wall or a padded border"). The capability matrix in section 3 becomes a machine-readable file the compiler enforces and the model reads, so the model cannot promise a feature the compiler refuses.

**One new numerical piece is needed for planar scenes: an initial connectivity.** FLUX keeps whatever topology it starts with unless reconnection is switched on, so the seed matters. For discrete fluxel footpoints on a plane, the potential field of point sources above a plane with image charges is analytic, so tracing seed lines through it needs no FFT. My draft 1 estimate of a few hundred lines was too low (section 12.1): the tracer also has to assign each line to a discrete opposite fluxel, send unbalanced flux to the open sphere, and avoid crossings. `make_world` also cannot be called as is for this: it always centres the open sphere at the origin and writes a force list without `b_eqa` (`make_world.pdl:74-80`). A spectral (Alissandrakis-style) extrapolation can come later for observed cutouts.

### 4.3 Layer 2: the Scene Designer conversation

Built into the Ask bar, using its proposal gate unchanged. New pieces:

- **Tools:** `get_capabilities` (read the matrix), `validate_scene` (dry-run compile, returns errors and a cost estimate), `preview_scene` (build the unrelaxed world and a footpoint map without relaxing; seconds, not minutes), and `propose_scene` (stores a validated spec as a proposal, like `submit_job` does today). Run and "save as scene preset" stay on the card, so your click is still the only approval.
- **Question protocol:** ask only questions whose answer changes the compiled world or the interpretation; each question carries a default, so you can answer "defaults" and get a build. At most about five per round. Unanswerable or out-of-scope asks get a plain statement of what FLUX cannot do, not a workaround presented as equivalent.
- **Physics review on every card:** a short "what this scene can and cannot tell you" block, written from the capability matrix (for example: force-free, no plasma, no gravity, topology fixed unless reconnection is on).
- **Model choice:** a spec proposal is cheap enough for the Ask bar's Sonnet. Writing a new recipe (Layer 3) is not; it should run as a full Claude Code session with repository tools in a worktree.

### 4.4 Layer 3: new recipes by pull request ("the plumber installs a new pipe")

When a request needs something the spec cannot express (a new driver, a new renderer, a new generator), the designer says so and offers to draft a recipe. That runs as a separate coding session that writes `fluxpype/recipes/<name>/` with a schema, a driver, a synthetic smoke-test world and tests, on a branch in `GillySpace27/fluxpype`, and opens a PR. You review and merge; Plumber picks the recipe up on its next update (Settings > Updates already does fast-forward and restart). Model-written Perl runs with your account's privileges, so it never runs before a merge.

## 5. Your two examples, walked through

### 5.1 "1 Mm cube, PIL down the center, 200 G average"

What the designer would say before asking anything:

- **Physics flag.** A 1 Mm box starting at the photosphere sits where plasma beta is near or above 1, so a force-free relaxation is a toy model there rather than a physical one. If the interest is coronal structure above a PIL, a box of tens of Mm is the regime FLUX describes. This is a judgment you might reasonably override (a toy PIL study is still useful); the card would state it, not refuse it.
- **"Cube" is not available as such.** FLUX has one lower boundary, at most one more (`B2`), and an open sphere. The closest match is a plane, a cylinder side wall, and an open sphere as the lid; or a plane plus a padded border of weak flux and an open sphere.

Questions, each with a default:

1. Does "200 G average" mean mean |Bz| (unsigned) or something else? [Unsigned. Arithmetic: 200 G over (1 Mm)^2 = 1e16 cm^2 gives 2e18 Mx unsigned, 1e18 Mx per polarity.]
2. How many fluxons, or what flux per fluxon? [100 per polarity, so 1e16 Mx per fluxon from the arithmetic above.]
3. PIL shape: a sharp step, or a tanh profile with what width? [tanh, width 0.1 of the box.]
4. Side walls: cylinder wall, padded border, or nothing (edge fluxels splay outward)? [padded border.]
5. Initial field: potential, or a sheared or twisted arcade across the PIL? [potential, then optionally a second run with footpoint shear along the PIL.]

Spec result: `geometry=plane, unit=Mm, flux_source=pil_linear, placement=hilbert, initial_field=point_source_potential, boundaries={lower: plane z=0, open: sphere centered on the box}`, outputs 3-D field lines and energy. Recipe: `planar_synthetic` (Phase 2 below).

### 5.2 "Investigate sunspot umbra magnetic evolution"

What the designer would say first: FLUX models the force-free field above a boundary, not the photospheric plasma, so it cannot model umbral dots, convection, or the umbra's own internal evolution. What it can model is how the field above a spot responds to changes at its footprint. Then it would offer:

| Option | Scene | Needs |
|---|---|---|
| A | Idealized spot (umbra and penumbra flux profile) that rotates; watch twist and energy build in the overlying field. | `sunspot` generator, rotation driver (`rotation_spiro` exists), `driven_sequence` recipe |
| B | Decaying spot: remove or open umbral fluxons frame by frame (moat-flow erosion analog); watch connectivity and open flux change. | flux removal driver (`delete`, `open`, `abdicate` exist) |
| C | Observed: a SHARP cutout sequence drives footpoint positions frame by frame. | SHARP fetch, cutout placement, frame-to-frame fluxon matching (new, and the hard part) |

Questions: which observable (twist, free energy, connectivity, open flux)? Idealized or a named AR/HARP? Time span and cadence? Reconnection on or off (off keeps topology fixed, which is cleaner for twist-buildup studies)?

## 6. Phases

Sizes are my estimates, not measured.

| Phase | Content | Depends on | Size |
|---|---|---|---|
| 0 | Recipe registry; `global_pfss` as recipe 1 behind a JSON task file; task key = recipe + parameter hash; per-recipe stage lists in the monitor. No user-visible change except honest cache reuse. | none | ~1 week |
| 1 | Verify the engine: relax three menagerie planar worlds (`potential.flux`, `twisted.flux`, `several-open-lines.flux`) with today's fluxpype relaxer on your Mac; confirm the 3-D viewer draws a plane world. Turns several **uncertain** rows into checked ones. | none | ~1 day |
| 2 | `planar_synthetic` recipe: synthetic Bz generators (bipole, PIL, sunspot, carpet), placement, point-source potential seed, `make_world` with plane and open sphere, relax, outputs. Hand-written specs only. | 0, 1 | ~2 weeks |
| 3 | Scene document: spec schema, compiler, capability file, the operation API with versions and undo, instant checks (section 9.4, tier 0). | 2 | ~1-2 weeks |
| 3b | Designer panel v1 (section 9): inspector generated from the schema, 2-D boundary editor, 3-D preview, toolbar, history, variants, tier 1 previews. No AI. | 3 | ~2-3 weeks |
| 4 | Claude as co-designer: the scene tools in the MCP server, suggestion layers, review and checks, the Ask bar handing a draft to the panel, question protocol, physics block, evaluation set (about 20 requests with known correct specs or refusals, including both examples above). | 3b | ~1-2 weeks |
| 4b | Probe runs and variant comparison (tier 2), with a CPU budget Claude may spend without asking. | 3b, 0 | ~1 week |
| 5 | `driven_sequence` recipe: rotation, translation, emergence, removal drivers; per-frame outputs and a movie; optional reconnection. | 2 | ~2 weeks |
| 6 | Recipe authoring by PR (Layer 3). | 0, 4 | ~1 week |
| 7 | Observed local boundaries (SHARP cutouts, spectral extrapolation). | 2 | open |
| L1-L4 | FLUXlight as the general light layer (section 10). L1 can start any time; it does not depend on recipes. | see section 10 | see section 10 |

Phase 0 and Phase 1 are worth doing even if the designer never ships: Phase 0 fixes the silent-reuse problem, and Phase 1 tells us whether FLUX's planar mode still works after years of global-only use.

## 7. Questions for you (defaults in brackets)

1. Should scene specs live in `config.ini` profiles or as separate JSON files next to them? [Separate JSON under the data folder; `config.ini` stays the global recipe's format.]
2. Is a time-dependent driven run (Phase 5) more important to you than observed local patches (Phase 7)? [Driven runs first: they need no new data source.]
3. Should Chris Lowder review the recipe registry (Phase 0), since it touches how the pipeline is launched? [Yes; it changes `config_runner` and the run script's arguments.]
4. Units for planar scenes: Mm and Mx throughout? [Yes, recorded in the spec and in a `.flux` header comment, since the file format has no unit field.]

## 8. Risks

- **Old code paths.** The planar mode, `B2`, reconnection and the duction scripts carry 2007-era headers and fluxpype has never exercised them. Phase 1 is the cheap way to find out what still works.
- **Overpromising.** A model asked about umbral evolution will be tempted to say yes. The capability file, the compiler's refusals and the physics block on every card are the guard; the evaluation set should include requests the designer must decline.
- **Seed topology decides the answer.** With reconnection off, the potential seed fixes connectivity for the whole run. The card should say which seed was used and why.
- **Units mismatch.** FLUX has none, so a spec written in Mm and a viewer assuming R_sun would render nonsense silently. Carry units explicitly end to end.
- **Model-written code.** Only through a PR you merge; never loaded live.

## 9. The Designer panel (draft 2)

What you asked for: a scene designer GUI with the same tools as the AI, that the AI can hand a draft to, with as many knobs as we can think of; usable directly through the API so changes appear as they happen; an editor toolbar; and a two-way link where Claude and the scene know about each other, so Claude can report what changed, judge whether it looks good, and propose, evaluate and apply adjustments.

### 9.1 What I changed in the idea, and why

1. **The scene is the shared object, not the conversation.** Each scene is a document (the spec from 4.2) with a version number and an append-only log of operations. The panel, Claude and scripts never write the spec directly; they send operations such as `set /placement/n_fluxons 200`, `add_source {type: sunspot, ...}`, `move_source s3 to (0.2, -0.1)`, `set_boundary b2 cylinder r=0.6`. "Same tools as the AI" then holds by construction: the panel's toolbar and Claude's MCP tools are two front ends to one operation API. Undo, redo, history and "who changed this" come free.
2. **Claude suggests by default; it edits only if you allow it.** Claude's operations land as a suggestion layer: shown as a ghost overlay in the canvases and a diff in the inspector, with Accept, Accept part, and Reject. Per scene, you can switch on "Claude may edit this draft directly". That is safe because draft edits are versioned and undoable and run nothing. Spending compute is a separate permission (9.4).
3. **"Looks good" has to come with numbers.** Before judging, Claude declares the checks it is judging against (flux balance, per-fluxon flux, footpoint spacing, seed lines leaving the box, stiffness trend), and its verdict lists pass or fail for each check, plus any qualitative judgment labelled as one. It can also look at rendered snapshots, but the computed checks win where they disagree, because a model judging a field-line picture is unreliable.
4. **Awareness is event-driven, not continuous.** A model watching every slider drag would be slow and expensive. Claude gets a compact summary of the changes since its last turn whenever you ask, click Review, or a preview finishes (only if "auto-review" is on, and debounced). Your edits always go into its next turn as a diff, so it never works from a stale scene.
5. **Knobs come from the schema, not hand-built forms.** "As many knobs as we can think of" turns into maintenance unless every knob is generated from the recipe's parameter schema (type, range, unit, default, help text, which tier of preview it invalidates). A new recipe parameter then gets a knob, a Claude tool argument and validation with no frontend work. Knobs are grouped Basic, Advanced and Expert, and a "changed from default" filter keeps the long list usable.
6. **Locks and pins.** You can lock any field or object ("keep the PIL where I drew it"); Claude's operations on locked paths are refused by the server, not just discouraged in the prompt.
7. **Variants instead of overwrites.** "Try it three ways" forks the scene into variants that share history. Previews and probes run per variant and show side by side, which turns "propose, evaluate, apply" into a comparison you can see.
8. **Scenes are separate from runs.** A run records the scene version hash it was built from, so a result always traces back to the exact knobs that made it (and this fixes the provenance gap in the iteration-loop scope).

### 9.2 Entry points

| Entry | How | What you see |
|---|---|---|
| Plumber page | `/scenes` list, `/scenes/<id>` editor | The panel. |
| From the Ask bar | A scene proposal card gets **Open in Designer**, which creates the scene from the draft spec | The draft, with Claude's question answers and caveats pinned in the side panel. |
| From Claude Code or another MCP agent | The existing MCP server (`plumber/mcp_server.py`) gains the scene tools | Operations appear live in any open panel, attributed to "agent". |
| From a script | REST `POST /api/scenes/{id}/ops`, and a small Python client (`fluxpype.scenes`) for notebooks | The same live view; useful for parameter sweeps written by hand. |

"Live" works because Plumber already has one app-wide WebSocket (`/api/ws`, `server.py:1653`) that pushes updates into the React query cache (`web/src/lib/live.tsx`); scene operations would be one more event kind on it (**checked** that the socket exists; whether it needs per-scene subscription to stay cheap is **uncertain**).

### 9.3 Layout and toolbar

- **Center:** two linked canvases: a 2-D boundary view (the Bz map, placed footpoints, PIL, sources with drag handles) and a 3-D view built on `FluxViewer.tsx` (three.js; it is sphere-only today, see 12.1) for seed lines, boundaries and, after a probe, relaxed lines.
- **Right:** the inspector (schema-generated knobs, with diff markers for suggestions and lock icons).
- **Left:** history (every operation, author, rationale), variants, and the checks panel.
- **Bottom:** the Claude dock: conversation scoped to this scene, its suggestion layers, and its check reports.

Toolbar groups:

| Group | Tools |
|---|---|
| Edit | select and move, undo, redo, lock, duplicate as variant |
| Sources | add monopole, bipole, sunspot (umbra and penumbra), flux tube, random carpet, import map (synoptic or later SHARP) |
| Shape | draw PIL (line or curve), flux brush (paint +/- flux on the map), smooth, balance flux |
| Domain | lower plane or sphere, B2 gizmo (plane, sphere, cylinder), open-sphere gizmo, units |
| Field | seed type (potential, twisted, tube, vertical), twist knob, sample field line at a clicked point |
| Drive | timeline of drivers (rotate, translate, emerge, remove) with keyframes |
| Measure | ruler, flux in a region, height of a line, energy readout |
| Evaluate | validate, preview, probe relax, compare variants, ask Claude to review |
| Run | queue full run (always your click) |

### 9.4 Cost tiers: what recomputes, and who may trigger it

| Tier | What | Time (estimate) | Who may trigger |
|---|---|---|---|
| 0 | Validation against the capability file, derived numbers (total and per-polarity flux, flux per fluxon, mean \|B\|, imbalance) | instant, on every operation | everyone |
| 1 | Placement, unrelaxed world, potential seed lines, boundary render | seconds | everyone, automatic after edits settle |
| 2 | Probe relax: a few hundred steps at reduced resolution, stiffness trend, energy | minutes | you; Claude within a per-scene CPU budget you set (default off) |
| 3 | Full run through the scheduler | minutes to hours | your click only, as today |

Each knob in the schema declares the lowest tier it invalidates, so changing a render option does not throw away a placement.

### 9.5 Claude's tools on a scene (mirrors the toolbar)

`get_scene` (spec, version, locks), `get_changes_since(version)` (diff with authors), `apply_ops(ops, base_version, rationale, mode=suggest|direct)` (refused on a stale base or a locked path), `validate`, `preview` (tier 1), `probe` (tier 2, budgeted), `get_checks`, `get_snapshot` (2-D or 3-D image), `fork_variant`, `compare_variants`, `withdraw_suggestion`. Running a full job stays a proposal card, as in the Ask bar today.

A review turn looks like: read the diff since its last turn; say what changed in a sentence; run tier 0 and tier 1; report checks with pass or fail; propose at most a few grouped operations, each with a one-line reason and the check it is meant to fix; optionally fork a variant and probe both if you allowed a budget.

### 9.6 Things this does not do (on purpose)

- It is not a free-form 3-D modeler. You can only make what the spec can express; anything else goes through the recipe-by-PR route (4.4).
- Claude never starts a full run, never edits a locked field, and never edits directly unless you turned that on for that scene.
- No multi-person editing in v1. The operation log would allow it later, but Plumber is single-user today.

### 9.7 Questions for you (defaults in brackets)

5. Should Claude edit directly by default, or suggest by default? [Suggest; direct is a per-scene switch.]
6. Should Claude get a probe CPU budget by default? [No; off until you set it per scene.]
7. Auto-review after each preview, or only when you ask? [Only when you ask; auto-review is a toggle.]
8. Which conversation backend for the dock? The Ask bar uses a short one-question `claude -p` call; a scene dock needs a longer conversation per scene. [Resume a CLI session per scene if the CLI supports it the way we need (to verify in a spike), else the Anthropic API behind the existing `ask_backend` seam.]

### 9.8 Risks added by the panel

- **Scope creep in the canvas.** The flux brush and driver timeline are the expensive tools; ship v1 with sources, PIL, domain gizmos and the inspector, and add painting and keyframes later.
- **Schema quality decides knob quality.** Bad units or missing ranges in a schema become bad knobs everywhere. Every recipe schema needs units, ranges and help text as a merge requirement.
- **Edit races.** You drag while Claude applies. The base-version check rejects Claude's stale operations, and Claude rebases on the next turn; your edits always win.

## 10. FLUXlight as fluxpype's light layer (draft 3)

Gilly's direction (2026-11-13): FLUXlight should hold any light-producing code fluxpype uses or needs, not only the global Thomson renderer.

### 10.1 What exists today (**checked**)

- `science/fluxlight.py` loads a relaxed world (`read_flux_world`), calls `simulate_thomson_scattering`, caches the result as a pickled `.light` file next to the world (`fluxlight.py:86-95`), and draws the quad and tri plots.
- `science/thomson_scattering_flux.py` holds the physics: a default radial electron density (`4.2e8 * 10**(4.32/r)` cm^-3, lines 25-31), Thomson geometry and incident intensity, and a line-of-sight integrator with a `method` switch (the function default is `voronoi`, `thomson_scattering_flux.py:62`, but `fluxlight.py:69` sets the pipeline default to `fluxel`) that uses the fluxon world to shape the density. Units are R_sun throughout; the camera is pole-on unless `observer_lon` is set.
- **GHOSTS in fluxpype is not an emission code.** `science/gen_fluxon_ghostsflow.pm` is a solar-wind flow model for fluxons with exactly one open end (line 88). The GHOSTS non-equilibrium ionization code itself is not in fluxpype; I have not looked at where it lives. My draft 1 phrase "pseudo-EUV via GHOSTS" overstated what is in the repo.

So today FLUXlight is one emission process, one density model and one camera, all global.

### 10.2 Proposed structure: four swappable layers

| Layer | Job | First members | Later |
|---|---|---|---|
| **Plasma** | n_e (and T where needed) at any point, from the world plus a model | the existing radial default; the existing fluxon-shaped density (`method`); densities from the wind stage (`gen_fluxon_*flow.pm` outputs) on open fluxons | a closed-loop model per closed fluxon (hydrostatic, or RTV-style scaling laws), needed for both EUV and any planar scene; GHOSTS ionization states if that code is brought in |
| **Emission** | emissivity per volume for a process | Thomson total and polarized brightness (existing code, moved behind the interface) | EUV and SXR via temperature response functions (needs atomic data, likely CHIANTI through a Python package; which package and its install cost are unchecked); NEI-corrected EUV with GHOSTS; possibly radio free-free |
| **Camera** | geometry and line-of-sight integration | the existing pole-on and `observer_lon` cameras in R_sun | arbitrary observer position (Earth, PUNCH, a STEREO-like vantage), top-down and oblique views of planar scenes in Mm, occulter and field-of-view masks |
| **Instrument** | turn ideal radiance into what an instrument records | none (ideal images, as now) | bandpasses, PSF, resolution, exposure for AIA, SUVI, PUNCH, LASCO, K-Cor and others. Every number comes from the sourced files in the vault's `instruments/` folder, never typed in from memory |

A FLUXlight product is one choice per layer, for example `plasma=fluxon_voronoi, emission=thomson_pB, camera=earth, instrument=none`. Each layer declares a parameter schema like a recipe does (section 4.1), so:

- the Designer panel gets knobs for light products for free (section 9.1 point 5);
- Claude gets matching tools and can propose "add an AIA 171 view from Earth" as an operation;
- the capability file can refuse impossible combinations in plain words ("EUV needs a temperature; this plasma model has none").

Outputs become FITS with WCS headers as well as PNGs, so results open in the HelioFITS tools and compare directly with data. The `.light` pickle stays as the cache format; cache keys add the four layer choices and their parameters.

### 10.3 Geometry

The plasma and camera layers carry units and geometry explicitly, so the same Thomson or EUV code runs on a global world (R_sun, sphere) and a planar scene (Mm, plane). Today's hard-coded `lower_bound=1.05` and the R_sun density law become defaults of the global plasma model, not assumptions of the integrator.

### 10.4 Phases

| Phase | Content | Size (estimate) |
|---|---|---|
| L1 | Refactor with no change in output: split `thomson_scattering_flux.py` into plasma, emission and camera pieces behind small interfaces; a regression test that today's CR 2150 render is reproduced to within float noise. FITS output with WCS. | ~1 week |
| L2 | Cameras: arbitrary observer, planar top-down and oblique. Thomson on a planar scene with a simple closed-loop density. | ~1 week |
| L3 | Closed-loop plasma model (n and T along closed fluxons) and EUV emission via a temperature response. Needs a decision on the atomic-data dependency. | ~2-3 weeks |
| L4 | Instrument layer from the vault's sourced instrument files; GHOSTS NEI emission if that code is brought into fluxpype. | open |

L1 can start now and does not depend on the recipe work. L3 is the piece SBIR Objective 2 (pseudo-EUV) needs, so it is the natural place to spend SBIR hours.

### 10.5 Questions for you (defaults in brackets)

9. Where does the GHOSTS code live, and should it come into fluxpype as a dependency or stay separate with an adapter? [Adapter first; move it in only if it is stable.]
10. Which atomic-data route for EUV: CHIANTI through a Python package, or precomputed response tables shipped with fluxpype? [Precomputed tables for the common bandpasses, so a plain install needs no large database; I have not checked sizes.]
11. Keep the name FLUXlight for the whole layer? [Yes; it already names the stage in Plumber (`monitor.py` STAGES).]

## 11. Draft 4: graphical editor, pipeline timeline, library, help (Gilly, 2026-11-13)

What you asked for: a graphical editor in the 3-D preview to place and finely edit magnetic structures; a live canvas panel to select and move them; a live preview of the job script; a drag-and-drop timeline of processing steps (Relax, Export, Wind, Light and so on); a "test run" smoke test with a handful of fluxons; a scene selector; the Ask bar staying where it is on every page; reusable building blocks; a gallery of the structures that already exist; and everything a user would expect that is still missing (tooltips, help, FAQ, recipes, get started, a download manager).

### 11.1 The 3-D editor: two editing levels

Direct manipulation in the 3-D view, with the 2-D top view, an outliner (scene tree) and the inspector all showing the same selection.

- **Structure mode (default).** A structure is a parametric object (a bipole, a sunspot, a twisted loop, a Gold-Hoyle tube, a carpet patch). You select it and get a gizmo to move, rotate and scale it; footpoint handles to drag each end along the photosphere (snapped to the plane or sphere); and numeric fields for exact values. Also: multi-select, align and distribute, mirror across a line, copy and paste, hide, lock, solo, group. Every drag is an operation on the scene document (9.1), so Claude and scripts see it and it can be undone. three.js ships `TransformControls` for the gizmo, and Plumber's viewer is already three.js.
- **Fluxon mode ("granular").** For fine edits you can "bake" a structure into explicit fluxons, then move a footpoint, drag a vertex, delete or duplicate a single fluxon, or re-route one. A baked structure is no longer parametric: its knobs disappear and the outliner marks it "baked", so it is never ambiguous which kind you are editing. Bake is undoable.
- **A guard FLUX needs.** FLUX preserves topology, so dragging one fluxon through another silently changes the physics you will get. On every drag the editor checks for crossings and for vertices below the photosphere (cheap geometry, tier 0) and flags them in place.

### 11.2 Structure gallery: what already exists, as building blocks

Every card carries a thumbnail, its parameters, the file it comes from and a badge saying whether Phase 1 has verified that it still relaxes.

| Card | Source (**checked** that the file exists and what its POD says; none run) |
|---|---|
| Twisted loop | `pdl/PDL/twisted_loop.pdl` (footpoints, N x M fluxons, spacing, twist in turns) |
| Vertical open fluxons | `pdl/PDL/vertical_fluxons.pdl` |
| Random twisted carpet | `pdl/PDL/random_carpet.pdl` (its POD calls it nonphysical) |
| Gold-Hoyle tube | `pdl/PDL/tubes/gold_hoyle_tube.pdl`, `gold_hoyle_simple.pdl`, `gold_hoyle_line_cart.pdl` |
| Lundquist tube | `pdl/PDL/tubes/lundquist_tube.pdl` |
| Twisted straight tube | `pdl/PDL/tubes/twisted_straight_tube.pdl` |
| Emerging bipole (single loop) | `Flux::World::emerge` (`World.pm:660-750`) |
| Coronal hole / whole Sun | `pdl/PDL/coronal_hole.pdl` |
| Example worlds | the 20 files in `pdl/menagerie/` (potential, twisted, emerging 1 to 4, plasmoid, open lines), importable as baked scenes |
| Global PFSS corona | fluxpype's current pipeline, as a structure that fills a sphere |
| New generators | bipole, PIL, sunspot (umbra and penumbra), from section 4.2 |

How the old Perl generators get used (corrected after the critique, section 12.1): each generator needs its own small adapter. Only `twisted_loop` and `vertical_fluxons` return a `.flux` text fragment; the tubes return lists of line arrays, and `random_carpet` and `coronal_hole` return a finished world with their own boundary and forces already set. Several print debug text, and concentration and fluxon labels have to be renumbered when structures share a world. Calling the originals still keeps one source of truth; porting them to Python would fork it.

### 11.3 Your own library: design once, reuse later

- Select one or more structures and **Save to library** as a named block with a thumbnail, a description and the parameters you choose to expose. Stored as JSON under the Plumber home (`library/`), versioned, and exportable as a file to share with a colleague.
- Inserting a block copies it into the scene and records which library item and version it came from. It does not stay linked, because a linked block that changes later would silently change old scenes. An "update available" marker lets you pull a newer version on purpose.
- Whole scenes can be saved as **templates** in the same library.
- Claude gets `list_library` and `insert_structure`, so "use my sheared-arcade block" works in conversation.

### 11.4 Pipeline timeline: drag-and-drop steps

The processing flow becomes a row of boxes you can add, remove, reorder and configure. This changes Layer 0 (4.1) in one way: **the unit that gets registered is a step, and a recipe is a default arrangement of steps.**

| Step | Needs | Makes | Exists today? |
|---|---|---|---|
| Fetch magnetogram | a CR or a cutout request | Bz map | yes (`get_magnetograms.py`) |
| Place footpoints | Bz map | footpoint list | yes (`fluxon_placement_hilbert.pm`) |
| Seed field | footpoints | field lines | PFSS yes; planar potential new |
| Build world | lines or structures | `.flux` world | sphere yes; plane via `make_world` |
| Relax | world | relaxed world | yes (`relax_pfss_world.pm`) |
| Reconnect | world | world | core only, never called by fluxpype |
| Drive (move footpoints) | world | world | core scripts only (`ductions/`) |
| Emerge or remove flux | world | world | core only |
| Repeat N times | a group of steps | a frame sequence | new |
| Field map | relaxed world | B along fluxons | yes (`map_fluxon_b*.pm`) |
| Wind | relaxed world with open fluxons | flow and density | yes (`gen_fluxon_*flow.pm` variants: flow, tflow, wsa, cranmer, schon, ghosts, tempest) |
| Light | world, optionally wind densities | FLUXlight product (section 10) | Thomson yes |
| Plots | any | figures | yes (plot catalog) |
| Export | any | `.flux`, FITS, movie, bundle | partly |

Rules that keep drag-and-drop honest:

- Each step declares what it needs and what it makes. A drop that breaks a dependency (Wind before Relax, Wind on a scene with no open fluxons) is refused at the drop target with the reason.
- Steps can repeat and branch: two Light boxes for two products; a Repeat box wrapping Drive and Relax for an evolution run.
- Clicking a box opens its settings (the same schema-generated knobs).
- Each box caches its output under a hash of its settings and everything upstream, so reordering or editing a late box never recomputes the early ones.
- The same boxes show live status while running, which replaces today's fixed ten-stage chip row for scene runs.

### 11.5 Live job-script preview

A read-only panel with three tabs that update on every edit, with the lines changed by the last edit highlighted:

1. **Scene** (the spec JSON),
2. **Script** (the generated driver: the actual `read_world`, force list, relaxer and step calls, in order),
3. **Command** (the one-line command that reproduces the run outside Plumber).

Copy and download on each. Hand-editing the script inside the panel is not offered, because arbitrary edits cannot be mapped back onto knobs. The offer is **Eject to custom recipe**, which copies the script into a new recipe on a branch (the PR route in 4.4).

### 11.6 Test run

One button that runs the whole timeline end to end, small: a handful of fluxons (default 20 per polarity or 1/50 of the count, whichever is larger), one short relaxation cycle, a low-resolution light product, into a scratch folder that never touches the cache. It reports pass or fail and time per box, shows the tiny results, and gives a rough cost estimate for the full run (labelled an estimate; how relaxation time scales with fluxon count is not measured). It answers "does every pipe connect", not "is the physics right": a 20-fluxon world relaxes differently from a 2000-fluxon one, and the panel says so. Claude may trigger it within the probe budget (9.4).

### 11.7 Scene selector, and the Ask bar

- **Scenes page** with thumbnails, search, tags, duplicate, archive, import and export of a scene file; plus a **scene switcher** in the Designer header, like the run picker in the Run view today. "New scene" offers blank, from template, from an Ask draft, or from an existing `.flux` world (imported as a baked scene).
- **The Ask bar stays in the top bar on every page**, unchanged. On a scene page it shows a chip naming the scene, and a question typed there lands in the scene's Claude dock, so there is one Claude with two doors rather than two assistants.

### 11.8 Help and guidance

Today Plumber has a tooltip component (`web/src/components/ui/index.tsx`, Radix) used in very few places, and no help menu, FAQ, tour or first-run guide (**checked** by search of `web/src`).

- **Tooltips on everything**, generated from the same schema help text as the knobs, each with units, range, default and "what it invalidates". A "What's this?" pointer mode for the canvases and toolbar.
- **Help menu** in the top bar: Get started (a checklist that tracks itself: install check, first global run, first scene, first test run, first light product); a guided tour of the Designer; **Recipes** (a cookbook of common tasks, each with "Open as scene": global corona for a CR, a sheared arcade, a rotating sunspot, a flux rope from a tube, compare two seeds, make a movie); FAQ; glossary (fluxon, fluxel, stiffness, flux concentration, open sphere), sourced from `ask/flux_reference.md` so it cannot drift from what Claude is told; keyboard shortcut sheet; "Ask Claude about this".
- **Empty states** that say the next action, and **error messages that name a fix** (the health checks in `doctor.py` and `health.py` already know many of them).

### 11.9 Download manager and bundles

Today data downloads happen inside the magnetogram stage, switched by `do_download` and needing a JSOC email (`science/get_magnetograms.py:54-63`), with no view of what is being fetched. Proposed:

- **Downloads panel:** queue, progress, retry, source status (JSOC email set or not, site reachable), prefetch a range of rotations, an offline indicator, and what is already on disk (linked to the Cache page).
- **Run bundle:** export a run as one archive (scene file, generated script, settings, outputs, provenance, software versions) for a collaborator or a paper's data statement.

### 11.10 Affordance review: what users will ask for

"Exists" is from file names and pages in `web/src` (**checked**); I did not exercise each feature.

| Affordance | Today | Proposal |
|---|---|---|
| Undo and redo | no | yes, from the operation log (9.1) |
| Autosave and crash recovery | scenes do not exist yet | every operation is saved; reopen restores the last version |
| Search everything | yes (Cmd+K palette) | add scenes, structures, library blocks, help articles |
| Notifications | yes (`Notifications.tsx`) | add test-run and probe results |
| Keyboard shortcuts | Cmd+K only | full set with a shortcut sheet; no global key handlers that fight text fields |
| Cost and time estimate before a run | no | from test-run timings, labelled as an estimate |
| Parameter sweeps | profile lists only (rotations, fluxon counts) | sweep any knob over a range, as variants or as jobs |
| Compare runs or variants | no | side-by-side view with synced cameras and a metric table |
| Notes, tags, stars on runs and scenes | no | yes |
| Provenance | partial ("Settings used") | scene version hash, step hashes, fluxon-mhd and fluxpype commits on every output |
| Figure export for papers | view and download | presets for size, DPI, fonts, colorblind-safe maps; vector output where possible |
| Movies | core script only (`make_flux_movie.pdl`) | Export box in the timeline |
| Citation helper | `CITATION.cff` in fluxon-mhd | "How to cite this run" with a draft methods paragraph for you to edit |
| Notebook hand-off | no | "Open in notebook": a starter notebook that loads the run's outputs |
| Units and coordinates readout | no | cursor position, field strength and flux under the cursor; unit toggle |
| Import an existing `.flux` world | viewer only | import as a baked scene |
| Scene file sharing | no | export and import of a scene file and library blocks |
| First-run setup | `flux-doctor`, health checks | guided setup in Get started |
| Demo without FLUX installed | yes (Simulated backend) | add two simulated scenes so the Designer can be explored |
| Accessibility | not reviewed | keyboard path for every canvas action, contrast, reduced motion |
| Bug report bundle | no | one button: logs, versions, scene, with secrets stripped |
| API docs | not checked | a page listing the operation API with examples |
| Cluster backends | deferred (NWRA scheduler unknown) | unchanged |

### 11.11 Phases added

| Phase | Content | Depends on | Size (estimate) |
|---|---|---|---|
| 0 (revised) | Registry of steps, with recipes as arrangements of steps; per-step cache hashes. This means splitting `magnetogram2wind.pdl` (one script with every stage inline) and migrating Plumber's `tasks` table, whose `cr`, `nflux`, `adapt`, `method` columns are required (`plumber/store.py:22-29`). | none | larger than first stated; several weeks, and it touches a script Chris Lowder also uses |
| 3c | Pipeline timeline UI, live script preview, test run. | 0, 3 | ~2 weeks |
| 3d | 3-D editor structure mode: gizmos, handles, outliner, gallery from the Perl generators. | 3b, 1 | ~2-3 weeks |
| 3e | Fluxon mode (bake and edit), crossing checks. | 3d | ~2 weeks |
| 3f | Library and templates; scenes page and switcher. | 3b | ~1 week |
| H | Help menu, tooltips from schema, Get started, Recipes, FAQ, glossary. | 3b for the scene parts; the rest can start now | ~1-2 weeks |
| D | Downloads panel and run bundles. | none | ~1 week |

### 11.12 Questions for you (defaults in brackets)

12. Fluxon mode: bake-then-edit as described, or no fluxon-level editing at all in v1? [Bake-then-edit, after structure mode ships.]
13. Library blocks: copied on insert (reproducible) or live-linked (always current)? [Copied, with an "update available" marker.]
14. Test run size? [20 fluxons per polarity or 1/50 of the count, one short cycle.]
15. Should the timeline allow steps FLUX cannot verify yet (Reconnect, Drive) from day one, marked experimental, or hide them until Phase 1 and Phase 5 verify them? [Show them greyed with "not verified yet".]

## 12. Independent critique (Fable, 2026-11-13) and what changed

A separate reviewer read the whole plan and spot-checked the code. I re-checked the factual points marked **confirmed** below myself. Its headline: the plan is a platform of roughly 28 to 32 weeks of estimated work (its sum of my estimates) with no first scientific result named, and the first thing you asked for (words to a new kind of run) arrives late. I agree with that criticism.

### 12.1 Facts I had wrong (now corrected in place)

| Claim in earlier drafts | What the code says |
|---|---|
| The Perl generators all return a `.flux` text fragment | Only `twisted_loop` and `vertical_fluxons` do. Tubes return line arrays; `random_carpet` returns a world with its own boundary and forces set (`random_carpet.pdl:47-50`, **confirmed**). Each needs an adapter. |
| `global_pfss` can stay one unchanged script and also be reorderable steps | It cannot be both. `magnetogram2wind.pdl` is one script with stages inline. Steps mean splitting it. |
| Phase 0 is about 1.5 weeks | Plumber's `tasks` table requires `cr`, `nflux`, `adapt`, `method` (`plumber/store.py:22-29`, **confirmed**), so Phase 0 includes a database migration as well as the script split. |
| The 3-D editor reuses `FluxViewer` | The viewer always draws a sphere and fits the camera to it (`FluxViewer.tsx:308-309`, **confirmed**). A plane editor is new work. |
| The scene dock uses the Ask bar's gate unchanged | Ask runs one short question: no session kept, 120 s limit, Sonnet (`plumber/ask.py:26-27, 113`, **confirmed**). A scene conversation with minutes-long probes needs a different backend. That spike must come before Phase 4. |
| FLUXlight's density method defaults to `voronoi` | The pipeline default is `fluxel` (`fluxlight.py:69`, **confirmed**). |
| The planar seed is a few hundred lines | Too low: it needs line-to-fluxel assignment, open lines for unbalanced flux, and no crossings. `make_world` also fixes the open sphere at the origin and omits `b_eqa` (`make_world.pdl:74-80`, **confirmed**). |

### 12.2 Physics guards added (I recommend treating these as required, not optional)

1. **Structures do not superpose.** Fluxons from two generators dropped into one scene are not the field lines of the combined field. Every scene declares a composition rule: either structures are isolated (no overlapping envelopes), or the scene is re-seeded from the combined boundary flux. Overlap raises a warning in place.
2. **Hand-set connectivity is labelled.** With reconnection off, connectivity you set by hand in fluxon mode is the answer. Baked, hand-edited scenes carry a "hand-set connectivity" mark on every output and in the bundle.
3. **Crossing checks test the drag, not the end state.** What matters is whether a move sweeps one fluxon through another; bake and import are checked the same way.
4. **Frames, not time.** Drive and Repeat produce a sequence of equilibria with no clock. Movies and the timeline label frames or the driver parameter, never seconds.
5. **Light on local scenes is illustrative.** White light of a Mm-scale scene is not an observable, and an EUV image needs a temperature FLUX does not have. Such products say "illustrative" on the image and name the plasma model in the FITS header.
6. **Test runs and probes are plumbing checks.** A 20-fluxon world relaxes differently. No "pass" badge on physics, and the cost estimate is shown as a range.
7. **Realized versus requested.** "200 G mean" is only met to within one fluxon's flux. The inspector shows the realized value and the quantization error.
8. **Claude reports "construction checks".** Passing geometry checks does not mean the scene answers the science question, so it never says "looks good" on the strength of them.

### 12.3 The smallest slice worth building first (the reviewer's proposal; I recommend it)

1. Phase 1: relax three menagerie planar worlds on your Mac (about a day).
2. One hand-written scene file for a planar bipole or PIL arcade, compiled by a script, relaxed with the existing relaxer.
3. The viewer learns to draw a plane.
4. One new Ask bar tool, `propose_scene`, that fills that scene file from your words.

No panel, no step registry beyond a second entry script. The reviewer's estimate is two to three weeks; I have not sized it independently. It delivers "words to a new kind of run" and tests every uncertain row in section 3 before anything large is built on them.

### 12.4 Approval list

Tick what you want. "Reviewer" is the independent critique; "Mine" is my recommendation after reading it. Everything stays in the mockup so you can see it before deciding.

| # | Feature | Reviewer | Mine |
|---|---|---|---|
| A1 | Smallest slice (12.3) | build first | build first |
| A2 | Physics guards (12.2) | required | required |
| A3 | Scene files, scenes page, scene switcher | keep | keep |
| A4 | Inspector with schema-generated knobs and tooltips | keep | keep |
| A5 | Top-view canvas: select and move structures | keep | keep |
| A6 | 3-D editor, structure mode (gizmo, footpoint handles, outliner) | not addressed directly | keep, after A5 |
| A7 | Fluxon mode (bake and hand-edit) | cut or defer indefinitely | defer; keep the design, build only if you would publish from hand-set topology |
| A8 | Gallery of existing structures, with per-generator adapters | implied keep, with validation badges | keep, one generator at a time |
| A9 | Library of your own blocks and templates | not cut | keep, simple copy-on-insert |
| A10 | Pipeline boxes | simplify: fixed order with toggles, repeats and extra Light boxes | start with the simplified form; free drag-and-drop later if the fixed order chafes |
| A11 | Live script preview | not cut | keep; it is cheap once the compiler exists |
| A12 | Test run button | keep, labelled as a plumbing check | keep |
| A13 | Operation log, locks, suggestion layers | simplify: whole-scene versions with a diff; Claude proposes a complete scene as one card | start simple as the reviewer says; move to operations when live co-editing is actually wanted |
| A14 | Variants with shared history | defer | defer; "duplicate scene" covers it at first |
| A15 | Claude direct-edit mode, CPU budgets, auto-review | defer | defer |
| A16 | Help menu, Get started, Recipes, FAQ, glossary | pays off only with other users | keep Get started, Recipes and tooltips; defer the guided tour and "What's this?" mode |
| A17 | Downloads panel | not addressed | keep; it is independent and small |
| A18 | Run bundle and provenance | not cut | keep |
| A19 | Flux brush, driver keyframes | cut | defer |
| A20 | FLUXlight layers L1 to L3 | keep, with the labels in 12.2 | keep |
| A21 | FLUXlight instrument layer (L4), citation helper, accessibility pass | defer | defer L4 and the citation helper; I would not drop accessibility, only schedule it with the panel |

### 12.5 Additions from the critique (ranked by the reviewer for value against effort; I agree with the top five)

Bells and whistles: (1) an "Explain this scene" card listing assumptions, seed and what the scene cannot show; (2) diff two scenes or runs in words and numbers; (3) a one-knob sweep with a small results table; (4) thumbnails of a library block across a parameter (twist 0, 0.5, 1); (5) Claude drafts a methods paragraph from provenance for you to edit; (6) a scrub bar over relaxation frames; (7) field-line colouring by twist, length or a current proxy; (8) a "Why did this fail" button that hands the log tail and scene to Claude; (9) a "validated against an analytic solution" badge on library blocks; (10) snapping footpoints to flux centroids of an imported magnetogram.

Affordances the plan was missing: convergence diagnostics with a "not converged" flag on every figure; a resolution study (same scene at twice the fluxons, with a difference metric); validation against analytic solutions (potential bipole, Gold-Hoyle), which is what gives the gallery badges meaning; random-seed control for placement; scene-file schema versions and migration; derived quantities (twist, free energy against the potential state, open flux, connectivity matrix); disk use and cleanup per scene; cancel and resume for sequences; a headless command to run a scene file without Plumber; stated coordinate conventions.

### 12.6 Questions the critique adds

16. What is the first scientific question a scene should answer? It should pick the first slice. [No default; this is yours.]
17. Who else will use this? Tours and FAQs only pay off with other users. [Assume you and a few collaborators.]
18. Is this SBIR-funded, and which pieces map to deliverables? [Assume FLUXlight L1 to L3 and planar scenes map to Objective 2; the rest is not SBIR work.]
19. Do you want several structures composed in one world at all, given they do not superpose, or one structure per scene plus an observed background? [Allow composition with the overlap warning.]
20. Would you publish from hand-set topology (fluxon mode)? [If no, drop A7.]
21. Is splitting `magnetogram2wind.pdl` into steps acceptable, given Chris Lowder also uses it? [Ask him before Phase 0.]
22. Are you willing to fix the 2007-era generators in your fluxon-mhd fork, accepting the drift from upstream? [Yes, in the fork only, as the standing rule already says.]

## 13. Decisions and build order (Gilly, 2026-11-13)

Answer to Q16 (first science question): Gilly picked all three candidates, in this order.

| Order | Slice | Why now | Maps to |
|---|---|---|---|
| **A** | Free energy over a sheared PIL: planar bipole or PIL arcade, footpoint shear, compared with its potential state | First step toward detecting non-potential configurations. This is what the SBIR needs most right now. | SBIR Objective 2 (non-potentiality metrics); sections 12.3, 5.1 |
| **B** | Sunspot twist and energy buildup as a driven sequence | Gilly wants actual time evolution. | Phase 5 (drivers), pulled forward |
| **C** | Better global corona runs: FLUXlight L1 to L3 and the pipeline-step split | New functionality for the existing pipeline, but not what the SBIR is about this quarter. | Section 10, revised Phase 0 |

### 13.1 Slice A, concretely

1. **Verify the engine (Phase 1).** Relax `potential.flux`, `twisted.flux` and `several-open-lines.flux` from `pdl/menagerie/` with today's relaxer and record energy, stiffness and wall-clock time. This has to run on Gilly's Mac, because the Flux build and PDL live there; the cloud container has Perl and gcc but no PDL (**checked**: `perl -MPDL` fails; apt has no `pdl` package here), so I cannot do it from the cloud without installing a toolchain first.
2. **One scene file** for a planar PIL arcade: plane at z = 0, tanh PIL, N fluxons per polarity, open-sphere lid, point-source potential seed.
3. **Two relaxations of the same footpoints:** potential (zero shear) and sheared (footpoints displaced along the PIL by a chosen amount).
4. **The non-potentiality readout:** free energy of the sheared state over the potential state, plus twist and open flux (the derived quantities in 12.5). Free energy needs a defined energy estimator; FLUX has `e_simple`, `e_simple2`, `e_open`, `e_eqa` (`lib/physics.c:50-53`), and which one is trustworthy for this comparison is **uncertain** and is part of the Phase 1 check (an analytic Gold-Hoyle or potential bipole is the reference).
5. **Viewer learns to draw a plane** (the viewer is sphere-only today, `FluxViewer.tsx:308-309`).
6. **Ask tool `propose_scene`** that fills the scene file from words.

### 13.2 Slice B: what "actual time evolution" can mean

FLUX is quasi-static: a relaxation step takes no physical time, and a "time step" is a boundary update followed by many relaxation steps (`ask/flux_reference.md`, section 1). So B gives a **sequence of equilibria driven by moving footpoints**, and the field only follows physical time if the driver is slow compared with the Alfven crossing time of the region. To put seconds on a frame, the scene must state the driver speed and the Alfven speed it assumes, and the movie axis says "driver time under that assumption". Without those, frames stay numbered. This does not model dynamics (waves, flares, reconnection outflows); FLUX has no inertia. If Gilly means those, FLUX is the wrong tool for that part and the plan should say so on the card.

B also needs: a driver (`rotate_poles`, `rotation_spiro` exist as scripts but were not run), a Repeat step, per-frame outputs, and optionally reconnection (core only, never called by fluxpype, unverified).

### 13.3 Remaining open questions

Of the 22, Q16 is answered. Q1 to Q15 and Q17 to Q22 keep their stated defaults until Gilly changes them, except these three, which do change what gets built first and will be asked one at a time: Q20 (hand-set topology, decides whether fluxon mode exists), Q19 (several structures in one world), and Q21 (splitting `magnetogram2wind.pdl`, which only matters once C starts).
