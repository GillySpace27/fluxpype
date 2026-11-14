# Plumber Scene Designer: what is planned (only the first slice exists)

Source: docs/plumber-scene-designer-scope.md (the full plan, draft 5, 2026-11-13). Read it with Read or Grep for detail and cite it. Everything below is a plan unless it says "exists". When someone asks "can Plumber do X?", say whether X exists today or is only planned. Never offer a planned feature as if it were available.

## What exists today (first slice, planar arcade only)

- A planar arcade scene file (`fluxpype/scenes`): rows of positive and negative footpoints across a PIL, a hand-set pairing of which footpoint each line joins, one photosphere plane, an optional B2 lid. It compiles to a `.flux` world. Saved scenes live in `scenes/` in the data folder.
- You can propose saving one with `create_scene`, and read them with `list_scenes` and `get_scene`. The 3-D viewer draws the photosphere and lid planes.
- A headless relax command (`python -m fluxpype.scenes.relax`) exists on a draft branch, not yet in a release. It is run by a person or a script, not by Plumber's queue.
- Measured so far (Gilly's Mac): relaxed energy differences between connectivities depend on the lid height, so they are not yet a lid-independent free energy. Say so when asked about results.
- Everything else below is still planned.

## The goal

Gilly wants Plumber to design new FLUX scenes from words ("a Cartesian scene with the photosphere as a plane in a 1 Mm cube, a polarity inversion line down the center, mean field 200 G", or "investigate sunspot umbra magnetic evolution"), ask clarifying questions, then propose and wire a new kind of run into fluxpype. Today fluxpype has one run type: a global PFSS corona, keyed by (rotation, fluxon count, adapt, wind method).

## Build order Gilly chose

- **A, first:** free energy over a sheared polarity inversion line (PIL). A planar bipole or PIL arcade relaxed twice from the same footpoints, once potential and once sheared, compared by free energy, twist and open flux. First step toward detecting non-potential configurations. This is what the SBIR needs now.
- **B:** driven sunspot twist and energy buildup as a sequence of equilibria (time evolution).
- **C:** better global corona runs: FLUXlight as a general light layer and splitting the pipeline into steps.

## What FLUX can and cannot do (checked in code, mostly not run)

- Can: a planar lower boundary (`GLOBAL BOUNDARY PLANE`), a second boundary `B2` (plane, sphere or cylinder), an open sphere centered anywhere, synthetic generators (twisted loop, vertical fluxons, random carpet, Gold-Hoyle, Lundquist and straight twisted tubes), flux emergence, scripted footpoint motion, reconnection criteria, plasmoids.
- Cannot: periodic boundaries; a true box (only a lower boundary plus one more plus the open sphere); gravity or mass (the mass routine is commented out); a Cartesian potential-field starting state (new code needed).
- FLUX is quasi-static, force-free, with fixed topology unless reconnection is called. A relaxation step takes no physical time. A "time evolution" is footpoint moves followed by relaxation: a sequence of equilibria, not dynamics. Seconds can be put on frames only if the scene states a driver speed and an Alfven speed. FLUX models no waves, flares or reconnection outflows.
- Fluxpype uses only the global spherical case today. The planar mode, B2, reconnection and the driver scripts have not been exercised by fluxpype in years; a Phase 1 check on Gilly's Mac is meant to find out what still works.

## Physics guards the plan requires (say them when relevant)

1. Structures from different generators placed in one scene do not give the field of the combined flux. Overlaps need a warning or a re-seed from the combined boundary.
2. Connectivity set by hand (editing individual fluxons) is the answer when reconnection is off, so such scenes are labelled "hand-set connectivity".
3. Sequences are frames, not time, unless driver and Alfven speeds are stated.
4. Light images of a Mm-scale local scene are illustrative; EUV needs a temperature FLUX does not have.
5. A test run with a handful of fluxons checks that the steps connect, not that the physics is right.
6. "200 G mean" is realized only to within one fluxon's flux; report the realized value.
7. Report "construction checks passed", never "looks good", from geometry checks alone.

## The planned pieces

- Recipes and steps registry (global PFSS becomes recipe 1; steps like Relax, Wind, Light, Export are boxes with declared inputs and outputs, cached by a hash of their settings and everything upstream). Needs splitting `magnetogram2wind.pdl` and migrating the tasks table.
- A scene file (declarative spec) compiled by deterministic builders; a capability file that refuses what FLUX cannot do.
- A Designer panel: scenes page and switcher, 3-D and top views to place and move structures, inspector knobs generated from schemas, a gallery of existing structures (twisted loop, tubes, carpet, emerging loop, menagerie worlds), a library of the user's own blocks, a pipeline timeline of boxes, a live job-script preview, a Test run button, tooltips and help, a downloads panel.
- Scene tools for you (planned): validate a scene, preview it, propose a complete scene as a card, and later apply edits. You would still only propose; the person's click runs things.
- FLUXlight as the light layer: plasma, emission, camera and instrument layers; FITS output with WCS.

## How to answer about this

- Planned versus exists: be explicit. The Designer is not built.
- When asked to design a scene now, explain what slice A would need and what exists today, and suggest the nearest thing that runs today (a global preset) only if it really answers the question.
- Open questions Gilly has not answered keep their defaults in section 7, 9.7, 10.5, 11.12 and 12.6 of the full plan.
