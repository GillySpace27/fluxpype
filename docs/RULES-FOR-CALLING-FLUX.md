# Rules for calling FLUX

FLUX core (fluxon-mhd) is frozen, so fluxpype follows these rules at every Perl entry point that calls it. The rules
live in one file, `fluxpype/science/flux_api.pm` (audit A1-01, A1-09); a new driver uses it instead of copying them.

```perl
use lib 'fluxpype/science';          # or `perl -I fluxpype/science`
use flux_api qw(load relax_cycle fix stats save done finish);
my $w = load($in);                   # read_world, kept alive
relax_cycle($w, $steps, $dt);        # neighbor init on step 0, then update_force + relax_step per step
fix($w, 0.3, 0.1, 2.5);              # fix_curvature(hi, lo), then fix_proximity(prox)
save($w, $out);                      # write_world: the last write
done(); finish();                    # print DONE, flush, POSIX::_exit(0)
```

| # | Rule | Why (core defect) | In the facade |
|---|---|---|---|
| 1 | Never free a world. Keep every Flux object alive, write `DONE` after the last write, flush, then `POSIX::_exit(0)`. | Freeing a world segfaults (exit 139) when it has a one-sided neighbor link (K04, A1-03; data.c teardown). | `load`, `read_str` and `keep` push to `@KEEP`; `done` and `finish` end the process. |
| 2 | A forked child leaves with `POSIX::_exit`, not a normal exit. Hand its result to the parent first. | Flux's DESTROY in a child frees the parent's world (Core.xs). | `fork_child()` marks the child; the facade's `END` block does the exit. |
| 3 | Do not run forked relax daughters unless needed: `relax_concurrency = 0` means serial. | Daughters share stdout and stderr and can SIGPIPE (model.c fork path). | Not enforced; `relax_cycle` is serial only. |
| 4 | Match `DONE` with `endswith`, not equality. | The core prints "Read '...'" with C `printf`, which is not ordered with Perl's `$|` output and can be glued in front of `DONE`. | Python side (`scenes/relax.py`); `done` only prints the line. |
| 5 | Call `update_neighbors(-1)` before the first `update_force`. | `update_force(0)` with no neighbors segfaults instead of failing (model.c `expand_lengthwise`). | `neighbors_init`, called by `relax_cycle` while `rel_step` is 0. |
| 6 | Order inside a cycle: steps, `fw_stats`, then `fix_curvature`, then `fix_proximity`. Read `dtau` once if one step size must hold across cycles. | The result depends on this order; the core changes `dtau` while it relaxes. | `relax_cycle`, `stats`, `fix`. |

Also true, not worked around: never use `exists` on a Flux object (always false, A1-09 #9); `str2world` ignores a
failed temp-file open, so check its result (A1-09); the core prints "second photosphere detected" to stderr on every
read of such a world (log noise, not an error).

## Status

Routed through the facade: the fork-child rule (rule 2) in `map_fluxon_flow_parallel_master.pm`. The facade's
`load`, `relax_cycle`, `fix`, `stats`, `save`, `done` and `finish` reproduce the scenes driver byte for byte
(`tests/realflux/test_facade.py`), but `scenes/relax.py` DRIVER, `simple_relaxer.pm`, `relax_pfss_world.pm` and the
other `read_world` and `write_world` callers still call Flux directly. They move one at a time; see the PR notes.
