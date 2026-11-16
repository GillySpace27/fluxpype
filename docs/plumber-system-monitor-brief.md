# Plumber system monitor: a brief for porting it to HelioFITS Studio

Written 2026-11-11 from the code at commit 13035d3 ("agent activity record, system monitor, FITS thumbnails") in the
fluxpype repo. Everything under "What it does" is read from that code. Everything under "Porting to Java" is a
suggestion; none of it has been built or run.

## What it is

Two pieces that talk over one JSON endpoint.

- A sidebar tile, always visible: six one-line rows (CPU, MEM, GPU, NET, DSK, STO), each with a label, a sparkline and
  a current value. It is one button; clicking it opens the detail popup.
- A popup ("System monitor"): a card per resource with a taller chart, a per-core bar strip, a storage list, and two
  process tables. Open it and it refreshes every 2 s.

Sources:

| Piece | File |
| --- | --- |
| Sampler (Python) | `fluxpype/plumber/sysmon.py` |
| Endpoint `GET /api/system?procs=0\|1` | `fluxpype/plumber/server.py` (search `sysmon`) |
| Tile, charts, popup (React) | `fluxpype/plumber/web/src/components/SysMon.tsx` |
| Polling hook `useSystem` | `fluxpype/plumber/web/src/lib/hooks.ts` |
| JSON shape `SysSnapshot` | `fluxpype/plumber/web/src/lib/types.ts` |
| Placement in the sidebar | `fluxpype/plumber/web/src/components/Layout.tsx` |

There are no tests for it. `psutil>=5.9` was added to the `plumber` and `mcp` extras in `pyproject.toml`.

## Why it works well (the approach worth copying)

1. **One sampler on the server, many viewers.** The UI never measures anything. It asks for a snapshot. The sampler
   returns the cached last sample if it is under 1 s old, so any number of open pages cost one measurement a second.
2. **History lives with the sampler, not the page.** The last 120 samples are kept in a deque and returned with every
   snapshot, so a page that has just opened draws full charts immediately instead of starting empty.
3. **Sampling is on demand.** There is no background thread. Nothing is measured while nobody is looking.
4. **Rates are counter deltas.** Network and disk I/O come from cumulative byte counters; the rate is
   `(now - previous) / elapsed seconds`. The first sample has no previous, so its rates are `null` and the UI shows "n/a".
5. **Every number may be missing, and the UI says so.** `null` means "this platform gives no reading". With no psutil
   the endpoint reports `psutil: false`, CPU falls back to the 1-minute load average divided by core count (labelled
   "CPU load" because it is a load, not a utilisation), storage still works, and the popup shows an install hint.
   Nothing crashes and nothing shows a fake zero.
6. **Attribution to the app's own work.** The popup has a "tasks running now" table: every process whose process group
   is a running Plumber task, summed per task (CPU, memory, process count), plus a `task` badge on matching rows in
   "Busiest processes". The server supplies a `{process-group id: label}` map; the sampler matches each process's
   group against it. This is the feature that makes the panel useful rather than generic, so keep it.
7. **Cheap heat colouring.** One rule for every percentage: accent below 70, amber from 70, red from 90.

## Sample and snapshot contract

One sample (all rates per second, `null` where unavailable):

```
{ t, cpu, mem, net_rx, net_tx, disk_r, disk_w, gpu }     // cpu, mem, gpu are 0-100
```

Snapshot returned by `GET /api/system` (`?procs=1` only when the popup is open):

```
psutil: bool
now: Sample                      // the newest sample
history: Sample[]                // up to 120, oldest first
cpu_count, load: [1m, 5m, 15m]
gpu: { name, pct, mem_used|null, mem_total|null } | null
storage: [{ label, path, total, used, free, pct }]
cores?: number[]                 // per-core percent, psutil only
memory?: { total, used, swap_used, swap_total }
processes?: { top: [{ pid, name, cpu, rss, task|null }],      // 8 busiest by CPU
              tasks: [{ label, cpu, rss, procs }] }           // grouped by app task
```

Details that matter when re-implementing:

- CPU is the mean of the per-core percentages. The sampler primes `cpu_percent` once at construction, because the first
  call has no interval to measure over.
- Memory "used" is `total - available`, not psutil's `used`, so cache is not counted as used. Swap is shown as text only.
- GPU: on macOS, parse `ioreg -r -d 1 -w0 -c IOAccelerator` for `"Device Utilization %"` and `"model"` (Apple
  silicon gives no memory figures). Elsewhere, if `nvidia-smi` exists, query
  `name,utilization.gpu,memory.used,memory.total` as CSV (memory is in MiB). Both calls have a 2 s timeout, and the GPU
  is re-read at most every 2 s, so a slow `ioreg` cannot stall the 1 s cadence. Any failure returns "no GPU", never an error.
- Storage: a list of `(label, path)` supplied by the app (Plumber gives: data folder, its own home, and `/`). Entries
  on the same device (`st_dev`) are collapsed to the first, so the same disk is not listed three times. The sidebar STO
  row shows the first entry only.
- Process CPU is per core, so a multi-threaded process can exceed 100%. It reads 0 on the first measurement of a
  process; the popup's footnote says so. psutil's `process_iter` keeps the process objects between calls, which is what
  makes the second reading valid. A process that vanishes mid-scan (no memory info, or `getpgid` raises) is skipped or
  left unattributed.

## The tile, exactly

- Grid per row: `30px` label, flexible chart, `74px` right-aligned value. Label is 10 px, uppercase, spaced, faint.
- CPU, MEM, GPU: one line, y-axis fixed 0-100, line coloured by the heat rule above, soft gradient fill under it.
- NET, DSK: two lines on one chart (down/read in accent, up/write in info), y-axis autoscaled to the largest value in
  view plus 15% headroom (minimum 1), value shown is the sum, formatted as bytes per second.
- STO: not a chart, a thin filled bar.
- The chart is a plain SVG polyline: x is the sample index (not the timestamp) spread across the width, a `null`
  sample is skipped, a baseline hairline is drawn, and the stroke width does not scale with the box.

Known weakness to fix in the port: x is the index, and samples only exist while a page is polling, so a page that was
closed for ten minutes draws the old and new samples side by side as if they were adjacent. Plotting by `t`, or
inserting a gap, would be better. With one viewer polling every 2 s the 120 samples span about 4 minutes.

## Popup contents

Cards: CPU (big value, core count and load averages, 52 px chart, per-core bar strip with one bar per core, each
coloured by the heat rule), Memory (used of total, swap), GPU (name, memory when known, else "No GPU readout on this
machine"), Network (down/up, legend), Disk I/O (read/write, legend), Storage (one bar per volume with free of total
and the path). Below them: "tasks running now" (hidden when empty) and "Busiest processes". Without psutil the cards
for unavailable data show "n/a" and a banner explains how to get them.

## Porting to Java (suggestions, unverified)

I do not know Studio's UI toolkit, so the drawing part is described by behaviour rather than API.

- **Sampler.** Keep the same shape: one object, `sample()` rate-limited to once a second, a ring buffer of 120, and a
  `snapshot(withProcesses)` that builds the structure above. Run it on demand from the UI timer (2 s), not on its own
  thread, so it costs nothing when the panel is hidden.
- **Metrics source.** The JDK alone (`OperatingSystemMXBean` for load, `FileStore` for storage, `ProcessHandle` for
  the process list and its CPU time) covers CPU load, storage and processes, but not network, disk I/O, per-core load
  or swap. The OSHI library (`com.github.oshi:oshi-core`) covers those; I have not checked its current version or
  how it behaves on Apple silicon, so verify before committing to it. Whichever is chosen, keep the "any field may be
  null" contract so a missing metric degrades to "n/a".
- **GPU.** Port the two shell-outs as they are (`ProcessBuilder`, 2 s timeout, cache 2 s). They are the same on any
  language.
- **Task attribution.** In Java this is easier than the process-group trick: take the pid of each job Studio launched
  and sum over `ProcessHandle.of(pid).descendants()` plus the process itself. Label with whatever Studio calls a job.
- **Heat colours and thresholds** (70 / 90), the 1 s / 2 s / 2 s cadences, and the 120-sample history can be copied
  verbatim.
- **If Studio has a web or embedded-browser UI**, `SysMon.tsx` and the `SysSnapshot` type could be reused nearly as
  is, with Java serving the same JSON from `/api/system`. That is the cheapest route if it fits.

## Not verified

I read the code but did not run the monitor for this brief, and there are no tests for it. The macOS GPU parse depends on
the exact `ioreg` output format of the installed macOS version. The 4-minute history figure is arithmetic from the
constants (120 samples, 2 s apart), not a measurement.
