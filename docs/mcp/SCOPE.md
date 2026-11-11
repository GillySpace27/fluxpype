# Fluxpype MCP server: scope

Status: scope draft 1 (2026-11-10). Builds on Fluxpype Plumber (PR #47); read `docs/gui/PLAN.md` first.

## 1. Goal

Let an MCP client (Claude Code, Claude Desktop, or any other MCP host) drive fluxpype the way Gilly does through Plumber. The client should be able to:

- read and preview profiles, and queue runs with overrides;
- watch stage progress and relaxation metrics;
- read logs to diagnose a failure, then retry;
- look at the output images;
- inspect the cache and clean it up.

Typical asks this should make one sentence long:

- "Run light_paper for CR 2150 with 1000 fluxons on ADAPT and tell me when the FLUXlight render is done."
- "Why did last night's wsa job fail?"
- "Which CRs already have relaxed worlds at 300 fluxons?"
- "Show me the newest FLUXlight image."

## 2. The main design decision: the MCP server is a client of Plumber, not a second scheduler

Plumber's `Scheduler` owns the job database (SQLite), the output-key locks, the CPU-slot budget, and reattachment to detached jobs. Suppose the MCP server embedded its own `Scheduler` while the GUI was also running. Then two schedulers would read the same queue and could launch the same pending task twice, which breaks the locking guarantees from PLAN.md.

So `flux-plumb-mcp` is a thin adapter over Plumber's existing REST API:

```
MCP host (Claude)  --stdio-->  flux-plumb-mcp  --HTTP + X-Flux-Token-->  flux-plumb server  -->  Scheduler / jobs
                                                                          (same one the browser uses)
```

- It finds the server in `~/.fluxpype/plumber/` (or `$FLUX_PLUMBER_HOME`): it reads the `token` file, plus a new `server.json` that `flux-plumb` writes at startup with host, port, pid and version.
- If no server is running, it starts one headless (`flux-plumb --no-browser`, detached) and waits for `/api/status`. A flag can turn that off (`--no-autostart`), in which case it returns a clear error instead.
- A job launched from Claude shows up in the browser immediately, and the reverse holds too. The GUI and the agent are two views of one queue.
- Remote or cluster use works with the same SSH-tunnel recipe as the GUI. Point the MCP server at `--url http://localhost:8765` and `--token-file` (or the `FLUX_PLUMBER_TOKEN` env var). No new network exposure.

The cost of this approach: the MCP server can only do what the REST API exposes. In the other direction, it gets every Plumber fix for free and adds no second code path to keep in sync.

## 3. Transport and packaging

- **Transport:** stdio, which is what Claude Code and Claude Desktop launch locally. Streamable HTTP is deferred, since the tunnel covers remote use.
- **Library:** the official Python MCP SDK (`mcp` on PyPI, FastMCP-style decorators). It supports Python 3.10 and later, matching fluxpype's range. I have not checked the exact minimum SDK version to pin; the plan will.
- **Packaging:**
  - module `fluxpype/plumber/mcp_server.py`;
  - entry point `flux-plumb-mcp`;
  - optional extra `[mcp]` = `mcp` + `httpx` (the `[plumber]` extra already carries FastAPI and uvicorn for autostart).
- **Registration** (documented in the README, with the `cd` spelled out):
  - Claude Code: `claude mcp add fluxpype -- <venv>/bin/flux-plumb-mcp`
  - Claude Desktop: a JSON snippet for its config file.

## 4. Tools

Each tool maps to one existing REST endpoint unless noted. The safety column uses MCP tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`) so hosts can ask before acting.

| Tool | Does | REST | Safety |
|---|---|---|---|
| `server_status` | Plumber version, home, queue state, backends | `GET /api/status`, `/api/queue` | read-only |
| `list_profiles` | Profile names with their task-product size and validation errors | `GET /api/profiles` | read-only |
| `get_profile` | Effective settings, the source of each value (DEFAULT or section), editable keys, task preview with cache hits | `GET /api/profiles/{name}` | read-only |
| `submit_job` | Queue a profile with optional overrides and a task subset. Returns the job id and task list | `POST /api/jobs` | writes; not destructive |
| `list_jobs` | Recent jobs with status, filterable by status or profile | `GET /api/jobs` | read-only |
| `get_job` | Job plus per-task stage summary | `GET /api/jobs/{id}?states=1` | read-only |
| `get_task` | Full task state: stages with source and verification, relax metrics, warnings, degraded flag, exit code | `GET /api/tasks/{id}` | read-only |
| `get_relax_series` | Relaxation series (step, stiffness, fmax, f_av, fs_av, cycle markers), downsampled to N points by default | `GET /api/tasks/{id}/relax` | read-only |
| `read_task_log` | Log tail or window, plus `grep` and `around_stage` options so the agent never pulls a 50k-line log | `GET /api/tasks/{id}/log` (+ server-side filter, **new**) | read-only |
| `wait_for_task` | Block until a task reaches a stage or finishes, or until a timeout (capped at about 10 min), streaming MCP progress notifications per stage | WebSocket `/api/ws`, polling fallback | read-only |
| `cancel` | Cancel a job or one task | `POST .../cancel` | **destructive** |
| `retry_failed` | Retry a job's failed tasks | `POST /api/jobs/{id}/retry` | writes |
| `set_queue` | Pause or resume; set max_parallel and cpu_budget | `PATCH /api/queue` | writes |
| `cache_summary` | Totals by kind; records filterable by kind, CR, n, batch, instrument | `GET /api/cache` | read-only |
| `cache_trash` / `cache_restore` | Move records to trash, or restore them | `POST /api/cache/trash[/restore]` | trash is **destructive** but reversible |
| `get_artifact_image` | Return a FLUXlight render, wind map or field map as MCP image content, downscaled to a max edge (default 1600 px), so Claude can actually look at results | `GET /api/artifacts` | read-only |

Deliberately **not** exposed:

- **Emptying the trash.** It is the one irreversible action; it stays a GUI-only button.
- **Deleting jobs.**
- **Anything that writes `config.ini`.** Plumber v1 never writes it, and the agent shouldn't either.

Overrides go through the same whitelist and value regex as the GUI. That matters because `pipe_helper.convert_value` calls `eval()` on config values.

## 5. Resources and prompts

- **Resources** (read-only context the host can attach):
  - `fluxpype://profiles/{name}`: effective profile;
  - `fluxpype://tasks/{id}/log`: log, tail by default;
  - `fluxpype://events-schema`: `docs/gui/EVENTS.md`;
  - `fluxpype://stages`: the 10 stage ids with descriptions.
- **Prompts** (canned workflows):
  - `diagnose_task(task_id)`: pull state, warnings, the failing stage, and the log around it; propose a fix.
  - `run_and_report(profile, overrides)`: submit, wait, return timings and the final image.
  - `compare_runs(task_a, task_b)`: stage durations and relax convergence side by side.

## 6. Output shaping for an LLM client

Tool results are compact JSON with stable field names and units:

- times in seconds;
- stiffness in percent, as fluxpype prints it;
- sizes in bytes plus a human string.

Large payloads are bounded:

- relax series are downsampled to 200 points unless the caller asks for more;
- logs are capped at 400 lines per call, with a cursor;
- cache listings are paginated (50 per page).

Every task result carries `source` and `degraded`, so the agent knows when progress was inferred from logs rather than events.

## 7. Changes needed in Plumber (small)

1. `flux-plumb` writes `server.json` (host, port, pid, version, started) next to `token`, and removes it on clean shutdown. The MCP server checks the pid is alive before trusting it.
2. `GET /api/tasks/{id}/log` gains `grep=` and `around_stage=` (byte-range around a stage's first log line) filters.
3. `GET /api/artifacts` gains `max_edge=` for server-side downscaling (Pillow is already pulled in by matplotlib in the science stack; if it's unavailable, fall back to returning the original image).
4. A `--headless` alias for `--no-browser --quiet` so autostart doesn't spam output.

## 8. Testing (no FLUX needed)

- Start `flux-plumb --demo` in a temp home inside pytest (same fixtures as the server tests).
- Drive the MCP server in-process with the SDK's client session over memory streams.
- Cover:
  - every tool's happy path;
  - auth failure (wrong token);
  - no-server autostart;
  - override rejection;
  - `wait_for_task` progress notifications;
  - log filtering limits;
  - image downscaling.
- One end-to-end check through a real stdio subprocess.
- Manual: register the server with Claude Code in this container and run the four example asks from section 1 against the demo.

## 9. Effort and phasing (estimates)

| Phase | Content | Size |
|---|---|---|
| M0 | Plumber additions in section 7, with tests | ~250 LOC |
| M1 | MCP server: read-only tools + resources | ~400 LOC |
| M2 | Write tools (submit, cancel, retry, queue, trash) with annotations; `wait_for_task` with progress | ~300 LOC |
| M3 | Prompts, README registration docs, e2e test | ~200 LOC |

This fits in one implementation pass of the kind used for Plumber. It can land on the same PR or a stacked follow-up PR. A follow-up is my recommendation, so #47 can be reviewed on its own.

## 10. Open questions (defaults in brackets)

- Should Claude be allowed to cancel jobs without asking? [Yes, but annotated destructive, so hosts prompt.]
- Should autostart be on by default? [Yes for localhost; off when `--url` points elsewhere.]
- Do we also want direct, science-level tools later that need the science stack? Examples: open a `.flux` world and report fluxon counts and open/closed flux, or compute a footpoint histogram. [Defer; they don't belong in the Plumber-client layer.]

## 11. Risks

- **The MCP SDK API is still moving.** Pin a version, and keep the adapter thin.
- **Long waits:** hosts time out long tool calls. `wait_for_task` caps its wait and returns `still_running` with the current stage, so the agent can call again.
- **Prompt injection via logs:** log text is returned as data inside a JSON field, never as instructions. Tool descriptions say so.
