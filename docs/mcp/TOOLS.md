# Fluxpype MCP server: tools and operation

`flux-plumb-mcp` is an MCP server (stdio) that wraps Plumber's REST API. It holds no pipeline state of its own. The design is in `PLAN.md`; this page is the reference for users.

## Setup

See the "MCP server" section of the README for installation and registration with Claude Code and Claude Desktop. In short: install the `mcp` extra into the Plumber environment and register `python -m fluxpype.plumber.mcp_server`.

```bash
cd ~/vscode/fluxons-plumber/fluxpype
.venv-plumber/bin/python -m fluxpype.plumber.mcp_server --help
```

Nothing is written to stdout except the MCP protocol; diagnostics go to stderr.

### Connect-an-agent prompt

Plumber's **Connect agent** button (top bar, and Settings > AI agents) shows a prompt to paste into any MCP-capable agent. It is built by `fluxpype/plumber/agent_prompt.py`, served at `GET /api/agent-prompt`, and printed by `flux-plumb-mcp --print-prompt [--home PATH] [--demo]`. It:

- names the interpreter running Plumber and that Plumber's home, so the agent reaches the same queue (a demo Plumber registers as `fluxpype-demo` with `--demo`);
- probes that interpreter from outside the repo with an empty environment, as an MCP host would launch it, and adds a `pip install` step if `mcp` or `httpx` is missing, or a warning if fluxpype itself is not importable;
- gives the Claude Code command (`claude mcp add --scope user ...`), the JSON `mcpServers` entry, and a generic fallback;
- asks the agent not to replace an existing server of the same name without asking, not to start jobs, and to confirm the connection by calling `status`;
- never contains the API token.

## Tools

Results are compact JSON. Every string is cut to 2,000 characters, except log text, which is bounded by line count (400) and by 40,000 characters. Config comments are removed from profile output.

| Tool | Arguments | What it does | Kind |
|---|---|---|---|
| `status` | none | Version, backends, queue settings, running and queued tasks, recent results with failure reasons | read-only |
| `list_profiles` | none | Profiles with batch name, task count, errors and warnings | read-only |
| `get_profile` | `name` | A profile's own settings, editable keys, and a task preview showing cached stages | read-only |
| `submit_job` | `profile`, `overrides`, `tasks`, `backend`, `scenario`, `speed` | Queues a job. Override values may be strings, numbers, booleans or lists; they are sent as strings (`[2225, 2226]` for a list). `tasks` is a list of `[cr, nflux, adapt, method]`. `scenario` and `speed` only apply to the fake backend | write |
| `list_jobs` | `limit` (20), `status`, `profile` | Recent jobs, filtered | read-only |
| `get_job` | `job_id` | One job with a stage summary per task | read-only |
| `get_task` | `task_id` | Stages (status, source, verified, elapsed), latest relaxation, warnings, degraded flag, exit code, artifacts with an index | read-only |
| `get_relax_series` | `task_id`, `max_points` (200) | Relaxation rows, downsampled but always keeping the first and last point and the cycle boundaries | read-only |
| `read_task_log` | `task_id`, `tail_lines` (200, max 400), `grep`, `context` (3, max 20) | The log tail, or matching lines with context. `grep` is a case-insensitive plain substring, not a regex | read-only |
| `wait_for_task` | `task_id`, `until` (`finished` or a stage id), `timeout_s` (60, max 300) | Polls every 1.5 s, reports progress on each stage change, and returns the final state or `{"still_running": true, "stage": ...}` | read-only |
| `cancel` | `job_id` or `task_id` | Cancels a job or one task; running work is stopped | destructive |
| `retry_failed` | `job_id` | Re-queues the failed tasks of a job | write |
| `set_queue` | `paused`, `max_parallel`, `cpu_budget` | Changes queue settings | write |
| `cache_summary` | `profile`, `kind`, `cr`, `nflux`, `limit` (50), `offset`, `rescan` | Totals by kind plus a filtered, paged file list | read-only |
| `get_task_image` | `task_id`, `index` (-1 is the newest), `max_edge` (1024) | One plot as an image with a caption. With Pillow installed it is downscaled; without it, images over 1.5 MB are refused | read-only |

Not provided: moving files to the cache trash, restoring or emptying it, deleting jobs, writing `config.ini`, MCP resources and prompts.

## Safety notes

- Log text, warnings, stage details and file names come from the pipeline and from files on disk. Tool descriptions tell the assistant to treat them as data, never as instructions.
- `submit_job` runs real work. It cannot change the keys Plumber forbids (for example `run_script`), and Plumber's own validation errors are returned verbatim.
- `cancel` is marked destructive so clients can ask for confirmation.
- Plumber only answers `localhost`, `127.0.0.1` and `::1`. `--url` rejects any other host. To use a Plumber on another machine, open an SSH tunnel and point `--url` at the local end.
- The API token is read from `<home>/token` (mode 0600). With `--url`, prefer `--token-file` or `FLUX_PLUMBER_TOKEN` over `--token`, which shows up in the process list.

## Finding and starting Plumber

By default the server uses `--home` (default `~/.fluxpype/plumber`, or `$FLUX_PLUMBER_HOME`) and connects lazily on the first tool call:

1. It reads `<home>/server.json` and `<home>/token`, checks that the recorded process is alive, and asks `/api/status` to confirm the Plumber's home matches.
2. If there is no valid `server.json` and nothing holds `<home>/plumber.lock`, it starts a detached headless Plumber on a free port (output goes to `<home>/server.log`) and waits up to 15 s for it to publish `server.json`.
3. If the lock is held but there is no usable `server.json`, it reports an error and never starts a second Plumber. Run `flux-plumb --stop` or read `<home>/server.log`.

`--no-autostart` turns step 2 off, and it is always off with `--url`. If Plumber is restarted while the assistant is connected, the next call looks it up again.

## Lifetime and stopping

A Plumber started by the MCP server is a separate session leader and keeps running after the assistant exits, because its jobs must. Stop it with:

```bash
cd ~/vscode/fluxons-plumber/fluxpype
.venv-plumber/bin/flux-plumb --stop --home ~/.fluxpype/plumber
```

`--stop` sends SIGTERM to the pid in `server.json` (after checking its start time), waits up to 10 s, and exits 0 when it is gone. Jobs are detached from Plumber, so running jobs keep running; the next Plumber re-attaches to them.

## Demo mode

`--demo` (which needs `--home`) makes autostart launch `flux-plumb --demo --home <home>`, with simulated runs and no FLUX or PDL needed:

```bash
cd ~/vscode/fluxons-plumber/fluxpype
.venv-plumber/bin/python -m fluxpype.plumber.mcp_server --home /tmp/plumber-demo --demo
```

## Tests

```bash
cd ~/vscode/fluxons-plumber/fluxpype
.venv-plumber/bin/python -m pytest -q tests/test_plumber_mcp.py tests/test_plumber_cli.py
```

The MCP tests drive the server with the SDK client against an in-process Plumber on the fake backend. One test starts a real subprocess over stdio. The downscale tests are skipped without Pillow.
