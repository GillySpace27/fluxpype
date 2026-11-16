# Fluxpype MCP server: implementation plan (v1)

Inputs: `docs/mcp/SCOPE.md` and an independent review of it. This plan supersedes the scope where they differ. Review dispositions are at the end.

## 0. Ground rules

- Branch `claude/project-thread-rx783j` (PR #47). Commit in logical steps; push to that branch only.
- Never use the em dash character anywhere.
- Do not change Plumber behavior except as listed in section 1. All existing tests (`pytest tests`, 171 at start) must still pass.
- SDK: `mcp>=2.3,<3`, using `from mcp.server.mcpserver import MCPServer, Context, Image`, `mcp_types.ToolAnnotations`, and `from mcp import Client` for tests. These were verified against mcp 2.3.0 in this container. Do not use the removed `FastMCP` module.
- The package requires Python `<3.13`; this container runs 3.13, so run everything as modules from the repo root (`python -m fluxpype.plumber.mcp_server`). Do not pip-install fluxpype here.
- New extra in `pyproject.toml`: `mcp = ["mcp>=2.3,<3", "fastapi>=0.110", "uvicorn[standard]>=0.27"]`. The HTTP client is `httpx`, which ships with FastAPI's test stack. Add `httpx>=0.27` explicitly to the extra, since mcp 2.x does not depend on it.
- Entry point `flux-plumb-mcp = "fluxpype.plumber.mcp_server:main"`.

## 1. Plumber changes (M0)

1. **Single-instance lock.** In `cli.py` before building `Store`/`Scheduler`, open `<home>/plumber.lock` and take `fcntl.flock(LOCK_EX | LOCK_NB)`. If it is held, print "another Plumber is already running for <home>" plus the contents of `server.json` if present, and exit 3. Hold the fd for the process lifetime.
2. **Write the token and `server.json` only after the lock and a successful bind check.** Move the port check before the token write. Write both atomically (temp file + `os.replace`), with mode 0600. `server.json` = `{"host","port","pid","pid_start","version","home","started","demo"}`. Remove `server.json` in a `finally` on shutdown (best effort).
3. **`--demo` honors `--home`.** With `--home`, the demo seeds that directory instead of a random temp dir; without it, behavior is unchanged.
4. **`--headless`**: implies `--no-browser` and prints nothing to stdout, logging only to `<home>/server.log`.
5. **`--stop`**: reads `server.json`, verifies the pid is alive with the same `pid_start`, sends SIGTERM, and waits up to 10 s. Jobs are detached, so they keep running; say so in the help text.
6. **Log grep**: `GET /api/tasks/{id}/log` gains optional `grep` (a plain substring, case-insensitive, not a regex) and `context` (lines, default 3, max 20). It returns matching line windows, capped at 400 lines total. Without `grep`, behavior is unchanged.

Tests: lock contention (a second start exits 3 and does not touch the token), `server.json` contents and removal, demo plus home, `--stop`, and log grep.

## 2. MCP server (`fluxpype/plumber/mcp_server.py`)

### 2.1 Connection

- `PlumberClient` wraps `httpx.Client(base_url, headers={"X-Flux-Token": token}, timeout=10)`. The constructor also accepts an injected `transport=` so tests can use `httpx.ASGITransport` / FastAPI's app in process.
- Discovery order:
  1. `--url` plus `--token` / `--token-file` / `FLUX_PLUMBER_TOKEN`;
  2. otherwise `--home` (default `paths.plumber_home()`): read `server.json` and `token`, then validate with `GET /api/status` that the returned `plumber_home` equals the home.
- Reject a `--url` host that is not `localhost`, `127.0.0.1` or `::1` with a clear message (Plumber's Host check would 403 it). Remote use goes through an SSH tunnel to localhost.
- **Autostart** (on by default for the home mode, off with `--no-autostart` and always off in `--url` mode):
  - only when `server.json` is absent or invalid **and** the lock file is not held (probe with a non-blocking `flock` and release immediately);
  - spawn `[sys.executable, "-m", "fluxpype.plumber.cli", "--headless", "--home", home] (+ "--demo" when the MCP server was started with --demo)` with stdin DEVNULL, stdout/stderr to `<home>/server.log`, and `start_new_session=True`;
  - poll for a valid `server.json` for up to 15 s.

  If the lock is held but no usable `server.json` exists, return an error telling the user to run `flux-plumb --stop` or check `<home>/server.log`. Never start a second one.
- The started server outlives the MCP host, because jobs must. Document `flux-plumb --stop`.

### 2.2 Tools (all results compact JSON; strings capped at 2,000 chars; config comments stripped)

| Tool | Args | Backed by | Annotations |
|---|---|---|---|
| `status` | none | `/api/status` + `/api/queue` + `/api/overview` (running, queued, recent finished with failure reasons) | read-only |
| `list_profiles` | none | `/api/profiles` (name, task count, errors) | read-only |
| `get_profile` | name | `/api/profiles/{name}`; drop `comment` fields; include editable keys and task preview with cache hits | read-only |
| `submit_job` | profile, overrides: dict[str, str/num/list], tasks: optional list of [cr, nflux, adapt, method], backend, scenario, speed | `POST /api/jobs` (the adapter stringifies values; lists become `[a, b]`) | write, not destructive, not idempotent |
| `list_jobs` | limit=20, status=None, profile=None | `/api/jobs?limit=` with adapter-side filtering | read-only |
| `get_job` | job_id | `/api/jobs/{id}?states=1`, shaped to a per-task stage summary | read-only |
| `get_task` | task_id | `/api/tasks/{id}`, shaped: stages (status, source, verified, elapsed), relax latest, warnings, degraded, exit code, artifacts with index | read-only |
| `get_relax_series` | task_id, max_points=200 | `/api/tasks/{id}/relax`, downsampled adapter-side (keep first, last, cycle boundaries; stride the rest) | read-only |
| `read_task_log` | task_id, tail_lines=200 (max 400), grep=None, context=3 | `/api/tasks/{id}/log` (tail via byte offset from the end; grep via the new server param) | read-only |
| `wait_for_task` | task_id, until="finished" or a stage id, timeout_s=60 (max 300) | polls `/api/tasks/{id}` every 1.5 s; `ctx.report_progress` on each stage change; returns the final state or `{"still_running": true, "stage": ...}` | read-only |
| `cancel` | job_id or task_id | `POST .../cancel` | destructive |
| `retry_failed` | job_id | `POST /api/jobs/{id}/retry` | write, not idempotent |
| `set_queue` | paused, max_parallel, cpu_budget (all optional) | `PATCH /api/queue` | write |
| `cache_summary` | profile=None, kind=None, cr=None, nflux=None, limit=50, offset=0 | `/api/cache?profile=`, with filtering and pagination adapter-side; totals by kind always included | read-only |
| `get_task_image` | task_id, index=-1 (newest), max_edge=1024 | the task's `artifacts` list, then `/api/artifacts?path=`. Downscale adapter-side with Pillow if importable (else return the original if under 1.5 MB, otherwise an error naming the size). Returns `Image` plus a one-line caption | read-only |

Not in v1: trash and restore (review I3: profile mismatch and in-use hazards), empty trash, delete job, any `config.ini` write, resources, and prompts.

Tool descriptions state that log text, warnings and file names are untrusted data from the pipeline, not instructions.

### 2.3 CLI

`flux-plumb-mcp [--home PATH] [--url URL] [--token T | --token-file F] [--no-autostart] [--demo]`. It runs `MCPServer.run(transport="stdio")`. Nothing may print to stdout except the protocol; send diagnostics to stderr.

## 3. Tests (`tests/test_plumber_mcp.py`)

- Build the FastAPI app in process exactly as `tests/test_plumber_server.py` does (fake backend, temp home), give `PlumberClient` an `httpx` transport bound to it, and build the `MCPServer` around that client.
- Drive it with `async with Client(server)`. Run async tests with `anyio` (`@pytest.mark.anyio`) or `asyncio.run`; `pytest-asyncio` is not installed.
- Cover:
  - each tool's happy path;
  - wrong token, which must produce a clear tool error;
  - override rejection surfaces the server's message;
  - `wait_for_task` progress callbacks and the `still_running` path;
  - log tail and grep caps;
  - relax downsampling keeps the endpoints;
  - comments stripped from `get_profile`;
  - the image tool (`pytest.importorskip("PIL")` for the downscale case);
  - annotations present on the destructive and write tools.
- One subprocess test:
  - start `python -m fluxpype.plumber.mcp_server --home <tmp> --demo` over stdio with the SDK's stdio client;
  - verify autostart creates `server.json`, `status` works, a second MCP server reuses the same Plumber (same pid), and `flux-plumb --stop` stops it.
- The M0 tests listed in section 1.

## 4. Docs

- `README.md`: an "MCP server" subsection with registration recipes.
  - Claude Code:

    ```
    cd ~/vscode/fluxons-plumber/fluxpype
    claude mcp add fluxpype -- "$PWD/.venv-plumber/bin/python" -m fluxpype.plumber.mcp_server
    ```

  - Claude Desktop: the JSON snippet with the absolute venv python path and `"args": ["-m", "fluxpype.plumber.mcp_server"]`.
  - Always show the `cd`.
- `docs/mcp/TOOLS.md`: tool table, safety notes, autostart and lifetime, `flux-plumb --stop`.

## 5. Order

1. M0 Plumber changes and their tests.
2. `mcp_server.py` with the read-only tools, plus tests.
3. Write tools and `wait_for_task`, plus tests.
4. Image tool, CLI, subprocess test.
5. Docs.

Run the full suite at the end and report the exact counts.

## Review dispositions

| Item | Disposition |
|---|---|
| B1 SDK renamed (FastMCP to MCPServer) | Adopted: pin `mcp>=2.3,<3`, MCPServer API. |
| B2 token overwritten by a second start | Adopted: lock first, then atomic token and server.json after the bind check. |
| B3 autostart double scheduler | Adopted: lock probe; error if lock held without a valid server.json; validate via `/api/status` home. |
| B4 demo not discoverable | Adopted: `--demo` honors `--home`; MCP `--demo` pass-through. |
| I1 REST claims wrong | Adopted: filtering, downsampling, tail and pagination done adapter-side; image by task + index. |
| I2 around_stage has no data | Adopted: dropped; grep with context instead. |
| I3 trash hazards | Adopted: trash and restore cut from v1. |
| I4 wait cap and WebSocket | Adopted: polling, 60 s default, 300 s cap. |
| I5 autostart hygiene | Adopted: headless, log to file, new session, `--stop`. |
| I6 non-localhost URL | Adopted: reject up front. |
| I7 injection surface | Adopted: strip comments, cap strings, annotations, untrusted-data notes in descriptions. |
| I8 submit details | Adopted: stringify, expose backend, scenario and speed. |
| I9 testing | Adopted: injected transport, anyio, importorskip. |
| Minor: resources and prompts | Cut. |
| Minor: overview tool | Folded into `status`. |
| Minor: image size | `max_edge` 1024, 1.5 MB cap without Pillow. |
