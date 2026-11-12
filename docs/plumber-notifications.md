# Plumber notifications: the event log

Plumber keeps one persisted, sequenced log of job events in its SQLite store (`events` table in `plumber.db`). The
server loop writes it whether or not a browser is open, so a job started by an agent with no web page open still
leaves its outcome in the log. The web app, the `watch` command and the MCP `get_notifications` tool all read this one
log. Code: `fluxpype/plumber/notify.py` (producer and line format), `store.py` (table, queries, retention),
`server.py` (loop wiring and API).

## What is recorded

Events carry only enums, ids and integers: never log text, file paths, failure detail or the token. Agents read
failure details through `get_task` and `read_task_log`, which already label pipeline text as untrusted.

| Kind | Severity | When | Fields (`f`) |
|---|---|---|---|
| `job_queued` | info | a new job is first seen | `tasks`, `backend` |
| `task_started` | info | a task starts (or ran and finished between two passes) | `attempt` |
| `task_ok`, `task_failed`, `task_timeout`, `task_crashed`, `task_cancelled` | success, error, error, error, info | a task attempt ends; exactly one per attempt | `attempt`, `exit`; `task_failed` adds `reason` (`exit`, `submit`, `stage`) and `stage`; `launcher_lost: 1` when only the launcher died and the runner's `run.end` decided |
| `job_ok`, `job_partial`, `job_failed`, `job_cancelled` | success, error, error, info | a job reaches an end status | counts `ok`, `failed`, `cancelled`, `timeout`, `crashed` |
| `task_stalled`, `task_resumed` | warn, info | no output for longer than the stall limit, then output again | `stage`, `quiet_s`, `limit_s` |
| `task_degraded` | warn | the task's events stopped while its log goes on | `stage` |
| `queue_paused`, `queue_resumed` | info | the queue setting changed | none |
| `backend_down`, `backend_up` | error, info | a backend answered `unknown` for a task, then a known state again | `backend` |
| `plumber_started` | info | the server started | `version`, `reattached` (running tasks found) |

Every event has a unique `key` (for example `task_end|<task>|<attempt>`); a second insert of the same key is ignored,
so two Plumbers on one home, or a restart, never duplicate an event. Outcomes come from store transitions (each task
and job row has a `notified` column that trails `status`), so every path is covered: a normal exit, a backend that
refused to start the task, a cancel while queued, a reattach after a restart. An upgraded database starts with
`notified = status`, so nothing old is announced.

**Stall limit**: `clamp(0.4 x task_timeout, 120, 1800)` seconds (800 s for the default 2000 s timeout), or the store
setting `stall_after_s` when set. "Output" is the newest modification time of `events.jsonl`, `run.log` and
`stdout.log` in the run directory. A gap of more than 30 s between telemetry passes (the laptop slept) suspends stall
checks for one full limit. Plumber never cancels or retries anything because of a stall. The limit is a starting
value that has not been checked against a real run's output gaps.

**Origin**: each job records who started it (`origin` on `GET /api/jobs*` and on its events): `web` (with the browser
tab id from `X-Flux-Tab`), `mcp` (with the MCP server's random session id from `X-Flux-Session` and the call id), or
`ask` (an approved Ask bar card, with question and proposal ids). A retry adds `retried_by`. Ids must match
`^[a-z0-9]{6,24}$` or are dropped. Origin is a routing label only: it grants nothing, and `web` does not prove that a
person acted.

**Retention**: at startup and hourly, info and success events older than 7 days and all events older than 30 days are
deleted, then the newest 50,000 are kept. Deleting a job acknowledges its events.

## API

- `GET /api/events?since=S | before=B &limit=100 (max 500) &min_sev= &job= &session= &wait=0..25` returns
  `{log_id, events, last_seq, unread, attention, read_upto}`. `since` lists ascending after `S`; `before` descending
  below `B`; neither gives the newest `limit`, descending. `wait` (with `since` only) holds the request until an event
  arrives or the wait ends. At most 4 waits per session (`X-Flux-Session`, else the client address) and 64 in all;
  beyond that the answer is 429 with `Retry-After: 5`.
- `POST /api/events/read {upto}` moves the read mark; `unread` counts warn and error events after it.
- `POST /api/events/ack {seqs: [...]}` or `{all: true}` or `{job: J}` acknowledges; `attention` counts unacknowledged
  errors. Both are writes: the token and origin checks apply, and the Ask bar's client is refused.
- WebSocket `/api/ws`: each new event arrives as `{"type": "event", "data": <event>}`. There is no replay on the
  socket; a client catches up with `GET /api/events?since=`.

Event JSON: `{seq, t, kind, sev, key, job_id, task_id, origin: {kind, session?, tab?}, f, acked}`. `log_id` changes
only if the database is replaced; a client that sees it change resets its cursor.

## Agents

The MCP server sends `X-Flux-Session` (a random id per MCP server process) on every request, so the jobs an agent
starts carry its session. `submit_job` and `retry_failed` return `notify.command`, a
`python -m fluxpype.plumber.watch --job J` line for Claude Code's Monitor tool; `get_notifications` is the pull
fallback; `status` reports `notifications: {last_seq, attention, unread}`. `watch` identifies itself with
`X-Flux-Client: watch`, which is neither logged as agent activity nor counted. Details and exit codes:
`docs/mcp/TOOLS.md`, "Being told when a job ends".

## Contracts

`tests/notify_line_golden.py`, `tests/notify_transitions_golden.py` and `tests/events_api_golden.py` write golden files
under `tests/golden/` (`--update`); `tests/test_notify_golden.py` names every entry that moved.
`tests/mcp_tools_golden.py` pins the MCP tool names, arguments and annotations of both server copies.
