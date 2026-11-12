# Plumber Ask bar and Cmd+K: scope

Status: phases 1 to 3 built with the Claude CLI (2026-11-11): palette and Cmd+K, `/api/ask` (`ask.py`), read-only tools, and click-to-run action cards. Phase 4 (history across sessions, a page context chip, the evaluation set) is not done.

Spike result that shaped the build: a bare `claude -p` loads every connected MCP server's tools (one trivial question used 234k tokens, about $0.94, 20 s). With `--tools ""`, `--strict-mcp-config` and an MCP config holding only Plumber's own server it is about 7 s and under a cent for a short question, one to two cents with the reference as the cached prefix.

## What it is

One box in the middle of the top bar. Press Cmd+K (Ctrl+K elsewhere) from any page to focus it; Esc closes it. It does three jobs, in this order of cost:

1. **Find and jump** (instant, no model). Fuzzy search over a command registry: pages, jobs and tasks by id or rotation, cache files by name, presets, backends, and actions (pause or resume the queue, open the system monitor, open the agent dialog, toggle theme, new run from a preset).
2. **Ask** (a Sonnet answer). Anything phrased as a question, or the "Ask Claude" row that is always last in the list, goes to a model that knows FLUX and Plumber. "What does the PFSS stage need?", "why would relaxation stall?", "what is a .flux VNEIGHBOR row?", "what can the agent do?".
3. **Act** (the same abilities as the MCP agent). "Queue the quickstart profile for CR2160", "why did the last job fail?". Reads run on their own. Anything that changes state comes back as a card with a Run button; the click is the approval.

## Design

**Registry (client).** `web/src/lib/commands.ts`: an array of `{id, title, keywords, group, run}`. Pages and dialogs register nothing; the file imports what it needs. Matching reuses `lib/fuzzy.ts`. Cache file names come from the existing cache query, jobs from `useJobs`. No new endpoint for step 1.

**Palette UI.** Radix Dialog (already used), a small trigger in the header that shows the Cmd+K hint, collapsing to an icon below the `md` breakpoint. Results grouped Pages, Jobs, Files, Actions, then Ask. Last 20 questions kept in localStorage.

**Ask endpoint.** `POST /api/ask` streams server-sent events. Same token and origin checks as every other write (it spends the user's quota, so it counts as one). One question at a time, a turn cap, a 120 s timeout.

**Prompt.** A short fixed preamble (what Plumber is, answer from the reference, say when you do not know, never invent file-format columns), then the reference documents, then a small live block (current page, active backend, queue counts, profile names). The reference is large (about 25 KB, roughly 7k tokens), so mark it as a cached prefix: after the first question it is cheap and fast.

Reference documents: ship a copy inside the package (`fluxpype/plumber/ask/flux_reference.md` plus a short `plumber_guide.md` covering pages, stage names and what the agent can do, which already lives in `monitor.STAGES` and `agent_log.TOOLS`). The copy in `~/.claude/FLUX-reference.md` is Gilly's personal file and will not exist for anyone else who runs Plumber. Decide which copy is canonical so they do not drift (suggest: the repo copy, with `~/.claude` linking to it).

**Acting.** Model-side tools are the MCP server's tools. `mcp_server.build_server` already lists names, descriptions and JSON schemas, and the tests drive it in process through `mcp.Client`, so the ask path can do the same with no second tool layer. Calls are logged in the same agent record, with a `via: "ask"` field so the Activity tab can tell them apart. Read tools run automatically. Write tools (`submit_job`, `retry_failed`, `set_queue`, `cancel`) are withheld from the model; it proposes them as a fenced `action` block with JSON arguments, the UI renders a confirm card, and the click calls the normal REST endpoint. The model never mutates anything on its own.

## The one real decision: how to call Sonnet

| | Claude Code CLI (`claude -p`) | Anthropic API |
|---|---|---|
| Credentials | the user's existing login, nothing for Plumber to store | an API key Plumber must hold (file in the Plumber home, 0600, never shown or logged) |
| Agent loop | the CLI runs it and talks to the MCP server itself | Plumber writes the loop (tool schema in, tool result out) |
| Streaming | `--output-format stream-json --include-partial-messages` | SDK stream, simplest to control |
| Cost per question | startup of a process, a second or two before the first token | no startup; prompt caching on the reference |
| Dependency | the CLI must be installed and logged in | the `anthropic` package |

Recommendation: the CLI for a first version (no key handling, no loop to write), behind a small `ask_backend` seam so the API can replace it. Flags I have confirmed exist in the installed CLI: `-p`, `--model`, `--output-format`, `--include-partial-messages`, `--append-system-prompt`, `--mcp-config`, `--strict-mcp-config`, `--allowedTools`, `--disallowedTools`, `--no-session-persistence`. I have not run a spike, so streaming shape, whether a system prompt of this size is accepted by `--append-system-prompt` or needs the file form, and how tool permission denials surface are unverified. Budget the spike first.

## Risks

- **Injection.** Log lines, file names and warnings reach the model through read tools. The MCP instructions already say this is data; repeat it in the preamble, and keep writes behind the click.
- **Wrong answers about FLUX.** The reference is only as good as it is. Build a short evaluation set (20 questions with known answers, such as the VNEIGHBOR columns and which end is the positive footpoint) and run it whenever the reference changes.
- **Quota.** Each question spends the user's quota. Show that it is asking, cap turns, and cache the prefix.
- **Offline or logged out.** Step 1 (find and jump) must keep working without a model; Ask shows a plain "not available: <reason>" row.

## Phases

1. Palette, registry and Cmd+K, no model. Useful on its own; about a day.
2. `/api/ask` with the CLI, reference prompt, streamed answer panel, read tools only; spike first.
3. Action cards with confirm, `via: "ask"` in the activity record.
4. History, a context chip for the current page or job, the evaluation set.

## Questions for Gilly

1. CLI first, or an API key from the start?
2. Should Ask ever be able to start a job after a click, or answer and navigate only?
3. Which copy of the FLUX reference is canonical?
