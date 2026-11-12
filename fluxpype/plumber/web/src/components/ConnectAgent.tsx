import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Bot, Check, ChevronRight, Copy } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { ago, clock, dateTime } from "../lib/format";
import { useAgentRecord, useStatus } from "../lib/hooks";
import { useLive } from "../lib/live";
import type { AgentCall, AgentTool } from "../lib/types";
import { Badge, Dialog, EmptyState, ErrorNote, Segmented, Skeleton, useToast } from "./ui";

interface AgentPrompt {
  name: string;
  command: string;
  args: string[];
  home: string;
  demo: boolean;
  fluxpype_importable: boolean | null;
  mcp_installed: boolean | null;
  install_command: string;
  claude_code: string;
  json_config: string;
  prompt: string;
}

async function copyText(text: string, fallback: HTMLTextAreaElement | null): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    if (!fallback) return false;
    fallback.select();
    return document.execCommand("copy");
  }
}

function CopyBlock({ text, label, primary, rows }: { text: string; label: string; primary?: boolean; rows?: number }) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const [done, setDone] = useState(false);
  const toast = useToast();
  const copy = async () => {
    if (await copyText(text, ref.current)) {
      setDone(true);
      setTimeout(() => setDone(false), 1800);
    } else toast("error", "Copy failed; select the text and copy it by hand");
  };
  return (
    <div className="relative">
      <textarea
        ref={ref}
        readOnly
        value={text}
        rows={rows ?? Math.min(18, text.split("\n").length + 1)}
        onFocus={(e) => e.currentTarget.select()}
        className="w-full resize-none rounded-xl border border-line bg-bg/60 px-3 pb-3 pt-11 font-mono text-[11.5px] leading-relaxed text-ink focus:outline-none focus:ring-1 focus:ring-accent/60"
        aria-label={label}
      />
      <button className={cn("btn btn-sm absolute right-2 top-2", primary && "btn-primary")} onClick={copy}>
        {done ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
        {done ? "Copied" : label}
      </button>
    </div>
  );
}

const KIND = { read: { word: "Looks", tone: "neutral" }, write: { word: "Changes things", tone: "accent" }, stop: { word: "Stops work", tone: "fail" } } as const;

/** Every tool the agent has, whether it has used it, and how often. */
function Abilities({ tools }: { tools: AgentTool[] }) {
  const used = tools.filter((t) => t.calls > 0).length;
  return (
    <div className="space-y-3">
      <p className="text-[13px] text-muted">
        The agent has used <span className="num font-medium text-ink">{used}</span> of <span className="num font-medium text-ink">{tools.length}</span> abilities.
      </p>
      <div className="divide-y divide-line/60 rounded-xl border border-line bg-bg/40">
        {tools.map((t) => (
          <div key={t.name} className={cn("flex items-start gap-3 px-4 py-2.5", t.calls === 0 && "opacity-70")}>
            <span className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", t.calls ? "bg-ok" : "bg-faint/50")} aria-label={t.calls ? "used" : "not used yet"} />
            <div className="min-w-0 flex-1">
              <p className="flex flex-wrap items-center gap-2 text-[13px] font-medium">
                <span className="font-mono">{t.name}</span>
                <Badge tone={KIND[t.kind].tone}>{KIND[t.kind].word}</Badge>
              </p>
              <p className="mt-0.5 text-xs text-muted">{t.what}</p>
            </div>
            <div className="shrink-0 text-right text-[11px] text-faint">
              {t.calls ? (
                <>
                  <p className="num text-ink">{t.calls} call{t.calls === 1 ? "" : "s"}</p>
                  {t.errors > 0 && <p className="text-fail">{t.errors} failed</p>}
                  <p>{ago(t.last)}</p>
                </>
              ) : (
                "not used yet"
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Every call the agent has made, newest first. */
function ActivityLog({ calls, total, since }: { calls: AgentCall[]; total: number; since: number | null }) {
  if (!calls.length) return <EmptyState title="No agent calls yet" body="Every command an agent sends through the MCP server is recorded here." />;
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted">
        {total} call{total === 1 ? "" : "s"} since {dateTime(since)}. Kept in the Plumber home folder (agent_log.jsonl), newest 1000.
      </p>
      <div className="overflow-hidden rounded-xl border border-line bg-bg/40">
        <table className="w-full text-[12px]">
          <tbody className="divide-y divide-line/50">
            {calls.map((c) => (
              <tr key={c.call} className="align-top">
                <td className="num whitespace-nowrap px-3 py-1.5 text-faint" title={dateTime(c.ts)}>
                  {clock(c.ts)}
                </td>
                <td className="px-2 py-1.5">
                  <span className="font-mono font-medium">{c.tool || "?"}</span>
                  {c.args && c.args !== "{}" && (
                    <span className="mt-0.5 block max-w-[360px] truncate font-mono text-[11px] text-muted" title={c.args}>
                      {c.args}
                    </span>
                  )}
                </td>
                <td className="whitespace-nowrap px-3 py-1.5 text-right">
                  <Badge tone={c.status >= 400 ? "fail" : "ok"}>{c.status >= 400 ? `error ${c.status}` : "ok"}</Badge>
                  <span className="num mt-0.5 block text-[10.5px] text-faint" title={c.paths.join("\n")}>
                    {c.ms} ms{c.requests > 1 ? ` · ${c.requests} requests` : ""}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

type Tab = "can" | "log" | "connect";

export function ConnectAgentDialog({ open, onOpenChange, idle, setup }: { open: boolean; onOpenChange: (o: boolean) => void; idle?: string; setup?: boolean }) {
  const [tab, setTab] = useState<Tab>("connect");
  useEffect(() => {
    if (open) setTab(setup ? "can" : "connect"); // an agent that has been here before gets its record first
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);
  const record = useAgentRecord(open);
  const { data, error, refetch } = useQuery({
    queryKey: ["agent-prompt"],
    queryFn: () => api.get<AgentPrompt>("/api/agent-prompt"),
    enabled: open,
    staleTime: 30_000,
  });
  const [more, setMore] = useState(false);
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      wide
      title="AI agent"
      description="An MCP-capable agent (Claude Code, Claude Desktop and others) works in this same queue: it can queue runs, watch progress, read logs and look at images."
    >
      <div className="mb-4">
        <Segmented<Tab>
          value={tab}
          onChange={setTab}
          label="Agent view"
          options={[
            { value: "can", label: "What it can do" },
            { value: "log", label: `Activity${record.data?.total ? ` (${record.data.total})` : ""}` },
            { value: "connect", label: "Connect" },
          ]}
        />
      </div>
      {tab !== "connect" &&
        (record.error ? (
          <ErrorNote error={record.error} retry={() => record.refetch()} />
        ) : !record.data ? (
          <Skeleton className="h-72" />
        ) : tab === "can" ? (
          <Abilities tools={record.data.tools} />
        ) : (
          <ActivityLog calls={record.data.calls} total={record.data.total} since={record.data.since} />
        ))}
      {tab === "connect" && (error ? (
        <ErrorNote error={error} retry={() => refetch()} />
      ) : !data ? (
        <Skeleton className="h-72" />
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
            <Badge tone="accent">{data.name}</Badge>
            {data.demo && <Badge>demo: simulated runs</Badge>}
            {data.mcp_installed === true && <Badge tone="ok">MCP SDK installed</Badge>}
            {data.mcp_installed === false && <Badge tone="fail">MCP SDK missing: the prompt includes the install step</Badge>}
            {data.fluxpype_importable === false && <Badge tone="fail">fluxpype not installed for this Python</Badge>}
          </div>
          {data.fluxpype_importable === false && (
            <p className="flex gap-2 rounded-xl border border-fail/30 bg-fail/10 p-3 text-xs leading-relaxed">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-fail" />
              <span>
                <span className="font-mono">{data.command}</span> cannot import fluxpype outside the repository, so an agent could not start the server. Run{" "}
                <span className="font-mono">pip install -e .</span> in the fluxpype checkout with that Python first.
              </span>
            </p>
          )}
          {idle && (
            <div className="space-y-2">
              <p className="text-[13px] leading-relaxed">
                An agent has been set up here (last seen {idle}) but is not connected right now. Plumber cannot wake it: open or restart Claude Code and it reconnects by itself, or paste this into a session that has it:
              </p>
              <CopyBlock text={`Call the ${data.name} status tool.`} label="Copy" primary rows={2} />
            </div>
          )}
          {(!idle || more) && (
            <>
              <CopyBlock text={data.prompt} label="Copy prompt" primary={!idle} rows={14} />
              <p className="text-xs leading-relaxed text-muted">
                The prompt holds no secrets: the server reads its token from <span className="font-mono">{data.home}</span>. The agent is asked to set things up and check the connection only, not to start jobs.
              </p>
            </>
          )}
          <button className="flex items-center gap-1 text-xs font-medium text-muted hover:text-ink" onClick={() => setMore((m) => !m)}>
            <ChevronRight className={cn("h-3.5 w-3.5 transition-transform", more && "rotate-90")} />
            {idle ? "Set it up again instead" : "Register it yourself instead"}
          </button>
          {more && (
            <div className="space-y-3">
              {data.mcp_installed !== true && (
                <div className="space-y-1.5">
                  <p className="text-xs font-medium">Install the MCP SDK</p>
                  <CopyBlock text={data.install_command} label="Copy" rows={2} />
                </div>
              )}
              <div className="space-y-1.5">
                <p className="text-xs font-medium">Claude Code (any directory)</p>
                <CopyBlock text={data.claude_code} label="Copy" rows={3} />
              </div>
              <div className="space-y-1.5">
                <p className="text-xs font-medium">Claude Desktop, Cursor and other JSON configs</p>
                <CopyBlock text={data.json_config} label="Copy" />
              </div>
            </div>
          )}
        </div>
      ))}
    </Dialog>
  );
}

/** An agent counts as connected while its last MCP request is this recent (the MCP link has no heartbeat). */
const AGENT_FRESH_MS = 10 * 60_000;
/** ...or while the MCP server's heartbeat (one a minute) is this recent. */
const BEAT_FRESH_MS = 150_000;

export function ConnectAgentButton({
  className,
  label = "Connect agent",
  live,
}: {
  className?: string;
  label?: string;
  /** show "Agent connected" and flash on agent activity (the top bar); otherwise a plain button */
  live?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const { agent } = useLive();
  const { data: status, dataUpdatedAt } = useStatus(live ? 30_000 : undefined); // the heartbeat only shows up in /api/status
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!live) return;
    const id = window.setInterval(() => setNow(Date.now()), 15_000);
    return () => window.clearInterval(id);
  }, [live]);

  // last request time: a live push, or what /api/status said when it was fetched
  const fromStatus = status?.agent ? dataUpdatedAt - status.agent.age_s * 1000 : 0;
  const seen = Math.max(agent?.ts ?? 0, fromStatus);
  const link = status?.agent_link;
  const beat = link?.alive != null ? dataUpdatedAt - link.alive * 1000 : 0; // the MCP server pings every minute
  const connected = !!live && (beat > 0 && now - beat < BEAT_FRESH_MS || (seen > 0 && Math.max(now, agent?.ts ?? 0) - seen < AGENT_FRESH_MS));
  // registered and seen before, but silent now: the button says so instead of offering the setup prompt again
  const lastSeen = link?.seen != null ? dataUpdatedAt - link.seen * 1000 : 0;
  const idle = !!live && !connected && lastSeen > 0;
  const idleAgo = idle ? ago(lastSeen / 1000, now / 1000) : undefined;
  const last = agent?.tool || status?.agent?.tool;

  return (
    <>
      <button
        className={cn("btn btn-sm relative", connected && "border-ok/50 text-ok", className)}
        onClick={() => setOpen(true)}
        title={connected ? `AI agent connected${last ? `; last call: ${last}` : ""}` : idle ? `AI agent set up, not connected; last seen ${idleAgo}` : undefined}
      >
        {connected && agent && <span key={agent.ts} className="pointer-events-none absolute inset-0 rounded-[inherit] animate-agent-flash" aria-hidden />}
        <Bot className="h-3.5 w-3.5" />
        <span className="hidden sm:inline">{connected ? "Agent connected" : idle ? "Agent idle" : label}</span>
        {connected && <span className="sr-only">: {last}</span>}
      </button>
      <ConnectAgentDialog open={open} onOpenChange={setOpen} idle={idleAgo} setup={connected || idle} />
    </>
  );
}
