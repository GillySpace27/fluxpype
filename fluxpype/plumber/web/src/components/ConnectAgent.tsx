import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Bot, Check, ChevronRight, Copy } from "lucide-react";
import { useRef, useState } from "react";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { Badge, Dialog, ErrorNote, Skeleton, useToast } from "./ui";

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

export function ConnectAgentDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
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
      title="Connect an AI agent"
      description="Paste this prompt into Claude Code, Claude Desktop or another MCP-capable agent. It registers Plumber's MCP server so the agent can queue runs, watch progress, read logs and look at images in this same queue."
    >
      {error ? (
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
          <CopyBlock text={data.prompt} label="Copy prompt" primary rows={14} />
          <p className="text-xs leading-relaxed text-muted">
            The prompt holds no secrets: the server reads its token from <span className="font-mono">{data.home}</span>. The agent is asked to set things up and check the connection only, not to start jobs.
          </p>
          <button className="flex items-center gap-1 text-xs font-medium text-muted hover:text-ink" onClick={() => setMore((m) => !m)}>
            <ChevronRight className={cn("h-3.5 w-3.5 transition-transform", more && "rotate-90")} />
            Register it yourself instead
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
      )}
    </Dialog>
  );
}

export function ConnectAgentButton({ className, label = "Connect agent" }: { className?: string; label?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button className={cn("btn btn-sm", className)} onClick={() => setOpen(true)}>
        <Bot className="h-3.5 w-3.5" />
        <span className="hidden sm:inline">{label}</span>
      </button>
      <ConnectAgentDialog open={open} onOpenChange={setOpen} />
    </>
  );
}
