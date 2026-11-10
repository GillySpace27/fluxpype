import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Cloud, Cpu, FlaskConical, Monitor, Moon, Server, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { Badge, Card, ErrorNote, PageHeader, Segmented, SectionTitle, Skeleton, Switch, useToast } from "../components/ui";
import { ConnectAgentButton } from "../components/ConnectAgent";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { useHealth, useQueue, useStatus } from "../lib/hooks";
import { Dot, useSetBackend } from "../components/SystemBar";
import { useTheme } from "../lib/theme";

function Row({ title, hint, children }: { title: string; hint?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3.5">
      <div className="min-w-0">
        <p className="text-[13px] font-medium">{title}</p>
        {hint && <p className="mt-0.5 max-w-md text-xs leading-relaxed text-muted">{hint}</p>}
      </div>
      {children}
    </div>
  );
}

function Stepper({ value, min, max, onCommit }: { value: number; min: number; max: number; onCommit: (v: number) => void }) {
  const [v, setV] = useState(String(value));
  useEffect(() => setV(String(value)), [value]);
  const commit = (n: number) => onCommit(Math.min(max, Math.max(min, Math.round(n) || min)));
  return (
    <div className="flex items-center gap-1">
      <button className="btn btn-icon" aria-label="Decrease" onClick={() => commit(value - 1)} disabled={value <= min}>-</button>
      <input className="input num w-14 text-center" value={v} inputMode="numeric" onChange={(e) => setV(e.target.value)} onBlur={() => commit(Number(v))} onKeyDown={(e) => e.key === "Enter" && commit(Number(v))} aria-label="Value" />
      <button className="btn btn-icon" aria-label="Increase" onClick={() => commit(value + 1)} disabled={value >= max}>+</button>
    </div>
  );
}

export function SettingsPage() {
  const { data: q, error, refetch } = useQueue();
  const { data: status } = useStatus();
  const { data: health } = useHealth();
  const setBackend = useSetBackend();
  const { choice, setChoice } = useTheme();
  const qc = useQueryClient();
  const toast = useToast();
  const patch = useMutation({
    mutationFn: (b: Record<string, unknown>) => api.patch("/api/queue", b),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["queue"] }),
    onError: (e) => toast("error", e instanceof Error ? e.message : String(e)),
  });
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!q || !status) return <Skeleton className="h-96" />;
  const paths: [string, string][] = [
    ["Plumber home", status.plumber_home],
    ["Config file", status.config_path],
    ["Data directory", status.data_dir],
  ];
  return (
    <div className="max-w-3xl animate-fade-in space-y-8">
      <PageHeader title="Settings" sub="Queue limits take effect immediately. Paths are read-only here; change them in the config file." />

      <section>
        <SectionTitle title="Queue" />
        <Card className="divide-y divide-line/60">
          <Row title="Run in parallel" hint="How many tasks may run at once. Each task also needs its CPU slots, so this is an upper bound.">
            <Stepper value={q.max_parallel} min={1} max={64} onCommit={(v) => patch.mutate({ max_parallel: v })} />
          </Row>
          <Row title="CPU budget" hint="Total CPU slots Plumber may hand out across running tasks. Defaults to leaving one core free.">
            <Stepper value={q.cpu_budget} min={1} max={512} onCommit={(v) => patch.mutate({ cpu_budget: v })} />
          </Row>
          <Row title="Pause queue" hint="Running tasks continue; nothing new starts until you resume.">
            <Switch checked={q.paused} onChange={(v) => patch.mutate({ paused: v })} label="Pause queue" />
          </Row>
        </Card>
      </section>

      <section>
        <SectionTitle title="Appearance" />
        <Card>
          <Row title="Theme">
            <Segmented
              value={choice}
              onChange={setChoice}
              options={[
                { value: "dark", label: <span className="flex items-center gap-1.5"><Moon className="h-3.5 w-3.5" />Dark</span> },
                { value: "light", label: <span className="flex items-center gap-1.5"><Sun className="h-3.5 w-3.5" />Light</span> },
                { value: "system", label: <span className="flex items-center gap-1.5"><Monitor className="h-3.5 w-3.5" />System</span> },
              ]}
            />
          </Row>
        </Card>
      </section>

      <section>
        <SectionTitle title="AI agents" />
        <Card>
          <Row title="Connect an agent" hint="Get a prompt to paste into Claude Code, Claude Desktop or another MCP agent. It registers Plumber's MCP server so the agent can queue runs, watch progress and read logs in this same queue.">
            <ConnectAgentButton label="Get prompt" />
          </Row>
        </Card>
      </section>

      <section>
        <SectionTitle title="Where things live" />
        <Card className="divide-y divide-line/60">
          {paths.map(([k, v]) => (
            <div key={k} className="grid gap-1 px-4 py-3 sm:grid-cols-[140px_minmax(0,1fr)]">
              <p className="text-[13px] font-medium">{k}</p>
              <p className="break-all font-mono text-xs text-muted">{v}</p>
            </div>
          ))}
        </Card>
      </section>

      <section>
        <SectionTitle title="Execution backends" hint="Where new runs go. The choice is also at the top left of every page." />
        <div className="grid gap-3 sm:grid-cols-2">
          {(health?.backends ?? []).map((b) => (
            <Card key={b.name} className={cn("p-4", !b.configured && "border-dashed opacity-70", b.active && "border-accent/50")}>
              <div className="flex items-center justify-between gap-2">
                <p className="flex items-center gap-2 text-[13.5px] font-semibold">
                  {b.kind === "fake" ? <FlaskConical className="h-4 w-4 text-accent" /> : b.kind === "slurm" ? <Cloud className="h-4 w-4" /> : b.kind === "ssh" ? <Server className="h-4 w-4" /> : <Cpu className="h-4 w-4 text-accent" />}
                  {b.label}
                </p>
                {b.active ? <Badge tone="accent">plumbing</Badge> : b.configured ? <button className="btn btn-sm" onClick={() => setBackend.mutate(b.name)}>Use this</button> : <Badge>coming next</Badge>}
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted">{b.description}</p>
              <p className="mt-2 flex items-center gap-1.5 text-xs text-faint">
                <Dot status={b.status} /> {b.checks.find((c) => c.status === "fail")?.detail ?? (b.configured ? `${b.slots} CPU slots` : "not set up")}
              </p>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
