import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Cloud, Cpu, Monitor, Moon, Server, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { Badge, Card, ErrorNote, PageHeader, Segmented, SectionTitle, Skeleton, Switch, useToast } from "../components/ui";
import { api } from "../lib/api";
import { useQueue, useStatus } from "../lib/hooks";
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
        <SectionTitle title="Execution backends" hint="Where tasks run." />
        <div className="grid gap-3 sm:grid-cols-2">
          {status.backends.map((b) => (
            <Card key={b.name} className="p-4">
              <div className="flex items-center justify-between">
                <p className="flex items-center gap-2 text-[13.5px] font-semibold"><Cpu className="h-4 w-4 text-accent" />{b.name}</p>
                {b.active && <Badge tone="ok">default</Badge>}
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted">{b.description}</p>
              <p className="num mt-2 text-xs text-faint">{b.slots} slots{b.shared_fs ? " · shared filesystem" : ""}</p>
            </Card>
          ))}
          {[{ n: "SSH host", i: <Server className="h-4 w-4" />, d: "Run on a workstation you can reach over SSH." }, { n: "Slurm cluster", i: <Cloud className="h-4 w-4" />, d: "Submit tasks as batch jobs on an HPC queue." }].map((b) => (
            <Card key={b.n} className="border-dashed p-4 opacity-70">
              <div className="flex items-center justify-between">
                <p className="flex items-center gap-2 text-[13.5px] font-semibold text-muted">{b.i}{b.n}</p>
                <Badge>coming next</Badge>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted">{b.d}</p>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
