import * as RMenu from "@radix-ui/react-dropdown-menu";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, ChevronDown, Cloud, Cpu, FlaskConical, RefreshCw, Server } from "lucide-react";
import { useState } from "react";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { ago } from "../lib/format";
import { useHealth, useQueue } from "../lib/hooks";
import { useLive } from "../lib/live";
import type { BackendHealth, Health, HealthCheck, HealthReport } from "../lib/types";
import { Badge, Dialog, Skeleton, Tip, useToast } from "./ui";

const DOT: Record<Health, string> = {
  ok: "bg-ok",
  warn: "bg-run",
  fail: "bg-fail",
  off: "bg-faint/50",
  unknown: "bg-faint",
};
const WORD: Record<Health, string> = { ok: "Ready", warn: "Needs attention", fail: "Not working", off: "Not set up", unknown: "Checking" };

export function Dot({ status, pulse, className }: { status: Health; pulse?: boolean; className?: string }) {
  return <span className={cn("inline-block h-2 w-2 shrink-0 rounded-full", DOT[status], pulse && "animate-pulse", className)} aria-hidden />;
}

function BackendIcon({ kind, className }: { kind: string; className?: string }) {
  const Icon = kind === "fake" ? FlaskConical : kind === "slurm" ? Cloud : kind === "ssh" ? Server : Cpu;
  return <Icon className={className} />;
}

export function useSetBackend() {
  const qc = useQueryClient();
  const toast = useToast();
  return useMutation({
    mutationFn: (name: string) => api.put<{ active_backend: string }>("/api/backend", { name }),
    onSuccess: (r) => {
      qc.invalidateQueries();
      toast("ok", r.active_backend === "fake" ? "Now plumbing simulated runs (sandboxed)" : "Backend switched");
    },
    onError: (e) => toast("error", e instanceof Error ? e.message : String(e)),
  });
}

/** The main switch: where new runs go. Lives at the left of the top bar. */
export function BackendSwitcher() {
  const { data } = useHealth();
  const { data: q } = useQueue();
  const set = useSetBackend();
  if (!data) return <Skeleton className="h-9 w-56" />;
  const active = data.backends.find((b) => b.active);
  const busy = (q?.running ?? 0) > 0;
  return (
    <RMenu.Root>
      <RMenu.Trigger asChild>
        <button
          className={cn(
            "group flex h-9 min-w-0 items-center gap-2.5 rounded-xl border px-3 text-left transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-accent/60",
            active?.sandboxed ? "border-accent/40 bg-accent/10 hover:bg-accent/15" : "border-line bg-raised/70 hover:bg-raised",
          )}
          aria-label="Choose where runs go"
        >
          <span className="hidden text-[11px] uppercase tracking-[0.12em] text-faint sm:inline">Plumbing</span>
          {active && <BackendIcon kind={active.kind} className={cn("h-4 w-4 shrink-0", active.sandboxed ? "text-accent" : "text-ink/80")} />}
          <span className="truncate text-[13px] font-semibold">{active?.label ?? "?"}</span>
          {active && <Dot status={active.status} pulse={busy && active.status === "ok"} />}
          <ChevronDown className="h-3.5 w-3.5 text-faint group-hover:text-muted" />
        </button>
      </RMenu.Trigger>
      <RMenu.Portal>
        <RMenu.Content align="start" sideOffset={6} className="z-50 w-[340px] animate-fade-in rounded-xl border border-line bg-surface p-1.5 shadow-pop">
          <p className="px-2.5 pb-1.5 pt-1 text-[11px] uppercase tracking-[0.12em] text-faint">New runs go to</p>
          {data.backends.map((b) => (
            <RMenu.Item
              key={b.name}
              disabled={!b.configured}
              onSelect={() => b.configured && !b.active && set.mutate(b.name)}
              className={cn(
                "flex cursor-pointer items-start gap-3 rounded-lg px-2.5 py-2 outline-none data-[highlighted]:bg-raised data-[disabled]:cursor-default data-[disabled]:opacity-55",
              )}
            >
              <BackendIcon kind={b.kind} className={cn("mt-0.5 h-4 w-4 shrink-0", b.sandboxed ? "text-accent" : "text-muted")} />
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-2 text-[13px] font-medium">
                  {b.label}
                  {b.sandboxed && <Badge tone="accent">sandbox</Badge>}
                </span>
                <span className="mt-0.5 block text-xs leading-snug text-muted">{b.configured ? b.description : "Not set up yet"}</span>
                <span className="mt-1 flex items-center gap-1.5 text-[11px] text-faint">
                  <Dot status={b.status} /> {WORD[b.status]}
                  {b.configured && b.slots > 0 && <span className="num">· {b.slots} CPU slots</span>}
                </span>
              </span>
              {b.active && <Check className="mt-0.5 h-4 w-4 text-accent" />}
            </RMenu.Item>
          ))}
          <p className="px-2.5 pb-1 pt-2 text-[11px] leading-snug text-faint">Running jobs stay on the backend they started on.</p>
        </RMenu.Content>
      </RMenu.Portal>
    </RMenu.Root>
  );
}

function CheckRow({ c }: { c: HealthCheck }) {
  return (
    <div className="flex items-start gap-3 px-4 py-2.5">
      <Dot status={c.status} className="mt-1.5" />
      <div className="min-w-0 flex-1">
        <p className="flex items-center justify-between gap-2 text-[13px] font-medium">
          {c.label}
          <span className={cn("text-[11px] font-normal", c.status === "fail" ? "text-fail" : c.status === "warn" ? "text-run" : "text-faint")}>{WORD[c.status]}</span>
        </p>
        {c.detail && <p className="mt-0.5 break-words font-mono text-[11px] leading-relaxed text-muted">{c.detail}</p>}
      </div>
    </div>
  );
}

function Section({ title, right, children }: { title: React.ReactNode; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section>
      <div className="mb-1.5 flex items-center justify-between px-1">
        <p className="text-[11px] uppercase tracking-[0.12em] text-faint">{title}</p>
        {right}
      </div>
      <div className="divide-y divide-line/60 rounded-xl border border-line bg-bg/40">{children}</div>
    </section>
  );
}

function BackendCard({ b }: { b: BackendHealth }) {
  const set = useSetBackend();
  return (
    <Section
      title={
        <span className="flex items-center gap-2 normal-case tracking-normal">
          <BackendIcon kind={b.kind} className="h-3.5 w-3.5" />
          <span className="text-[12px] font-semibold text-ink">{b.label}</span>
          {b.active && <Badge tone="accent">plumbing</Badge>}
          {b.sandboxed && <Badge>sandbox</Badge>}
        </span>
      }
      right={
        b.configured && !b.active ? (
          <button className="btn btn-sm" onClick={() => set.mutate(b.name)}>
            Use this
          </button>
        ) : undefined
      }
    >
      {b.checks.length === 0 ? <p className="px-4 py-2.5 text-xs text-muted">Checking...</p> : b.checks.map((c) => <CheckRow key={c.id} c={c} />)}
      {b.data_dir && (
        <div className="px-4 py-2 text-[11px] text-faint">
          Data folder: <span className="break-all font-mono">{b.data_dir}</span>
        </div>
      )}
    </Section>
  );
}

function StatusPanel({ data, open, onOpenChange }: { data: HealthReport; open: boolean; onOpenChange: (o: boolean) => void }) {
  const { connected } = useLive();
  const qc = useQueryClient();
  const [busy, setBusy] = useState(false);
  const recheck = async () => {
    setBusy(true);
    try {
      qc.setQueryData(["health"], await api.get<HealthReport>("/api/health?refresh=1"));
    } finally {
      setBusy(false);
    }
  };
  const live: HealthCheck = {
    id: "live",
    label: "Live updates",
    status: connected ? "ok" : "warn",
    detail: connected ? "connected; screens update as things happen" : "reconnecting; screens fall back to polling",
  };
  const configured = data.backends.filter((b) => b.configured);
  const others = data.backends.filter((b) => !b.configured);
  return (
    <Dialog
      side
      open={open}
      onOpenChange={onOpenChange}
      title="System status"
      description={`Checked ${data.checked_at ? ago(data.checked_at) : "just now"}. Backend checks run the job Python and Perl the way a job would.`}
      footer={
        <button className="btn" onClick={recheck} disabled={busy}>
          <RefreshCw className={cn("h-3.5 w-3.5", busy && "animate-spin")} /> {busy ? "Checking..." : "Check again"}
        </button>
      }
    >
      <div className="space-y-5">
        {configured.map((b) => (
          <BackendCard key={b.name} b={b} />
        ))}
        <Section title="Plumber">
          {[data.system[0], live, ...data.system.slice(1)].map((c) => (
            <CheckRow key={c.id} c={c} />
          ))}
        </Section>
        <Section title="Not set up yet">
          {others.map((b) => (
            <div key={b.name} className="flex items-center gap-3 px-4 py-2.5 text-[13px] text-muted">
              <BackendIcon kind={b.kind} className="h-4 w-4" /> {b.label}
              <span className="ml-auto text-[11px] text-faint">{b.description}</span>
            </div>
          ))}
        </Section>
      </div>
    </Dialog>
  );
}

/** One light for the whole system, at the right of the top bar; opens the status panel. */
export function StatusLight() {
  const { data } = useHealth();
  const { connected } = useLive();
  const [open, setOpen] = useState(false);
  if (!data) return null;
  const overall: Health = !connected && data.overall === "ok" ? "warn" : data.overall;
  const problems = [...data.system, ...(data.backends.find((b) => b.active)?.checks ?? [])].filter((c) => c.status === "fail" || c.status === "warn");
  return (
    <>
      <Tip label={problems.length ? problems.map((p) => p.label).join(", ") : "All systems ready"}>
        <button className="flex h-8 items-center gap-2 rounded-full border border-line bg-raised/70 px-3 text-xs hover:bg-raised" onClick={() => setOpen(true)}>
          <Dot status={overall} />
          <span className="hidden text-muted sm:inline">{overall === "ok" ? "All systems ready" : problems.length ? `${problems.length} issue${problems.length === 1 ? "" : "s"}` : WORD[overall]}</span>
        </button>
      </Tip>
      {open && <StatusPanel data={data} open={open} onOpenChange={setOpen} />}
    </>
  );
}
