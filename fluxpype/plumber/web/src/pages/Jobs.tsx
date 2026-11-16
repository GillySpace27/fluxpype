import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Ban, ListChecks, PauseCircle, Plus, RotateCw, Search, Trash2, X } from "lucide-react";
import { Link } from "react-router-dom";
import { ActivityList, OriginTag } from "../components/Notifications";
import { StageStrip, toneOf } from "../components/StageStrip";
import { Cover, coverUrl } from "../components/Cover";
import { jobSummary, RunCard } from "../components/RunCard";
import { cn } from "../lib/cn";
import { Badge, Card, EmptyState, ErrorNote, PageHeader, SectionTitle, Skeleton, StatusBadge, Tip, useConfirm, useToast } from "../components/ui";
import { api } from "../lib/api";
import { stalledTasks } from "../lib/events";
import { duration, instOf } from "../lib/format";
import { useJobs, useNow, useOverview } from "../lib/hooks";
import { useEvents } from "../lib/live";
import type { Job } from "../lib/types";
import { waitLabel } from "./Overview";

export function JobActions({ job, onDeleted }: { job: Job; onDeleted?: () => void }) {
  const qc = useQueryClient();
  const toast = useToast();
  const { ask, node } = useConfirm();
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["jobs"] });
    qc.invalidateQueries({ queryKey: ["job", job.id] });
    qc.invalidateQueries({ queryKey: ["overview"] });
  };
  const fail = (e: unknown) => toast("error", e instanceof Error ? e.message : String(e));
  const cancel = useMutation({ mutationFn: () => api.post(`/api/jobs/${job.id}/cancel`), onSuccess: () => { toast("info", "Cancelling..."); refresh(); }, onError: fail });
  const retry = useMutation({ mutationFn: () => api.post(`/api/jobs/${job.id}/retry`), onSuccess: () => { toast("ok", "Retrying failed tasks"); refresh(); }, onError: fail });
  const del = useMutation({ mutationFn: () => api.del(`/api/jobs/${job.id}`), onSuccess: () => { toast("ok", "Job removed from history"); refresh(); onDeleted?.(); }, onError: fail });
  const active = job.status === "running" || job.status === "pending";
  const retryable = !active && (job.counts.failed || job.counts.cancelled || job.counts.timeout || job.counts.crashed);
  return (
    <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
      {active && (
        <button
          className="btn btn-sm btn-danger"
          onClick={async () => {
            if (await ask({ title: "Cancel this job?", body: "Running tasks are stopped (SIGTERM, then SIGKILL after 10 s) and queued tasks are dropped. Files already written stay on disk.", confirm: "Cancel job", danger: true })) cancel.mutate();
          }}
        >
          <Ban className="h-3.5 w-3.5" /> Cancel
        </button>
      )}
      {retryable ? (
        <button className="btn btn-sm" onClick={() => retry.mutate()} disabled={retry.isPending}>
          <RotateCw className="h-3.5 w-3.5" /> Retry failed
        </button>
      ) : null}
      {!active && (
        <Tip label="Remove from history (files on disk are kept)">
          <button
            className="btn btn-sm btn-ghost btn-icon"
            aria-label="Delete job"
            onClick={async () => {
              if (await ask({ title: "Remove this job from history?", body: "Only the record is removed. Output files stay on disk and can be managed from the Cache page.", confirm: "Remove", danger: true })) del.mutate();
            }}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </Tip>
      )}
      {node}
    </div>
  );
}

/** Running and queued tasks, live: stage strip, elapsed time, stall badge; queue order with why each waits. */
function LiveBands({ origins }: { origins: Map<string, Job["origin"]> }) {
  const { data } = useOverview();
  const ev = useEvents();
  const now = useNow(1000);
  if (!data) return <Skeleton className="h-24" />;
  const stalled = stalledTasks(ev.events);
  return (
    <>
      <section>
        <SectionTitle title="Running" hint={data.running.length ? undefined : "Nothing is running."} />
        {data.running.length > 0 && (
          <Card className="divide-y divide-line/60">
            {data.running.map((t) => (
              <Link key={t.id} to={`/jobs/${t.job_id}/${t.id}`} className="flex items-center gap-3.5 px-3 py-3 hover:bg-raised/50">
                <Cover src={coverUrl.task(t.id, t.current_stage)} seed={t.id} className="h-14 w-20 shrink-0 rounded-lg border border-line/70" />
                <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center justify-between gap-2 text-[13px]">
                  <span className="flex items-center gap-2 font-medium">
                    CR {t.cr} <span className="font-normal text-muted">· {t.nflux} fluxons · {instOf(t.adapt)} · {t.method} · {t.profile}</span>
                    <OriginTag origin={origins.get(t.job_id)} />
                    {stalled.has(t.id) && <Badge tone="run">No output: stalled?</Badge>}
                  </span>
                  <span className="num text-xs text-muted">
                    {t.current_stage ?? "starting"} · {duration(t.started ? now - t.started : null)}
                  </span>
                </div>
                <StageStrip className="mt-2" labels={false} items={t.stages.map((s) => ({ id: s.id, label: s.id, tone: toneOf(s.status) }))} />
                </div>
              </Link>
            ))}
          </Card>
        )}
      </section>
      <section>
        <SectionTitle
          title="Queued"
          hint={
            <span className="flex items-center gap-1.5">
              {data.queue.paused && <PauseCircle className="h-3.5 w-3.5 text-run" />}
              Queued tasks are the only future runs; Plumber has no scheduled runs yet.
            </span>
          }
        />
        {data.pending.length > 0 && (
          <Card className="divide-y divide-line/60">
            {data.pending.map((t, i) => (
              <Link key={t.id} to={`/jobs/${t.job_id}`} className="flex items-center justify-between gap-3 px-3 py-2 text-[13px] hover:bg-raised/50">
                <span className="flex min-w-0 items-center gap-2.5">
                  <span className="num w-5 text-right text-xs text-faint">{i + 1}</span>
                  <Cover src={coverUrl.task(t.id)} seed={t.id} className="h-8 w-11 shrink-0 rounded-md border border-line/70" />
                  <span className="shrink-0">CR {t.cr}</span>
                  <span className="truncate text-muted">· {t.nflux} fluxons · {t.method} · {t.profile}</span>
                  <OriginTag origin={t.origin} />
                </span>
                <span className="num shrink-0 text-xs text-muted">
                  {waitLabel(t.wait_reason)} · {t.slots} slot{t.slots === 1 ? "" : "s"}
                </span>
              </Link>
            ))}
          </Card>
        )}
      </section>
    </>
  );
}

type ExternalRun = {
  id: string;
  title?: string;
  status?: string;
  host?: string;
  cwd?: string;
  pid?: number;
  pid_alive?: boolean;
  started?: string;
  updated?: string;
  log?: string;
  report?: string;
  summary?: string;
  progress?: { label: string; value: number | string | null }[];
};

function ExternalRunCard({ run }: { run: ExternalRun }) {
  const [open, setOpen] = useState(false);
  const tail = useQuery({
    queryKey: ["external-run", run.id],
    queryFn: () => api.get<{ log_tail: string | null }>(`/api/external-runs/${encodeURIComponent(run.id)}`),
    enabled: open,
    refetchInterval: open ? 5000 : false,
  });
  const gone = run.status === "running" && run.pid_alive === false;
  return (
    <Card className="p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-2.5 text-[14px] font-semibold">
          {run.title || run.id}
          <StatusBadge status={run.status && run.status !== "note" ? run.status : "skipped"} />
          {gone && <Badge tone="fail">process gone</Badge>}
        </p>
        <p className="text-xs text-muted">
          {[run.host, run.pid ? `pid ${run.pid}` : "", run.updated ? `updated ${run.updated.replace("T", " ").slice(0, 19)}` : ""].filter(Boolean).join(" · ")}
        </p>
      </div>
      {run.summary && <p className="mt-2 text-[13px]">{run.summary}</p>}
      {run.progress && run.progress.length > 0 && (
        <p className="mt-2 flex flex-wrap gap-1.5">
          {run.progress.map((r) => (
            <Badge key={r.label}>{r.label}: {String(r.value ?? "-")}</Badge>
          ))}
        </p>
      )}
      <p className="mt-2 break-all text-xs text-muted">{[run.cwd, run.report ? `report ${run.report}` : ""].filter(Boolean).join(" · ")}</p>
      {run.log && (
        <button className="btn btn-sm mt-2" onClick={() => setOpen(!open)}>
          {open ? "Hide log" : "Show end of log"}
        </button>
      )}
      {open && <pre className="mt-2 max-h-64 overflow-auto rounded bg-raised p-2 text-[11.5px]">{tail.data?.log_tail ?? "(no log yet)"}</pre>}
    </Card>
  );
}

/** Runs started outside Plumber (scripts, terminals, other Claudes) that registered themselves; not jobs, no controls. */
function ExternalRuns() {
  const { data } = useQuery({ queryKey: ["external-runs"], queryFn: () => api.get<{ runs: ExternalRun[] }>("/api/external-runs"), refetchInterval: 5000 });
  if (!data || data.runs.length === 0) return null;
  return (
    <div className="space-y-2.5">
      <SectionTitle title="Outside the queue" hint="Started by a script or a terminal and registered here. Plumber only shows them; it cannot cancel or retry them." />
      {data.runs.slice(0, 8).map((r) => (
        <ExternalRunCard key={r.id} run={r} />
      ))}
    </div>
  );
}

type RunFilter = "all" | "running" | "ok" | "failed" | "cancelled";
const RUN_FILTERS: { key: RunFilter; label: string; test: (j: Job) => boolean }[] = [
  { key: "all", label: "All", test: () => true },
  { key: "running", label: "Active", test: (j) => j.status === "running" || j.status === "pending" },
  { key: "ok", label: "Succeeded", test: (j) => j.status === "ok" },
  { key: "failed", label: "Failed", test: (j) => ["failed", "partial", "timeout", "crashed"].includes(j.status) },
  { key: "cancelled", label: "Cancelled", test: (j) => j.status === "cancelled" },
];

function loadRunFilter(): RunFilter {
  try {
    const v = localStorage.getItem("plumber.runFilter") as RunFilter | null;
    return v && RUN_FILTERS.some((f) => f.key === v) ? v : "all";
  } catch {
    return "all";
  }
}

/** Every job as a gallery card with its picture, filterable by status and searchable by preset, rotation or id. */
function RecentRuns({ jobs, now }: { jobs?: Job[]; now: number }) {
  const [filter, setFilterState] = useState<RunFilter>(loadRunFilter);
  const [q, setQ] = useState("");
  const setFilter = (f: RunFilter) => {
    setFilterState(f);
    try {
      localStorage.setItem("plumber.runFilter", f);
    } catch {
      /* per-browser convenience only */
    }
  };
  const visible = useMemo(() => {
    const words = q.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const test = RUN_FILTERS.find((f) => f.key === filter)?.test ?? (() => true);
    return (jobs ?? []).filter((j) => {
      if (!test(j)) return false;
      if (!words.length) return true;
      const hay = `${j.profile} ${j.id} ${jobSummary(j)} ${j.tasks.map((t) => `cr${t.cr} ${t.cr} ${t.nflux} ${t.method}`).join(" ")} ${j.status}`.toLowerCase();
      return words.every((w) => hay.includes(w));
    });
  }, [jobs, filter, q]);
  return (
    <section className="space-y-3">
      <SectionTitle title="All runs" hint={jobs ? `${jobs.length} in history` : undefined} />
      {jobs && jobs.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-[200px] flex-1">
            <Search className="pointer-events-none absolute left-2.5 top-2 h-4 w-4 text-faint" />
            <input className="input pl-8 pr-8" placeholder="Search preset, CR, fluxons, method" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search runs" />
            {q && (
              <button className="absolute right-2 top-2 text-faint hover:text-ink" aria-label="Clear search" onClick={() => setQ("")}>
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Filter runs by status">
            {RUN_FILTERS.map((f) => {
              const n = jobs.filter(f.test).length;
              return (
                <button
                  key={f.key}
                  role="radio"
                  aria-checked={filter === f.key}
                  onClick={() => setFilter(f.key)}
                  className={cn(
                    "rounded-full border px-2.5 py-0.5 text-[11.5px] transition-colors",
                    filter === f.key ? "border-accent/60 bg-accent/15 text-ink" : "border-line text-muted hover:border-faint/60 hover:text-ink",
                  )}
                >
                  {f.label} <span className="num text-faint">{n}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
      {!jobs ? (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="aspect-[4/3]" />)}</div>
      ) : jobs.length === 0 ? (
        <EmptyState icon={<ListChecks className="h-5 w-5" />} title="No jobs yet" body="Start a run from a preset and it will be tracked here." action={<Link to="/presets" className="btn btn-primary">Choose a preset</Link>} />
      ) : visible.length === 0 ? (
        <p className="py-8 text-center text-[13px] text-muted">No runs match.</p>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {visible.map((j) => (
            <RunCard key={j.id} job={j} now={now} actions={<JobActions job={j} />} className="animate-rise" />
          ))}
        </div>
      )}
    </section>
  );
}

export function JobsPage() {
  const { data, error, refetch } = useJobs();
  const now = useNow(5000);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  const origins = new Map((data?.jobs ?? []).map((j) => [j.id, j.origin]));
  return (
    <div className="animate-fade-in">
      <PageHeader
        title="Runs"
        sub="What is running, what waits and why, and every job you have started."
        right={
          <Link to="/jobs/new" className="btn btn-primary">
            <Plus className="h-3.5 w-3.5" /> New run
          </Link>
        }
      />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="min-w-0 space-y-6">
      <LiveBands origins={origins} />
      <ExternalRuns />
      <RecentRuns jobs={data?.jobs} now={now} />
      </div>
      <aside className="min-w-0">
        <SectionTitle title="Activity" hint="Live job events" />
        <Card className="p-1">
          <ActivityList limit={30} />
        </Card>
      </aside>
      </div>
    </div>
  );
}
