import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Ban, ListChecks, PauseCircle, Plus, RotateCw, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import { ActivityList, OriginTag } from "../components/Notifications";
import { StageStrip, toneOf } from "../components/StageStrip";
import { Badge, Card, EmptyState, ErrorNote, PageHeader, SectionTitle, Skeleton, StatusBadge, Tip, useConfirm, useToast } from "../components/ui";
import { api } from "../lib/api";
import { stalledTasks } from "../lib/events";
import { ago, dateTime, duration, instOf } from "../lib/format";
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

function Segments({ job }: { job: Job }) {
  const total = Math.max(1, job.tasks.length);
  const seg = (n: number, cls: string) => (n ? <div className={cls} style={{ width: `${(n / total) * 100}%` }} /> : null);
  const failed = (job.counts.failed || 0) + (job.counts.timeout || 0) + (job.counts.crashed || 0);
  return (
    <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-raised">
      {seg(job.counts.ok || 0, "bg-ok")}
      {seg(job.counts.running || 0, "bg-run animate-pulse")}
      {seg(failed, "bg-fail")}
      {seg(job.counts.cancelled || 0, "bg-skip/60")}
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
              <Link key={t.id} to={`/jobs/${t.job_id}/${t.id}`} className="block px-4 py-3 hover:bg-raised/50">
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
              <Link key={t.id} to={`/jobs/${t.job_id}`} className="flex items-center justify-between gap-3 px-4 py-2.5 text-[13px] hover:bg-raised/50">
                <span className="flex min-w-0 items-center gap-2">
                  <span className="num w-5 text-right text-xs text-faint">{i + 1}</span>
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
      <SectionTitle title="Recent" />
      {!data ? (
        <div className="space-y-2">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-20" />)}</div>
      ) : data.jobs.length === 0 ? (
        <EmptyState icon={<ListChecks className="h-5 w-5" />} title="No jobs yet" body="Start a run from a preset and it will be tracked here." action={<Link to="/presets" className="btn btn-primary">Choose a preset</Link>} />
      ) : (
        <div className="space-y-2.5">
          {data.jobs.map((j) => {
            const end = j.ended ?? now;
            return (
              <Link key={j.id} to={`/jobs/${j.id}`} className="block focus:outline-none">
                <Card className="p-4 transition-colors hover:border-faint/50">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="min-w-0">
                      <p className="flex items-center gap-2.5 text-[14px] font-semibold">
                        {j.profile}
                        <StatusBadge status={j.status} />
                        <OriginTag origin={j.origin} />
                      </p>
                      <p className="mt-1 text-xs text-muted">
                        {j.tasks.length} task{j.tasks.length === 1 ? "" : "s"} · {dateTime(j.created)} ({ago(j.created, now)}) · {j.started ? duration(end - j.started) : "not started"}
                        {Object.keys(j.overrides).length > 0 && ` · ${Object.keys(j.overrides).length} override${Object.keys(j.overrides).length === 1 ? "" : "s"}`}
                      </p>
                    </div>
                    <JobActions job={j} />
                  </div>
                  <div className="mt-3">
                    <Segments job={j} />
                  </div>
                </Card>
              </Link>
            );
          })}
        </div>
      )}
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
