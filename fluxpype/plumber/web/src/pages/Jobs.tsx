import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Ban, ListChecks, RotateCw, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Card, EmptyState, ErrorNote, PageHeader, Skeleton, StatusBadge, Tip, useConfirm, useToast } from "../components/ui";
import { api } from "../lib/api";
import { ago, dateTime, duration } from "../lib/format";
import { useJobs, useNow } from "../lib/hooks";
import type { Job } from "../lib/types";

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

export function JobsPage() {
  const { data, error, refetch } = useJobs();
  const now = useNow(5000);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  return (
    <div className="animate-fade-in">
      <PageHeader title="Jobs" sub="Every run you have started, newest first." />
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
  );
}
