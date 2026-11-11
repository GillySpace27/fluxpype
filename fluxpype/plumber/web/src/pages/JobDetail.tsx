import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Ban, PanelsTopLeft } from "lucide-react";
import { useEffect, useMemo } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { LogPane } from "../components/LogPane";
import { PipelineChips } from "../components/PipelineChips";
import { ResultsPane } from "../components/ResultsPane";
import { StagePanel } from "../components/StagePanel";
import { RelaxCharts } from "../components/RelaxCharts";
import { SourcesPanel } from "../components/SourcesPanel";
import { StageTimeline } from "../components/StageTimeline";
import { Banner, Card, EmptyState, ErrorNote, SectionTitle, Skeleton, StatusBadge, Tab, TabList, TabPanel, Tabs, useConfirm, useToast } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { dateTime, duration, instOf, int, num } from "../lib/format";
import { useJob, useNow, useRelax, useTask } from "../lib/hooks";
import type { Stage, TaskRow, TaskState } from "../lib/types";
import { JobActions } from "./Jobs";

function Metric({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div className="min-w-0 rounded-lg border border-line/70 bg-bg/40 px-3 py-2">
      <p className="label truncate">{label}</p>
      <p className="num mt-0.5 truncate text-[15px] font-semibold tracking-tight">{value}</p>
      {sub && <p className="truncate text-[11px] text-faint">{sub}</p>}
    </div>
  );
}

/** One task: header with metrics and banners, then tabs. `results` adds the Results tab (the run view shows results in its own tab). */
export function TaskView({ row, results = true, defaultTab = "progress" }: { row: TaskRow; results?: boolean; defaultTab?: string }) {
  const active = row.status === "running" || row.status === "pending";
  const { data: t, error, refetch } = useTask(row.id, true);
  const live = t ? t.status === "running" || t.status === "pending" : active;
  const { data: relax } = useRelax(row.id, t?.relax.n_points ?? 0, live);
  const now = useNow(live ? 1000 : 60000);
  const qc = useQueryClient();
  const toast = useToast();
  const { ask, node } = useConfirm();
  const [sp, setSp] = useSearchParams();
  const cancel = useMutation({
    mutationFn: () => api.post(`/api/tasks/${row.id}/cancel`),
    onSuccess: () => {
      toast("info", "Cancelling task...");
      qc.invalidateQueries({ queryKey: ["job", row.job_id] });
    },
    onError: (e) => toast("error", String(e)),
  });
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!t) return <Skeleton className="h-96" />;
  const elapsed = t.started ? (t.ended ?? now) - t.started : t.elapsed_s;
  // the selected stage: the one in the URL, else what is running, else what failed, else the last one that finished
  const pick = (f: (x: Stage) => boolean) => t.stages.find(f)?.id;
  const stageId = (t.stages.some((x) => x.id === sp.get("stage")) ? sp.get("stage") : null) ?? pick((x) => x.status === "running") ?? pick((x) => x.status === "failed") ?? [...t.stages].reverse().find((x) => x.status === "ok")?.id ?? t.stages[0]?.id ?? "";
  const select = (id: string) => setSp((p) => { const n = new URLSearchParams(p); n.set("stage", id); return n; }, { replace: true });
  const bad = ["failed", "timeout", "crashed"].includes(t.status);
  const failedStage = t.stages.find((x) => x.status === "failed");
  const reason = failedStage?.detail ?? (t.outcome && t.outcome !== "failed" ? t.outcome : null) ?? t.warnings[0] ?? null;
  const extraWarnings = [...new Set(t.warnings)].filter((w) => w !== reason);
  const nVert = t.metrics.find((m) => m.name === "n_actual");
  return (
    <div className="space-y-5">
      <Card className="p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2.5 text-[17px]">
              CR {t.cr} <span className="font-normal text-muted">· {t.nflux} fluxons</span>
              <StatusBadge status={t.status} />
            </h2>
            <p className="mt-1 text-xs text-muted">
              {instOf(t.adapt)} · {t.method} · attempt {t.attempt} · {t.started ? `started ${dateTime(t.started)}` : "waiting for a slot"}
            </p>
          </div>
          {live && t.status === "running" && (
            <button
              className="btn btn-sm btn-danger"
              onClick={async () => {
                if (await ask({ title: "Cancel this task?", body: "The pipeline process group is stopped. Files already written stay on disk.", confirm: "Cancel task", danger: true })) cancel.mutate();
              }}
            >
              <Ban className="h-3.5 w-3.5" /> Cancel task
            </button>
          )}
        </div>
        <PipelineChips stages={t.stages} className="mt-4" selected={stageId} onSelect={select} />
        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          <Metric label="Elapsed" value={duration(elapsed)} />
          <Metric label="Stiffness" value={num(t.relax.latest?.stiffness)} sub={t.relax.threshold != null ? `target ${num(t.relax.threshold)}` : undefined} />
          <Metric label="Relax step" value={int(t.relax.latest?.step)} sub={t.relax.steps_per_s ? `${num(t.relax.steps_per_s, 3)} steps/s` : undefined} />
          <Metric label="Cycles" value={t.relax.cycles || "-"} sub={t.relax.cycle?.nc_curve != null ? `${t.relax.cycle.nc_curve} curvature fixes` : undefined} />
          <Metric label="Vertices" value={int(t.relax.latest?.vertices)} />
          <Metric label="Fluxons out" value={nVert ? String(nVert.value) : "-"} sub={nVert ? `of ${t.nflux} asked` : undefined} />
        </div>
      </Card>

      {bad && <Banner tone="error" title={failedStage ? `${failedStage.label} failed` : "Task failed"}>{reason ? `${reason}. ` : ""}Exit code {t.exit_code ?? "?"}. The Log tab shows where it stopped.</Banner>}
      {t.degraded && <Banner tone="warn" title="Progress is partly inferred">{t.degraded_reason ?? "Structured events stopped arriving, so stages are read from the log."}</Banner>}
      {extraWarnings.map((w) => (
        <Banner key={w} tone="warn">{w}</Banner>
      ))}

      <Tabs defaultValue={defaultTab}>
        <TabList>
          <Tab value="progress">Progress</Tab>
          <Tab value="log">Log</Tab>
          {results && (
            <Tab value="results">
              Results
            </Tab>
          )}
          <Tab value="sources">Evidence</Tab>
        </TabList>
        <TabPanel value="progress" className="space-y-6 pt-5">
          <section>
            <SectionTitle title="Stages" hint="Pick a stage above to see its progress and what it made." />
            <Card className="p-4">
              <StageTimeline task={t} />
            </Card>
          </section>
          <StagePanel
            task={t}
            stageId={stageId}
            onSelect={select}
            relaxNode={
            <Card className="p-4">
                  <RelaxCharts
                    points={relax?.points ?? []}
                    cycles={relax?.cycles ?? []}
                    idle={(() => {
                      const rs = t.stages.find((s) => s.id === "relax")?.status;
                      if (rs === "skipped") return { title: "Relaxation skipped", body: "This task did not run the relaxation stage." };
                      if (rs === "ok") return { title: "No relaxation steps recorded", body: "The stage finished without reporting steps, usually because an existing relaxed world was reused from the cache." };
                      if (rs === "failed") return { title: "Relaxation failed before reporting steps", body: "The Log tab shows where it stopped." };
                      if (!["pending", "running"].includes(t.status)) return { title: "Relaxation did not run", body: "The task ended before reaching the relaxation stage." };
                      return undefined;
                    })()}
                  />
                </Card>
            }
          />
        </TabPanel>
        <TabPanel value="log" className="pt-5">
          <LogPane taskId={t.id} active={live} />
        </TabPanel>
        {results && (
          <TabPanel value="results" className="pt-5">
            <ResultsPane task={t} />
          </TabPanel>
        )}
        <TabPanel value="sources" className="pt-5">
          <SourcesPanel task={t} />
        </TabPanel>
      </Tabs>
      {node}
    </div>
  );
}

export function JobDetail() {
  const { id, taskId } = useParams();
  const nav = useNavigate();
  const { data: job, error, refetch } = useJob(id);
  const rows = job?.tasks ?? [];
  const selected = useMemo(() => {
    if (!rows.length) return undefined;
    return (
      rows.find((r) => r.id === taskId) ??
      rows.find((r) => r.status === "running") ??
      rows.find((r) => ["failed", "timeout", "crashed"].includes(r.status)) ??
      rows[0]
    );
  }, [rows, taskId]);
  useEffect(() => {
    if (error && (error as { status?: number }).status === 404) nav("/jobs", { replace: true });
  }, [error, nav]);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!job) return <Skeleton className="h-96" />;
  return (
    <div className="animate-fade-in space-y-5">
      <div>
        <Link to="/jobs" className="mb-2 inline-flex items-center gap-1 text-xs text-muted hover:text-ink">
          <ArrowLeft className="h-3.5 w-3.5" /> Jobs
        </Link>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="flex items-center gap-3 text-[22px]">
              {job.profile} <StatusBadge status={job.status} />
            </h1>
            <p className="mt-1 text-[13px] text-muted">
              Started {dateTime(job.created)} · backend {job.backend}
              {job.extra.scenario && ` · scenario ${job.extra.scenario}`}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Link to={`/run/${job.id}/progress${selected ? `?task=${selected.id}` : ""}`} className="btn btn-sm">
              <PanelsTopLeft className="h-3.5 w-3.5" /> Open as run
            </Link>
            <JobActions job={job} onDeleted={() => nav("/jobs")} />
          </div>
        </div>
      </div>

      {rows.length > 1 && (
        <div className="-mx-1 flex gap-2 overflow-x-auto px-1 pb-1" role="tablist" aria-label="Tasks">
          {rows.map((r) => (
            <Link
              key={r.id}
              role="tab"
              aria-selected={r.id === selected?.id}
              to={`/jobs/${job.id}/${r.id}`}
              replace
              className={cn(
                "flex shrink-0 items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors",
                r.id === selected?.id ? "border-accent/50 bg-raised text-ink" : "border-line text-muted hover:text-ink",
              )}
            >
              <StatusDot status={r.status} />
              CR {r.cr} · {r.nflux}
            </Link>
          ))}
        </div>
      )}

      {selected ? <TaskView key={selected.id} row={selected} /> : <EmptyState title="This job has no tasks" />}
    </div>
  );
}

export function StatusDot({ status }: { status: string }) {
  const c = status === "ok" ? "bg-ok" : status === "running" ? "bg-run animate-pulse" : ["failed", "timeout", "crashed"].includes(status) ? "bg-fail" : "bg-faint";
  return <span className={cn("h-1.5 w-1.5 rounded-full", c)} />;
}

export type { TaskState };
