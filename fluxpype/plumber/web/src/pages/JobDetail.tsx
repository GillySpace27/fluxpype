import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Ban, Copy, Sparkles } from "lucide-react";
import { openAsk } from "../components/AskBar";
import { useEffect, useMemo } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { LogPane } from "../components/LogPane";
import { PipelineChips } from "../components/PipelineChips";
import { ResultsPane } from "../components/ResultsPane";
import { RunFiles, RunSettings } from "../components/RunTabs";
import { SameRotation } from "../components/SameRotation";
import { StagePanel } from "../components/StagePanel";
import { RelaxCharts } from "../components/RelaxCharts";
import { SourcesPanel } from "../components/SourcesPanel";
import { StageTimeline, useProgress } from "../components/StageTimeline";
import { Banner, Card, EmptyState, ErrorNote, Skeleton, StatusBadge, StatusIcon, Tab, TabList, TabPanel, Tabs, useConfirm, useToast } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { dateTime, duration, instOf, int, num } from "../lib/format";
import { useJob, useNow, useRelax, useTask, useTaskResults } from "../lib/hooks";
import type { Job, Stage, TaskRow, TaskState } from "../lib/types";
import { JobActions } from "./Jobs";

/** The stage chips as radio selectors; while wind runs its chip fills with the progress. */
function Strip({ task, selected, onSelect }: { task: TaskState; selected: string; onSelect: (id: string) => void }) {
  const { wind } = useProgress(task);
  const windRunning = task.stages.find((s) => s.id === "wind")?.status === "running";
  // a finished stage whose files all predate this run was reused from the cache, so its chip goes purple
  const { data } = useTaskResults(task.id, task.status === "running" || task.status === "pending");
  const reused = new Set<string>();
  const byStage = new Map<string, { origin: string | null }[]>();
  data?.products.forEach((p) => byStage.set(p.id === "relaxed_world" ? "relax" : p.id, p.files));
  data?.plots.forEach((f) => f.stage && byStage.set(f.stage, [...(byStage.get(f.stage) ?? []), f]));
  byStage.forEach((files, id) => {
    if (files.length && files.every((f) => f.origin === "reused") && task.stages.find((s) => s.id === id)?.status === "ok") reused.add(id);
  });
  return <PipelineChips stages={task.stages} className="mt-3" selected={selected} onSelect={onSelect} fill={windRunning && wind ? { wind: wind.frac } : undefined} reused={reused} />;
}

/** One task: header with metrics and banners, then tabs. `results` adds the Results tab (the run view shows results in its own tab). */
export function TaskView({ row, job, results = true, defaultTab = "progress" }: { row: TaskRow; job?: Job; results?: boolean; defaultTab?: string }) {
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
  // ?view=compact swaps the strip for the stage rail beside the panel (kept for a compact view elsewhere)
  const compact = sp.get("view") === "compact";
  // the tab lives in the URL (?tab=) so a link opens on it
  const tabs = ["progress", "log", ...(results ? ["results"] : []), ...(job ? ["files", "settings"] : []), "sources"];
  const tab = tabs.includes(sp.get("tab") ?? "") ? (sp.get("tab") as string) : defaultTab;
  const setTab = (v: string) => setSp((p) => { const n = new URLSearchParams(p); n.set("tab", v); return n; }, { replace: true });
  const bad = ["failed", "timeout", "crashed"].includes(t.status);
  const failedStage = t.stages.find((x) => x.status === "failed");
  const reason = failedStage?.detail ?? (t.outcome && t.outcome !== "failed" ? t.outcome : null) ?? t.warnings[0] ?? null;
  const extraWarnings = [...new Set(t.warnings)].filter((w) => w !== reason);
  const nVert = t.metrics.find((m) => m.name === "n_actual");
  // one line of numbers instead of six tiles; a metric with no value yet is left out
  const metrics = [
    ["Elapsed", duration(elapsed)],
    ["Stiffness", num(t.relax.latest?.stiffness) + (t.relax.threshold != null ? ` / ${num(t.relax.threshold)}` : "")],
    ["Step", int(t.relax.latest?.step)],
    ["Cycles", t.relax.cycles ? String(t.relax.cycles) : "-"],
    ["Vertices", int(t.relax.latest?.vertices)],
    ["Fluxons out", nVert ? `${nVert.value} of ${t.nflux}` : "-"],
  ].filter(([, v]) => v && !v.startsWith("-"));
  const warnings = [...(t.degraded ? [t.degraded_reason ?? "Structured events stopped arriving, so stages are read from the log."] : []), ...extraWarnings];
  // the stage list stays open while it is useful (running, failed) and on wide screens; on a phone a finished task shows just the selected row
  const railOpen = live || bad || window.matchMedia("(min-width: 1024px)").matches;
  return (
    <div className="space-y-4">
      <Card className="p-4">
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
        {!compact && <Strip task={t} selected={stageId} onSelect={select} />}
        <p className="num mt-3 flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-muted">
          {metrics.map(([k, v]) => (
            <span key={k}>
              {k} <span className="font-medium text-ink">{v}</span>
            </span>
          ))}
        </p>
        {bad && (
          <button className="mt-2 block text-left text-[13px] text-fail hover:underline" onClick={() => failedStage && select(failedStage.id)}>
            {failedStage ? `${failedStage.label} failed` : "Task failed"}: {reason ? `${reason}. ` : ""}Exit code {t.exit_code ?? "?"}. The Log tab shows where it stopped.
          </button>
        )}
        {bad && (
          <button
            className="btn btn-sm mt-2"
            onClick={() =>
              openAsk(`Why did task ${t.id} of job ${t.job_id} (CR ${t.cr}, ${t.nflux} fluxons) ${failedStage ? `fail at the ${failedStage.label} stage` : `end as ${t.status}`}? Read its log and suggest a fix.`)
            }
          >
            <Sparkles className="h-3.5 w-3.5" /> Ask why this failed
          </button>
        )}
        {warnings.length > 0 && (
          <details className="mt-2 text-[13px]">
            <summary className="cursor-pointer text-run">{warnings.length} warning{warnings.length === 1 ? "" : "s"}</summary>
            <div className="mt-2 space-y-2">
              {warnings.map((w) => (
                <Banner key={w} tone="warn">{w}</Banner>
              ))}
            </div>
          </details>
        )}
      </Card>

      <Tabs value={tab} onValueChange={setTab}>
        <TabList>
          <Tab value="progress">Progress</Tab>
          <Tab value="log">Log</Tab>
          {results && (
            <Tab value="results">
              Results
            </Tab>
          )}
          {job && <Tab value="files">Files</Tab>}
          {job && <Tab value="settings">Settings used</Tab>}
          <Tab value="sources">Evidence</Tab>
        </TabList>
        {compact ? (
          <TabPanel value="progress" className="grid gap-4 pt-4 lg:grid-cols-[22rem_minmax(0,1fr)]">
          <Card className="p-3 lg:sticky lg:top-20 lg:self-start">
            <details open={railOpen}>
              <summary className="flex cursor-pointer items-center gap-2 text-[13px]">
                <span className="label">Stages</span>
                <StatusIcon status={t.stages.find((x) => x.id === stageId)?.status ?? "pending"} className="h-3.5 w-3.5" />
                {t.stages.find((x) => x.id === stageId)?.label}
              </summary>
              <div className="mt-2">
                <StageTimeline task={t} selected={stageId} onSelect={select} />
              </div>
            </details>
          </Card>
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
        ) : (
          <TabPanel value="progress" className="space-y-4 pt-4">
            <Card className="p-3">
              <details open={live || bad}>
                <summary className="flex cursor-pointer items-center gap-2 text-[13px]">
                  <span className="label">Timeline</span>
                  <span className="num text-xs text-muted">{duration(elapsed)}</span>
                </summary>
                <div className="mt-2">
                  <StageTimeline task={t} selected={stageId} onSelect={select} />
                </div>
              </details>
            </Card>
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
            <SameRotation task={t} />
          </TabPanel>
        )}
        <TabPanel value="log" className="pt-5">
          <LogPane taskId={t.id} active={live} />
        </TabPanel>
        {results && (
          <TabPanel value="results" className="pt-5">
            <ResultsPane task={t} />
          </TabPanel>
        )}
        {job && (
          <TabPanel value="files" className="pt-5">
            <RunFiles job={job} task={row} />
          </TabPanel>
        )}
        {job && (
          <TabPanel value="settings" className="pt-5">
            <RunSettings job={job} />
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
            <Link to={`/jobs/new?from=${job.id}`} className="btn btn-sm">
              <Copy className="h-3.5 w-3.5" /> Duplicate as new run
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

      {selected ? <TaskView key={selected.id} row={selected} job={job} /> : <EmptyState title="This job has no tasks" />}
    </div>
  );
}

export function StatusDot({ status }: { status: string }) {
  const c = status === "ok" ? "bg-ok" : status === "running" ? "bg-run animate-pulse" : ["failed", "timeout", "crashed"].includes(status) ? "bg-fail" : "bg-faint";
  return <span className={cn("h-1.5 w-1.5 rounded-full", c)} />;
}

export type { TaskState };
