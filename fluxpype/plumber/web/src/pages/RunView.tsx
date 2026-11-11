import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Play, Plus, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, Navigate, NavLink, useNavigate, useParams, useSearchParams } from "react-router-dom";

import { ResultsPane } from "../components/ResultsPane";
import { RunForm, changedOverrides, emptyDraft, jobBody, pickedTasks, taskKey, type RunDraft } from "../components/RunForm";
import { Badge, Banner, Card, EmptyState, ErrorNote, Skeleton, StatusBadge, useToast } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { dateTime } from "../lib/format";
import { useJob, useJobConfig, useJobs, useProfile, useProfiles, useStatus, useTask } from "../lib/hooks";
import { loadDraft, saveDraft } from "../lib/layoutMode";
import type { Job, TaskRow } from "../lib/types";
import { RunFiles, RunSettings } from "../components/RunTabs";
import { StatusDot, TaskView } from "./JobDetail";
import { JobActions } from "./Jobs";

const enc = encodeURIComponent;
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

const TABS = [
  { id: "settings", label: "Settings" },
  { id: "progress", label: "Run" },
  { id: "cache", label: "Cache" },
  { id: "results", label: "Results" },
] as const;
type TabId = (typeof TABS)[number]["id"];

function runLabel(j: Job) {
  const crs = [...new Set(j.tasks.map((t) => t.cr))];
  const n = [...new Set(j.tasks.map((t) => t.nflux))];
  return `${j.profile} · CR ${crs.slice(0, 3).join(", ")}${crs.length > 3 ? ` +${crs.length - 3}` : ""} · ${n.join("/")} fluxons`;
}

/* ------------------------------------------------------------------ header: which run */
function RunPicker({ current }: { current?: string }) {
  const { data } = useJobs();
  const nav = useNavigate();
  const jobs = data?.jobs ?? [];
  return (
    <div className="flex flex-wrap items-center gap-2">
      <select
        className="input h-9 w-auto min-w-0 max-w-full py-0 text-[13px] sm:max-w-[460px]"
        aria-label="Choose a run"
        value={current ?? ""}
        onChange={(e) => e.target.value && nav(`/run/${e.target.value}/progress`)}
      >
        {!current && <option value="">Choose a run...</option>}
        {jobs.map((j) => (
          <option key={j.id} value={j.id}>
            {dateTime(j.created)} · {runLabel(j)} · {j.status}
          </option>
        ))}
      </select>
      <Link to="/run/new" className="btn btn-sm">
        <Plus className="h-3.5 w-3.5" /> New run
      </Link>
    </div>
  );
}

function TabBar({ base, tab, disabled, search }: { base: string; tab: TabId; disabled?: boolean; search?: string }) {
  return (
    <div className="flex gap-1 overflow-x-auto border-b hair" role="tablist" aria-label="Run">
      {TABS.map((t) => {
        const off = disabled && t.id !== "settings";
        const cls = cn(
          "-mb-px shrink-0 border-b-2 px-3.5 py-2 text-[13px] font-medium transition-colors",
          t.id === tab ? "border-accent text-ink" : "border-transparent text-muted hover:text-ink",
          off && "pointer-events-none opacity-40",
        );
        return off ? (
          <span key={t.id} className={cls} title="Available once the run is launched" aria-disabled>
            {t.label}
          </span>
        ) : (
          <NavLink key={t.id} role="tab" aria-selected={t.id === tab} to={`${base}/${t.id}${search ?? ""}`} replace className={cls}>
            {t.label}
          </NavLink>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ Results tab */
function RunResults({ row }: { row: TaskRow }) {
  const active = row.status === "running" || row.status === "pending";
  const { data: t, error, refetch } = useTask(row.id, active);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!t) return <Skeleton className="h-96" />;
  return <ResultsPane task={t} />;
}

/* ------------------------------------------------------------------ one run */
export function RunPage() {
  const { id, tab: tabParam } = useParams();
  const [params] = useSearchParams();
  const nav = useNavigate();
  const { data: job, error, refetch } = useJob(id);
  const tab = (TABS.some((t) => t.id === tabParam) ? tabParam : "progress") as TabId;
  const rows = job?.tasks ?? [];
  const taskId = params.get("task");
  const selected = useMemo(
    () => rows.find((r) => r.id === taskId) ?? rows.find((r) => r.status === "running") ?? rows.find((r) => ["failed", "timeout", "crashed"].includes(r.status)) ?? rows[0],
    [rows, taskId],
  );
  useEffect(() => {
    if (error && (error as { status?: number }).status === 404) nav("/run", { replace: true });
  }, [error, nav]);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!job) return <Skeleton className="h-96" />;
  const search = selected && rows.length > 1 ? `?task=${selected.id}` : "";
  return (
    <div className="animate-fade-in space-y-5">
      <RunPicker current={job.id} />
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="flex flex-wrap items-center gap-3 text-[22px]">
            {job.profile} <StatusBadge status={job.status} />
          </h1>
          <p className="mt-1 text-[13px] text-muted">
            Started {dateTime(job.created)} · backend {job.backend} · {rows.length} task{rows.length === 1 ? "" : "s"}
            {job.extra.scenario && ` · scenario ${job.extra.scenario}`}
          </p>
        </div>
        <JobActions job={job} onDeleted={() => nav("/run")} />
      </div>
      {rows.length > 1 && (
        <div className="-mx-1 flex gap-2 overflow-x-auto px-1 pb-1" role="tablist" aria-label="Tasks">
          {rows.map((r) => (
            <Link
              key={r.id}
              role="tab"
              aria-selected={r.id === selected?.id}
              to={`/run/${job.id}/${tab}?task=${r.id}`}
              replace
              className={cn("flex shrink-0 items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors", r.id === selected?.id ? "border-accent/50 bg-raised text-ink" : "border-line text-muted hover:text-ink")}
            >
              <StatusDot status={r.status} />
              CR {r.cr} · {r.nflux}
            </Link>
          ))}
        </div>
      )}
      <TabBar base={`/run/${job.id}`} tab={tab} search={search} />
      <div className="pt-1">
        {tab === "settings" && <RunSettings job={job} />}
        {tab === "progress" && (selected ? <TaskView key={selected.id} row={selected} results={false} /> : <EmptyState title="This run has no tasks" />)}
        {tab === "cache" && <RunFiles job={job} task={selected} />}
        {tab === "results" && (selected ? <RunResults key={selected.id} row={selected} /> : <EmptyState title="This run has no tasks" />)}
      </div>
    </div>
  );
}

/** /run: open the most recent run, or start a new one. */
export function RunIndex() {
  const { data, error, refetch } = useJobs();
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!data) return <Skeleton className="h-96" />;
  const latest = data.jobs[0];
  return <Navigate to={latest ? `/run/${latest.id}/progress` : "/run/new"} replace />;
}

/* ------------------------------------------------------------------ a run before launch */
interface StoredDraft extends RunDraft {
  preset: string;
}

export function NewRun() {
  const [params, setParams] = useSearchParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const toast = useToast();
  const { data: status } = useStatus();
  const { data: presets } = useProfiles();
  const fromJob = params.get("from") ?? undefined;
  const { data: srcCfg } = useJobConfig(fromJob);
  const { data: srcJob } = useJob(fromJob);
  const [draft, setDraftState] = useState<StoredDraft | null>(() => {
    if (params.get("preset") || fromJob) return null;
    return loadDraft<StoredDraft>();
  });
  const setDraft = useCallback((d: StoredDraft | null) => {
    setDraftState(d);
    saveDraft(d);
  }, []);

  // ?preset=X starts a fresh draft on that preset
  useEffect(() => {
    const p = params.get("preset");
    if (p) {
      setDraft({ preset: p, ...emptyDraft() });
      setParams({}, { replace: true });
    }
  }, [params, setParams, setDraft]);
  // ?from=<job> copies that run's preset, changes and tasks
  useEffect(() => {
    if (!fromJob || !srcCfg || !srcJob) return;
    setDraft({
      preset: srcCfg.profile,
      ...emptyDraft(),
      overrides: { ...srcCfg.overrides },
      picked: srcJob.tasks.map(taskKey),
      scenario: srcJob.extra.scenario ?? "ok",
      speed: srcJob.extra.speed ?? "20",
    });
    setParams({}, { replace: true });
    toast("info", `Copied ${srcJob.profile} (${srcJob.id})`);
  }, [fromJob, srcCfg, srcJob, setParams, setDraft, toast]);

  const visible = (presets?.profiles ?? []).filter((p) => !p.hidden || p.name === draft?.preset);
  const { data: detail, error } = useProfile(draft?.preset);
  const sim = !!status?.sandboxed;

  // picked tasks the preset no longer has are dropped; if none remain, fall back to all
  const effective: RunDraft | null = useMemo(() => {
    if (!draft || !detail) return null;
    if (draft.picked === null) return draft;
    const have = new Set(detail.tasks.map(taskKey));
    const kept = draft.picked.filter((k) => have.has(k));
    return { ...draft, picked: kept.length ? kept : null };
  }, [draft, detail]);

  const launch = useMutation({
    mutationFn: () => api.post<Job>("/api/jobs", jobBody(detail!, effective!, sim)),
    onSuccess: (job) => {
      setDraft(null);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["overview"] });
      toast("ok", `Started ${job.tasks.length} task${job.tasks.length === 1 ? "" : "s"}`);
      nav(`/run/${job.id}/progress`);
    },
    onError: (e) => toast("error", errText(e)),
  });

  if (fromJob && !draft) return <Skeleton className="h-96" />;
  const nTasks = detail && effective ? pickedTasks(detail, effective).length : 0;
  const nChanged = detail && effective ? changedOverrides(detail, effective).length : 0;

  return (
    <div className="animate-fade-in space-y-5">
      <RunPicker />
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-3 text-[22px]">
            New run <Badge tone="accent">draft</Badge>
          </h1>
          <p className="mt-1 text-[13px] text-muted">Kept in this browser until you launch it or discard it.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {draft && (
            <button className="btn btn-sm" onClick={() => setDraft(null)}>
              <X className="h-3.5 w-3.5" /> Discard draft
            </button>
          )}
          <button className="btn btn-primary" disabled={!detail || !effective || nTasks === 0 || launch.isPending || (detail?.errors.length ?? 0) > 0} onClick={() => launch.mutate()}>
            <Play className="h-3.5 w-3.5" /> {launch.isPending ? "Starting..." : `Launch${nTasks ? ` ${nTasks} task${nTasks === 1 ? "" : "s"}` : ""}`}
          </button>
        </div>
      </div>
      <TabBar base="/run/new" tab="settings" disabled />
      <div className="space-y-5 pt-1">
        <Card className="flex flex-wrap items-center gap-3 p-4">
          <label className="flex min-w-0 flex-1 items-center gap-3 text-[13px]">
            <span className="label shrink-0">Preset</span>
            <select
              className="input h-9 max-w-sm py-0"
              value={draft?.preset ?? ""}
              onChange={(e) => setDraft(e.target.value ? { preset: e.target.value, ...emptyDraft() } : null)}
            >
              <option value="">Choose a preset...</option>
              {visible.map((p) => (
                <option key={p.name} value={p.name} disabled={p.errors.length > 0}>
                  {p.name} ({p.task_count} task{p.task_count === 1 ? "" : "s"}){p.errors.length ? " · has errors" : ""}
                </option>
              ))}
            </select>
          </label>
          {draft && (
            <span className="text-[12px] text-muted">
              {nChanged > 0 ? `${nChanged} change${nChanged === 1 ? "" : "s"} from the preset` : "Preset as saved"}
              {" · "}
              <Link to={`/presets/${enc(draft.preset)}`} className="hover:text-ink">
                open preset
              </Link>
            </span>
          )}
        </Card>
        {!draft && <EmptyState title="Pick a preset to start from" body="Its settings fill in below. Change what you need for this run, choose tasks, then launch." />}
        {draft && error && <ErrorNote error={error} />}
        {draft && !error && !detail && <Skeleton className="h-80" />}
        {detail && detail.errors.length > 0 && (
          <Banner tone="error" title="This preset can't run">
            {detail.errors.join("; ")}
          </Banner>
        )}
        {detail && effective && <RunForm detail={detail} draft={effective} onChange={(d) => setDraft({ ...d, preset: draft!.preset })} wide />}
      </div>
    </div>
  );
}
